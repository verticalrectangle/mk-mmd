//! mkr: the scene bake `mk build` writes, drawn on the GPU as Blender's EEVEE Next draws it.
//!
//!   mkr render --bake DIR --output NAME [--frames SPEC] --out DIR [--layers FILE.json] [--samples 64] [--percent 100]
//!              [--motion-blur] [--shutter 0.35] [--params FILE.json] [--threads N]
//!   mkr info --bake DIR [--output NAME]... [--frames SPEC] [--layers FILE.json]
//!   mkr params
//!
//! `--layers` is a request for the layers mk render writes beside the frames (mkr/mkr/src/layers.rs). Each command
//! prints one JSON report on stdout; progress and diagnostics go to stderr.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::json;

use mkr::bake::Bake;
use mkr::frames::parse_spec;
use mkr::gpu::Gpu;
use mkr::info::{blocker, info};
use mkr::layers::{Counts, Request};
use mkr::looks_gpu::LookCounts;
use mkr::materials::Support;
use mkr::output::Writer;
use mkr::params::Params;
use mkr::render::{Job, Options, Renderer, Stats};

/// Frames drawn before their layers and looks: a chunk's lit frames, then its flat layers and look passes. A chunk
/// boundary costs a frame's overlap of the CPU and the GPU (Renderer::render drains its last frame).
const CHUNK: usize = 48;

/// The stats of two runs of Renderer::render, one after the other.
fn merged(mut a: Stats, b: Stats) -> Stats {
    a.rendered += b.rendered;
    a.gpu_ms.extend(b.gpu_ms);
    a.phases.extend(b.phases);
    a.encode_ms.extend(b.encode_ms);
    a.stage_ms.extend(b.stage_ms);
    a.submit_ms.extend(b.submit_ms);
    a.wait_ms.extend(b.wait_ms);
    a.collect_ms.extend(b.collect_ms);
    a.warnings.extend(b.warnings);
    a
}

/// The look counts of two runs of Renderer::draw_looks.
fn look_sum(mut a: LookCounts, b: LookCounts) -> LookCounts {
    a.frames += b.frames;
    a.plates += b.plates;
    a.mattes += b.mattes;
    a.silhouette += b.silhouette;
    a.vector += b.vector;
    a.seconds += b.seconds;
    a
}

#[derive(Parser)]
#[command(name = "mkr", about = "mk-mmd's renderer: the scene bake drawn on the GPU at EEVEE parity")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Draw frames of one output as 8-bit RGB PNGs, OUT/<frame:05>.png, and the layers a request asks for beside them.
    Render {
        #[arg(long)]
        bake: PathBuf,
        #[arg(long)]
        output: String,
        /// Blender frames: a, a:b (inclusive), a:b:step, joined by commas (none: only the request's layers).
        #[arg(long)]
        frames: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 64)]
        samples: u32,
        #[arg(long, default_value_t = 100)]
        percent: u32,
        #[arg(long)]
        motion_blur: bool,
        /// Motion blur shutter in frames, centred on the frame.
        #[arg(long, default_value_t = 0.35)]
        shutter: f32,
        /// Tunables for hill climbing (`mkr params` lists them).
        #[arg(long)]
        params: Option<PathBuf>,
        /// PNG writer threads.
        #[arg(long)]
        threads: Option<usize>,
        /// The layers to draw (mk render writes the request: layers.rs).
        #[arg(long)]
        layers: Option<PathBuf>,
    },
    /// Which frames of each output mkr draws, why Blender must draw the others, and what mkr cannot draw; with
    /// `--layers`, the same for every layer the request lists.
    Info {
        #[arg(long)]
        bake: PathBuf,
        #[arg(long)]
        output: Vec<String>,
        #[arg(long)]
        frames: Option<String>,
        #[arg(long)]
        layers: Option<PathBuf>,
    },
    /// The tunables `--params` takes, with their defaults.
    Params,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Render { bake, output, frames, out, samples, percent, motion_blur, shutter, params, threads, layers } => {
            render(bake, output, frames, out, samples, percent, motion_blur.then_some(shutter), params, threads, layers)
        }
        Cmd::Info { bake, output, frames, layers } => run_info(bake, output, frames, layers),
        Cmd::Params => serde_json::to_value(Params::default()).map_err(Into::into),
    };
    match result {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("mkr: {e:#}");
            std::process::exit(1);
        }
    }
}

fn run_info(bake: PathBuf, outputs: Vec<String>, frames: Option<String>, layers: Option<PathBuf>)
            -> Result<serde_json::Value> {
    let bake = Bake::open(&bake)?;
    let outputs = if outputs.is_empty() { bake.doc.outputs.iter().map(|o| o.name.clone()).collect() } else { outputs };
    let frames = match frames {
        Some(s) => parse_spec(&s)?,
        None => (0..bake.doc.frames as i64).map(|k| bake.doc.frame0 + k).collect(),
    };
    let req = layers.map(|p| Request::load(&p)).transpose()?;
    Ok(serde_json::to_value(info(&bake, &outputs, &frames, req.as_ref())?)?)
}

#[allow(clippy::too_many_arguments)]
fn render(bake: PathBuf, output: String, frames: Option<String>, out: PathBuf, samples: u32, percent: u32,
          shutter: Option<f32>, params: Option<PathBuf>, threads: Option<usize>, layers: Option<PathBuf>)
          -> Result<serde_json::Value> {
    let t0 = Instant::now();
    if samples == 0 {
        bail!("--samples must be at least 1");
    }
    if !(1..=1000).contains(&percent) {
        bail!("--percent {percent}: 1 to 1000");
    }
    if let Some(s) = shutter {
        if !(0.0..=2.0).contains(&s) {
            bail!("--shutter {s}: 0 to 2 frames");
        }
    }
    let params = Params::load(params.as_deref())?;
    let bake = Bake::open(&bake)?;
    bake.lit()?;
    let size = bake.output(&output).with_context(|| format!("the bake has no output {output}"))?.size;
    let frames = match &frames {
        Some(s) => parse_spec(s)?,
        None if layers.is_some() => Vec::new(),
        None => bail!("--frames: no frames, and no --layers"),
    };
    let req = layers.map(|p| Request::load(&p)).transpose()?;
    std::fs::create_dir_all(&out).with_context(|| format!("cannot make {}", out.display()))?;
    let support = Support::of(&bake)?;
    let mut warnings = Vec::new();
    let mut jobs = Vec::new();
    let looks = req.as_ref().and_then(|r| r.looks_of(&output));
    let mut look_frames = Vec::new();
    for f in frames {
        let Some(k) = bake.k(f) else {
            warnings.push(format!("frame {f}: outside the clip, skipped"));
            continue;
        };
        match blocker(&bake, &support, &output, f, looks)? {
            // a frame in a look: its look job's passes (looks.rs)
            Ok(_) if looks.is_some_and(|l| l.frame_job(f).is_some()) => look_frames.push(f),
            Ok(camera) => jobs.push(Job { frame: f, k, camera, path: out.join(format!("{f:05}.png")), hidden: Vec::new() }),
            Err(why) => {
                // a look or a medium window is drawn on top of the lit picture: mkr can still draw the lit frame
                let shot = mkr::frames::shot_at(&bake.doc.shots, f);
                let camera = shot.and_then(|s| s.cameras.get(&output)).cloned();
                let drawable = camera.as_ref().is_some_and(|c| {
                    bake.camera(c).is_some_and(|r| r.fisheye.is_none())
                        && bake.doc.lit.as_ref().is_some_and(|l| l.cameras.get(c).is_none_or(|x| x.kind == "PERSP" || x.kind == "ORTHO"))
                }) && (why.contains(" look") || why.contains("[[medium]]"));
                if drawable {
                    warnings.push(format!("frame {f}: {why} (drawn lit; mk render gives it to Blender)"));
                    jobs.push(Job { frame: f, k, camera: camera.unwrap(), path: out.join(format!("{f:05}.png")), hidden: Vec::new() });
                } else {
                    warnings.push(format!("frame {f}: {why}, skipped"));
                }
            }
        }
    }
    // the layers: plates and bare frames are lit frames too; points are written now; the flat passes come after, then
    // the looks' passes
    let mut counts = Counts::default();
    let mut flats = Vec::new();
    let mut look_ids = Vec::new();
    if let Some(req) = &req {
        let lit = mkr::layers::lit_jobs(&bake, &support, &output, req, &out, &mut warnings)?;
        for j in &lit {
            if j.hidden.is_empty() { counts.plate += 1 } else { counts.bare += 1 }
        }
        jobs.extend(lit);
        flats = mkr::layers::plan_flat(&bake, &support, &output, req, &out, &mut counts, &mut warnings)?;
        look_ids = mkr::layers::look_jobs(&bake, &support, &output, req, &look_frames, &mut warnings)?;
    }
    jobs.sort_by_key(|j| j.k);
    if jobs.is_empty() && flats.is_empty() && look_ids.is_empty() {
        // nothing left for the GPU: points alone, or every layer skipped
        return Ok(json!({
            "rendered": 0,
            "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
            "s_per_frame": 0.0,
            "gpu_ms_per_frame": serde_json::Value::Null,
            "setup_seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
            "layers": counts,
            "flat_seconds": 0.0,
            "adapter": serde_json::Value::Null,
            "peak_rss_mb": peak_rss_mb(),
            "warnings": warnings,
        }));
    }
    let gpu = Gpu::new()?;
    let mut renderer = Renderer::new(gpu, &bake, params)?;
    let adapter = renderer.adapter().to_string();
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1));
    let mut writer = Writer::new(threads)?;
    let t_draw = Instant::now();
    let opts = Options { samples, percent, shutter };
    // a chunk of frames at a time: their lit frames, then their flat layers and their looks' passes, so every file of
    // a frame is on disk soon after its frame is drawn (mk post --follow takes them while the next ones draw)
    let kept = req.as_ref().and_then(|r| r.keep.as_ref()).map(|k| k.materials.clone()).unwrap_or_default();
    let mut kit = renderer.flat_kit(&output, &flats, &kept)?;
    flats.sort_by_key(|f| f.k);
    let looks_req = match (look_ids.is_empty(), &req) {
        (false, Some(r)) => {
            let lr = r.looks.as_ref().context("the request has look jobs and no looks")?;
            let spool = lr.spool.as_deref().context("the request's looks name no spool to write their passes to")?;
            Some((r.looks_of(&output).context("no looks of this output")?, spool, lr.limit))
        }
        _ => None,
    };
    let look_k = |id: usize| -> usize {
        looks_req.as_ref().and_then(|(l, ..)| l.jobs.get(id)).and_then(|j| bake.k(j.frame)).unwrap_or(0)
    };
    look_ids.sort_by_key(|&id| look_k(id));
    let mut ks: Vec<usize> = jobs.iter().map(|j| j.k).chain(flats.iter().map(|f| f.k)).chain(look_ids.iter().map(|&i| look_k(i)))
        .collect();
    ks.sort_unstable();
    ks.dedup();
    let (mut j0, mut f0, mut l0) = (0, 0, 0);
    let (mut draw_s, mut flat_s) = (0.0, 0.0);
    let mut stats: Option<Stats> = None;
    let mut looked: Option<LookCounts> = None;
    for chunk in ks.chunks(CHUNK) {
        let last = chunk[chunk.len() - 1];
        let j1 = jobs.partition_point(|j| j.k <= last);
        if j1 > j0 {
            let t = Instant::now();
            let s = renderer.render(&output, size, &jobs[j0..j1], &opts, &mut writer)?;
            stats = Some(match stats.take() { Some(a) => merged(a, s), None => s });
            draw_s += t.elapsed().as_secs_f64();
            j0 = j1;
        }
        let f1 = flats.partition_point(|f| f.k <= last);
        if let (true, Some(kit)) = (f1 > f0, kit.as_mut()) {
            let t = Instant::now();
            let done = renderer.draw_flats(kit, &output, size, percent, &flats[f0..f1], &mut writer)?;
            counts.figure += done.figure;
            counts.keep += done.keep;
            counts.screen += done.screen;
            flat_s += t.elapsed().as_secs_f64();
            f0 = f1;
        }
        let l1 = look_ids.partition_point(|&id| look_k(id) <= last);
        if let (true, Some((l, spool, limit))) = (l1 > l0, looks_req.as_ref()) {
            let c = renderer.draw_looks(&output, size, percent, l, &look_ids[l0..l1], spool, *limit)?;
            looked = Some(match looked.take() { Some(a) => look_sum(a, c), None => c });
            l0 = l1;
        }
    }
    writer.finish()?;
    let stats = stats.unwrap_or_else(|| Stats {
        rendered: 0, gpu_ms: Vec::new(), phases: Vec::new(), encode_ms: Vec::new(), stage_ms: Vec::new(),
        submit_ms: Vec::new(), wait_ms: Vec::new(), collect_ms: Vec::new(), warnings: Vec::new(),
    });
    let look_counts = match looked {
        Some(c) => serde_json::to_value(c)?,
        None => serde_json::Value::Null,
    };
    warnings.extend(stats.warnings);
    let gpu_ms = if stats.gpu_ms.is_empty() {
        serde_json::Value::Null
    } else {
        json!((stats.gpu_ms.iter().sum::<f64>() / stats.gpu_ms.len() as f64 * 100.0).round() / 100.0)
    };
    // the mean GPU milliseconds per frame of each phase (pose, shadow maps, prepass, ...)
    let phases = if stats.phases.is_empty() {
        serde_json::Value::Null
    } else {
        let n = stats.phases.len() as f64;
        let mut m = serde_json::Map::new();
        for (i, name) in mkr::render::PHASES.iter().enumerate() {
            let v = stats.phases.iter().map(|p| p[i]).sum::<f64>() / n;
            m.insert(name.to_string(), json!((v * 100.0).round() / 100.0));
        }
        serde_json::Value::Object(m)
    };
    let n = stats.rendered.max(1) as f64;
    let rendered = stats.rendered - counts.plate - counts.bare;
    Ok(json!({
        "rendered": rendered,
        "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        "s_per_frame": (draw_s / n * 1000.0).round() / 1000.0,
        "gpu_ms_per_frame": gpu_ms,
        "gpu_ms_phases": phases,
        "cpu_ms_encode": mean_ms(&stats.encode_ms),
        "cpu_ms_stage": mean_ms(&stats.stage_ms),
        "cpu_ms_submit": mean_ms(&stats.submit_ms),
        "cpu_ms_wait": mean_ms(&stats.wait_ms),
        "cpu_ms_collect": mean_ms(&stats.collect_ms),
        "setup_seconds": ((t_draw - t0).as_secs_f64() * 100.0).round() / 100.0,
        "compile_seconds": (renderer.compile_seconds * 100.0).round() / 100.0,
        "layers": counts,
        "looks": look_counts,
        "flat_seconds": (flat_s * 100.0).round() / 100.0,
        "adapter": adapter,
        "peak_rss_mb": peak_rss_mb(),
        "warnings": warnings,
    }))
}

/// The mean of per-frame milliseconds, to 0.1 ms (null when nothing was timed).
fn mean_ms(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    json!((v.iter().sum::<f64>() / v.len() as f64 * 10.0).round() / 10.0)
}

/// The process's peak resident memory so far (VmHWM in /proc/self/status), in MB; None where the kernel does not say.
fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some((kb / 1024.0).round())
}
