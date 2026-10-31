//! mkr-shade lab, GPU side: evaluate the node graphs `lab/blender_lab.py` exported with mkr-shade's WGSL on the GPU and
//! compare per quad with Blender's renders.
//!
//!   cargo run --release -p mkr-shade --example lab -- LAB_DIR [--only S[,S...]] [--json OUT.json] [--dump] [--verbose]
//!
//! Errors are measured as EEVEE's film shows values (negative and NaN radiance as 0), absolute below 1 and relative
//! above (the film is half float). Closure cases light the closures as EEVEE's uniform world does (see
//! tests/common/gpu.rs), with the diffuse response per row measured by `closure_diffuse`. One line per case, then a
//! table per node family. `--dump` writes <case>.ours.bin next to the references.

#[path = "../tests/common/gpu.rs"]
mod gpu;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use gpu::{Gpu, Kind, film_error};

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytemuck::cast_slice(&b).to_vec())
}

struct Stats {
    max: f64,
    mean: f64,
    p99: f64,
    worst: usize,
}

fn stats(out: &[f32], reference: &[f32], quads: usize) -> Stats {
    let errs: Vec<f64> = (0..quads)
        .map(|q| (0..3).map(|c| film_error(out[q * 4 + c], reference[q * 4 + c])).fold(0.0, f64::max))
        .collect();
    let mean = errs.iter().map(|e| e.min(1e9)).sum::<f64>() / quads.max(1) as f64;
    let (worst, max) = errs.iter().cloned().enumerate().fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
    let mut sorted = errs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = sorted[((quads as f64 * 0.99) as usize).min(quads - 1)];
    Stats { max, mean, p99, worst }
}

fn family(name: &str) -> String {
    let parts: Vec<&str> = name.split('_').collect();
    match parts[0] {
        "closure" => format!("closure {}", parts.get(1).copied().unwrap_or("")),
        "noise" | "voronoi" | "wave" => format!("{} {}", parts[0], parts.get(1).copied().unwrap_or("")),
        "color" | "vector" => "conversion".to_string(),
        "rgb" => "rgb_to_bw".to_string(),
        "white" => "white_noise".to_string(),
        "hue" => "hue_sat".to_string(),
        "bright" => "bright_contrast".to_string(),
        p => p.to_string(),
    }
}

/// The world's diffuse response per row (rgb), from the `closure_diffuse` case (its colour is (0.8, 0.5, 0.2)).
fn diffuse_response(dir: &Path, n: usize) -> Vec<f32> {
    let col = [0.8f32, 0.5, 0.2];
    match read_f32(&dir.join("closure_diffuse.ref.bin")) {
        Ok(r) if r.len() == n * n * 4 => (0..n)
            .flat_map(|j| (0..3).map(move |c| (j, c)))
            .map(|(j, c)| (0..n).map(|i| r[(j * n + i) * 4 + c] / col[c]).sum::<f32>() / n as f32)
            .collect(),
        _ => vec![1.0; 3 * n],
    }
}

/// The world's specular response per quad (rgb), from `closure_glossy_multi_ggx`: a white multiple-scattering glossy
/// whose closure colour is exactly 1, so the render is what the world's reflection light gives a unit lobe.
fn specular_response(dir: &Path, n: usize) -> Vec<f32> {
    match read_f32(&dir.join("closure_glossy_multi_ggx.ref.bin")) {
        Ok(r) if r.len() == n * n * 4 => (0..n * n).flat_map(|q| [r[q * 4], r[q * 4 + 1], r[q * 4 + 2]]).collect(),
        _ => vec![1.0; 3 * n * n],
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().ok_or_else(|| anyhow!("usage: lab LAB_DIR [--only S] [--json OUT]"))?);
    let only = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)).cloned();
    let json_out = args.iter().position(|a| a == "--json").and_then(|i| args.get(i + 1)).cloned();
    let verbose = args.iter().any(|a| a == "--verbose");
    let dump = args.iter().any(|a| a == "--dump");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
        .filter(|p| {
            let s = p.file_stem().unwrap().to_string_lossy().to_string();
            only.as_ref().map(|o| o.split(',').any(|o| s.contains(o))).unwrap_or(true)
        })
        .collect();
    cases.sort();
    let gpu = Gpu::new()?;
    eprintln!("lab: GPU {}", gpu.adapter);
    let mut fam: BTreeMap<String, (usize, f64, f64, String)> = BTreeMap::new();
    let mut rows_json = Vec::new();
    for path in &cases {
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let name = doc["name"].as_str().unwrap_or("?").to_string();
        let n = doc["n"].as_u64().unwrap_or(48) as u32;
        let kind = match doc["kind"].as_str() {
            Some("closure") => Kind::Closure,
            Some("shade") => Kind::Shade,
            Some("lit") => {
                // lamp-lit cases measure EEVEE's lighting of closures (the renderer's side), not mkr-shade
                println!("{name}: lamp-lit lighting measurement, not compared here");
                continue;
            }
            _ => Kind::Field,
        };
        let material: mkr_shade::MaterialDesc = serde_json::from_value(doc["material"].clone())?;
        let groups: BTreeMap<String, mkr_shade::GroupDesc> = serde_json::from_value(doc["groups"].clone())?;
        let images = mkr_shade::images_from_lit(&doc);
        let m = match mkr_shade::compile("lab", &material, &groups, &images, 0) {
            Ok(m) => m,
            Err(e) => {
                println!("{name}: compile error {e}");
                continue;
            }
        };
        let stem = path.with_extension("");
        let inputs = read_f32(&stem.with_extension("in.bin"))?;
        let reference = read_f32(&stem.with_extension("ref.bin"))?;
        let mut rows: Vec<f32> =
            doc["rows"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(1.0) as f32).collect()).unwrap_or_default();
        rows.resize(n as usize, 1.0);
        rows.extend(diffuse_response(&dir, n as usize));
        if kind == Kind::Closure {
            rows.extend(specular_response(&dir, n as usize));
        }
        let image = match doc["images"].as_object().and_then(|m| m.values().next()) {
            Some(rec) => {
                let file = rec["file"].as_str().unwrap_or("");
                let path = if Path::new(file).is_absolute() { PathBuf::from(file) } else { dir.join(file) };
                let img = image::open(&path).with_context(|| format!("loading {}", path.display()))?.to_rgba8();
                Some(gpu::LabImage {
                    width: img.width(),
                    height: img.height(),
                    rgba: img.into_raw(),
                    srgb: rec["colorspace"] == "sRGB",
                })
            }
            None => None,
        };
        let out = match gpu.eval(&m, n, kind, &inputs, &rows, image.as_ref()) {
            Ok(o) => o,
            Err(e) => {
                println!("{name}: GPU error {e}");
                if verbose {
                    println!("{}", m.wgsl);
                }
                continue;
            }
        };
        if dump {
            std::fs::write(stem.with_extension("ours.bin"), bytemuck::cast_slice(&out))?;
        }
        let quads = (n * n) as usize;
        let st = stats(&out, &reference, quads);
        let w = st.worst;
        println!(
            "{name:<44} max {:.5} mean {:.6} p99 {:.5}  worst #{w} ours {:.4} {:.4} {:.4} ref {:.4} {:.4} {:.4} in {:.4} {:.4} {:.4}{}",
            st.max,
            st.mean,
            st.p99,
            out[w * 4],
            out[w * 4 + 1],
            out[w * 4 + 2],
            reference[w * 4],
            reference[w * 4 + 1],
            reference[w * 4 + 2],
            inputs[w * 9],
            inputs[w * 9 + 1],
            inputs[w * 9 + 2],
            if m.unsupported.is_empty() { String::new() } else { format!("  UNSUPPORTED {:?}", m.unsupported) }
        );
        if verbose {
            for wline in &m.warnings {
                println!("    warning: {wline}");
            }
        }
        let f = fam.entry(family(&name)).or_insert((0, 0.0, 0.0, String::new()));
        f.0 += 1;
        if st.max >= f.1 {
            f.1 = st.max;
            f.3 = name.clone();
        }
        f.2 = f.2.max(st.mean);
        rows_json.push(serde_json::json!({"case": name, "family": family(&name), "max": st.max, "mean": st.mean,
                                          "p99": st.p99, "quads": quads}));
    }
    println!("\n{:<24} {:>5} {:>9} {:>10}  worst case", "family", "cases", "max err", "worst mean");
    for (k, (c, mx, mn, worst)) in &fam {
        println!("{k:<24} {c:>5} {mx:>9.5} {mn:>10.6}  {worst}");
    }
    if let Some(p) = json_out {
        std::fs::write(p, serde_json::to_string_pretty(&rows_json)?)?;
    }
    Ok(())
}
