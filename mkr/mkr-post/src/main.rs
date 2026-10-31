//! `mkr-post`: mk post's frames on the GPU (see the library's docs).
//!
//!     mkr-post serve --setup SETUP.json [--threads N] [--stats FD]   finish one output's frames: jobs on stdin,
//!                                                                    frames on stdout
//!     mkr-post info                                                  the adapter and the frame effects it draws, as JSON

use std::fs::File;
use std::io::{BufRead, BufWriter, StdoutLock, Write};
use std::os::fd::FromRawFd;
use std::sync::mpsc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use mkr_post::engine::{self, Engine, Inputs, Picture};
use mkr_post::gpu::Gpu;
use mkr_post::job::{Job, Setup};

/// Jobs read ahead of the one on the GPU (their PNGs decoded meanwhile).
const AHEAD: usize = 6;

#[derive(Parser)]
#[command(name = "mkr-post", about = "mk post's frames on the GPU")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Finish one output's frames: one JSON job per line on stdin, one record per frame on stdout.
    Serve {
        #[arg(long)]
        setup: String,
        /// Threads decoding the jobs' PNGs.
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// A file descriptor (a pipe this process inherits) for each frame's clip index (u32 LE) and darkest luma (f32
        /// LE); stdout then carries the frames' bytes alone, as the encoder takes them.
        #[arg(long)]
        stats: Option<i32>,
    },
    /// The adapter and the frame effects it draws, as JSON.
    Info,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve { setup, threads, stats } => serve(&setup, threads, stats),
        Cmd::Info => {
            let gpu = Gpu::new()?;
            let a = &gpu.adapter;
            println!("{}", serde_json::json!({
                "adapter": a.name, "backend": format!("{:?}", a.backend), "device_type": format!("{:?}", a.device_type),
                "driver": a.driver, "driver_info": a.driver_info, "ops": mkr_post::fx::OPS,
            }));
            Ok(())
        }
    }
}

/// A frame on its way out, in the jobs' order: drawn on the GPU (its bytes still to be taken back), or finished by
/// Python and handed on as it is.
enum Out {
    Drawn(engine::Pending),
    Finished { k: u32, darkest: f32, rgb: Vec<u8> },
}

/// Where the frames go: stdout, each frame's record (clip index, darkest luma, bytes) on it, or with a stats pipe the
/// bytes alone on stdout and the rest on the pipe.
struct Sink<'a> {
    out: BufWriter<StdoutLock<'a>>,
    stats: Option<File>,
}

impl Sink<'_> {
    fn put(&mut self, k: u32, darkest: f32, rgb: &[u8]) -> Result<()> {
        if self.stats.is_none() {
            self.out.write_all(&k.to_le_bytes())?;
            self.out.write_all(&darkest.to_le_bytes())?;
        }
        self.out.write_all(rgb)?;
        self.out.flush()?;
        if let Some(s) = &mut self.stats {
            let mut rec = [0u8; 8];
            rec[..4].copy_from_slice(&k.to_le_bytes());
            rec[4..].copy_from_slice(&darkest.to_le_bytes());
            s.write_all(&rec)?;
        }
        Ok(())
    }
}

fn serve(setup: &str, threads: usize, stats: Option<i32>) -> Result<()> {
    let setup: Setup = serde_json::from_slice(&std::fs::read(setup).with_context(|| format!("{setup}: cannot read"))?)
        .with_context(|| format!("{setup}: not a setup"))?;
    // SAFETY: the descriptor is one mk post opened for this process (a pipe's write end) and nothing else here uses it
    let stats = stats.map(|fd| unsafe { File::from_raw_fd(fd) });
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build_global().ok();
    let engine = Engine::new(Gpu::new()?, setup)?;
    let (w, h) = engine.size();
    let (tx, rx) = mpsc::sync_channel::<mpsc::Receiver<Result<(Job, Inputs)>>>(AHEAD);
    let device = engine.gpu.device.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let (otx, orx) = mpsc::channel();
            if tx.send(orx).is_err() {
                break;
            }
            let device = device.clone();
            rayon::spawn(move || {
                let got = serde_json::from_str::<Job>(&line)
                    .with_context(|| format!("not a job: {line}"))
                    .and_then(|job| engine::inputs(&job, w, h, &device).map(|i| (job, i)));
                otx.send(got).ok();
            });
        }
    });
    let profile = std::env::var_os("MKR_POST_PROFILE").is_some();
    std::thread::scope(|s| {
        // each frame's bytes taken back and written out on a thread of their own, in the order the frames were
        // submitted, while the GPU goes on with the frames after it (Engine::SLOTS in flight at most)
        let (ptx, prx) = mpsc::sync_channel::<Out>(engine::SLOTS - 2);
        let engine = &engine;
        let writer = s.spawn(move || -> Result<()> {
            let stdout = std::io::stdout();
            let mut sink = Sink { out: BufWriter::with_capacity(1 << 22, stdout.lock()), stats };
            for o in prx {
                match o {
                    Out::Drawn(p) => engine.finish_with(p, |k, darkest, rgb| sink.put(k, darkest, rgb))?,
                    Out::Finished { k, darkest, rgb } => sink.put(k, darkest, &rgb)?,
                }
            }
            Ok(())
        });
        let run = (|| -> Result<()> {
            for orx in rx {
                let t0 = std::time::Instant::now();
                let (job, inputs) = orx.recv().context("a job's reader stopped")??;
                let t1 = std::time::Instant::now();
                let out = match inputs.picture {
                    Picture::Finished(rgb) => {
                        let darkest = job.finished.as_ref().map_or(f32::NAN, |f| f.darkest);
                        Out::Finished { k: job.k, darkest, rgb }
                    }
                    picture => Out::Drawn(engine.submit(&job, Inputs { picture, ..inputs })?),
                };
                let t2 = std::time::Instant::now();
                if ptx.send(out).is_err() {
                    break;                  // the writer stopped: its error says why
                }
                if profile {
                    let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                    eprintln!("profile frame {} inputs {:.2} run {:.2} out {:.2}", job.k, ms(t0, t1), ms(t1, t2),
                              ms(t2, std::time::Instant::now()));
                }
            }
            Ok(())
        })();
        drop(ptx);
        let wrote = writer.join().map_err(|_| anyhow::anyhow!("the frames' writer panicked"))?;
        wrote.and(run)
    })
}
