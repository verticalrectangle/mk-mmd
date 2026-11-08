//! One output's frames on the GPU: the picture (the cut's frame or a worker's), the frame effects still to draw, the
//! screen type, the keep mask, the grade, the bytes the encoder takes and the darkest luma (mkmmd.postframes,
//! mkmmd.post, mkmmd.core.screentype as the Python side runs them).

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

use crate::fx::{self, Ctx, Decoded, Drawn, Resources, STAMPS};
use crate::gpu::{Gpu, tiles};
use crate::job::{Grade, Job, Setup};
use crate::pngio::{self, Kind};

const UNPACK: &str = include_str!("shaders/unpack.wgsl");
const COMPOSITE: &str = include_str!("shaders/composite.wgsl");
const GRADE: &str = include_str!("shaders/grade.wgsl");
const QUANTIZE: &str = include_str!("shaders/quantize.wgsl");

/// Where a frame's picture comes from, read on the CPU.
pub enum Picture {
    /// Packed RGBA8, one word per pixel (pngio::read), staged for the GPU by the thread that read it (`staged`).
    Packed(wgpu::Buffer),
    /// A worker's float RGB picture, w * h * 3 float32 little endian.
    Raw(Vec<u8>),
    /// A frame Python finished (Job::finished): the w * h * 3 bytes the encoder takes, handed on as they are.
    Finished(Vec<u8>),
}

/// A job's files, read (on any thread) before the GPU works on it: the frame, its screen layer, its negative layer and
/// its keep mask staged for the GPU there (the frame's submit only copies them), what its ops read decoded.
pub struct Inputs {
    pub picture: Picture,
    pub layer: Option<wgpu::Buffer>,
    pub negative: Option<wgpu::Buffer>,
    pub keep: Option<wgpu::Buffer>,
    /// Each op's inputs (fx::Effect::inputs), in the ops' order.
    pub files: Vec<Vec<Arc<Decoded>>>,
}

/// `words` in a buffer the GPU copies from, written by the calling thread (a reader's: not the thread that submits).
fn staged(device: &wgpu::Device, words: &[u32]) -> Result<wgpu::Buffer> {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staged"),
        size: (words.len() as u64 * 4).max(4),
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut().context("staging a frame's file")?.copy_from_slice(bytemuck::cast_slice(words));
    buf.unmap();
    Ok(buf)
}

/// A finished frame: rgb24 bytes and the darkest luma of the graded picture.
pub struct Frame {
    pub k: u32,
    pub darkest: f32,
    pub rgb: Vec<u8>,
}

/// A frame handed to the GPU (Engine::submit), its bytes still to be taken back (Engine::finish).
pub struct Pending {
    k: u32,
    slot: usize,
    bytes: u64,
    index: wgpu::SubmissionIndex,
    done: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    /// MKR_POST_PROFILE: what each GPU timestamp of the frame closes (Stamps).
    marks: Vec<String>,
}

/// Frames in flight at most: submitted, or being taken back (one readback buffer each).
pub const SLOTS: usize = 3;

/// MKR_POST_PROFILE's GPU timestamps: per readback slot a query set, its resolve and read buffers.
struct Stamps {
    sets: [wgpu::QuerySet; SLOTS],
    resolve: [wgpu::Buffer; SLOTS],
    read: [wgpu::Buffer; SLOTS],
    period_ms: f64,
}

impl Stamps {
    fn new(gpu: &Gpu) -> Option<Stamps> {
        if std::env::var_os("MKR_POST_PROFILE").is_none() || !gpu.timestamps {
            return None;
        }
        let buf = |usage| gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile stamps"), size: STAMPS as u64 * 8, usage, mapped_at_creation: false,
        });
        Some(Stamps {
            sets: [0; SLOTS].map(|_| gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile"), ty: wgpu::QueryType::Timestamp, count: STAMPS,
            })),
            resolve: [0; SLOTS].map(|_| buf(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC)),
            read: [0; SLOTS].map(|_| buf(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)),
            period_ms: gpu.queue.get_timestamp_period() as f64 / 1e6,
        })
    }
}

/// Read a job's files: the frame (or the worker's picture, then removed), its screen layer and keep mask (staged for the
/// GPU on `device`), and what its ops read (in parallel, through fx::read's cache); or the bytes of a frame Python
/// finished (then removed).
pub fn inputs(job: &Job, w: u32, h: u32, device: &wgpu::Device) -> Result<Inputs> {
    if let Some(f) = &job.finished {
        let bytes = std::fs::read(&f.path).with_context(|| format!("{}: cannot read", f.path))?;
        std::fs::remove_file(&f.path).ok();
        if bytes.len() != (w * h * 3) as usize {
            bail!("{}: {} bytes, not a {w}x{h} rgb24 frame", f.path, bytes.len());
        }
        return Ok(Inputs { picture: Picture::Finished(bytes), layer: None, negative: None, keep: None, files: Vec::new() });
    }
    let picture = match (&job.png, &job.raw) {
        (Some(p), None) => Picture::Packed(staged(device, &pngio::read(Path::new(p), w, h, Kind::Picture)?)?),
        (None, Some(r)) => {
            let bytes = std::fs::read(r).with_context(|| format!("{r}: cannot read"))?;
            std::fs::remove_file(r).ok();
            if bytes.len() != (w * h * 12) as usize {
                bail!("{r}: {} bytes, not a {w}x{h} float RGB picture", bytes.len());
            }
            Picture::Raw(bytes)
        }
        _ => bail!("frame {}: one of png and raw", job.k),
    };
    let layer = job.layer.as_ref()
        .map(|p| staged(device, &pngio::read(Path::new(p), w, h, Kind::Layer)?)).transpose()?;
    let negative = job.negative.as_ref()
        .map(|p| staged(device, &pngio::read(Path::new(p), w, h, Kind::Layer)?)).transpose()?;
    let keep = job.keep.as_ref()
        .map(|p| staged(device, &pngio::read(Path::new(p), w, h, Kind::Mask)?)).transpose()?;
    let files = job.ops.par_iter()
        .map(|op| op.effect().inputs().par_iter().map(|i| fx::read(i, w, h)).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;
    Ok(Inputs { picture, layer, negative, keep, files })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims {
    w: u32,
    h: u32,
    a: u32,
    b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradeParams {
    w: u32,
    h: u32,
    keep: u32,
    split: u32,
    floor: u32,
    grain_offset: u32,
    pad0: u32,
    pad1: u32,
    sat: f32,
    keep_delta: f32,
    pivot: f32,
    contrast: f32,
    amount: f32,
    yf: f32,
    lim: f32,
    lim_yf: f32,
    grain_amount: f32,
    lo: f32,
    pad2: f32,
    pad3: f32,
    shadows: [f32; 4],
    highlights: [f32; 4],
    floor_rgb: [f32; 4],
}

struct Kernels {
    unpack_picture: wgpu::ComputePipeline,
    unpack_mask: wgpu::ComputePipeline,
    unpack_alpha: wgpu::ComputePipeline,
    expand: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,
    negate: wgpu::ComputePipeline,
    grade: wgpu::ComputePipeline,
    quantize: wgpu::ComputePipeline,
}

pub struct Engine {
    pub gpu: Gpu,
    setup: Setup,
    w: u32,
    h: u32,
    k: Kernels,
    lut: wgpu::Buffer,
    vig: wgpu::Buffer,
    grain: wgpu::Buffer,
    packed: [wgpu::Buffer; 4],
    raw: wgpu::Buffer,
    img: [wgpu::Buffer; 2],
    layer: wgpu::Buffer,
    mask: wgpu::Buffer,
    /// The negative layer's coverage, one float a pixel.
    neg: wgpu::Buffer,
    rgb: wgpu::Buffer,
    darkest: wgpu::Buffer,
    /// Where frames' bytes are read back from, round in turn: the GPU works on the frames after one while its bytes are
    /// taken back and written out (serve's writer thread).
    readback: [wgpu::Buffer; SLOTS],
    slot: std::sync::atomic::AtomicUsize,
    res: Resources,
    stamps: Option<Stamps>,
}

fn floats(path: &str, count: usize) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("{path}: cannot read"))?;
    if bytes.len() != count * 4 {
        bail!("{path}: {} bytes, not {count} floats", bytes.len());
    }
    Ok(bytes)
}

impl Engine {
    pub fn new(gpu: Gpu, setup: Setup) -> Result<Engine> {
        let [w, h] = setup.size;
        if w % 2 != 0 || h % 2 != 0 || w == 0 || h == 0 {
            bail!("size {w}x{h}: mk post encodes even sizes");
        }
        if setup.lut.len() != 256 {
            bail!("lut: {} entries, not 256", setup.lut.len());
        }
        if setup.lut.windows(2).any(|p| p[1] < p[0]) {
            bail!("lut: not rising (v / 255)");
        }
        if setup.screen != "graded" && setup.screen != "over" {
            bail!("screen {:?}: graded or over", setup.screen);
        }
        let n = (w * h) as u64;
        let k = Kernels {
            unpack_picture: gpu.kernel("unpack", UNPACK, "unpack_picture"),
            unpack_mask: gpu.kernel("unpack", UNPACK, "unpack_mask"),
            unpack_alpha: gpu.kernel("unpack", UNPACK, "unpack_alpha"),
            expand: gpu.kernel("unpack", UNPACK, "expand"),
            composite: gpu.kernel("composite", COMPOSITE, "composite"),
            negate: gpu.kernel("composite", COMPOSITE, "negate"),
            grade: gpu.kernel("grade", GRADE, "grade"),
            quantize: gpu.kernel("quantize", QUANTIZE, "quantize"),
        };
        let lut = gpu.storage_init("lut", bytemuck::cast_slice(&setup.lut));
        let vig = gpu.storage_init("vignette", &floats(&setup.vignette, n as usize)?);
        let grain = gpu.storage_init("grain", &floats(&setup.grain.file, n as usize * setup.grain.count as usize)?);
        let packed = [gpu.storage("frame", n * 4), gpu.storage("layer8", n * 4), gpu.storage("mask8", n * 4),
                      gpu.storage("negative8", n * 4)];
        let raw = gpu.storage("raw", n * 12);
        let img = [gpu.storage("img0", n * 16), gpu.storage("img1", n * 16)];
        let layer = gpu.storage("layer", n * 16);
        let mask = gpu.storage("mask", n * 4);
        let neg = gpu.storage("negative", n * 4);
        let rgb = gpu.storage("rgb", n * 3);
        let darkest = gpu.storage("darkest", 4);
        let out_bytes = if setup.floats { n * 16 } else { n * 3 };
        let readback = [0; SLOTS].map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes + 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let stamps = Stamps::new(&gpu);
        Ok(Engine {
            gpu, setup, w, h, k, lut, vig, grain, packed, raw, img, layer, mask, neg, rgb, darkest, readback,
            slot: std::sync::atomic::AtomicUsize::new(0), res: Resources::default(), stamps,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    /// The frame of `job` finished from its `inputs`.
    pub fn run(&self, job: &Job, inputs: Inputs) -> Result<Frame> {
        let pending = self.submit(job, inputs)?;
        self.finish(pending)
    }

    /// The frame of `job` handed to the GPU, its bytes to be taken back by `finish` (frames finish in the order they
    /// were submitted; up to SLOTS - 1 may be pending while the next is submitted). A frame Python finished is not.
    pub fn submit(&self, job: &Job, inputs: Inputs) -> Result<Pending> {
        if matches!(inputs.picture, Picture::Finished(_)) {
            bail!("frame {}: finished by Python, nothing to draw", job.k);
        }
        let (w, h, gpu) = (self.w, self.h, &self.gpu);
        let groups = tiles(w, h);
        let dims = gpu.uniform(bytemuck::bytes_of(&Dims { w, h, a: 0, b: 0 }));
        let slot = self.slot.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % SLOTS;
        // MKR_POST_PROFILE: a GPU timestamp after each step, named by what it closes
        let mut marks: Vec<String> = Vec::new();
        let stamp = |enc: &mut wgpu::CommandEncoder, marks: &mut Vec<String>, what: &str| {
            if let Some(s) = &self.stamps {
                if (marks.len() as u32) < STAMPS {
                    enc.write_timestamp(&s.sets[slot], marks.len() as u32);
                    marks.push(what.to_string());
                }
            }
        };
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        stamp(&mut enc, &mut marks, "start");
        gpu.queue.write_buffer(&self.darkest, 0, bytemuck::bytes_of(&f32::MAX.to_bits()));
        match &inputs.picture {
            Picture::Packed(p) => {
                enc.copy_buffer_to_buffer(p, 0, &self.packed[0], 0, p.size());
                gpu.pass(&mut enc, &self.k.unpack_picture,
                         &[(0, &dims), (1, &self.packed[0]), (2, &self.lut), (3, &self.img[0])], groups);
            }
            Picture::Raw(r) => {
                gpu.queue.write_buffer(&self.raw, 0, r);
                gpu.pass(&mut enc, &self.k.expand, &[(0, &dims), (3, &self.img[0]), (5, &self.raw)], groups);
            }
            Picture::Finished(_) => unreachable!(),
        }
        stamp(&mut enc, &mut marks, "unpack");
        let mut cur = 0usize;
        for (op, files) in job.ops.iter().zip(&inputs.files) {
            let t0 = std::time::Instant::now();
            let mut cx = Ctx {
                gpu, enc: &mut enc, w, h, src: &self.img[cur], dst: &self.img[1 - cur], lut: &self.lut,
                levels: &self.setup.lut, res: &self.res, stamps: self.stamps.as_ref().map(|s| (&s.sets[slot], &mut marks)),
            };
            if op.effect().draw(&mut cx, files).with_context(|| format!("frame {}: {op:?}", job.k))? == Drawn::Dst {
                cur = 1 - cur;
            }
            if self.stamps.is_some() {      // its CPU time here (with any reading back it waited for), its GPU's below
                let name = format!("{op:?}");
                let name = name.split('(').next().unwrap_or_default().to_lowercase();
                eprintln!("profile frame {} op {name} cpu {:.2} ms", job.k, t0.elapsed().as_secs_f64() * 1e3);
                stamp(&mut enc, &mut marks, &name);
            }
        }
        let plain = self.setup.plain;
        let graded = self.setup.screen == "graded" && !plain;
        let keep = inputs.keep.is_some() && !plain;
        if let Some(m) = &inputs.keep {
            enc.copy_buffer_to_buffer(m, 0, &self.packed[2], 0, m.size());
            gpu.pass(&mut enc, &self.k.unpack_mask, &[(0, &dims), (1, &self.packed[2]), (2, &self.lut), (4, &self.mask)],
                     groups);
        }
        if let (Some(l), false) = (&inputs.layer, plain) {
            enc.copy_buffer_to_buffer(l, 0, &self.packed[1], 0, l.size());
            gpu.pass(&mut enc, &self.k.unpack_picture,
                     &[(0, &dims), (1, &self.packed[1]), (2, &self.lut), (3, &self.layer)], groups);
        }
        let negative = inputs.negative.is_some() && !plain;
        if let (Some(n), false) = (&inputs.negative, plain) {
            enc.copy_buffer_to_buffer(n, 0, &self.packed[3], 0, n.size());
            gpu.pass(&mut enc, &self.k.unpack_alpha,
                     &[(0, &dims), (1, &self.packed[3]), (2, &self.lut), (4, &self.neg)], groups);
        }
        if graded {
            self.composite(&mut enc, cur, keep, inputs.layer.is_some(), negative);
        }
        if !plain {
            self.grade(&mut enc, job.k, job.grade.as_ref().unwrap_or(&self.setup.grade), keep, cur);
            cur = 1 - cur;
            if !graded {
                self.composite(&mut enc, cur, false, inputs.layer.is_some(), negative);
            }
        }
        stamp(&mut enc, &mut marks, "grade");
        let bytes = if self.setup.floats { (w * h * 16) as u64 } else { (w * h * 3) as u64 };
        let readback = &self.readback[slot];
        if self.setup.floats {
            enc.copy_buffer_to_buffer(&self.img[cur], 0, readback, 0, bytes);
        } else {
            let quads = w * h / 4;
            let qdims = gpu.uniform(bytemuck::bytes_of(&Dims { w, h, a: quads, b: 0 }));
            gpu.pass(&mut enc, &self.k.quantize, &[(0, &qdims), (1, &self.img[cur]), (2, &self.rgb), (3, &self.darkest)],
                     (quads.div_ceil(256), 1));
            enc.copy_buffer_to_buffer(&self.rgb, 0, readback, 0, bytes);
        }
        enc.copy_buffer_to_buffer(&self.darkest, 0, readback, bytes, 4);
        stamp(&mut enc, &mut marks, "out");
        if let Some(s) = &self.stamps {
            let n = marks.len() as u32;
            enc.resolve_query_set(&s.sets[slot], 0..n, &s.resolve[slot], 0);
            enc.copy_buffer_to_buffer(&s.resolve[slot], 0, &s.read[slot], 0, n as u64 * 8);
        }
        let index = gpu.queue.submit([enc.finish()]);
        let (tx, done) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        Ok(Pending { k: job.k, slot, bytes, index, done, marks })
    }

    /// The bytes of a submitted frame, once the GPU has them.
    pub fn finish(&self, p: Pending) -> Result<Frame> {
        self.finish_with(p, |k, darkest, rgb| Ok(Frame { k, darkest, rgb: rgb.to_vec() }))
    }

    /// A submitted frame handed to `take` once the GPU has it: its clip index, darkest luma and bytes (read where the
    /// GPU left them, the buffer free again after).
    pub fn finish_with<R>(&self, p: Pending, take: impl FnOnce(u32, f32, &[u8]) -> Result<R>) -> Result<R> {
        let gpu = &self.gpu;
        gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(p.index.clone()), timeout: None }).context("GPU wait")?;
        p.done.recv().context("GPU readback")?.context("GPU readback")?;
        let readback = &self.readback[p.slot];
        let got = {
            let view = readback.slice(..).get_mapped_range().context("GPU readback")?;
            let darkest = f32::from_bits(u32::from_le_bytes(view[p.bytes as usize..].try_into().unwrap()));
            take(p.k, darkest, &view[..p.bytes as usize])
        };
        readback.unmap();
        if let Some(s) = &self.stamps {
            self.report_stamps(s, &p.index, p.k, p.slot, &p.marks)?;
        }
        got
    }

    /// MKR_POST_PROFILE: one line per frame on stderr, each step's GPU milliseconds ("profile gpu frame K step=ms ...").
    fn report_stamps(&self, s: &Stamps, index: &wgpu::SubmissionIndex, k: u32, slot: usize, marks: &[String]) -> Result<()> {
        let n = marks.len() as u64;
        if n < 2 {
            return Ok(());
        }
        let slice = s.read[slot].slice(..n * 8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(index.clone()), timeout: None })
            .context("GPU wait")?;
        rx.recv().context("timestamps")?.context("timestamps")?;
        let v: Vec<u64> = bytemuck::pod_collect_to_vec(&slice.get_mapped_range().context("timestamps")?);
        s.read[slot].unmap();
        let mut line = format!("profile gpu frame {k}");
        for i in 1..v.len() {
            line.push_str(&format!(" {}={:.3}", marks[i], v[i].wrapping_sub(v[i - 1]) as f64 * s.period_ms));
        }
        eprintln!("{line}");
        Ok(())
    }

    /// The screen type over the picture `img[cur]` (mkmmd.core.screentype.composite): the screen layer laid over it
    /// (`layer`), then the negative type inverting what is under it (`negative`: it goes last, as on the CPU).
    fn composite(&self, enc: &mut wgpu::CommandEncoder, cur: usize, keep: bool, layer: bool, negative: bool) {
        if !layer && !negative {
            return;
        }
        let p = self.gpu.uniform(bytemuck::bytes_of(&Dims { w: self.w, h: self.h, a: keep as u32, b: 0 }));
        if layer {
            self.gpu.pass(enc, &self.k.composite, &[(0, &p), (1, &self.img[cur]), (2, &self.layer), (3, &self.mask)],
                          tiles(self.w, self.h));
        }
        if negative {
            self.gpu.pass(enc, &self.k.negate, &[(0, &p), (1, &self.img[cur]), (3, &self.mask), (4, &self.neg)],
                          tiles(self.w, self.h));
        }
    }

    fn grade(&self, enc: &mut wgpu::CommandEncoder, k: u32, g: &Grade, keep: bool, cur: usize) {
        let n = self.w * self.h;
        let mut p = GradeParams {
            w: self.w,
            h: self.h,
            keep: keep as u32,
            split: g.split.is_some() as u32,
            floor: g.floor.is_some() as u32,
            grain_offset: (k % self.setup.grain.count) * n,
            sat: g.saturation,
            keep_delta: g.keep_delta,
            pivot: g.pivot,
            contrast: g.contrast,
            grain_amount: self.setup.grain.amount,
            ..Zeroable::zeroed()
        };
        if let Some(s) = &g.split {
            p.amount = s.amount;
            p.shadows = [s.shadows[0], s.shadows[1], s.shadows[2], 0.0];
            p.highlights = [s.highlights[0], s.highlights[1], s.highlights[2], 0.0];
        }
        if let Some(f) = &g.floor {
            p.yf = f.yf;
            p.lim = f.lim;
            p.lim_yf = f.lim_yf;
            p.lo = f.lo;
            p.floor_rgb = [f.rgb[0], f.rgb[1], f.rgb[2], 0.0];
        }
        let u = self.gpu.uniform(bytemuck::bytes_of(&p));
        self.gpu.pass(enc, &self.k.grade,
                      &[(0, &u), (1, &self.img[cur]), (2, &self.img[1 - cur]), (3, &self.vig), (4, &self.grain),
                        (5, &self.mask)],
                      tiles(self.w, self.h));
    }
}
