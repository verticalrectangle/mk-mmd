//! The frame effects mkr-post draws (docs/design.md: Post; mkmmd/gpufx works out their ops): one module and one kernel
//! file (shaders/fx_<kind>.wgsl) per kind. An op is a JSON object tagged "op" whose parameters Python worked out with
//! the same code the CPU path draws with. It names the files it reads besides the picture (`Effect::inputs`: decoded on
//! the reader threads, through a cache every job shares) and draws from the picture as it reaches it (`Ctx::src`) into
//! `Ctx::dst`, or over `Ctx::src` in place.
//!
//! A kind is four lines here: its `mod`, its name in `OPS`, its `Op` variant and its arm in `Op::effect`.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use bytemuck::Pod;
use serde::Deserialize;

use crate::gpu::{Dispatch, Gpu, tiles};
use crate::pngio::{self, Kind};

/// The unpack kernels (shaders/unpack.wgsl): `widen_luma` makes a figure layer's luma bytes the words kernels read.
const UNPACK: &str = include_str!("../shaders/unpack.wgsl");

pub mod warp;
pub mod figure;
pub mod plane;
pub mod echo;
pub mod slitscan;
pub mod lens;
pub mod shock;
pub mod push;
pub mod slab;
pub mod split;
pub mod melt;
pub mod footage;
pub mod fields;
pub mod flood;
pub mod spiral;
pub mod masked;
pub mod kaleido;
pub mod liquid;
pub mod tunnel;
pub mod mandala;
pub mod trails;
pub mod scope;
pub mod grid;
pub mod stack;
pub mod pixelsort;
pub mod doodle;
pub mod mosh;
pub mod film;
pub mod tint;
pub mod halftone;
pub mod invert;
pub mod inks;
pub mod camcorder;
pub mod silk;

/// The ops mkr-post draws (`mkr-post info`).
pub const OPS: &[&str] = &["warp", "echo", "slitscan", "lens", "shock", "push", "slab", "split", "melt", "footage", "flood",
                           "spiral", "kaleido", "liquid", "tunnel", "mandala", "trails", "scope", "grid", "stack",
                           "pixelsort", "doodle", "mosh", "film", "tint", "halftone", "invert", "inks", "camcorder"];

/// One frame effect's picture work.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    Warp(warp::Warp),
    Echo(echo::Echo),
    Slitscan(slitscan::Slitscan),
    Lens(lens::Lens),
    Shock(shock::Shock),
    Push(push::Push),
    Slab(slab::Slab),
    Split(split::Split),
    Melt(melt::Melt),
    Footage(footage::Footage),
    Flood(flood::Flood),
    Spiral(spiral::Spiral),
    Kaleido(kaleido::Kaleido),
    Liquid(liquid::Liquid),
    Tunnel(tunnel::Tunnel),
    Mandala(mandala::Mandala),
    Trails(trails::Trails),
    Scope(scope::Scope),
    Grid(grid::Grid),
    Stack(stack::Stack),
    Pixelsort(pixelsort::Pixelsort),
    Doodle(doodle::Doodle),
    Mosh(mosh::Mosh),
    Film(film::Film),
    Tint(tint::Tint),
    Halftone(halftone::Halftone),
    Invert(invert::Invert),
    Inks(inks::Inks),
    Camcorder(camcorder::Camcorder),
}

impl Op {
    pub fn effect(&self) -> &dyn Effect {
        match self {
            Op::Warp(o) => o,
            Op::Echo(o) => o,
            Op::Slitscan(o) => o,
            Op::Lens(o) => o,
            Op::Shock(o) => o,
            Op::Push(o) => o,
            Op::Slab(o) => o,
            Op::Split(o) => o,
            Op::Melt(o) => o,
            Op::Footage(o) => o,
            Op::Flood(o) => o,
            Op::Spiral(o) => o,
            Op::Kaleido(o) => o,
            Op::Liquid(o) => o,
            Op::Tunnel(o) => o,
            Op::Mandala(o) => o,
            Op::Trails(o) => o,
            Op::Scope(o) => o,
            Op::Grid(o) => o,
            Op::Stack(o) => o,
            Op::Pixelsort(o) => o,
            Op::Doodle(o) => o,
            Op::Mosh(o) => o,
            Op::Film(o) => o,
            Op::Tint(o) => o,
            Op::Halftone(o) => o,
            Op::Invert(o) => o,
            Op::Inks(o) => o,
            Op::Camcorder(o) => o,
        }
    }
}

/// A frame effect on the GPU.
pub trait Effect {
    /// The files it reads besides the picture, decoded before the GPU takes the frame; `draw` gets them in this order.
    fn inputs(&self) -> Vec<Input> {
        Vec::new()
    }

    /// Draw it: from `cx.src` into `cx.dst` (Drawn::Dst), or over `cx.src` (Drawn::Src: in place, or nothing to do).
    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn>;
}

/// Which picture buffer holds an op's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawn {
    Src,
    Dst,
}

/// A file an op reads besides the picture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Input {
    /// An 8-bit PNG of the output's size, read as mk post reads it (pngio::Kind); cached by path.
    Png(String, Kind),
    /// `count` float32 (little endian) Python wrote for this frame, removed once read; never cached.
    F32 { path: String, count: usize },
    /// A picture Python wrote for this frame as rgb24 bytes, `size` (w, h) of them (a clip's frame decoded and fitted:
    /// mkmmd.gpufx.footage), removed once read unless `keep` (one the run's outputs share); never cached. Read as packed
    /// RGBA8, alpha 255.
    Rgb { path: String, size: [u32; 2], keep: bool },
    /// `count` float32 Python wrote once for many frames (a sticker tile), kept on disk; cached as a PNG is.
    Kept { path: String, count: usize },
}

/// An input once read.
pub enum Decoded {
    /// One packed RGBA8 word per pixel (pngio::read).
    Packed(Vec<u32>),
    /// A figure layer (read as pngio::Kind::Luma): its luma, one byte a pixel (uploaded so, widened on the GPU to the
    /// words the kernels read: Ctx::upload), and where it covers anything.
    Figure(Vec<u8>, Cover),
    Floats(Vec<f32>),
}

impl Decoded {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Decoded::Packed(v) => bytemuck::cast_slice(v),
            Decoded::Figure(v, _) => v,
            Decoded::Floats(v) => bytemuck::cast_slice(v),
        }
    }

    /// A figure layer's Cover (None: another input).
    pub fn cover(&self) -> Option<&Cover> {
        match self {
            Decoded::Figure(_, c) => Some(c),
            _ => None,
        }
    }
}

/// Where a figure layer covers anything, from its luma bytes (what the unpack kernels widen by the 1/255 table): each
/// row's and each column's largest byte, and a bit per pixel set where the byte is 1 or more. The boxes the masked
/// remaps and the collages cut by come from it, and the support of a figure carried by an affine map that keeps the
/// axes (figure::carried_support), with no reading back.
pub struct Cover {
    w: u32,
    h: u32,
    rows: Vec<u8>,
    cols: Vec<u8>,
    /// Row by row, `stride` words a row: bit x % 64 of word x / 64.
    bits: Vec<u64>,
    stride: usize,
}

impl Cover {
    fn new(px: &[u8], w: u32, h: u32) -> Cover {
        let (wu, hu) = (w as usize, h as usize);
        let stride = wu.div_ceil(64);
        let mut rows = vec![0u8; hu];
        let mut cols = vec![0u8; wu];
        let mut bits = vec![0u64; stride * hu];
        for (y, row) in px.chunks_exact(wu).enumerate().take(hu) {
            let mut most = 0u8;
            for (word, run) in bits[y * stride..(y + 1) * stride].iter_mut().zip(row.chunks(64)) {
                let mut b = 0u64;
                for (i, &v) in run.iter().enumerate() {
                    b |= ((v != 0) as u64) << i;
                    most = most.max(v);
                }
                *word = b;
            }
            for (m, &v) in cols.iter_mut().zip(row) {
                *m = (*m).max(v);
            }
            rows[y] = most;
        }
        Cover { w, h, rows, cols, bits, stride }
    }

    /// The layer's size (w, h).
    pub fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    /// The box (x0, y0, x1, y1; x1, y1 past it) of the pixels whose byte is `t` or more; None when there are none.
    pub fn bbox(&self, t: u8) -> Option<[u32; 4]> {
        let x0 = self.cols.iter().position(|&v| v >= t)? as u32;
        let x1 = self.cols.iter().rposition(|&v| v >= t)? as u32 + 1;
        let y0 = self.rows.iter().position(|&v| v >= t)? as u32;
        let y1 = self.rows.iter().rposition(|&v| v >= t)? as u32 + 1;
        Some([x0, y0, x1, y1])
    }

    /// Row `y`'s bits (byte 1 or more).
    pub fn row_bits(&self, y: usize) -> &[u64] {
        &self.bits[y * self.stride..(y + 1) * self.stride]
    }
}

/// The smallest byte whose value in `levels` (the 1/255 table, rising) is over `cut`: a packed input's sample stands
/// for more than `cut` exactly when its byte is at least this. None when none is.
pub fn first_over(levels: &[f32], cut: f32) -> Option<u8> {
    levels.iter().position(|&v| v > cut).map(|b| b as u8)
}

// ---------------------------------------------------------------------------------------------------- caches
/// Least recently used values by key, at most `cap` bytes of them (each value's size given when it is put).
struct Lru<K, V> {
    cap: usize,
    used: usize,
    map: HashMap<K, (V, usize)>,
    order: VecDeque<K>,
}

impl<K: Eq + Hash + Clone, V: Clone> Lru<K, V> {
    fn new(cap: usize) -> Self {
        Lru { cap, used: 0, map: HashMap::new(), order: VecDeque::new() }
    }

    fn get(&mut self, k: &K) -> Option<V> {
        let v = self.map.get(k)?.0.clone();
        if let Some(i) = self.order.iter().position(|x| x == k) {
            let key = self.order.remove(i).unwrap();
            self.order.push_back(key);
        }
        Some(v)
    }

    fn put(&mut self, k: K, v: V, bytes: usize) {
        if let Some((_, b)) = self.map.remove(&k) {
            self.used -= b;
            self.order.retain(|x| x != &k);
        }
        while self.used + bytes > self.cap {
            let Some(old) = self.order.pop_front() else { break };
            if let Some((_, b)) = self.map.remove(&old) {
                self.used -= b;
            }
        }
        self.used += bytes;
        self.map.insert(k.clone(), (v, bytes));
        self.order.push_back(k);
    }
}

/// Decoded PNGs kept for the jobs after (an echo's, a slitscan's earlier frames, a figure layer several ops read).
const DECODED_BYTES: usize = 512 << 20;
/// Uploaded inputs kept on the GPU: a few frames' worth (24 packed inputs of 1080p: a stack's layers and plates for
/// the frames after).
const UPLOADED_BYTES: usize = 192 << 20;

static DECODED: Mutex<Option<Lru<Input, Arc<Decoded>>>> = Mutex::new(None);

/// Read `input` (`w` x `h` the output's size), through the decode cache.
pub fn read(input: &Input, w: u32, h: u32) -> Result<Arc<Decoded>> {
    match input {
        Input::Png(..) | Input::Kept { .. } => {
            if let Some(got) = DECODED.lock().unwrap().get_or_insert_with(|| Lru::new(DECODED_BYTES)).get(input) {
                return Ok(got);
            }
            let t0 = std::time::Instant::now();
            let got = Arc::new(match input {
                Input::Png(path, Kind::Luma) => {
                    let px = pngio::read_luma(std::path::Path::new(path), w, h)?;
                    let cover = Cover::new(&px, w, h);          // on the reader's thread, beside the bytes
                    Decoded::Figure(px, cover)
                }
                Input::Png(path, kind) => Decoded::Packed(pngio::read(std::path::Path::new(path), w, h, *kind)?),
                Input::Kept { path, count } => Decoded::Floats(floats(path, *count)?),
                Input::F32 { .. } | Input::Rgb { .. } => unreachable!(),
            });
            if std::env::var_os("MKR_POST_PROFILE").is_some() {
                let what = if matches!(input, Input::Png(..)) { "png" } else { "kept" };
                eprintln!("profile read {what} {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            let bytes = got.bytes().len();
            DECODED.lock().unwrap().get_or_insert_with(|| Lru::new(DECODED_BYTES)).put(input.clone(), got.clone(), bytes);
            Ok(got)
        }
        Input::F32 { path, count } => {
            let got = floats(path, *count)?;
            std::fs::remove_file(path).ok();
            Ok(Arc::new(Decoded::Floats(got)))
        }
        Input::Rgb { path, size, keep } => {
            let got = rgb24(path, *size)?;
            if !keep {
                std::fs::remove_file(path).ok();
            }
            Ok(Arc::new(Decoded::Packed(got)))
        }
    }
}

/// The rgb24 picture at `path` (w x h) as packed RGBA8 words, alpha 255.
fn rgb24(path: &str, [w, h]: [u32; 2]) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("{path}: cannot read"))?;
    let (w, h) = (w as usize, h as usize);
    if bytes.len() != w * h * 3 {
        bail!("{path}: {} bytes, not a {w}x{h} rgb24 picture", bytes.len());
    }
    Ok(pngio::pack(&bytes, w * 3, w, h, 3, Kind::Picture))
}

/// `count` float32 (little endian) from the file at `path`.
fn floats(path: &str, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("{path}: cannot read"))?;
    if bytes.len() != count * 4 {
        bail!("{path}: {} bytes, not {count} float32", bytes.len());
    }
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}

/// What the ops share across frames: their kernels (compiled once, when first drawn, with their entry points' names),
/// scratch buffers (kept for the frames after, grown as needed), the inputs already on the GPU, and the pictures an
/// effect holds from frame to frame (a mosh's), each with the frame it is of.
pub struct Resources {
    kernels: Mutex<HashMap<(&'static str, &'static str), wgpu::ComputePipeline>>,
    names: Mutex<HashMap<wgpu::ComputePipeline, &'static str>>,
    scratch: Mutex<HashMap<(&'static str, usize), wgpu::Buffer>>,
    uploaded: Mutex<Lru<Input, wgpu::Buffer>>,
    held: Mutex<HashMap<String, (Option<i64>, wgpu::Buffer)>>,
    /// MKR_POST_PROFILE=passes: a timestamp after every pass, named by its kernel's entry point.
    passes: bool,
}

impl Default for Resources {
    fn default() -> Self {
        Resources {
            kernels: Mutex::new(HashMap::new()),
            names: Mutex::new(HashMap::new()),
            scratch: Mutex::new(HashMap::new()),
            uploaded: Mutex::new(Lru::new(UPLOADED_BYTES)),
            held: Mutex::new(HashMap::new()),
            passes: std::env::var("MKR_POST_PROFILE").is_ok_and(|v| v == "passes"),
        }
    }
}

// ---------------------------------------------------------------------------------------------------- drawing
/// What an op draws with: the device, the frame's command encoder, the pictures (one vec4<f32> per pixel, rgb and a
/// fourth component the kernels leave alone) and the shared resources.
pub struct Ctx<'a> {
    pub gpu: &'a Gpu,
    pub enc: &'a mut wgpu::CommandEncoder,
    pub w: u32,
    pub h: u32,
    /// The picture as it reaches the op.
    pub src: &'a wgpu::Buffer,
    /// A picture buffer of the same size to draw into.
    pub dst: &'a wgpu::Buffer,
    /// 256 floats, v / 255 as numpy makes it: what an 8-bit sample of a packed input stands for (`levels` on the
    /// CPU, rising).
    pub lut: &'a wgpu::Buffer,
    pub levels: &'a [f32],
    pub res: &'a Resources,
    /// MKR_POST_PROFILE: the frame's timestamp queries and what each one written so far closes (`mark`).
    pub stamps: Option<(&'a wgpu::QuerySet, &'a mut Vec<String>)>,
}

/// Timestamps a frame may write under MKR_POST_PROFILE (the rest of a frame with more steps goes untimed).
pub const STAMPS: u32 = 2048;

impl Ctx<'_> {
    /// The pipeline of entry point `entry` of the kernel source `src` (shaders/fx_<kind>.wgsl), compiled once.
    pub fn kernel(&self, name: &'static str, src: &str, entry: &'static str) -> wgpu::ComputePipeline {
        self.res.kernels.lock().unwrap().entry((name, entry)).or_insert_with(|| {
            let k = self.gpu.kernel(name, src, entry);
            self.res.names.lock().unwrap().insert(k.clone(), entry);
            k
        }).clone()
    }

    /// A storage buffer of at least `bytes` known by `label` (one label per buffer an op needs at once), kept for the
    /// frames after; its contents are whatever the last user left.
    pub fn scratch(&self, label: &'static str, bytes: u64) -> wgpu::Buffer {
        self.scratch_at(label, 0, bytes)
    }

    /// The `i`th of the scratch buffers known by `label` (an op that needs a varying number of them at once).
    pub fn scratch_at(&self, label: &'static str, i: usize, bytes: u64) -> wgpu::Buffer {
        let mut pool = self.res.scratch.lock().unwrap();
        if let Some(b) = pool.get(&(label, i)) {
            if b.size() >= bytes {
                return b.clone();
            }
        }
        let b = self.gpu.storage(label, bytes);
        pool.insert((label, i), b.clone());
        b
    }

    /// The picture known by `key` an effect holds from frame to frame (`bytes` long) and the frame it is of (None:
    /// none yet, or a picture of another size).
    pub fn held(&self, key: &str, bytes: u64) -> (wgpu::Buffer, Option<i64>) {
        let mut pool = self.res.held.lock().unwrap();
        if let Some((frame, b)) = pool.get(key) {
            if b.size() == bytes.max(16) {
                return (b.clone(), *frame);
            }
        }
        let b = self.gpu.storage(key, bytes);
        pool.insert(key.to_owned(), (None, b.clone()));
        (b, None)
    }

    /// Say which frame the picture `key` holds is of now.
    pub fn hold(&self, key: &str, frame: i64) {
        if let Some(entry) = self.res.held.lock().unwrap().get_mut(key) {
            entry.0 = Some(frame);
        }
    }

    /// `input` on the GPU (as `decoded` holds it): uploaded once while it stays in the cache (a PNG), or for this frame
    /// (floats Python wrote for it). A figure layer goes over as its luma bytes and is widened here, in the frame's
    /// work, to the packed words the kernels read (the byte in r, g and b, alpha 255: pngio::Kind::Luma's).
    pub fn upload(&mut self, input: &Input, decoded: &Decoded) -> wgpu::Buffer {
        let cached = matches!(input, Input::Png(..) | Input::Kept { .. });
        if cached {
            if let Some(b) = self.res.uploaded.lock().unwrap().get(input) {
                return b;
            }
        }
        let b = match decoded {
            Decoded::Figure(luma, _) => {
                let n = luma.len() as u32;
                let bytes = self.gpu.storage_init("luma", luma);
                let wide = self.gpu.storage("input", n as u64 * 4);
                let k = self.kernel("unpack", UNPACK, "widen_luma");
                let u = self.gpu.uniform(bytemuck::bytes_of(&[n, 0u32, 0, 0]));
                self.pass(&k, &[(0, &u), (1, &bytes), (6, &wide)], (n.div_ceil(4).div_ceil(256), 1));
                wide
            }
            _ => self.gpu.storage_init("input", decoded.bytes()),
        };
        if cached {
            self.res.uploaded.lock().unwrap().put(input.clone(), b.clone(), b.size() as usize);
        }
        b
    }

    /// A uniform buffer holding `v` (a pass's parameters).
    pub fn uniform<T: Pod>(&self, v: &T) -> wgpu::Buffer {
        self.gpu.uniform(bytemuck::bytes_of(v))
    }

    /// A storage buffer holding `data` for this frame (a table of parameters).
    pub fn table<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        self.gpu.storage_init(label, bytemuck::cast_slice(data))
    }

    /// One dispatch of `pipeline` over `groups` with the buffers `bind` (binding number, buffer) in its group 0
    /// (MKR_POST_PROFILE=passes: a timestamp after it, `k_<entry point>`).
    pub fn pass(&mut self, pipeline: &wgpu::ComputePipeline, bind: &[(u32, &wgpu::Buffer)], groups: (u32, u32)) {
        self.dispatch(pipeline, bind, Dispatch::Groups(groups.0, groups.1));
    }

    /// One dispatch of `pipeline` as `how` says (a count, or one a kernel before it wrote), as `pass`.
    pub fn dispatch(&mut self, pipeline: &wgpu::ComputePipeline, bind: &[(u32, &wgpu::Buffer)], how: Dispatch) {
        self.gpu.dispatch(self.enc, pipeline, bind, how);
        if self.res.passes && self.stamps.is_some() {
            let name = self.res.names.lock().unwrap().get(pipeline).copied().unwrap_or("?");
            self.mark(&format!("k_{name}"));
        }
    }

    /// A buffer of `n` indirect dispatches known by `label` (Gpu::dispatches), kept for the frames after.
    pub fn dispatches(&self, label: &'static str, n: u64) -> wgpu::Buffer {
        let mut pool = self.res.scratch.lock().unwrap();
        if let Some(b) = pool.get(&(label, usize::MAX)) {
            if b.size() >= 12 * n {
                return b.clone();
            }
        }
        let b = self.gpu.dispatches(label, n);
        pool.insert((label, usize::MAX), b.clone());
        b
    }

    /// Workgroups of 16 x 16 covering the picture.
    pub fn tiles(&self) -> (u32, u32) {
        tiles(self.w, self.h)
    }

    /// With MKR_POST_PROFILE: a GPU timestamp closing a stage of an op (`what`), reported with the frame's others;
    /// nothing otherwise.
    pub fn mark(&mut self, what: &str) {
        if let Some((set, marks)) = &mut self.stamps {
            if (marks.len() as u32) < STAMPS {
                self.enc.write_timestamp(set, marks.len() as u32);
                marks.push(what.replace(' ', "_"));
            }
        }
    }

    /// The first `bytes` of `buf` as the GPU has them once everything encoded so far has run: what was encoded is
    /// submitted and waited for (a frame's work that depends on what the GPU found, a figure's box), and the frame's
    /// encoding goes on in a new encoder.
    pub fn read_back(&mut self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_back"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        let next = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let done = std::mem::replace(&mut *self.enc, next);
        self.gpu.queue.submit([done.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).context("GPU wait")?;
        rx.recv().context("GPU read back")?.context("GPU read back")?;
        let out = slice.get_mapped_range().context("GPU read back")?[..bytes as usize].to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Copy the picture as it reaches the op into `dst` (an op that draws over part of it).
    pub fn copy_src_to_dst(&mut self) {
        let bytes = self.w as u64 * self.h as u64 * 16;
        self.enc.copy_buffer_to_buffer(self.src, 0, self.dst, 0, bytes);
    }
}
