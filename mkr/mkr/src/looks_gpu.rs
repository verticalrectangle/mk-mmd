//! The looks' passes on the GPU (looks.rs says what each draws and the file they go to), drawn on the scene
//! `Renderer::stage` poses on the frame's own time (no motion blur: styles.Looks._film turns it off):
//!
//!   flat    Workbench, flat light: the subject and the hard accents opaque in their colours (linear, the Object.color
//!           the look sets), depth-tested, back faces culled when the scene's shading culls them; straight RGBA through
//!           the Standard view
//!   matte   the subject's coverage alone, the same way, at twice the frame's size
//!   type    EEVEE with the lights off and a black world: the type objects' materials (their emission and their
//!   knock   transparency, as the lit pass draws them), the type held out by the subject and the hard accents, the
//!           knock-out objects over everything; the look's samples and EEVEE's film filter; straight RGBA through the
//!           Standard view
//!   soft    EEVEE: the soft accents' coverage (hashed as the lit pass tests it) and the opaque share of their Mix
//!           Shader (an AOV), held out by the subject and the hard accents
//!   id      the vector look at twice the frame's size: each material's id where it is drawn, no antialiasing, and the
//!           planar depth of the same draw (a material under half opaque writes the depth alone; of two surfaces at one
//!           depth the first drawn stays)
//!   shade   Workbench's studio light (the scene's, fixed in the world) on white times its shadow factor; nothing inside
//!           the stencil shadow volumes the look's light casts, extruded from every casting mesh's edges as Workbench
//!           extrudes a mesh that is not closed (every face casts, an edge between two faces counts twice)
//!   tex     the texture colour: each material's Workbench texture (cut where its alpha is under 0.1) or its viewport
//!           colour, materials not wholly opaque blended over; through the Standard view
//!
//! Workbench's antialiasing (render_aa "16"): its 16 sample points (BLI_jitter_init's, the one nearest the middle moved
//! onto it), each spread over its 3 x 3 neighbours with the Blackman-Harris weight (2 px wide) of twice its offset,
//! summed in log2 in half floats rounded toward zero and resolved as exp2(sum / weights) - 0.5 (look_film.wgsl), the
//! samples in the order that gives Blender's own levels (a whole pixel's alpha 254 of 255, a flat #1f1d2e 32 30 47:
//! measured on Blender's passes of a Crimson silhouette).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::Vec2;

use crate::camera::{Cam, render_size};
use crate::film::{self, Sample, TAP_COUNT};
use crate::looks::{Class, Kind, LookJob, LookSet, OutputLooks, PassData, Role, casts_shadow, wait_for_room,
                   workbench_material, write_passes};
use crate::lut::ViewLut;
use crate::pipelines::{self, COLOR_FORMAT, DEPTH_FORMAT, Immediate};
use crate::render::{Renderer, SampleGpu, mat, sized, storage};
use crate::scene::ShapeKind;

const LOOK_DRAW_WGSL: &str = include_str!("shaders/look_draw.wgsl");
const LOOK_FILM_WGSL: &str = include_str!("shaders/look_film.wgsl");
const STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTHV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Workbench draws a material transparent when its alpha is not 255 of 255.
const OPAQUE_TEX: f32 = 254.5 / 255.0;
/// The vector look's passes are drawn at this many times the frame's size (mkmmd.core.shotstyle.VECTOR_SCALE), and so
/// is a matte.
const SCALE: u32 = 2;

// ================================================================================================ Workbench's samples
/// One of Workbench's antialiasing samples: where its point sits from the pixel centre (pixels, x right, y down) and
/// the weight it counts for in each of the 3 x 3 neighbours, entry (dx + 1) * 3 + (dy + 1).
#[derive(Clone, Debug)]
pub(crate) struct WbSample {
    pub jitter: Vec2,
    pub weights: [f32; 9],
}

fn blackman_harris(r: f64, width: f64) -> f64 {
    if r > width * 0.5 {
        return 0.0;
    }
    let t = std::f64::consts::TAU * (r / width + 0.5).clamp(0.0, 1.0);
    0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
}

/// Workbench's 16 samples (BLI_jitter_init: its 48-bit generator and relaxation, in single precision as Blender
/// computes them; the sample nearest the middle moved onto it) in the order they are accumulated, and the sum of all
/// their weights.
pub(crate) fn workbench_samples() -> (Vec<WbSample>, f32) {
    const NUM: usize = 16;
    let mut state: u64 = ((31415926u64 + NUM as u64) << 16) | 0x330E;
    let mut rand = || {
        state = (0x5DEECE66Du64.wrapping_mul(state).wrapping_add(0xB)) & 0xFFFF_FFFF_FFFF;
        (state >> 17) as f64 / 2147483648.0
    };
    let sq = (NUM as f64).sqrt() as f32;
    let rad1 = 1.0 / sq;
    let rad2 = 1.0 / NUM as f32;
    let rad3 = sq / NUM as f32;
    let mut jit: Vec<[f32; 2]> = Vec::with_capacity(NUM);
    let mut x = 0.0f32;
    for i in 0..NUM {
        let a = x + rad1 * (0.5 - rand()) as f32;
        let b = (i as f64 / NUM as f64) as f32 + rad1 * (0.5 - rand()) as f32;
        jit.push([a, b]);
        x += rad3;
        x -= x.floor();
    }
    let near = |vx: f32, vy: f32, r: f32, d: &mut [f32; 2]| {
        if vx.abs() < r && vy.abs() < r {
            let mut len = (vx * vx + vy * vy).sqrt();
            if len > 0.0 && len < r {
                len /= r;
                d[0] += vx / len;
                d[1] += vy / len;
            }
        }
    };
    let jitterate1 = |jit: &mut Vec<[f32; 2]>, r: f32| {
        let old = jit.clone();
        for (i, p) in old.iter().enumerate() {
            let mut d = [0.0f32; 2];
            for j in (0..NUM).rev() {
                if i == j {
                    continue;
                }
                let mut vx = (old[j][0] - p[0]) - 1.0;
                let mut vy = (old[j][1] - p[1]) - 1.0;
                for _ in 0..3 {
                    near(vx, vy, r, &mut d);
                    vx += 1.0;
                    near(vx, vy, r, &mut d);
                    vx += 1.0;
                    near(vx, vy, r, &mut d);
                    vx -= 2.0;
                    vy += 1.0;
                }
            }
            let nx = p[0] - d[0] / 18.0;
            let ny = p[1] - d[1] / 18.0;
            jit[i] = [nx - nx.floor(), ny - ny.floor()];
        }
    };
    let jitterate2 = |jit: &mut Vec<[f32; 2]>, r: f32| {
        let old = jit.clone();
        for (i, p) in old.iter().enumerate() {
            let (mut dx, mut dy) = (0.0f32, 0.0f32);
            for j in (0..NUM).rev() {
                if i == j {
                    continue;
                }
                let mut vx = (old[j][0] - p[0]) - 1.0;
                let mut vy = (old[j][1] - p[1]) - 1.0;
                for _ in 0..3 {
                    if vx.abs() < r {
                        dx += vx * r;
                    }
                    vx += 1.0;
                }
                for _ in 0..3 {
                    if vy.abs() < r {
                        dy += vy * r;
                    }
                    vy += 1.0;
                }
            }
            let nx = p[0] - dx / 2.0;
            let ny = p[1] - dy / 2.0;
            jit[i] = [nx - nx.floor(), ny - ny.floor()];
        }
    };
    for _ in 0..24 {
        jitterate1(&mut jit, rad1);
        jitterate1(&mut jit, rad1);
        jitterate2(&mut jit, rad2);
    }
    for p in &mut jit {
        *p = [p[0] - 0.5, p[1] - 0.5];
    }
    let (mut ci, mut cd) = (0usize, 1.0f32);
    for (i, p) in jit.iter().enumerate() {
        let d = p[0] * p[0] + p[1] * p[1];
        if d < cd {
            cd = d;
            ci = i;
        }
    }
    let c = jit[ci];
    let mut out: Vec<WbSample> = jit
        .iter()
        .map(|p| {
            let o = [((p[0] - c[0]) + 0.5) as f64 % 1.0 * 2.0 - 1.0, ((p[1] - c[1]) + 0.5) as f64 % 1.0 * 2.0 - 1.0];
            let mut weights = [0.0f32; 9];
            for dx in -1i32..=1 {
                for dy in -1i32..=1 {
                    let r = (dx as f64 - o[0]).hypot(dy as f64 + o[1]);
                    weights[((dx + 1) * 3 + (dy + 1)) as usize] = blackman_harris(r, 2.0) as f32;
                }
            }
            WbSample { jitter: Vec2::new((-o[0] / 2.0) as f32, (o[1] / 2.0) as f32), weights }
        })
        .collect();
    // the accumulation order matters to the half floats' rounding: this one gives Blender's levels
    out.reverse();
    let total = out.iter().map(|s| s.weights.iter().sum::<f32>()).sum();
    (out, total)
}

// ===================================================================================================== GPU records
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct LookDrawGpu {
    color: [f32; 4],
    info: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LookLightGpu {
    dirs: [[f32; 4]; 4],
    diffuse: [[f32; 4]; 4],
    ambient: [f32; 4],
    shadow: [f32; 4],
    params: [f32; 4],
    plane: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VolEdgeGpu {
    v: [u32; 4],
    m: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WbAccumulateGpu {
    w: [f32; 12],
    width: u32,
    height: u32,
    first: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EeAccumulateGpu {
    taps: [f32; 28],
    width: u32,
    height: u32,
    first: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveGpu {
    width: u32,
    height: u32,
    mode: u32,
    lut_n: u32,
    lo: f32,
    hi: f32,
    total: f32,
    pad: f32,
    zero: [f32; 4],
}

/// The film resolves (look_film.wgsl `resolve`).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Flat = 0,
    Coverage = 1,
    Shade = 2,
    Tex = 3,
    Rgba = 4,
    Soft = 5,
}

// ======================================================================================================== the kit
/// A frame's targets at one size.
struct Targets {
    w: u32,
    h: u32,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    stencil: wgpu::TextureView,
    normals: wgpu::TextureView,
    id: wgpu::Texture,
    depthv: wgpu::Texture,
    _film: wgpu::Buffer,
    pixels: wgpu::Buffer,
    readback: wgpu::Buffer,
    acc_bg: wgpu::BindGroup,
    res_bg: wgpu::BindGroup,
    draw_bg: wgpu::BindGroup,
    draw_bg_pre: wgpu::BindGroup,
}

/// An emission pass's pipelines per material (EEVEE: type, knock): the hashed prepass of a transparent one, the
/// forward pass, a blended one's pass keeping its alpha.
#[derive(Default)]
struct Emissive {
    pre: BTreeMap<usize, wgpu::RenderPipeline>,
    main: BTreeMap<usize, wgpu::RenderPipeline>,
    blend: BTreeMap<usize, wgpu::RenderPipeline>,
    soft: BTreeMap<usize, wgpu::RenderPipeline>,
}

pub(crate) struct LookKit {
    flat: [wgpu::RenderPipeline; 2],
    id: [wgpu::RenderPipeline; 2],
    id_depth: [wgpu::RenderPipeline; 2],
    normals: [wgpu::RenderPipeline; 2],
    vol_fail: [wgpu::RenderPipeline; 2],
    vol_pass: wgpu::RenderPipeline,
    light: [wgpu::RenderPipeline; 2],
    tex: [wgpu::RenderPipeline; 2],
    tex_blend: [wgpu::RenderPipeline; 2],
    emissive: Emissive,
    draw_bgl: wgpu::BindGroupLayout,
    draws_buf: wgpu::Buffer,
    light_buf: wgpu::Buffer,
    edges: wgpu::Buffer,
    caps: wgpu::Buffer,
    wb_acc: wgpu::ComputePipeline,
    ee_acc: wgpu::ComputePipeline,
    acc_bgl: wgpu::BindGroupLayout,
    resolve: wgpu::ComputePipeline,
    res_bgl: wgpu::BindGroupLayout,
    lut: ViewLut,
    lut_buf: wgpu::Buffer,
    targets: Vec<Targets>,
    /// Per shape: its shadow volumes' edges and caps (vertex indices of the shape, triangles drawn: the indices
    /// they came from, a frame-by-frame shape's lit.bin offset).
    volumes: BTreeMap<usize, (usize, Vec<[u32; 4]>, Vec<[u32; 3]>)>,
    /// The screen type's kit (layers_gpu.rs), made for the first silhouette frame with knock-out type on screen.
    screen: Option<crate::layers_gpu::LayerKit>,
}

/// The pipelines of one shader module with the looks' layout (the scene's group and the looks' own).
struct Maker<'a> {
    device: &'a wgpu::Device,
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    cache: Option<&'a wgpu::PipelineCache>,
}

struct PipeDesc<'a> {
    vs: &'a str,
    fs: Option<&'a str>,
    targets: Vec<Option<wgpu::ColorTargetState>>,
    cull: bool,
    depth: Option<wgpu::DepthStencilState>,
}

impl Maker<'_> {
    fn pipe(&self, label: &str, d: PipeDesc<'_>) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState { module: &self.module, entry_point: Some(d.vs), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: d.cull.then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: d.depth,
            multisample: Default::default(),
            fragment: d.fs.map(|fs| wgpu::FragmentState {
                module: &self.module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &d.targets,
            }),
            multiview_mask: None,
            cache: self.cache,
        })
    }
}

fn target(format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>, write: wgpu::ColorWrites) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState { format, blend, write_mask: write })
}

fn depth_state(format: wgpu::TextureFormat, write: bool, compare: wgpu::CompareFunction, stencil: wgpu::StencilState)
               -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil,
        bias: Default::default(),
    })
}

fn face(compare: wgpu::CompareFunction, depth_fail: wgpu::StencilOperation, pass: wgpu::StencilOperation) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState { compare, fail_op: wgpu::StencilOperation::Keep, depth_fail_op: depth_fail, pass_op: pass }
}

fn stencil(front: wgpu::StencilFaceState, back: wgpu::StencilFaceState, write: bool) -> wgpu::StencilState {
    wgpu::StencilState { front, back, read_mask: 0xff, write_mask: if write { 0xff } else { 0 } }
}

/// The layer passes' "over" for a blended material keeping its alpha: premultiplied colour plus the destination times
/// the transmittance (dual source), the alpha likewise.
const OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::Src1, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::Src1Alpha, operation: wgpu::BlendOperation::Add },
};

/// Workbench's transparent texture colour over what is drawn: alpha-blended colour, the alpha accumulated.
const TEX_OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
};

fn storage_entry(binding: u32, read_only: bool, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn compute(device: &wgpu::Device, module: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout, entry: &str, immediate: u32,
           cache: Option<&wgpu::PipelineCache>) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: immediate,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache,
    })
}

/// The look counts mkr's report gives.
#[derive(serde::Serialize, Default, Debug)]
pub struct LookCounts {
    pub frames: usize,
    pub plates: usize,
    pub mattes: usize,
    pub silhouette: usize,
    pub vector: usize,
    pub seconds: f64,
}

/// The draws (indices into the renderer's draws) of the meshes a class of a look holds that show on the frame.
struct Draws {
    by_class: BTreeMap<u8, Vec<usize>>,
}

impl Draws {
    fn of(&self, classes: &[Class]) -> Vec<usize> {
        let mut out: Vec<usize> = classes.iter().flat_map(|c| self.by_class.get(&(*c as u8)).cloned().unwrap_or_default()).collect();
        out.sort_unstable();
        out
    }
}

impl Renderer<'_> {
    /// Draw the passes of look jobs `ids` of `output` (a `size` frame at `percent`) and write them into `spool`, waiting
    /// while `limit` pass files are there (looks.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_looks(&mut self, output: &str, size: [u32; 2], percent: u32, looks: &OutputLooks, ids: &[usize],
                      spool: &Path, limit: usize) -> Result<LookCounts> {
        let t0 = std::time::Instant::now();
        let [w, h] = render_size(size, percent);
        if w == 0 || h == 0 {
            bail!("--percent {percent} leaves no pixels");
        }
        let mut counts = LookCounts::default();
        if ids.is_empty() {
            return Ok(counts);
        }
        let sets: Vec<&LookSet> = ids.iter().map(|&i| looks.job(i).map(|j| looks.set_of(j))).collect::<Result<_>>()?;
        let mut kit = self.look_kit(&sets)?;
        let (wb, total) = workbench_samples();
        let ee_max = sets.iter().map(|s| s.samples as usize).max().unwrap_or(16);
        self.render_bind_group(wb.len().max(ee_max).max(1));
        let mut order: Vec<usize> = ids.to_vec();
        order.sort_by_key(|&i| looks.jobs[i].frame);
        for id in order {
            let job = looks.job(id)?;
            let set = looks.set_of(job);
            let passes = self.look_job(&mut kit, output, job, set, &wb, total, w, h)?;
            wait_for_room(spool, limit)?;
            write_passes(spool, id, &passes)?;
            match job.role {
                Role::Frame => counts.frames += 1,
                Role::Matte => counts.mattes += 1,
                Role::Plate | Role::Bare => counts.plates += 1,
            }
            match set.kind {
                Kind::Silhouette => counts.silhouette += 1,
                Kind::Vector => counts.vector += 1,
            }
        }
        counts.seconds = (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0;
        Ok(counts)
    }

    /// One job's passes (module docstring).
    #[allow(clippy::too_many_arguments)]
    fn look_job(&mut self, kit: &mut LookKit, output: &str, job: &LookJob, set: &LookSet, wb: &[WbSample], total: f32,
                w: u32, h: u32) -> Result<Vec<PassData>> {
        let k = self.bake.k(job.frame).with_context(|| format!("frame {} is outside the clip", job.frame))?;
        let kf = k as f64;
        let seed = (job.frame as u32).wrapping_mul(0x9e37_79b9);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("look stage") });
        let rays = !self.blas.iter().flatten().any(|b| b.built);
        let staged = self.stage(&mut enc, output, k, [kf, kf, kf], false, rays, &[])?;
        self.gpu.queue.submit(Some(enc.finish()));
        let cam = self.camera_at(&job.camera, kf)?;
        let draws = self.look_draws(set, &staged.shown);
        let mut out = Vec::new();
        match set.kind {
            Kind::Silhouette => {
                self.write_look_draws(kit, set, None)?;
                let flat = draws.of(&[Class::Subject, Class::Hard]);
                out.push(self.wb_pass(kit, &cam, w, h, wb, total, seed, Mode::Flat, &flat, "flat")?);
                if job.role == Role::Matte {
                    out.push(self.wb_pass(kit, &cam, w * SCALE, h * SCALE, wb, total, seed, Mode::Coverage,
                                          &draws.of(&[Class::Subject]), "matte")?);
                }
                if job.role == Role::Frame && crate::layers::screen_knockout_live(self.bake, output, job.frame) {
                    // the cut's own frame: its knock-out screen type, reversed out of the figure (styles.Looks)
                    if kit.screen.is_none() {
                        let flat = crate::layers::Flat { frame: job.frame, k, figures: Vec::new(), keep: None,
                                                          screen: Some((PathBuf::new(), true)), negative: None };
                        kit.screen = self.flat_kit(output, std::slice::from_ref(&flat), &[])?;
                    }
                    let sk = kit.screen.as_mut().context("no screen kit")?;
                    let bytes = self.screen_knock(sk, output, w, h, job.frame, k)?;
                    out.push(PassData { name: "screen_knock", shape: vec![h as usize, w as usize, 4], kind: "u8", bytes });
                }
                let holdout = draws.of(&[Class::Subject, Class::Hard]);
                let soft = draws.of(&[Class::Soft]);
                if !soft.is_empty() {
                    out.push(self.ee_pass(kit, &cam, w, h, set.samples, seed, &holdout, &soft, Mode::Soft, "soft")?);
                }
                let typ = draws.of(&[Class::Type]);
                if !typ.is_empty() {
                    out.push(self.ee_pass(kit, &cam, w, h, set.samples, seed, &holdout, &typ, Mode::Rgba, "type")?);
                }
                let knock = draws.of(&[Class::Knock]);
                if !knock.is_empty() {
                    out.push(self.ee_pass(kit, &cam, w, h, set.samples, seed, &[], &knock, Mode::Rgba, "knock")?);
                }
            }
            Kind::Vector => {
                let (vw, vh) = (w * SCALE, h * SCALE);
                let subject = draws.of(&[Class::Subject]);
                self.write_look_draws(kit, set, Some(&staged.shown))?;
                let (ids, depth) = self.id_pass(kit, &cam, vw, vh, seed, &subject)?;
                out.push(ids);
                out.push(depth);
                out.push(self.shade_pass(kit, &cam, vw, vh, wb, total, seed, set, &subject, &staged.shown)?);
                if set.tex {
                    out.push(self.wb_pass(kit, &cam, vw, vh, wb, total, seed, Mode::Tex, &subject, "tex")?);
                }
                if job.role == Role::Matte {
                    // a matte's figure: the subject's coverage at twice the frame's size, as a silhouette's
                    out.push(self.wb_pass(kit, &cam, vw, vh, wb, total, seed, Mode::Coverage, &subject, "matte")?);
                }
                let typ = draws.of(&[Class::Type]);
                if !typ.is_empty() {
                    out.push(self.ee_pass(kit, &cam, w, h, set.samples, seed, &subject, &typ, Mode::Rgba, "type")?);
                }
            }
        }
        Ok(out)
    }

    /// Per class of the look, the draws of its meshes shown on the frame (camera-visible, with triangles).
    fn look_draws(&self, set: &LookSet, shown: &[bool]) -> Draws {
        let lit = self.bake.doc.lit.as_ref().expect("lit bake");
        let mut by_class: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            if !shown.get(mi).copied().unwrap_or(false) || !m.visible.camera {
                continue;
            }
            let Some(class) = set.class_of(lit.meshes[mi].record_name()) else { continue };
            let base = self.draw_base[mi] as usize;
            for di in base..base + m.slots.len() {
                if self.draws[di].tri_count > 0 {
                    by_class.entry(class as u8).or_default().push(di);
                }
            }
        }
        Draws { by_class }
    }

    /// The looks' per-draw records: a silhouette's flat colours; a vector look's material ids and Workbench texture
    /// colour (its image, UV slot and sampling, or its viewport colour and alpha).
    fn write_look_draws(&self, kit: &LookKit, set: &LookSet, _shown: Option<&[bool]>) -> Result<()> {
        let lit = self.bake.lit()?;
        let mut recs = vec![LookDrawGpu::default(); self.draws.len().max(1)];
        let image_index: BTreeMap<&str, usize> = self.mats.images.iter().enumerate().map(|(i, (n, _))| (n.as_str(), i)).collect();
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            let Some(class) = set.class_of(lit.meshes[mi].record_name()) else { continue };
            let shape = &self.geo.shapes[m.shape];
            let base = self.draw_base[mi] as usize;
            for (si, slot) in m.slots.iter().enumerate() {
                let r = &mut recs[base + si];
                match set.kind {
                    Kind::Silhouette => {
                        let c = if class == Class::Hard { set.accent_color } else { set.subject_color };
                        r.color = [c[0], c[1], c[2], 1.0];
                    }
                    Kind::Vector => {
                        let name = slot.material.map(|i| self.geo.materials[i].as_str());
                        r.info = [name.and_then(|n| set.ids.get(n)).copied().unwrap_or(0), u32::MAX, u32::MAX, 0];
                        r.color = [0.8, 0.8, 0.8, 1.0];
                        let Some(name) = name else { continue };
                        let Some(wbm) = workbench_material(self.bake, name) else { continue };
                        r.color = [wbm.color[0], wbm.color[1], wbm.color[2], wbm.alpha];
                        if wbm.texture == "none" || wbm.texture.is_empty() {
                            continue;
                        }
                        let Some(img) = wbm.image.as_deref().and_then(|n| image_index.get(n)) else { continue };
                        let uv = shape.attr_names.iter().position(|a| a == "uv:").or_else(|| shape.attr_names.iter().position(|a| a.starts_with("uv:")));
                        let Some(uv) = uv else { continue };
                        let ext = match wbm.extension.as_deref() {
                            Some("EXTEND") | Some("CLIP") => 1u32,
                            Some("MIRROR") => 2,
                            _ => 0,
                        };
                        let flags = (wbm.premul as u32) | ((wbm.cutoff as u32) << 1)
                            | (((wbm.extension.as_deref() == Some("CLIP")) as u32) << 2) | ((wbm.closest as u32) << 3) | (ext << 4);
                        r.info = [r.info[0], *img as u32, uv as u32, flags];
                    }
                }
            }
        }
        self.gpu.queue.write_buffer(&kit.draws_buf, 0, bytemuck::cast_slice(&recs));
        Ok(())
    }

    /// The frame record and the samples of a pass through `cam` on a `w` x `h` picture (lights off, no motion blur).
    fn look_frame(&self, cam: &Cam, w: u32, h: u32, jitter: &[Vec2], seed: u32) {
        let frame = self.frame_record(cam, w, h, 0, jitter.len() as u32, false, seed);
        let queue = &self.gpu.queue;
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        let view = cam.view();
        let sgpu: Vec<SampleGpu> = jitter.iter().enumerate().map(|(si, j)| {
            let n = si as u32 + 1;
            let vp = cam.projection(w as f32, h as f32, *j) * view;
            SampleGpu {
                vp: mat(&vp),
                vp_inv: mat(&vp.as_dmat4().inverse().as_mat4()),
                info: [1, si as u32, 0, 0],
                t: [0.0, j.x, j.y, 0.0],
                rand: [film::halton(n, 7), film::halton(n, 11), film::halton(n, 13), film::halton(n, 17)],
                ..bytemuck::Zeroable::zeroed()
            }
        }).collect();
        queue.write_buffer(&self.samples_buf, 0, bytemuck::cast_slice(&sgpu));
    }

    /// The looks' light record: Workbench's studio light (the scene's shading, rotated about Z, in glTF's axes), the
    /// look's shadow direction and the planar depth of `cam`.
    fn write_look_light(&self, kit: &LookKit, cam: &Cam, set: &LookSet) {
        let wbr = self.bake.raw.get("workbench");
        let rot = wbr.and_then(|w| w.get("rotate_z")).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let (sr, cr) = rot.sin_cos();
        let gl = |v: [f64; 3]| [v[0] as f32, v[2] as f32, -v[1] as f32];
        let mut dirs = [[1.0f32, 0.0, 0.0, 0.0]; 4];
        let mut diffuse = [[0.0f32; 4]; 4];
        if let Some(lights) = wbr.and_then(|w| w.get("lights")).and_then(|l| l.as_array()) {
            for (i, l) in lights.iter().take(4).enumerate() {
                if !l.get("use").and_then(|u| u.as_bool()).unwrap_or(false) {
                    continue;
                }
                let d = l.get("direction").and_then(|d| d.as_array()).map(|a| [0, 1, 2].map(|k| a[k].as_f64().unwrap_or(0.0))).unwrap_or([0.0; 3]);
                // the studio light's axes in Blender's world: (x, -z, y), turned about Z
                let b = [d[0], -d[2], d[1]];
                let g = gl([b[0] * cr + b[1] * sr, -b[0] * sr + b[1] * cr, b[2]]);
                let wrap = l.get("wrap").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                dirs[i] = [g[0], g[1], g[2], wrap];
                let red = l.get("diffuse").and_then(|d| d.as_array()).and_then(|a| a.first()).and_then(|v| v.as_f64()).unwrap_or(0.0);
                diffuse[i] = [red as f32, 0.0, 0.0, 0.0];
            }
        }
        let ambient = wbr.and_then(|w| w.get("ambient")).and_then(|a| a.as_array()).and_then(|a| a.first())
            .and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        // Workbench turns light_direction (x, y, z) into (-x, z, -y): its direction of travel, Blender's axes
        let l = set.light;
        let lt = gl([-l[0] as f64, l[2] as f64, -l[1] as f64]);
        // the view axis from the view matrix (glTF world -> Blender camera axes): planar depth is minus its third row
        let v = cam.view();
        let fwd = glam::Vec3::new(-v.x_axis.z, -v.y_axis.z, -v.z_axis.z);

        // as styles.Looks sets them: shadow_shift 0.1, shadow_focus 0 (Workbench's 1 - focus (1 - shift) of 1e-4)
        let (shift, focus) = (0.1f32, 1e-4f32);
        let rec = LookLightGpu {
            dirs,
            diffuse,
            ambient: [ambient, 0.0, 0.0, 0.0],
            shadow: [lt[0], lt[1], lt[2], shift],
            params: [1.0 - focus * (1.0 - shift), 1.0, 0.0, cam.far],
            plane: [fwd.x, fwd.y, fwd.z, -v.w_axis.z],
        };
        self.gpu.queue.write_buffer(&kit.light_buf, 0, bytemuck::bytes_of(&rec));
    }

    // ------------------------------------------------------------------------------------------------ the passes
    /// A Workbench pass (flat colours, coverage, texture colour) of `draws` through its 16 samples, resolved to bytes.
    #[allow(clippy::too_many_arguments)]
    fn wb_pass(&mut self, kit: &mut LookKit, cam: &Cam, w: u32, h: u32, wb: &[WbSample], total: f32, seed: u32, mode: Mode,
               draws: &[usize], name: &'static str) -> Result<PassData> {
        let ti = self.look_targets(kit, w, h);
        let jitter: Vec<Vec2> = wb.iter().map(|s| s.jitter).collect();
        self.look_frame(cam, w, h, &jitter, seed);
        let cull = self.workbench_cull() as usize;
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
        let t = &kit.targets[ti];
        let bg = self.render_bg.as_ref().context("no render bind group")?;
        let (opaque, blended): (Vec<usize>, Vec<usize>) = if mode == Mode::Tex {
            draws.iter().partition(|&&di| self.tex_alpha(di) >= OPAQUE_TEX)
        } else {
            (draws.to_vec(), Vec::new())
        };
        for (si, s) in wb.iter().enumerate() {
            {
                let mut pass = colour_pass(&mut enc, &t.color, Some(&t.depth), true);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &t.draw_bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_pipeline(if mode == Mode::Tex { &kit.tex[cull] } else { &kit.flat[cull] });
                for &di in &opaque {
                    self.look_draw(&mut pass, di, si);
                }
                if !blended.is_empty() {
                    pass.set_pipeline(&kit.tex_blend[cull]);
                    for &di in &blended {
                        self.look_draw(&mut pass, di, si);
                    }
                }
            }
            wb_accumulate(kit, &mut enc, ti, s, si == 0);
        }
        self.resolve_into(kit, &mut enc, ti, mode, total);
        let px = self.read_pixels(kit, enc, ti)?;
        Ok(match mode {
            Mode::Coverage => PassData { name, shape: vec![h as usize, w as usize], kind: "u8", bytes: px.iter().map(|p| *p as u8).collect() },
            Mode::Tex => PassData {
                name,
                shape: vec![h as usize, w as usize, 3],
                kind: "u8",
                bytes: px.iter().flat_map(|p| [*p as u8, (*p >> 8) as u8, (*p >> 16) as u8]).collect(),
            },
            _ => PassData { name, shape: vec![h as usize, w as usize, 4], kind: "u8", bytes: px.iter().flat_map(|p| p.to_le_bytes()).collect() },
        })
    }

    /// The vector look's material ids and depth: one unjittered sample, opaque materials first, those under half
    /// opaque into the depth alone (they only hide what is behind them).
    fn id_pass(&mut self, kit: &mut LookKit, cam: &Cam, w: u32, h: u32, seed: u32, draws: &[usize])
               -> Result<(PassData, PassData)> {
        let ti = self.look_targets(kit, w, h);
        self.look_frame(cam, w, h, &[Vec2::ZERO], seed);
        self.write_look_light(kit, cam, &LookSet { kind: Kind::Vector, hide: vec![], subject: vec![], hard: vec![],
            soft: vec![], kind_type: vec![], knock: vec![], subject_color: [0.0; 3], accent_color: [0.0; 3], samples: 1,
            ids: BTreeMap::new(), tex: false, light: [0.0, 0.0, 1.0] });
        let cull = self.workbench_cull() as usize;
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("look ids") });
        let t = &kit.targets[ti];
        let bg = self.render_bg.as_ref().context("no render bind group")?;
        let (opaque, faint): (Vec<usize>, Vec<usize>) = draws.iter().partition(|&&di| self.tex_alpha(di) >= 0.5);
        {
            let idv = t.id.create_view(&Default::default());
            let dv = t.depthv.create_view(&Default::default());
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("look ids"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &idv, depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } }),
                    Some(wgpu::RenderPassColorAttachment { view: &dv, depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &t.draw_bg, &[]);
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_pipeline(&kit.id[cull]);
            for &di in &opaque {
                self.look_draw(&mut pass, di, 0);
            }
            pass.set_pipeline(&kit.id_depth[cull]);
            for &di in &faint {
                self.look_draw(&mut pass, di, 0);
            }
        }
        let ids = self.read_texture(kit, enc, ti, true)?;
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("look depth") });
        enc.insert_debug_marker("depth readback");
        let depth = self.read_texture(kit, enc, ti, false)?;
        let n = (w * h) as usize;
        let mut id16 = Vec::with_capacity(n * 2);
        for v in &ids {
            if *v > u16::MAX as u32 {
                bail!("material id {v} beyond 65535");
            }
            id16.extend_from_slice(&(*v as u16).to_le_bytes());
        }
        let dbytes: Vec<u8> = depth.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok((PassData { name: "id", shape: vec![h as usize, w as usize], kind: "u16", bytes: id16 },
            PassData { name: "depth", shape: vec![h as usize, w as usize], kind: "f32", bytes: dbytes }))
    }

    /// Workbench's studio light on white times its shadow factor, nothing inside the look's light's shadow volumes.
    #[allow(clippy::too_many_arguments)]
    fn shade_pass(&mut self, kit: &mut LookKit, cam: &Cam, w: u32, h: u32, wb: &[WbSample], total: f32, seed: u32,
                  set: &LookSet, draws: &[usize], shown: &[bool]) -> Result<PassData> {
        let ti = self.look_targets(kit, w, h);
        let jitter: Vec<Vec2> = wb.iter().map(|s| s.jitter).collect();
        self.look_frame(cam, w, h, &jitter, seed);
        self.write_look_light(kit, cam, set);
        let (ne, nc) = self.upload_volumes(kit, set, shown)?;
        let ti = if kit.targets[ti].w == w && kit.targets[ti].h == h { ti } else { self.look_targets(kit, w, h) };
        let cull = self.workbench_cull() as usize;
        let l = set.light;
        let lt = [-l[0], -l[1], -l[2]];
        let v = cam.view();
        let fwd = glam::Vec3::new(-v.x_axis.z, -v.y_axis.z, -v.z_axis.z);
        let fail = fwd.x * lt[0] + fwd.y * lt[1] + fwd.z * lt[2] > 0.0;
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("look shade") });
        let t = &kit.targets[ti];
        let bg = self.render_bg.as_ref().context("no render bind group")?;
        for (si, s) in wb.iter().enumerate() {
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("look normals"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &t.normals, depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &t.stencil,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &t.draw_bg_pre, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_pipeline(&kit.normals[cull]);
                for &di in draws {
                    self.look_draw(&mut pass, di, si);
                }
            }
            if ne + nc > 0 {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("look volumes"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &t.stencil,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &t.draw_bg, &[]);
                let imm = Immediate { draw: 0, sample: si as u32 };
                let pipes: Vec<(&wgpu::RenderPipeline, u32)> =
                    if fail { vec![(&kit.vol_fail[0], ne), (&kit.vol_fail[1], nc)] } else { vec![(&kit.vol_pass, ne)] };
                for (pipe, count) in pipes {
                    pass.set_pipeline(pipe);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm));
                    pass.draw(0..6, 0..count);
                }
            }
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("look light"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &t.color, depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &t.stencil,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &t.draw_bg, &[]);
                pass.set_stencil_reference(0);
                for shadow in [false, true] {
                    pass.set_pipeline(&kit.light[shadow as usize]);
                    pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: shadow as u32, sample: si as u32 }));
                    pass.draw(0..3, 0..1);
                }
            }
            wb_accumulate(kit, &mut enc, ti, s, si == 0);
        }
        self.resolve_into(kit, &mut enc, ti, Mode::Shade, total);
        let px = self.read_pixels(kit, enc, ti)?;
        Ok(PassData { name: "shade", shape: vec![h as usize, w as usize], kind: "u8", bytes: px.iter().map(|p| *p as u8).collect() })
    }

    /// An EEVEE pass (type, knock, soft) of `draws` held out by `holdout`: the look's samples, EEVEE's film filter.
    #[allow(clippy::too_many_arguments)]
    fn ee_pass(&mut self, kit: &mut LookKit, cam: &Cam, w: u32, h: u32, samples: u32, seed: u32, holdout: &[usize],
               draws: &[usize], mode: Mode, name: &'static str) -> Result<PassData> {
        let ti = self.look_targets(kit, w, h);
        let smp: Vec<Sample> = film::samples(samples.max(1), &self.filter, self.params.jitter, None);
        let jitter: Vec<Vec2> = smp.iter().map(|s| s.jitter).collect();
        self.look_frame(cam, w, h, &jitter, seed);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
        let t = &kit.targets[ti];
        let bg = self.render_bg.as_ref().context("no render bind group")?;
        let view = cam.view();
        let mut solid = Vec::new();
        let mut blended = Vec::new();
        for &di in draws {
            let mat = self.draws[di].material as usize;
            if kit.emissive.blend.contains_key(&mat) && mode != Mode::Soft {
                let d = &self.draws[di];
                let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
                let c = (shape.lo + shape.hi) * 0.5;
                blended.push(((view * c.extend(1.0)).z, di));
            } else {
                solid.push(di);
            }
        }
        blended.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (si, s) in smp.iter().enumerate() {
            {
                let mut pass = depth_pass(&mut enc, &t.depth);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &t.draw_bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &di in holdout.iter().chain(&solid) {
                    let mat = self.draws[di].material as usize;
                    let pipe = match (&self.pipes.prepass[mat], kit.emissive.pre.get(&mat)) {
                        (_, Some(p)) => p,
                        (Some(p), None) => p,
                        _ => &self.pipes.depth[self.mats.list[mat].cull as usize],
                    };
                    pass.set_pipeline(pipe);
                    self.look_draw(&mut pass, di, si);
                }
            }
            {
                let mut pass = colour_pass(&mut enc, &t.color, Some(&t.depth), false);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &t.draw_bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &di in &solid {
                    let mat = self.draws[di].material as usize;
                    let pipe = if mode == Mode::Soft { kit.emissive.soft.get(&mat) } else { kit.emissive.main.get(&mat) };
                    let Some(pipe) = pipe else { continue };
                    pass.set_pipeline(pipe);
                    self.look_draw(&mut pass, di, si);
                }
                for &(_, di) in &blended {
                    let mat = self.draws[di].material as usize;
                    let m = &self.mats.list[mat];
                    if !m.show_back {
                        pass.set_pipeline(&self.pipes.depth_blend[m.cull as usize]);
                        self.look_draw(&mut pass, di, si);
                    }
                    pass.set_pipeline(&kit.emissive.blend[&mat]);
                    self.look_draw(&mut pass, di, si);
                }
            }
            ee_accumulate(kit, &mut enc, ti, s, si == 0);
        }
        self.resolve_into(kit, &mut enc, ti, mode, 1.0);
        let px = self.read_pixels(kit, enc, ti)?;
        Ok(match mode {
            Mode::Soft => PassData {
                name,
                shape: vec![h as usize, w as usize, 2],
                kind: "u8",
                bytes: px.iter().flat_map(|p| [*p as u8, (*p >> 8) as u8]).collect(),
            },
            _ => PassData { name, shape: vec![h as usize, w as usize, 4], kind: "u8", bytes: px.iter().flat_map(|p| p.to_le_bytes()).collect() },
        })
    }

    /// Draw `di` of the scene's draws with sample `si` (the draw index rides as the instance).
    fn look_draw(&self, pass: &mut wgpu::RenderPass<'_>, di: usize, si: usize) {
        let d = &self.draws[di];
        let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
        let first = shape.index_base + d.first_tri * 3;
        pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: di as u32, sample: si as u32 }));
        pass.draw_indexed(first..first + d.tri_count * 3, shape.vertex_base as i32, di as u32..di as u32 + 1);
    }

    /// The alpha Workbench draws a draw's material with (its viewport alpha; 1 without a record).
    fn tex_alpha(&self, di: usize) -> f32 {
        let d = &self.draws[di];
        let m = &self.geo.meshes[d.mesh as usize];
        let slot = m.slots.iter().zip(self.draw_base[d.mesh as usize] as usize..).find(|(_, i)| *i == di).map(|(s, _)| s);
        slot.and_then(|s| s.material).and_then(|i| workbench_material(self.bake, &self.geo.materials[i])).map(|w| w.alpha).unwrap_or(1.0)
    }

    /// Whether the scene's shading culls back faces (bake.json `workbench.cull`).
    fn workbench_cull(&self) -> bool {
        self.bake.raw.get("workbench").and_then(|w| w.get("cull")).and_then(|c| c.as_bool()).unwrap_or(false)
    }

    /// Upload the shadow volumes of the look's casting subject meshes shown on the frame: (edge instances, caps).
    fn upload_volumes(&self, kit: &mut LookKit, set: &LookSet, shown: &[bool]) -> Result<(u32, u32)> {
        let lit = self.bake.lit()?;
        let mut edges: Vec<VolEdgeGpu> = Vec::new();
        let mut twice: Vec<VolEdgeGpu> = Vec::new();
        let mut caps: Vec<[u32; 4]> = Vec::new();
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            if !shown.get(mi).copied().unwrap_or(false) || !m.visible.camera {
                continue;
            }
            let rec = lit.meshes[mi].record_name();
            if set.class_of(rec) != Some(Class::Subject) || !casts_shadow(self.bake, rec) {
                continue;
            }
            let shape = &self.geo.shapes[m.shape];
            let key = match shape.kind {
                ShapeKind::ByFrame { .. } => self.loaded_tris[m.shape].unwrap_or(usize::MAX),
                _ => 0,
            };
            if kit.volumes.get(&m.shape).is_none_or(|v| v.0 != key) {
                let tris = self.shape_triangles(mi)?;
                let (e, c) = volumes(&tris);
                kit.volumes.insert(m.shape, (key, e, c));
            }
            let (_, e, c) = &kit.volumes[&m.shape];
            let vb = shape.vertex_base;
            for x in e {
                let rec = VolEdgeGpu { v: [x[0] + vb, x[1] + vb, x[2] + vb, x[3] + vb], m: [mi as u32, 0, 0, 0] };
                edges.push(rec);
                if x[0] != x[3] {
                    twice.push(rec);
                }
            }
            for t in c {
                caps.push([t[0] + vb, t[1] + vb, t[2] + vb, mi as u32]);
            }
        }
        edges.extend(twice);
        let device = &self.gpu.device;
        let need_e = (edges.len().max(1) * std::mem::size_of::<VolEdgeGpu>()) as u64;
        let need_c = (caps.len().max(1) * 16) as u64;
        if kit.edges.size() < need_e || kit.caps.size() < need_c {
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if kit.edges.size() < need_e {
                kit.edges = sized(device, "look edges", need_e.next_power_of_two(), usage);
            }
            if kit.caps.size() < need_c {
                kit.caps = sized(device, "look caps", need_c.next_power_of_two(), usage);
            }
            let sizes: Vec<(u32, u32)> = kit.targets.iter().map(|t| (t.w, t.h)).collect();
            kit.targets.clear();
            for (w, h) in sizes {
                self.look_targets(kit, w, h);
            }
        }
        let queue = &self.gpu.queue;
        if !edges.is_empty() {
            queue.write_buffer(&kit.edges, 0, bytemuck::cast_slice(&edges));
        }
        if !caps.is_empty() {
            queue.write_buffer(&kit.caps, 0, bytemuck::cast_slice(&caps));
        }
        Ok((edges.len() as u32, caps.len() as u32))
    }

    /// The triangles (vertex indices of its shape) mesh `mi` draws on this frame: its shape's, or for a frame-by-frame
    /// shape the geometry loaded now.
    fn shape_triangles(&self, mi: usize) -> Result<Vec<u32>> {
        let m = &self.geo.meshes[mi];
        let shape = &self.geo.shapes[m.shape];
        let base = self.draw_base[mi] as usize;
        let mut tris = Vec::new();
        match shape.kind {
            ShapeKind::ByFrame { rec } => {
                let lit = self.bake.lit()?;
                let Some(off) = self.loaded_tris[m.shape] else { return Ok(tris) };
                let bf = lit.meshes[rec].by_frame.as_ref().context("frame-by-frame mesh without geometries")?;
                let g = bf.geoms.iter().find(|g| g.triangles.offset == off).context("its loaded geometry")?;
                let bytes = self.bake.bytes(g.triangles.offset, g.triangles.count * 12)?;
                let all: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                for di in base..base + m.slots.len() {
                    let d = &self.draws[di];
                    tris.extend_from_slice(&all[(d.first_tri * 3) as usize..((d.first_tri + d.tri_count) * 3) as usize]);
                }
            }
            _ => {
                for di in base..base + m.slots.len() {
                    let d = &self.draws[di];
                    let at = (shape.index_base + d.first_tri * 3) as usize;
                    tris.extend_from_slice(&self.geo.indices[at..at + (d.tri_count * 3) as usize]);
                }
            }
        }
        Ok(tris)
    }

    // ------------------------------------------------------------------------------------------------ the film
    /// Resolve the film of targets `ti` into its pixels.
    fn resolve_into(&self, kit: &LookKit, enc: &mut wgpu::CommandEncoder, ti: usize, mode: Mode, total: f32) {
        let t = &kit.targets[ti];
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("look resolve"), timestamp_writes: None });
        pass.set_pipeline(&kit.resolve);
        pass.set_bind_group(0, &t.res_bg, &[]);
        let r = ResolveGpu {
            width: t.w,
            height: t.h,
            mode: mode as u32,
            lut_n: kit.lut.n as u32,
            lo: kit.lut.lo,
            hi: kit.lut.hi,
            total,
            pad: 0.0,
            zero: [kit.lut.zero[0], kit.lut.zero[1], kit.lut.zero[2], 0.0],
        };
        pass.set_immediates(0, bytemuck::bytes_of(&r));
        pass.dispatch_workgroups(t.w.div_ceil(8), t.h.div_ceil(8), 1);
    }

    /// Submit `enc` with the copy of targets `ti`'s pixels and wait for them.
    fn read_pixels(&self, kit: &LookKit, mut enc: wgpu::CommandEncoder, ti: usize) -> Result<Vec<u32>> {
        let t = &kit.targets[ti];
        let bytes = (t.w * t.h * 4) as u64;
        enc.copy_buffer_to_buffer(&t.pixels, 0, &t.readback, 0, bytes);
        self.wait_map(enc, &t.readback, bytes, |data| data.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    /// Submit `enc` with the copy of targets `ti`'s id (or depth) texture and wait for it: one u32 (or f32 bits) per
    /// pixel, top row first.
    fn read_texture(&self, kit: &LookKit, mut enc: wgpu::CommandEncoder, ti: usize, ids: bool) -> Result<Vec<u32>> {
        let t = &kit.targets[ti];
        let row = (t.w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: if ids { &t.id } else { &t.depthv }, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &t.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(t.h) } },
            wgpu::Extent3d { width: t.w, height: t.h, depth_or_array_layers: 1 },
        );
        let (w, h) = (t.w as usize, t.h as usize);
        self.wait_map(enc, &t.readback, row as u64 * t.h as u64, |data| {
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h {
                let r = &data[y * row as usize..y * row as usize + w * 4];
                out.extend(r.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            }
            out
        })
    }

    fn wait_map<T>(&self, enc: wgpu::CommandEncoder, buf: &wgpu::Buffer, bytes: u64, read: impl FnOnce(&[u8]) -> T) -> Result<T> {
        let index = self.gpu.queue.submit(Some(enc.finish()));
        let slice = buf.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .map_err(|e| anyhow::anyhow!("GPU wait: {e}"))?;
        rx.recv()?.map_err(|e| anyhow::anyhow!("readback: {e}"))?;
        let out = {
            let data = slice.get_mapped_range().map_err(|e| anyhow::anyhow!("readback: {e:?}"))?;
            read(&data)
        };
        buf.unmap();
        Ok(out)
    }

    // ------------------------------------------------------------------------------------------------ the kit
    /// The looks' pipelines: the drawing module (the scene's core, the materials the EEVEE passes draw, the looks' own
    /// entry points), the films and the Standard view.
    fn look_kit(&self, sets: &[&LookSet]) -> Result<LookKit> {
        let device = &self.gpu.device;
        let bake = self.bake;
        let lit = bake.lit()?;
        let mats = &self.mats;
        let n = mats.list.len();
        // the materials of the EEVEE passes: type, knock, soft
        let mut used = vec![false; n];
        let mut soft = vec![false; n];
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            let rec = lit.meshes[mi].record_name();
            for set in sets {
                let class = set.class_of(rec);
                if !matches!(class, Some(Class::Type | Class::Knock | Class::Soft)) {
                    continue;
                }
                for s in &m.slots {
                    let i = s.material.map(|g| self.mat_of[g]).unwrap_or(mats.default_index());
                    used[i] = true;
                    soft[i] |= class == Some(Class::Soft);
                }
            }
        }
        let args = "in: VsOut, @builtin(front_facing) front: bool, @builtin(primitive_index) prim: u32";
        let mut src = pipelines::module_source(mats, &used);
        for (i, m) in mats.list.iter().enumerate() {
            let id = &m.id;
            if used[i] && m.blended {
                src.push_str(&format!(
                    "\n@fragment\nfn fls_{id}({args}) -> Blended {{\n    let s = fragment_surface(in, front, prim);\n    let c = mat_{id}(s);\n    let t = clamp(c.transmittance, vec3<f32>(0.0), vec3<f32>(1.0));\n    let o = 1.0 - mks_avg3(t);\n    return Blended(vec4<f32>(light_closures(c, s), o), vec4<f32>(t, 1.0 - o));\n}}\n"
                ));
            }
        }
        let soft_src = crate::looks_soft::soft_entries(self, &soft)?;
        src.push_str(&soft_src);
        src.push('\n');
        src.push_str(LOOK_DRAW_WGSL);
        if let Ok(path) = std::env::var("MKR_DUMP_LOOK_WGSL") {
            std::fs::write(&path, &src).with_context(|| format!("cannot write {path}"))?;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("looks"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("the look shaders do not compile: {e}");
        }
        let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("looks"),
            entries: &[storage_entry(0, true, vis), storage_entry(1, true, vis), storage_entry(2, true, vis), storage_entry(3, true, vis),
                       texture_entry(4, wgpu::ShaderStages::FRAGMENT)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("looks"),
            bind_group_layouts: &[Some(&self.render_bgl), Some(&draw_bgl)],
            immediate_size: std::mem::size_of::<Immediate>() as u32,
        });
        let mk = Maker { device, module, layout, cache: self.gpu.cache.as_ref() };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let gt = wgpu::CompareFunction::Greater;
        let all = wgpu::ColorWrites::ALL;
        let nost = wgpu::StencilState::default();
        let both = |f: &dyn Fn(bool) -> wgpu::RenderPipeline| [f(false), f(true)];
        let flat = both(&|cull| mk.pipe("look flat", PipeDesc { vs: "vs_look", fs: Some("fs_look_flat"),
            targets: vec![target(COLOR_FORMAT, None, all)], cull, depth: depth_state(DEPTH_FORMAT, true, gt, nost.clone()) }));
        let tex = both(&|cull| mk.pipe("look tex", PipeDesc { vs: "vs_look", fs: Some("fs_look_tex"),
            targets: vec![target(COLOR_FORMAT, None, all)], cull, depth: depth_state(DEPTH_FORMAT, true, gt, nost.clone()) }));
        let tex_blend = both(&|cull| mk.pipe("look tex blend", PipeDesc { vs: "vs_look", fs: Some("fs_look_tex"),
            targets: vec![target(COLOR_FORMAT, Some(TEX_OVER), all)], cull, depth: depth_state(DEPTH_FORMAT, false, gt, nost.clone()) }));
        let id = both(&|cull| mk.pipe("look id", PipeDesc { vs: "vs_look", fs: Some("fs_look_id"),
            targets: vec![target(ID_FORMAT, None, all), target(DEPTHV_FORMAT, None, all)], cull,
            depth: depth_state(DEPTH_FORMAT, true, gt, nost.clone()) }));
        let id_depth = both(&|cull| mk.pipe("look id depth", PipeDesc { vs: "vs_look", fs: Some("fs_look_id"),
            targets: vec![target(ID_FORMAT, None, wgpu::ColorWrites::empty()), target(DEPTHV_FORMAT, None, all)], cull,
            depth: depth_state(DEPTH_FORMAT, true, gt, nost.clone()) }));
        let normals = both(&|cull| mk.pipe("look normals", PipeDesc { vs: "vs_look", fs: Some("fs_look_normal"),
            targets: vec![target(COLOR_FORMAT, None, all)], cull, depth: depth_state(STENCIL_FORMAT, true, gt, nost.clone()) }));
        use wgpu::StencilOperation as Op;
        let always = wgpu::CompareFunction::Always;
        // the light runs away from the camera: counted where the volumes lie behind what is drawn (with caps)
        let fail = stencil(face(always, Op::DecrementWrap, Op::Keep), face(always, Op::IncrementWrap, Op::Keep), true);
        // else where they lie in front of it
        let pass_ = stencil(face(always, Op::Keep, Op::IncrementWrap), face(always, Op::Keep, Op::DecrementWrap), true);
        let vol = |label: &str, vs: &str, st: wgpu::StencilState| mk.pipe(label, PipeDesc { vs, fs: Some("fs_vol"), targets: vec![],
            cull: false, depth: depth_state(STENCIL_FORMAT, false, gt, st) });
        let vol_fail = [vol("look volume edges", "vs_vol_edge", fail.clone()), vol("look volume caps", "vs_vol_cap", fail)];
        let vol_pass = vol("look volume edges", "vs_vol_edge", pass_);
        let light = [wgpu::CompareFunction::Equal, wgpu::CompareFunction::NotEqual].map(|cmp| {
            mk.pipe("look light", PipeDesc { vs: "vs_screen", fs: Some("fs_look_light"), targets: vec![target(COLOR_FORMAT, None, all)],
                cull: false, depth: depth_state(STENCIL_FORMAT, false, always,
                    stencil(face(cmp, Op::Keep, Op::Keep), face(cmp, Op::Keep, Op::Keep), false)) })
        });
        let mut emissive = Emissive::default();
        let eq = wgpu::CompareFunction::Equal;
        for (i, m) in mats.list.iter().enumerate() {
            if !used[i] {
                continue;
            }
            if m.blended {
                let cmp = if m.show_back { gt } else { eq };
                let fs = format!("fls_{}", m.id);
                emissive.blend.insert(i, mk.pipe(&fs, PipeDesc { vs: "vs_main", fs: Some(&fs), targets: vec![target(COLOR_FORMAT, Some(OVER), all)],
                    cull: m.cull, depth: depth_state(DEPTH_FORMAT, false, cmp, nost.clone()) }));
            } else {
                let fs = format!("fs_{}", m.id);
                emissive.main.insert(i, mk.pipe(&fs, PipeDesc { vs: "vs_main", fs: Some(&fs), targets: vec![target(COLOR_FORMAT, None, all)],
                    cull: m.cull, depth: depth_state(DEPTH_FORMAT, false, eq, nost.clone()) }));
                if m.transparent {
                    let fa = format!("fa_{}", m.id);
                    emissive.pre.insert(i, mk.pipe(&fa, PipeDesc { vs: "vs_main", fs: Some(&fa), targets: vec![],
                        cull: m.cull, depth: depth_state(DEPTH_FORMAT, true, gt, nost.clone()) }));
                }
            }
            if soft[i] && !m.blended {
                let fs = format!("fsoft_{}", m.id);
                emissive.soft.insert(i, mk.pipe(&fs, PipeDesc { vs: "vs_main", fs: Some(&fs), targets: vec![target(COLOR_FORMAT, None, all)],
                    cull: m.cull, depth: depth_state(DEPTH_FORMAT, false, eq, nost.clone()) }));
            }
        }
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("the look pipelines cannot be made: {e}");
        }
        // the films
        let film_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("look film"),
            source: wgpu::ShaderSource::Wgsl(LOOK_FILM_WGSL.into()) });
        let c = wgpu::ShaderStages::COMPUTE;
        let acc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("look accumulate"),
            entries: &[texture_entry(0, c), storage_entry(1, false, c)],
        });
        let res_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("look resolve"),
            entries: &[storage_entry(0, true, c), storage_entry(1, true, c), storage_entry(2, false, c)],
        });
        let cache = self.gpu.cache.as_ref();
        let wb_acc = compute(device, &film_module, &acc_bgl, "wb_accumulate", std::mem::size_of::<WbAccumulateGpu>() as u32, cache);
        let ee_acc = compute(device, &film_module, &acc_bgl, "ee_accumulate", std::mem::size_of::<EeAccumulateGpu>() as u32, cache);
        let resolve = compute(device, &film_module, &res_bgl, "resolve", std::mem::size_of::<ResolveGpu>() as u32, cache);
        if let Err(e) = self.gpu.save_cache() {
            eprintln!("mkr: the pipeline cache is not kept: {e:#}");
        }
        // the Standard view the passes go through
        let rec = bake.raw.get("looks").and_then(|l| l.get("lut")).context("the bake holds no looks' Standard view (`mk build --bake`)")?;
        let file = rec.get("file").and_then(|f| f.as_str()).context("looks.lut: no file")?;
        let get = |k: &str| rec.get(k).and_then(|v| v.as_f64()).with_context(|| format!("looks.lut: no {k}"));
        let zero = rec.get("zero").and_then(|z| z.as_array()).map(|a| [0, 1, 2].map(|i| a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32));
        let lut = ViewLut::load(&bake.dir.join(file), get("n")? as usize, get("lo")? as f32, get("hi")? as f32, zero)?;
        let lut_buf = storage(device, "looks lut", bytemuck::cast_slice(&lut.data), wgpu::BufferUsages::empty());
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let draws_buf = sized(device, "look draws", (self.draws.len().max(1) * std::mem::size_of::<LookDrawGpu>()) as u64, usage);
        let light_buf = sized(device, "look light", std::mem::size_of::<LookLightGpu>() as u64, usage);
        let edges = sized(device, "look edges", 1 << 16, usage);
        let caps = sized(device, "look caps", 1 << 16, usage);
        Ok(LookKit { flat, id, id_depth, normals, vol_fail, vol_pass, light, tex, tex_blend, emissive, draw_bgl, draws_buf,
                     light_buf, edges, caps, wb_acc, ee_acc, acc_bgl, resolve, res_bgl, lut, lut_buf, targets: Vec::new(),
                     volumes: BTreeMap::new(), screen: None })
    }

    /// The index of the kit's targets at `w` x `h`, made when there are none of that size.
    fn look_targets(&self, kit: &mut LookKit, w: u32, h: u32) -> usize {
        if let Some(i) = kit.targets.iter().position(|t| t.w == w && t.h == h) {
            return i;
        }
        let device = &self.gpu.device;
        let tex = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let color = tex("look sample", COLOR_FORMAT, rt | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let depth = tex("look depth", DEPTH_FORMAT, rt).create_view(&Default::default());
        let stencil = tex("look stencil", STENCIL_FORMAT, rt).create_view(&Default::default());
        let normals = tex("look normals", COLOR_FORMAT, rt | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let id = tex("look ids", ID_FORMAT, rt | wgpu::TextureUsages::COPY_SRC);
        let depthv = tex("look planar depth", DEPTHV_FORMAT, rt | wgpu::TextureUsages::COPY_SRC);
        let n = (w * h) as u64;
        let film = sized(device, "look film", n * 32, wgpu::BufferUsages::STORAGE);
        let pixels = sized(device, "look pixels", n * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u64;
        let readback = sized(device, "look readback", row * h as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        fn e(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> { wgpu::BindGroupEntry { binding, resource } }
        let acc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("look accumulate"),
            layout: &kit.acc_bgl,
            entries: &[e(0, wgpu::BindingResource::TextureView(&color)), e(1, film.as_entire_binding())],
        });
        let res_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("look resolve"),
            layout: &kit.res_bgl,
            entries: &[e(0, film.as_entire_binding()), e(1, kit.lut_buf.as_entire_binding()), e(2, pixels.as_entire_binding())],
        });
        let group = |label: &str, nview: &wgpu::TextureView| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &kit.draw_bgl,
            entries: &[e(0, kit.draws_buf.as_entire_binding()), e(1, kit.light_buf.as_entire_binding()), e(2, kit.edges.as_entire_binding()),
                       e(3, kit.caps.as_entire_binding()), e(4, wgpu::BindingResource::TextureView(nview))],
        });
        let draw_bg = group("looks", &normals);
        // the normals pass draws into the normals: its group holds the sample picture there instead (unread)
        let draw_bg_pre = group("looks' normals pass", &color);
        kit.targets.push(Targets { w, h, color, depth, stencil, normals, id, depthv, _film: film, pixels, readback, acc_bg, res_bg,
                                   draw_bg, draw_bg_pre });
        kit.targets.len() - 1
    }
}

/// The edges and faces of a mesh's triangles for its shadow volumes, as Workbench extrudes a mesh that is not closed:
/// per edge [far vertex of its first face, its ends, far vertex of its second face] (an edge of one face: its own far
/// vertex twice; the caller counts an edge of two faces twice), and every triangle a cap.
fn volumes(tris: &[u32]) -> (Vec<[u32; 4]>, Vec<[u32; 3]>) {
    let mut at: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut recs: Vec<([u32; 4], u32)> = Vec::new();
    for t in tris.chunks_exact(3) {
        for e in 0..3 {
            let (a, b, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            match at.get(&key) {
                None => {
                    at.insert(key, recs.len());
                    recs.push(([c, a, b, u32::MAX], 1));
                }
                Some(&i) => {
                    if recs[i].1 == 1 {
                        recs[i].0[3] = c;
                    }
                    recs[i].1 += 1;
                }
            }
        }
    }
    let edges = recs.into_iter().map(|(r, n)| if n == 2 { r } else { [r[0], r[1], r[2], r[0]] }).collect();
    let caps = tris.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    (edges, caps)
}

/// The sample's picture, cleared transparent, over a depth attachment cleared to far (or loaded from the prepass).
fn colour_pass<'e>(enc: &'e mut wgpu::CommandEncoder, color: &wgpu::TextureView, depth: Option<&wgpu::TextureView>, clear_depth: bool)
                   -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("look colour"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: depth.map(|d| wgpu::RenderPassDepthStencilAttachment {
            view: d,
            depth_ops: Some(wgpu::Operations {
                load: if clear_depth { wgpu::LoadOp::Clear(0.0) } else { wgpu::LoadOp::Load },
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// A depth-only pass, cleared to far (reversed depth: 0).
fn depth_pass<'e>(enc: &'e mut wgpu::CommandEncoder, depth: &wgpu::TextureView) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("look prepass"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Add Workbench sample `s`'s picture into targets `ti`'s film (in log2, half floats rounded toward zero).
fn wb_accumulate(kit: &LookKit, enc: &mut wgpu::CommandEncoder, ti: usize, s: &WbSample, first: bool) {
    let t = &kit.targets[ti];
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("look wb accumulate"), timestamp_writes: None });
    pass.set_pipeline(&kit.wb_acc);
    pass.set_bind_group(0, &t.acc_bg, &[]);
    let mut w = [0.0f32; 12];
    w[..9].copy_from_slice(&s.weights);
    let a = WbAccumulateGpu { w, width: t.w, height: t.h, first: first as u32, pad: 0 };
    pass.set_immediates(0, bytemuck::bytes_of(&a));
    pass.dispatch_workgroups(t.w.div_ceil(8), t.h.div_ceil(8), 1);
}

/// Spread EEVEE sample `s`'s picture into targets `ti`'s film with its taps.
fn ee_accumulate(kit: &LookKit, enc: &mut wgpu::CommandEncoder, ti: usize, s: &Sample, first: bool) {
    let t = &kit.targets[ti];
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("look ee accumulate"), timestamp_writes: None });
    pass.set_pipeline(&kit.ee_acc);
    pass.set_bind_group(0, &t.acc_bg, &[]);
    let mut taps = [0.0f32; 28];
    taps[..TAP_COUNT].copy_from_slice(&s.taps);
    let a = EeAccumulateGpu { taps, width: t.w, height: t.h, first: first as u32, pad: 0 };
    pass.set_immediates(0, bytemuck::bytes_of(&a));
    pass.dispatch_workgroups(t.w.div_ceil(8), t.h.div_ceil(8), 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workbench_samples_sit_where_layers_measured_them() {
        // Layers' WORKBENCH_JITTER (layers_gpu.rs), measured on Blender's renders to 0.005 px
        let measured: [[f32; 2]; 16] = [
            [-0.045, -0.485], [0.415, -0.405], [-0.325, -0.345], [0.175, -0.315], [-0.065, -0.235], [0.475, -0.155],
            [-0.275, -0.065], [0.235, -0.065], [0.0, 0.0], [-0.465, 0.095], [-0.215, 0.165], [0.275, 0.185],
            [0.025, 0.275], [0.455, 0.355], [-0.275, 0.405], [0.215, 0.445],
        ];
        let (s, total) = workbench_samples();
        assert_eq!(s.len(), 16);
        for m in measured {
            let best = s.iter().map(|x| (x.jitter - Vec2::new(m[0], m[1])).length()).fold(f32::MAX, f32::min);
            assert!(best < 0.006, "{m:?}: nearest sample {best} px away");
        }
        // the centre sample counts for its own pixel alone, at the window's peak
        let c = s.iter().find(|x| x.jitter == Vec2::ZERO).unwrap();
        assert!((c.weights[4] - 1.0).abs() < 1e-4 && c.weights.iter().sum::<f32>() < 1.001);
        assert!((total - 8.1364).abs() < 1e-3, "{total}");
    }

    #[test]
    fn an_edge_between_two_faces_has_both_far_vertices() {
        // two triangles sharing the edge 1-2: 0 1 2 and 2 1 3
        let (e, c) = volumes(&[0, 1, 2, 2, 1, 3]);
        assert_eq!(c.len(), 2);
        assert_eq!(e.len(), 5);
        let shared = e.iter().find(|x| (x[1], x[2]) == (1, 2) || (x[1], x[2]) == (2, 1)).unwrap();
        assert_eq!((shared[0], shared[3]), (0, 3));
        // an open edge's far vertex stands in for the missing face's
        let open = e.iter().find(|x| (x[1], x[2]) == (0, 1)).unwrap();
        assert_eq!(open[0], open[3]);
    }
}
