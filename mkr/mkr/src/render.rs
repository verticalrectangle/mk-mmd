//! Drawing frames, deferred as EEVEE Next draws them: per frame the scene posed (on the frame's own time, and with motion
//! blur on the shutter's two ends) and the lights' shadow maps (or, `shadow_mode: "rays"`, the shadow rays'
//! acceleration structures); then N samples (jittered, at a time inside the shutter), each a depth prepass with hashed
//! alpha, the G-buffer of the opaque and dithered surfaces, the lighting pass over it, the blended surfaces back to
//! front, and the film's accumulation; then the view transform, dither and 8 bits, read back while the next frame draws.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::bake::{Bake, LIGHT_FLOATS, LightRec};
use crate::camera::{Cam, render_size};
use crate::dof::{self, DofPost, DofTargets, SceneDof};
use crate::film::{self, Filter, Sample};
use crate::gpu::Gpu;
use crate::lut::ViewLut;
use crate::materials::{MaterialSet, ParamLayout};
use crate::motion_blur::{self, MotionPost, MotionTargets, Shutter};
use crate::output::Writer;
use crate::params::{MotionBlurMode, Params, ShadowMode, ShadowSequence};
use crate::pipelines::{self, DEPTH_FORMAT, Immediate, Pipelines};
use crate::scene::{self, Geometry, Pose, ShapeKind, blender_matrix, rows};
use crate::shadows::{self, ShadowFaceGpu, ShadowMaps};

const SKIN_WGSL: &str = include_str!("shaders/skin.wgsl");
const ACCUMULATE_WGSL: &str = include_str!("shaders/accumulate.wgsl");
const RESOLVE_WGSL: &str = include_str!("shaders/resolve.wgsl");

// ================================================================================================= GPU-side records
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FrameGpu {
    pub(crate) view: [[f32; 4]; 4],
    pub(crate) camera: [[f32; 4]; 4],
    pub(crate) camera_inv: [[f32; 4]; 4],
    pub(crate) cam_pos: [f32; 4],
    pub(crate) cam_fwd: [f32; 4],
    pub(crate) size: [f32; 4],
    pub(crate) light_count: u32,
    pub(crate) sample_count: u32,
    pub(crate) flags: u32,
    pub(crate) frame_seed: u32,
    pub(crate) shadow: [f32; 4],
    pub(crate) scale: [f32; 4],
    pub(crate) trace: [f32; 4],
    pub(crate) bias: [f32; 4],
    pub(crate) shadow_map: [f32; 4],
    pub(crate) shadow_lod: [f32; 4],
    pub(crate) shadow_pool: [f32; 4],
    pub(crate) world: [f32; 4],
    pub(crate) world_probe: [f32; 4],
    pub(crate) rt: [f32; 4],
    pub(crate) rt2: [f32; 4],
    pub(crate) rt3: [f32; 4],
    pub(crate) rt4: [f32; 4],
    pub(crate) sampling: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SampleGpu {
    pub(crate) vp: [[f32; 4]; 4],
    pub(crate) vp_inv: [[f32; 4]; 4],
    pub(crate) info: [u32; 4],
    pub(crate) t: [f32; 4],
    pub(crate) rand: [f32; 4],
    /// The post motion blur's cameras (no jitter, no lens): on the sample's time, the shutter's open and its close.
    pub(crate) vp_cur: [[f32; 4]; 4],
    pub(crate) vp_prev: [[f32; 4]; 4],
    pub(crate) vp_next: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LightGpu {
    pub(crate) pos: [f32; 4],
    pub(crate) fwd: [f32; 4],
    pub(crate) right: [f32; 4],
    pub(crate) up: [f32; 4],
    pub(crate) color: [f32; 4],
    pub(crate) spot: [f32; 4],
    pub(crate) factor: [f32; 4],
    pub(crate) map: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MeshGpu {
    pub(crate) object: [[f32; 4]; 4],
    pub(crate) object_inv: [[f32; 4]; 4],
    pub(crate) m_prev: [[f32; 4]; 3],
    pub(crate) m_cur: [[f32; 4]; 3],
    pub(crate) m_next: [[f32; 4]; 3],
    pub(crate) n_cur: [[f32; 4]; 3],
    pub(crate) object_color: [f32; 4],
    pub(crate) object_info: [f32; 4],
    pub(crate) vertex_base: u32,
    pub(crate) attr_base: u32,
    pub(crate) attr_stride: u32,
    pub(crate) index_base: u32,
    pub(crate) draw_base: u32,
    pub(crate) generated: u32,
    pub(crate) flags: u32,
    pub(crate) pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DrawGpu {
    pub(crate) mesh: u32,
    pub(crate) material: u32,
    pub(crate) first_tri: u32,
    pub(crate) tri_count: u32,
    pub(crate) remap: [u32; 8],
    pub(crate) param_base: u32,
    pub(crate) image_base: u32,
    /// Material::draw_flags (materials.rs DRAW_UNSHADOWED).
    pub(crate) flags: u32,
    pub(crate) pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinJob {
    m: [[f32; 4]; 3],
    vertex_base: u32,
    vertex_count: u32,
    skin_base: u32,
    joint_base: u32,
    morph_base: u32,
    morph_first: u32,
    morph_count: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumulateGpu {
    width: u32,
    height: u32,
    first_sample: u32,
    count: u32,
    first: u32,
    pad: [u32; 3],
}

/// Samples whose pictures the film takes in one pass.
pub(crate) const SAMPLES_PER_FILM_PASS: usize = 4;
/// The most samples a frame can have (the film's taps buffer).
pub const MAX_SAMPLES: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveGpu {
    width: u32,
    height: u32,
    lut_n: u32,
    per_channel: u32,
    lo: f32,
    hi: f32,
    dither: f32,
    bias: f32,
    zero: [f32; 4],
    exposure: f32,
    frame: u32,
    pad2: [u32; 2],
}

pub(crate) fn mat(m: &Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

pub(crate) fn storage(device: &wgpu::Device, label: &str, bytes: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
    // wgpu refuses empty bindings: keep at least 16 bytes
    let mut data = bytes.to_vec();
    if data.len() < 16 {
        data.resize(16, 0);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
    })
}

pub(crate) fn sized(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
}

// ===================================================================================================== the options
pub struct Options {
    pub samples: u32,
    pub percent: u32,
    /// Motion blur shutter (frames), None: off.
    pub shutter: Option<f32>,
}

/// One frame to draw: its Blender frame, bake frame and shot camera, the PNG it becomes, and the lit meshes (indices into
/// the bake's `lit.meshes`) left out of it: neither drawn nor casting shadows.
#[derive(Clone, Debug)]
pub struct Job {
    pub frame: i64,
    pub k: usize,
    pub camera: String,
    pub path: std::path::PathBuf,
    pub hidden: Vec<usize>,
}

/// The parts of a frame the GPU times (timestamps between passes), in the report's order.
pub const PHASES: [&str; 10] =
    ["pose", "shadow maps", "prepass", "gbuffer", "lighting", "blended", "motion blur", "film", "dof", "resolve"];

pub struct Stats {
    pub rendered: usize,
    pub gpu_ms: Vec<f64>,
    /// Per frame, milliseconds per PHASES entry.
    pub phases: Vec<[f64; 10]>,
    /// Per frame, the CPU milliseconds spent recording and submitting it (of which posing it, `stage`, and finishing
    /// and submitting its commands), waiting for the GPU to finish it, and handing its pixels to the writer (readback
    /// copy, and the wait while the writer's queue is full).
    pub encode_ms: Vec<f64>,
    pub stage_ms: Vec<f64>,
    pub submit_ms: Vec<f64>,
    pub wait_ms: Vec<f64>,
    pub collect_ms: Vec<f64>,
    pub warnings: Vec<String>,
}

/// The per-frame GPU targets of one picture size.
pub(crate) struct Targets {
    pub(crate) w: u32,
    pub(crate) h: u32,
    /// The samples' pictures, four at a time: the film takes them in one pass (with EEVEE's motion blur, the post
    /// process's blurred copies of them instead: motion_blur.rs).
    pub(crate) colors: [wgpu::TextureView; SAMPLES_PER_FILM_PASS],
    /// The depth and stencil the passes draw with, and its depth alone (what the lighting and the shadow windows read).
    pub(crate) depth: wgpu::TextureView,
    pub(crate) depth_read: wgpu::TextureView,
    pub(crate) gb_base: wgpu::TextureView,
    pub(crate) gb_motion: wgpu::TextureView,
    pub(crate) pixels: wgpu::Buffer,
    pub(crate) readback: [wgpu::Buffer; 2],
    /// With `MKR_LINEAR` set: per slot, the film read back (written as `<frame>.npy` beside the PNG).
    pub(crate) film_read: Option<[wgpu::Buffer; 2]>,
    pub(crate) accumulate_bg: wgpu::BindGroup,
    pub(crate) resolve_bg: wgpu::BindGroup,
    /// Group 1 of the G-buffer pass (the extras) and of the lighting pass (the G-buffer, depth, the sample's picture).
    pub(crate) gbuffer_bg: wgpu::BindGroup,
    pub(crate) light_bg: [wgpu::BindGroup; SAMPLES_PER_FILM_PASS],
    /// The samples' film taps (28 floats each), written by `render`.
    pub(crate) taps: wgpu::Buffer,
}

/// A shape's BLAS: one geometry per material slot run.
pub(crate) struct ShapeBlas {
    pub(crate) blas: wgpu::Blas,
    pub(crate) sizes: Vec<wgpu::BlasTriangleGeometrySizeDescriptor>,
    /// (first triangle, count) of each geometry
    pub(crate) runs: Vec<(u32, u32)>,
    pub(crate) built: bool,
}

/// What `Renderer::stage` leaves for a frame's passes.
pub(crate) struct Staged {
    /// Per mesh: drawn (and casting shadows) on this frame.
    pub(crate) shown: Vec<bool>,
    /// The scene on the frame's own time.
    pub(crate) cur: Pose,
    /// Per mesh: its world box over the frame's times (None: it deforms, no box is known without posing it).
    pub(crate) boxes: Vec<Option<(Vec3, Vec3)>>,
}

/// The geometry a bind group of `render_bgl` reads (bindings 4-10): posed vertices on the frame's time, the shutter's
/// open and its close; attributes, triangles, mesh and draw records.
pub(crate) struct SceneBuffers<'a> {
    pub(crate) posed: [&'a wgpu::Buffer; 3],
    pub(crate) attrs: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
    pub(crate) meshes: &'a wgpu::Buffer,
    pub(crate) draws: &'a wgpu::Buffer,
}

/// One indirect draw (the draw's index rides as its instance: the shaders' `instance_index`).
pub(crate) type IndirectArgs = wgpu::util::DrawIndexedIndirectArgs;
const INDIRECT_STRIDE: u64 = std::mem::size_of::<IndirectArgs>() as u64;

/// A pipeline multi-draws use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pipe {
    /// The shared depth-only prepass of opaque materials (back faces culled or not).
    Depth(bool),
    Prepass(usize),
    GBuffer(usize),
    ShadowDepth,
    /// A hashed caster's opaque part into a shadow map's depth, and its partly transparent part into its colour layers.
    Shadow(usize),
    ShadowHashed(usize),
}

/// Consecutive indirect draws of one pipeline.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Group {
    pub(crate) pipe: Pipe,
    pub(crate) first: u32,
    pub(crate) count: u32,
}

/// The indirect record of triangles `first`.. `first + count` of a shape, drawn as draw `di`.
fn indirect(shape: &crate::scene::Shape, first: u32, count: u32, di: usize) -> IndirectArgs {
    IndirectArgs {
        index_count: count * 3,
        instance_count: 1,
        first_index: shape.index_base + first * 3,
        base_vertex: shape.vertex_base as i32,
        first_instance: di as u32,
    }
}

/// Append an indirect draw, extending the last group when it has the same pipeline.
fn push_group(groups: &mut Vec<Group>, args: &mut Vec<IndirectArgs>, pipe: Pipe, a: IndirectArgs) {
    if let Some(g) = groups.last_mut()
        && g.pipe == pipe
        && g.first + g.count == args.len() as u32
    {
        g.count += 1;
        args.push(a);
        return;
    }
    groups.push(Group { pipe, first: args.len() as u32, count: 1 });
    args.push(a);
}

/// Per frame slot: the timestamp queries, their resolve and read buffers, and which phase each stamp closes.
pub(crate) struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    capacity: u32,
    marks: Vec<u8>,
}

pub struct Renderer<'b> {
    pub(crate) gpu: Gpu,
    pub(crate) bake: &'b Bake,
    pub(crate) params: Params,
    pub(crate) geo: Geometry,
    pub(crate) mats: MaterialSet,
    /// The parameter buffer's blocks: the materials', and a draw's own where its material reads object attributes.
    pub(crate) param_layout: ParamLayout,
    /// Geometry's material index to the compiled material.
    pub(crate) mat_of: Vec<usize>,
    pub(crate) lut: ViewLut,
    pub(crate) filter: Filter,
    pub(crate) lights: Vec<LightRec>,
    pub(crate) light_threshold: f32,
    /// Steps along a shadow ray (the scene's shadow_step_count unless the params say).
    pub(crate) shadow_steps: u32,
    /// Shadow rays per sample and light: the scene's shadow_ray_count times the params' shadow_rays.
    pub(crate) shadow_rays: u32,
    pub(crate) rest: wgpu::Buffer,
    pub(crate) attrs: wgpu::Buffer,
    pub(crate) indices: wgpu::Buffer,
    pub(crate) posed: [wgpu::Buffer; 3],
    pub(crate) joints: [wgpu::Buffer; 3],
    pub(crate) morphs: [wgpu::Buffer; 3],
    pub(crate) skin_bg: [wgpu::BindGroup; 3],
    pub(crate) skin_pipe: wgpu::ComputePipeline,
    /// The bending meshes' normal rebuild (bake `bend` records) on the frame's posed vertices.
    pub(crate) bend: Option<crate::bend::Bend>,
    pub(crate) frame_buf: wgpu::Buffer,
    pub(crate) samples_buf: wgpu::Buffer,
    pub(crate) lights_buf: wgpu::Buffer,
    pub(crate) meshes_buf: wgpu::Buffer,
    pub(crate) draws: Vec<DrawGpu>,
    pub(crate) draw_base: Vec<u32>,
    pub(crate) draws_buf: wgpu::Buffer,
    pub(crate) params_buf: wgpu::Buffer,
    pub(crate) image_map_buf: wgpu::Buffer,
    pub(crate) lut_buf: wgpu::Buffer,
    /// The frame's indirect draws (shadow faces, prepass, G-buffer), grown as needed.
    pub(crate) indirect: wgpu::Buffer,
    pub(crate) film: Option<wgpu::Buffer>,
    pub(crate) blas: Vec<Option<ShapeBlas>>,
    pub(crate) tlas: wgpu::Tlas,
    pub(crate) shadow_maps: ShadowMaps,
    /// Per frame-by-frame shape: the triangles offset now in its index range.
    pub(crate) loaded_tris: Vec<Option<usize>>,
    pub(crate) render_bgl: wgpu::BindGroupLayout,
    pub(crate) gbuffer_bgl: wgpu::BindGroupLayout,
    pub(crate) light_bgl: wgpu::BindGroupLayout,
    pub(crate) render_bg: Option<wgpu::BindGroup>,
    /// The same with a stand-in for the shadow maps (bound while they are drawn into).
    pub(crate) shadow_bg: Option<wgpu::BindGroup>,
    /// The shadow windows' setup passes' bind group (over the targets' depth; None: none made yet or no windows).
    pub(crate) shadow_setup_bg: Option<wgpu::BindGroup>,
    pub(crate) images: Vec<wgpu::TextureView>,
    pub(crate) samplers: Vec<wgpu::Sampler>,
    pub(crate) shade_lut: wgpu::TextureView,
    pub(crate) ltc: wgpu::TextureView,
    pub(crate) pipes: Pipelines,
    pub(crate) accumulate_pipe: wgpu::ComputePipeline,
    pub(crate) accumulate_bgl: wgpu::BindGroupLayout,
    pub(crate) resolve_pipe: wgpu::ComputePipeline,
    pub(crate) resolve_bgl: wgpu::BindGroupLayout,
    pub(crate) targets: Option<Targets>,
    /// Per frame slot: GPU timing.
    pub(crate) timers: Option<[Timer; 2]>,
    /// Seconds Renderer::new spent making pipelines.
    pub compile_seconds: f64,
    /// The scene's EEVEE depth of field settings, the post process's pipelines (made when a frame first needs them)
    /// and its targets over the film of the current picture size.
    pub(crate) dof_scene: SceneDof,
    pub(crate) dof_post: Option<DofPost>,
    pub(crate) dof_targets: Option<DofTargets>,
    /// The scene's world (world.rs; None: it has none), the stand-ins bound without one, and group 0 with those
    /// stand-ins for the world probe's own passes (the probe cannot be read while it is drawn).
    pub(crate) world: Option<crate::world::WorldGpu>,
    pub(crate) world_stand: crate::world::WorldViews,
    pub(crate) world_bg: Option<wgpu::BindGroup>,
    /// The scene's motion blur (EEVEE's post process: where its shutter sits, its steps, its soft depth test's slope),
    /// the post process's pipelines (made when a frame first needs them) and its targets of the current picture size.
    pub(crate) mb_position: String,
    pub(crate) mb_steps: u32,
    pub(crate) mb_depth_scale: f32,
    pub(crate) mb_post: Option<MotionPost>,
    pub(crate) mb_targets: Option<MotionTargets>,
    /// The scene's screen tracing (raytrace.rs; None: it traces nothing).
    pub(crate) raytrace: Option<crate::raytrace::RayTrace>,
}

pub(crate) fn compute_pipe(device: &wgpu::Device, module: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout,
                           entry: &str, immediate: u32, cache: Option<&wgpu::PipelineCache>) -> wgpu::ComputePipeline {
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

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl<'b> Renderer<'b> {
    pub fn new(gpu: Gpu, bake: &'b Bake, params: Params) -> Result<Renderer<'b>> {
        let lit = bake.lit()?;
        let t0 = Instant::now();
        let mats = MaterialSet::compile(bake)?;
        eprintln!("mkr: {} materials compiled ({:.1} s)", mats.list.len(), t0.elapsed().as_secs_f64());
        let geo = Geometry::load(bake, &|m: &str| mats.attributes(m))?;
        eprintln!("mkr: geometry loaded ({:.1} s)", t0.elapsed().as_secs_f64());
        let mat_of: Vec<usize> = geo.materials.iter().map(|n| mats.index(n)).collect::<Result<_>>()?;
        let lut_rec = &lit.view.lut;
        let lut = ViewLut::load(&bake.dir.join(&lut_rec.file), lut_rec.n, lut_rec.lo as f32, lut_rec.hi as f32,
                                lut_rec.zero.or(lit.view.zero))?;
        let filter = Filter { radius: lit.film.filter_size as f32 * params.filter_scale, sigma: params.filter_sigma };
        let eevee_f = |key: &str, d: f64| lit.eevee.get(key).and_then(|v| v.as_f64()).unwrap_or(d);
        let light_threshold = eevee_f("light_threshold", 0.01) as f32;
        let shadow_steps = if params.shadow_steps > 0 { params.shadow_steps } else { eevee_f("shadow_step_count", 6.0) as u32 };
        let shadow_rays = (eevee_f("shadow_ray_count", 1.0) as u32).max(1) * params.shadow_rays;
        let rays = params.shadow_mode == ShadowMode::Rays;
        let device = &gpu.device;

        // geometry: rigid shapes keep their own-space vertices in every posed buffer
        let rest = storage(device, "rest", bytemuck::cast_slice(&geo.rest), wgpu::BufferUsages::BLAS_INPUT);
        let skin = storage(device, "skin", bytemuck::cast_slice(&geo.skin), wgpu::BufferUsages::empty());
        let deltas = storage(device, "deltas", bytemuck::cast_slice(&geo.deltas), wgpu::BufferUsages::empty());
        let attrs = storage(device, "attrs", bytemuck::cast_slice(&geo.attrs), wgpu::BufferUsages::empty());
        let indices = storage(device, "indices", bytemuck::cast_slice(&geo.indices),
                              wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::INDEX);
        let posed = [0, 1, 2].map(|i| {
            storage(device, ["posed prev", "posed", "posed next"][i], bytemuck::cast_slice(&geo.rest),
                    wgpu::BufferUsages::BLAS_INPUT)
        });
        let bones: usize = bake.doc.armatures.iter().map(|a| a.bones.len()).sum();
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let joints = [0, 1, 2].map(|_| sized(device, "joints", (bones * 48) as u64, rw));
        let morph_cap: usize = geo.shapes.iter().map(|s| match &s.kind {
            ShapeKind::Deforming { morphs: Some(k), .. } => k.count,
            _ => 0,
        }).sum();
        let morphs = [0, 1, 2].map(|_| sized(device, "morphs", (morph_cap * 8) as u64, rw));

        let skin_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skin"),
            source: wgpu::ShaderSource::Wgsl(SKIN_WGSL.into()),
        });
        let skin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin"),
            entries: &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, true), storage_entry(3, true),
                       storage_entry(4, true), storage_entry(5, false)],
        });
        let cache = gpu.cache.as_ref();
        let skin_pipe = compute_pipe(device, &skin_module, &skin_bgl, "main", std::mem::size_of::<SkinJob>() as u32, cache);
        let bend = crate::bend::Bend::new(device, bake, &geo, &posed[1])?;
        let skin_bg = [0, 1, 2].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skin"),
                layout: &skin_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: rest.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: skin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: deltas.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: joints[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: morphs[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: posed[i].as_entire_binding() },
                ],
            })
        });

        // draws: one per (mesh, slot); the materials used
        let mut draws = Vec::new();
        let mut draw_base = Vec::with_capacity(geo.meshes.len());
        let mut used = vec![false; mats.list.len()];
        let mut param_layout = ParamLayout::new(&mats);
        for (mi, m) in geo.meshes.iter().enumerate() {
            draw_base.push(draws.len() as u32);
            let shape = &geo.shapes[m.shape];
            for s in &m.slots {
                let mat = s.material.map(|i| mat_of[i]).unwrap_or(mats.default_index());
                used[mat] = true;
                let mut remap = [u32::MAX; 8];
                for (i, a) in mats.list[mat].attributes.iter().enumerate().take(8) {
                    if let Some(p) = shape.attr_names.iter().position(|x| x == a) {
                        remap[i] = p as u32;
                    }
                }
                let ml = &mats.list[mat];
                draws.push(DrawGpu {
                    mesh: mi as u32,
                    material: mat as u32,
                    first_tri: s.first,
                    tri_count: s.count,
                    remap,
                    param_base: param_layout.draw_base(&mats, mat, mi),
                    image_base: ml.image_base,
                    flags: ml.draw_flags(),
                    pad: 0,
                });
            }
        }
        let draws_buf = storage(device, "draws", bytemuck::cast_slice(&draws), wgpu::BufferUsages::empty());
        let meshes_buf = sized(device, "meshes", (geo.meshes.len() * std::mem::size_of::<MeshGpu>()) as u64, rw);
        let lights_buf = sized(device, "lights", (lit.lights.len().max(1) * std::mem::size_of::<LightGpu>()) as u64, rw);
        let frame_buf = sized(device, "frame", std::mem::size_of::<FrameGpu>() as u64, rw);
        let samples_buf = sized(device, "samples", 16, rw);
        let params_buf = sized(device, "material params", (param_layout.count.max(1) * 16) as u64, rw);
        let image_map_buf = storage(device, "image map", bytemuck::cast_slice(&mats.image_map), wgpu::BufferUsages::empty());
        let lut_buf = storage(device, "view lut", bytemuck::cast_slice(&lut.data), wgpu::BufferUsages::empty());
        let indirect = sized(device, "indirect draws", (draws.len() * 8 + 64) as u64 * INDIRECT_STRIDE,
                             wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST);

        // shadows: maps per shadow-casting light, or (rays) one BLAS per shape (its first mesh's slots) and a TLAS with
        // one instance per mesh
        let kinds: Vec<Option<&str>> = lit.lights.iter()
            .map(|l| (l.use_shadow && !rays && params.shadows).then_some(l.kind.as_str()))
            .collect();
        let shadow_maps = ShadowMaps::new(device, &kinds, params.shadow_map_size, params.shadow_face_size, params.shadow_windows,
                                          gpu.cache.as_ref())?;
        let mut blas: Vec<Option<ShapeBlas>> = (0..geo.shapes.len()).map(|_| None).collect();
        if rays {
            for m in &geo.meshes {
                let shape = &geo.shapes[m.shape];
                if blas[m.shape].is_some() || !m.visible.shadow || matches!(shape.kind, ShapeKind::ByFrame { .. }) {
                    continue;
                }
                let runs: Vec<(u32, u32)> = m.slots.iter().map(|s| (s.first, s.count)).filter(|r| r.1 > 0).collect();
                if runs.is_empty() {
                    continue;
                }
                let opaque: Vec<bool> = m.slots.iter().filter(|s| s.count > 0).map(|s| {
                    let mat = s.material.map(|i| mat_of[i]).unwrap_or(mats.default_index());
                    !mats.list[mat].casts_alpha_shadow()
                }).collect();
                blas[m.shape] = Some(make_blas(device, &shape_label(shape), shape.vertex_count, &runs, &opaque,
                                               matches!(shape.kind, ShapeKind::Rigid)));
            }
        }
        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene"),
            max_instances: geo.meshes.len().max(1) as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        if !(rays && params.shadows) {
            // bound but never traced: built once, empty
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("empty tlas") });
            enc.build_acceleration_structures(std::iter::empty(), std::iter::once(&tlas));
            gpu.queue.submit(Some(enc.finish()));
        }

        // images, samplers, tables
        let mut images = Vec::new();
        for (name, rec) in &mats.images {
            let px = crate::images::load(&bake.dir, rec).with_context(|| format!("image {name}"))?;
            images.push(pipelines::upload_texture(&gpu, name, &px));
        }
        if images.is_empty() {
            images.push(pipelines::upload_texture(&gpu, "none", &crate::images::magenta()));
        }
        let samplers = pipelines::samplers(device, params.anisotropy);
        let (lw, lh, ll, ldata) = mats.shade_lut();
        let shade_lut = pipelines::table_texture(&gpu, "mkr-shade lut", lw, lh, ll.max(2), &pad_layers(&ldata, lw, lh, ll));
        let n = crate::ltc::LTC_SIZE as u32;
        let ltc_data: Vec<f32> = crate::ltc::table(params.ltc_glossy).into_iter().flatten().collect();
        let ltc = pipelines::table_texture(&gpu, "ltc", n, n, 1, &ltc_data);

        eprintln!("mkr: {} images uploaded ({:.1} s)", images.len(), t0.elapsed().as_secs_f64());
        let layouts = pipelines::layouts(device, images.len() as u32);
        let t_compile = Instant::now();
        let threads = crate::gpu::compile_threads();
        let pipes = pipelines::make(&gpu, &layouts, &mats, &used, params.shadow_mode, threads, lit.lights.len())?;
        let compile_seconds = t_compile.elapsed().as_secs_f64();
        eprintln!("mkr: pipelines made ({:.1} s, {threads} threads)", t0.elapsed().as_secs_f64());

        // the film
        let accumulate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accumulate"),
            source: wgpu::ShaderSource::Wgsl(ACCUMULATE_WGSL.into()),
        });
        let sample_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let accumulate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accumulate"),
            entries: &[sample_tex(0), sample_tex(1), sample_tex(2), sample_tex(3), storage_entry(4, false),
                       storage_entry(5, true)],
        });
        let accumulate_pipe = compute_pipe(device, &accumulate_module, &accumulate_bgl, "accumulate",
                                           std::mem::size_of::<AccumulateGpu>() as u32, cache);
        let resolve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resolve"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_WGSL.into()),
        });
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve"),
            entries: &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false), storage_entry(3, true)],
        });
        let resolve_pipe = compute_pipe(device, &resolve_module, &resolve_bgl, "resolve",
                                        std::mem::size_of::<ResolveGpu>() as u32, cache);
        if let Err(e) = gpu.save_cache() {
            eprintln!("mkr: the pipeline cache is not kept: {e:#}");
        }
        let loaded_tris = vec![None; geo.shapes.len()];
        eprintln!(
            "mkr: scene ready in {:.1} s ({} meshes, {} shapes, {} vertices, {} of {} materials, {} images, {} shadow faces)",
            t0.elapsed().as_secs_f64(), geo.meshes.len(), geo.shapes.len(), geo.rest.len(),
            used.iter().filter(|&&u| u).count(), mats.list.len(), mats.images.len(), shadow_maps.entry_count()
        );
        let world = crate::world::WorldGpu::new(&gpu, &mats, &layouts.render, lit.film.transparent)?;
        let world_stand = crate::world::stand_ins(&gpu.device);
        let raytrace = crate::raytrace::RayTrace::of(&lit.eevee, &params);
        let pipelines::Layouts { render: render_bgl, gbuffer: gbuffer_bgl, light: light_bgl } = layouts;
        Ok(Renderer {
            gpu,
            bake,
            params,
            lights: lit.lights.clone(),
            light_threshold,
            shadow_steps,
            shadow_rays,
            geo,
            mats,
            param_layout,
            mat_of,
            lut,
            filter,
            rest,
            attrs,
            indices,
            posed,
            joints,
            morphs,
            skin_bg,
            skin_pipe,
            bend,
            frame_buf,
            samples_buf,
            lights_buf,
            meshes_buf,
            draws,
            draw_base,
            draws_buf,
            params_buf,
            image_map_buf,
            lut_buf,
            indirect,
            film: None,
            blas,
            tlas,
            shadow_maps,
            loaded_tris,
            render_bgl,
            gbuffer_bgl,
            light_bgl,
            render_bg: None,
            shadow_bg: None,
            shadow_setup_bg: None,
            images,
            samplers,
            shade_lut,
            ltc,
            pipes,
            accumulate_pipe,
            accumulate_bgl,
            resolve_pipe,
            resolve_bgl,
            targets: None,
            timers: None,
            compile_seconds,
            dof_scene: SceneDof::of(&lit.eevee),
            dof_post: None,
            dof_targets: None,
            world,
            world_stand,
            world_bg: None,
            mb_position: lit.render.get("motion_blur_position").and_then(|v| v.as_str()).unwrap_or("CENTER").to_string(),
            mb_steps: lit.eevee.get("motion_blur_steps").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
            mb_depth_scale: lit.eevee.get("motion_blur_depth_scale").and_then(|v| v.as_f64()).unwrap_or(100.0) as f32,
            mb_post: None,
            mb_targets: None,
            raytrace,
        })
    }

    pub fn adapter(&self) -> &str {
        &self.gpu.adapter
    }

    /// Whether motion blur samples time (mkr's own, `motion_blur: "quality"`) rather than smearing each sample in
    /// EEVEE's post process. A scene of more than one motion blur step (EEVEE draws that many segments of the shutter,
    /// each smeared over its own) gets the time samples too: they are closer to EEVEE's segments than one smear.
    fn time_sampled(&self) -> bool {
        self.params.motion_blur == MotionBlurMode::Quality || self.mb_steps > 1
    }

    fn targets(&mut self, w: u32, h: u32) {
        if self.targets.as_ref().is_some_and(|t| t.w == w && t.h == h) {
            return;
        }
        // the post process's targets read the old film and depth
        self.dof_targets = None;
        self.mb_targets = None;
        let device = &self.gpu.device;
        let tex = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let att = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let colors = [0, 1, 2, 3].map(|_| tex("sample", pipelines::COLOR_FORMAT, att | wgpu::TextureUsages::STORAGE_BINDING));
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: att,
            view_formats: &[],
        });
        let depth = depth_tex.create_view(&Default::default());
        let depth_read = depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let gb_base = tex("gbuffer", pipelines::GB_BASE_FORMAT, att);
        let gb_motion = tex("gbuffer motion", pipelines::GB_MOTION_FORMAT, att);
        let n = (w * h) as u64;
        let gb_extras = sized(device, "gbuffer extras", n * 32, wgpu::BufferUsages::STORAGE);
        let film = sized(device, "film", n * 16, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let pixels = sized(device, "pixels", n * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = [0, 1].map(|_| sized(device, "readback", n * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST));
        let film_read = std::env::var_os("MKR_LINEAR").map(|_| {
            [0, 1].map(|_| sized(device, "film read", n * 16, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST))
        });
        let dither = storage(device, "dither", bytemuck::cast_slice(&crate::dither::pattern(w, h)), wgpu::BufferUsages::empty());
        let taps = sized(device, "film taps", (MAX_SAMPLES * 28 * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let accumulate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate"),
            layout: &self.accumulate_bgl,
            entries: &[
                b(0, wgpu::BindingResource::TextureView(&colors[0])),
                b(1, wgpu::BindingResource::TextureView(&colors[1])),
                b(2, wgpu::BindingResource::TextureView(&colors[2])),
                b(3, wgpu::BindingResource::TextureView(&colors[3])),
                b(4, film.as_entire_binding()),
                b(5, taps.as_entire_binding()),
            ],
        });
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve"),
            layout: &self.resolve_bgl,
            entries: &[
                b(0, film.as_entire_binding()),
                b(1, self.lut_buf.as_entire_binding()),
                b(2, pixels.as_entire_binding()),
                b(3, dither.as_entire_binding()),
            ],
        });
        let gbuffer_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer"),
            layout: &self.gbuffer_bgl,
            entries: &[b(0, gb_extras.as_entire_binding())],
        });
        // the screen tracing's light (raytrace.rs): two pictures, sample s writing one and reading the other, the one
        // sample s - 1 wrote (a texel each without tracing); and so the traced reflections' history (rt_temporal)
        let (rw, rh) = if self.raytrace.is_some() { (w, h) } else { (1, 1) };
        let pair = |label: &str| [0, 1].map(|_| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: pipelines::COLOR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&Default::default()));
        let radiance = pair("traced light");
        let history = pair("traced reflections' history");
        let light_bg = [0, 1, 2, 3].map(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting"),
                layout: &self.light_bgl,
                entries: &[
                    b(0, wgpu::BindingResource::TextureView(&gb_base)),
                    b(1, wgpu::BindingResource::TextureView(&gb_motion)),
                    b(2, wgpu::BindingResource::TextureView(&depth_read)),
                    b(3, gb_extras.as_entire_binding()),
                    b(4, wgpu::BindingResource::TextureView(&colors[k])),
                    b(5, wgpu::BindingResource::TextureView(&radiance[(k + 1) % 2])),
                    b(6, wgpu::BindingResource::TextureView(&radiance[k % 2])),
                    b(7, wgpu::BindingResource::TextureView(&history[(k + 1) % 2])),
                    b(8, wgpu::BindingResource::TextureView(&history[k % 2])),
                ],
            })
        });
        self.film = Some(film);
        self.shadow_setup_bg = None;
        self.targets = Some(Targets {
            w, h, colors, depth, depth_read, gb_base, gb_motion, pixels, readback, film_read, accumulate_bg,
            resolve_bg, gbuffer_bg, light_bg, taps,
        });
    }

    /// Timestamp queries for frames of `samples` samples.
    fn timers(&mut self, samples: usize) {
        if !self.gpu.timestamps {
            return;
        }
        let need = (8 + 6 * samples) as u32;
        if self.timers.as_ref().is_some_and(|t| t[0].capacity >= need) {
            return;
        }
        let device = &self.gpu.device;
        self.timers = Some([0, 1].map(|_| Timer {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("time"), ty: wgpu::QueryType::Timestamp, count: need }),
            resolve: sized(device, "time", need as u64 * 8, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: sized(device, "time read", need as u64 * 8, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            capacity: need,
            marks: Vec::new(),
        }));
    }

    pub(crate) fn render_bind_group(&mut self, samples: usize) {
        let needed = (samples * std::mem::size_of::<SampleGpu>()) as u64;
        if self.samples_buf.size() < needed {
            self.samples_buf = sized(&self.gpu.device, "samples", needed, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            self.render_bg = None;
            self.shadow_setup_bg = None;
        }
        if self.render_bg.is_some() {
            return;
        }
        let geo = || SceneBuffers {
            posed: [&self.posed[1], &self.posed[0], &self.posed[2]],
            attrs: &self.attrs,
            indices: &self.indices,
            meshes: &self.meshes_buf,
            draws: &self.draws_buf,
        };
        let render = self.scene_bind_group("render", geo());
        let sm = &self.shadow_maps;
        let shadow = self.bind_group("shadow maps' passes", geo(), sm.dummy.views(), sm.dummy.views(), self.world_views());
        let world = self.bind_group("world probe", geo(), sm.coarse.views(), sm.fine.views(), &self.world_stand);
        self.render_bg = Some(render);
        self.shadow_bg = Some(shadow);
        self.world_bg = Some(world);
    }

    /// A bind group of `render_bgl` over the geometry `geo` (bindings 4-10) and the renderer's own frame, samples,
    /// lights, shadow, material, table and world bindings.
    pub(crate) fn scene_bind_group(&self, label: &str, geo: SceneBuffers<'_>) -> wgpu::BindGroup {
        let sm = &self.shadow_maps;
        self.bind_group(label, geo, sm.coarse.views(), sm.fine.views(), self.world_views())
    }

    /// The world's probe and harmonics (bindings 30, 31), or their stand-ins without a world.
    fn world_views(&self) -> &crate::world::WorldViews {
        self.world.as_ref().map(|w| &w.views).unwrap_or(&self.world_stand)
    }

    /// `coarse`, `fine`: the shadow maps' (depths, transparency) arrays: the cube faces' (bindings 23, 26) and the suns'
    /// maps' and windows' (28, 29); `world`: the world's probe (30, 31).
    fn bind_group(&self, label: &str, geo: SceneBuffers<'_>, coarse: [&wgpu::TextureView; 2], fine: [&wgpu::TextureView; 2],
                  world: &crate::world::WorldViews) -> wgpu::BindGroup {
        let views: Vec<&wgpu::TextureView> = self.images.iter().collect();
        let s = &self.samplers;
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.render_bgl,
            entries: &[
                b(0, self.frame_buf.as_entire_binding()),
                b(1, self.samples_buf.as_entire_binding()),
                b(2, self.lights_buf.as_entire_binding()),
                b(3, self.tlas.as_binding()),
                b(4, geo.posed[0].as_entire_binding()),
                b(5, geo.posed[1].as_entire_binding()),
                b(6, geo.posed[2].as_entire_binding()),
                b(7, geo.attrs.as_entire_binding()),
                b(8, geo.indices.as_entire_binding()),
                b(9, geo.meshes.as_entire_binding()),
                b(10, geo.draws.as_entire_binding()),
                b(11, self.params_buf.as_entire_binding()),
                b(12, wgpu::BindingResource::TextureViewArray(&views)),
                b(13, wgpu::BindingResource::Sampler(&s[0])),
                b(14, wgpu::BindingResource::Sampler(&s[1])),
                b(15, wgpu::BindingResource::Sampler(&s[2])),
                b(16, wgpu::BindingResource::Sampler(&s[3])),
                b(17, wgpu::BindingResource::Sampler(&s[4])),
                b(18, wgpu::BindingResource::Sampler(&s[5])),
                b(19, wgpu::BindingResource::TextureView(&self.shade_lut)),
                b(20, wgpu::BindingResource::Sampler(&s[6])),
                b(21, wgpu::BindingResource::TextureView(&self.ltc)),
                b(22, self.image_map_buf.as_entire_binding()),
                b(23, wgpu::BindingResource::TextureView(coarse[0])),
                b(24, self.shadow_maps.faces_buf.as_entire_binding()),
                b(25, self.shadow_maps.windows_buf.as_entire_binding()),
                b(26, wgpu::BindingResource::TextureView(coarse[1])),
                b(28, wgpu::BindingResource::TextureView(fine[0])),
                b(29, wgpu::BindingResource::TextureView(fine[1])),
                b(30, wgpu::BindingResource::TextureView(&world.cube)),
                b(31, world.harmonics.as_entire_binding()),
            ],
        })
    }

    /// One light of the bake on bake frame `k`: its per-frame floats (None: hidden on that frame).
    fn light_frame(&self, l: &LightRec, k: usize) -> Result<Option<&[f32]>> {
        let kk = if l.frames == 1 { 0 } else { k };
        let d = self.bake.f(l.offset + LIGHT_FLOATS * kk, LIGHT_FLOATS)?;
        Ok((d[21] >= 0.5).then_some(d))
    }

    /// The scene's lights on bake frame `k`, in mkr's record (radiance by Blender's conventions, EEVEE's influence,
    /// where its shadow map is; `sun_maps`: per light, a sun map's depth span and texel size this frame).
    pub(crate) fn lights_at(&self, k: usize, sun_maps: &[Option<[f32; 2]>]) -> Result<Vec<LightGpu>> {
        let p = &self.params;
        let mut out = Vec::new();
        for (li, l) in self.lights.iter().enumerate() {
            let Some(d) = self.light_frame(l, k)? else { continue };
            let v3 = |i: usize| Vec3::new(d[i], d[i + 1], d[i + 2]);
            let (pos, fwd, right, up, color) = (v3(0), v3(3).normalize(), v3(6), v3(9), v3(12));
            let energy = d[15];
            let (size_x, size_y, spot_size, spot_blend, radius) = (d[16], d[17], d[18], d[19], d[20]);
            let prop = |name: &str, default: f32| l.props.get(name).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default);
            let vis = |name: &str| l.visible.get(name).copied().unwrap_or(true);
            let bits = (vis("diffuse") as u32) | ((vis("glossy") as u32) << 1) | ((vis("transmission") as u32) << 2);
            let pi = std::f32::consts::PI;
            let (kind, radiance) = match l.kind.as_str() {
                "AREA" => {
                    let ellipse = matches!(l.shape.as_deref(), Some("DISK") | Some("ELLIPSE"));
                    let area = size_x * size_y * if ellipse { pi / 4.0 } else { 1.0 };
                    (if ellipse { 4.0 } else { 3.0 }, energy / (pi * area.max(1e-8)) * p.light_area)
                }
                "POINT" | "SPOT" => {
                    let r = radius.max(1e-4);
                    let rad = energy / (4.0 * pi * pi * r * r);
                    if l.kind == "SPOT" { (2.0, rad * p.light_spot) } else { (0.0, rad * p.light_point) }
                }
                "SUN" => {
                    let half = (radius * 0.5).max(1e-3);
                    let s = half.sin();
                    (1.0, energy / (pi * s * s) * p.light_sun)
                }
                other => bail!("light {}: type {other}", l.name),
            };
            // EEVEE fades a light out toward its influence radius (suns never fade)
            let inv_r2 = if p.light_influence && kind != 1.0 {
                let custom = l.props.get("use_custom_distance").and_then(|v| v.as_bool()).unwrap_or(false);
                let factor = prop("diffuse_factor", 1.0).max(prop("specular_factor", 1.0)).max(prop("transmission_factor", 1.0));
                influence_inv_r2(custom.then(|| prop("cutoff_distance", 40.0)), energy * color.max_element() * factor,
                                 self.light_threshold)
            } else {
                0.0
            };
            let (half_x, half_y, w) = if kind == 1.0 { (0.0, 0.0, radius * 0.5) } else { (size_x * 0.5, size_y * 0.5, radius) };
            let rn = right.normalize_or_zero();
            let un = up.normalize_or_zero();
            let sm = &self.shadow_maps;
            let map = match sm.slots[li] {
                Some(s) if s.sun => {
                    let [span, texel] = sun_maps.get(li).copied().flatten().unwrap_or([0.0; 2]);
                    [s.first as f32, span, sm.fine.size as f32, texel]
                }
                Some(s) => [s.first as f32, shadow_near(radius), sm.coarse.size as f32,
                            prop("shadow_maximum_resolution", 0.001).max(0.0)],
                None => [-1.0, 0.0, 0.0, 0.0],
            };
            out.push(LightGpu {
                pos: [pos.x, pos.y, pos.z, kind],
                fwd: [fwd.x, fwd.y, fwd.z, w],
                right: [rn.x, rn.y, rn.z, half_x],
                up: [un.x, un.y, un.z, half_y],
                color: [color.x * radiance, color.y * radiance, color.z * radiance, inv_r2],
                spot: [(spot_size * 0.5).cos(), spot_blend, if l.use_shadow { 1.0 } else { 0.0 }, bits as f32],
                factor: [prop("diffuse_factor", 1.0), prop("specular_factor", 1.0), prop("transmission_factor", 1.0),
                         prop("shadow_filter_radius", 1.0).max(0.0)],
                map,
            });
        }
        Ok(out)
    }

    /// The camera of `name` on fractional bake frame `time`.
    pub(crate) fn camera_at(&self, name: &str, time: f64) -> Result<Cam> {
        let rec = self.bake.camera(name).with_context(|| format!("the bake has no camera {name}"))?;
        let lit = self.bake.doc.lit.as_ref().and_then(|l| l.cameras.get(name));
        let (k0, k1, t) = scene::bracket(time, self.bake.doc.frames);
        let a = Cam::at(self.bake, rec, lit, k0)?;
        if t == 0.0 {
            return Ok(a);
        }
        Ok(Cam::lerp(&a, &Cam::at(self.bake, rec, lit, k1)?, t))
    }

    /// Draw `jobs` of output `output` (size `size`), each into its own PNG.
    pub fn render(&mut self, output: &str, size: [u32; 2], jobs: &[Job], opts: &Options, writer: &mut Writer)
                  -> Result<Stats> {
        let [w, h] = render_size(size, opts.percent);
        if w == 0 || h == 0 {
            bail!("--percent {} leaves no pixels", opts.percent);
        }
        if opts.samples as usize > MAX_SAMPLES {
            bail!("--samples {}: at most {MAX_SAMPLES}", opts.samples);
        }
        self.targets(w, h);
        self.render_bind_group(opts.samples as usize);
        self.timers(opts.samples as usize);
        // mkr's own motion blur draws each sample at its time in the shutter; EEVEE's all at the shutter's middle
        let timed = opts.shutter.filter(|_| self.time_sampled());
        let samples = film::samples(opts.samples, &self.filter, self.params.jitter, timed);
        let mut taps = vec![0.0f32; samples.len() * 28];
        for (si, s) in samples.iter().enumerate() {
            taps[si * 28..si * 28 + film::TAP_COUNT].copy_from_slice(&s.taps);
        }
        self.gpu.queue.write_buffer(&self.targets.as_ref().unwrap().taps, 0, bytemuck::cast_slice(&taps));
        let mut stats = Stats { rendered: 0, gpu_ms: Vec::new(), phases: Vec::new(), encode_ms: Vec::new(), stage_ms: Vec::new(),
                                submit_ms: Vec::new(), wait_ms: Vec::new(), collect_ms: Vec::new(), warnings: Vec::new() };
        let mut pending: Option<(usize, std::path::PathBuf, wgpu::SubmissionIndex)> = None;
        for (i, job) in jobs.iter().enumerate() {
            let slot = i % 2;
            let cam = self.camera_at(&job.camera, job.k as f64)?;
            if cam.dof.is_some() && cam.ortho.is_some() {
                stats.warnings.push(format!("frame {}: {} is orthographic: its depth of field is Blender's, not drawn",
                                            job.frame, job.camera));
            }
            let t_enc = Instant::now();
            let (index, stage_ms, submit_ms) = self.encode_frame(output, job, &samples, opts, w, h, slot)?;
            stats.encode_ms.push(t_enc.elapsed().as_secs_f64() * 1e3);
            stats.stage_ms.push(stage_ms);
            stats.submit_ms.push(submit_ms);
            if let Some((ps, path, idx)) = pending.take() {
                self.collect(ps, &path, idx, w, h, writer, &mut stats)?;
            }
            pending = Some((slot, job.path.clone(), index));
        }
        if let Some((ps, path, idx)) = pending.take() {
            self.collect(ps, &path, idx, w, h, writer, &mut stats)?;
        }
        Ok(stats)
    }

    /// Wait for a frame and hand its pixels to the writer; its GPU times when timestamps are on.
    #[allow(clippy::too_many_arguments)]
    fn collect(&mut self, slot: usize, path: &Path, index: wgpu::SubmissionIndex, w: u32, h: u32, writer: &mut Writer,
               stats: &mut Stats) -> Result<()> {
        let t = self.targets.as_ref().unwrap();
        let buf = &t.readback[slot];
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let timing = self.timers.as_ref().map(|q| {
            let (ttx, trx) = std::sync::mpsc::channel();
            let n = q[slot].marks.len() as u64 * 8;
            q[slot].read.slice(..n.max(8)).map_async(wgpu::MapMode::Read, move |r| {
                let _ = ttx.send(r);
            });
            trx
        });
        let t_wait = Instant::now();
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .map_err(|e| anyhow::anyhow!("GPU wait: {e}"))?;
        stats.wait_ms.push(t_wait.elapsed().as_secs_f64() * 1e3);
        let t_collect = Instant::now();
        if let (Some(trx), Some(q)) = (timing, &self.timers) {
            trx.recv()?.map_err(|e| anyhow::anyhow!("timestamps: {e}"))?;
            let timer = &q[slot];
            let n = timer.marks.len();
            let v: Vec<u64> = {
                let view = timer.read.slice(..(n as u64 * 8).max(8)).get_mapped_range()
                    .map_err(|e| anyhow::anyhow!("timestamps: {e:?}"))?;
                bytemuck::cast_slice(&view).to_vec()
            };
            timer.read.unmap();
            let period = self.gpu.timestamp_period as f64 / 1e6;
            let mut phases = [0.0f64; 10];
            for i in 1..n.min(v.len()) {
                phases[timer.marks[i] as usize] += v[i].wrapping_sub(v[i - 1]) as f64 * period;
            }
            if n >= 2 {
                stats.gpu_ms.push(v[n - 1].wrapping_sub(v[0]) as f64 * period);
            }
            stats.phases.push(phases);
        }
        rx.recv()?.map_err(|e| anyhow::anyhow!("readback: {e}"))?;
        let rgbx = buf.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("readback: {e:?}"))?
            [..(w * h * 4) as usize].to_vec();
        buf.unmap();
        writer.push(path.to_path_buf(), w, h, crate::output::Pixels::Rgbx(rgbx), 8)?;
        stats.collect_ms.push(t_collect.elapsed().as_secs_f64() * 1e3);
        if let Some(fr) = &t.film_read {
            // MKR_LINEAR: the scene-linear film (weighted sum over weights), before the view transform
            let fbuf = &fr[slot];
            let (ftx, frx) = std::sync::mpsc::channel();
            fbuf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = ftx.send(r);
            });
            self.gpu.wait()?;
            frx.recv()?.map_err(|e| anyhow::anyhow!("film readback: {e}"))?;
            let lin: Vec<f32> = {
                let data = fbuf.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("film readback: {e:?}"))?;
                let px: &[[f32; 4]] = bytemuck::cast_slice(&data);
                px.iter().take((w * h) as usize)
                    .flat_map(|p| if p[3] > 0.0 { [p[0] / p[3], p[1] / p[3], p[2] / p[3]] } else { [0.0; 3] })
                    .collect()
            };
            fbuf.unmap();
            crate::output::write_npy(&path.with_extension("npy"), &[h as usize, w as usize, 3], &lin)?;
        }
        stats.rendered += 1;
        Ok(())
    }

    /// Upload the geometry frame-by-frame shapes show on bake frame `k` (and their posed vertices on `times`); the
    /// draws' runs follow the frame's geometry.
    fn load_by_frame(&mut self, k: usize, times: &[f64; 3], motion: bool, shown: &[bool]) -> Result<bool> {
        let lit = self.bake.lit()?;
        let rays = self.params.shadow_mode == ShadowMode::Rays;
        let mut draws_changed = false;
        for (si, shape) in self.geo.shapes.iter().enumerate() {
            let ShapeKind::ByFrame { rec } = shape.kind else { continue };
            let meshes: Vec<usize> = (0..self.geo.meshes.len()).filter(|&mi| self.geo.meshes[mi].shape == si).collect();
            if !meshes.iter().any(|&mi| shown[mi]) {
                continue;
            }
            let r = &lit.meshes[rec];
            let fresh = |time: f64, data: bool| scene::by_frame_geom(self.bake, r, shape, k, time, data);
            let Some(cur) = fresh(times[1], true)? else { continue };
            let queue = &self.gpu.queue;
            let at = shape.vertex_base as u64 * 32;
            queue.write_buffer(&self.posed[1], at, bytemuck::cast_slice(&cur.posed));
            for (ti, buf) in [(0usize, &self.posed[0]), (2, &self.posed[2])] {
                let g = if motion { fresh(times[ti], false)?.map(|g| g.posed) } else { None };
                queue.write_buffer(buf, at, bytemuck::cast_slice(g.as_ref().unwrap_or(&cur.posed)));
            }
            if !cur.attrs.is_empty() {
                queue.write_buffer(&self.attrs, shape.attr_base as u64 * 16, bytemuck::cast_slice(&cur.attrs));
            }
            if self.loaded_tris[si] != Some(cur.triangles_offset) {
                queue.write_buffer(&self.indices, shape.index_base as u64 * 4, bytemuck::cast_slice(&cur.indices));
                self.loaded_tris[si] = Some(cur.triangles_offset);
            }
            for &mi in &meshes {
                let base = self.draw_base[mi] as usize;
                for (di, s) in self.geo.meshes[mi].slots.iter().enumerate() {
                    let run = cur.slots.iter().find(|x| x[0] == s.slot).map(|x| (x[1] as u32, x[2] as u32)).unwrap_or((0, 0));
                    let d = &mut self.draws[base + di];
                    if (d.first_tri, d.tri_count) != run {
                        d.first_tri = run.0;
                        d.tri_count = run.1;
                        draws_changed = true;
                    }
                }
            }
            // its BLAS follows the frame's runs
            let mi = meshes[0];
            let m = &self.geo.meshes[mi];
            if rays && m.visible.shadow {
                let runs: Vec<(u32, u32)> = self.draws[self.draw_base[mi] as usize..][..m.slots.len()]
                    .iter().map(|d| (d.first_tri, d.tri_count)).filter(|r| r.1 > 0).collect();
                let fits = self.blas[si].as_ref().is_some_and(|b| b.runs.iter().map(|r| r.1).eq(runs.iter().map(|r| r.1)));
                if !fits && !runs.is_empty() {
                    let opaque: Vec<bool> = m.slots.iter().zip(&self.draws[self.draw_base[mi] as usize..])
                        .filter(|(_, d)| d.tri_count > 0)
                        .map(|(s, _)| {
                            let mat = s.material.map(|i| self.mat_of[i]).unwrap_or(self.mats.default_index());
                            !self.mats.list[mat].casts_alpha_shadow()
                        })
                        .collect();
                    self.blas[si] = Some(make_blas(&self.gpu.device, &shape_label(shape), shape.vertex_count, &runs, &opaque, false));
                } else if let Some(b) = &mut self.blas[si] {
                    b.runs = runs;
                }
            }
        }
        if draws_changed {
            self.gpu.queue.write_buffer(&self.draws_buf, 0, bytemuck::cast_slice(&self.draws));
        }
        Ok(draws_changed)
    }

    /// Pose the scene on bake frame `k` for `output` on `times` (the shutter's open, the frame, its close; the frame's own
    /// three times without motion blur): the frame-by-frame shapes' geometry loaded, joints and shape keys uploaded and
    /// the deforming shapes posed by compute passes in `enc` (their normals rebuilt as Blender's on the frame's time),
    /// the meshes' and materials' per-frame records uploaded; with `shadows`, the shadow rays' acceleration structures
    /// built. `hidden` meshes (indices into the bake's `lit.meshes`) are left out of the frame: neither drawn nor
    /// casting shadows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage(&mut self, enc: &mut wgpu::CommandEncoder, output: &str, k: usize, times: [f64; 3], motion: bool,
                        shadows: bool, hidden: &[usize]) -> Result<Staged> {
        let bake = self.bake;
        let lit = bake.lit()?;
        let mut poses: Vec<Option<Pose>> = Vec::with_capacity(3);
        for (ti, &t) in times.iter().enumerate() {
            poses.push(if ti == 1 || motion { Some(Pose::at(bake, &self.geo, t)?) } else { None });
        }
        let mut shown: Vec<bool> = self.geo.meshes.iter().enumerate()
            .map(|(mi, m)| scene::shown(bake, &lit.meshes[mi], m, k, output))
            .collect::<Result<_>>()?;
        for &mi in hidden {
            if let Some(s) = shown.get_mut(mi) {
                *s = false;
            }
        }
        self.load_by_frame(k, &times, motion, &shown)?;
        let queue = &self.gpu.queue;
        for (ti, p) in poses.iter().enumerate() {
            if let Some(p) = p {
                queue.write_buffer(&self.joints[ti], 0, bytemuck::cast_slice(&p.joints));
                if !p.morphs.is_empty() {
                    queue.write_buffer(&self.morphs[ti], 0, bytemuck::cast_slice(&p.morphs));
                }
            }
        }
        let cur = poses[1].as_ref().unwrap();
        let prev = poses[0].as_ref().unwrap_or(cur);
        let next = poses[2].as_ref().unwrap_or(cur);
        let mut meshes = Vec::with_capacity(self.geo.meshes.len());
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            let shape = &self.geo.shapes[m.shape];
            let ob_gl = match &m.object_matrix {
                Some(mr) => scene::matrix_at(bake, mr, times[1])?,
                None => Mat4::IDENTITY,
            };
            let ob = blender_matrix(&ob_gl);
            let mc = cur.matrices[mi];
            let nm = Mat3::from_mat4(mc).inverse().transpose();
            meshes.push(MeshGpu {
                object: mat(&ob),
                object_inv: mat(&ob.inverse()),
                m_prev: rows(&prev.matrices[mi]),
                m_cur: rows(&mc),
                m_next: rows(&next.matrices[mi]),
                n_cur: rows(&Mat4::from_mat3(nm)),
                object_color: m.color,
                object_info: [m.pass_index, unit_hash(&m.name), 0.0, 0.0],
                vertex_base: shape.vertex_base,
                attr_base: shape.attr_base,
                attr_stride: shape.attr_stride,
                index_base: shape.index_base,
                draw_base: self.draw_base[mi],
                generated: shape.generated.unwrap_or(u32::MAX),
                flags: 0,
                pad: 0,
            });
        }
        queue.write_buffer(&self.meshes_buf, 0, bytemuck::cast_slice(&meshes));
        let params = self.param_layout.values(&self.mats, bake, k)?;
        if !params.is_empty() {
            queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&params));
        }

        // posing the deforming shapes
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("skin"), timestamp_writes: None });
            pass.set_pipeline(&self.skin_pipe);
            for (ti, p) in poses.iter().enumerate() {
                let Some(p) = p else { continue };
                pass.set_bind_group(0, &self.skin_bg[ti], &[]);
                for (si, s) in self.geo.shapes.iter().enumerate() {
                    let ShapeKind::Deforming { armature, morphs } = &s.kind else { continue };
                    if s.vertex_count == 0 {
                        continue;
                    }
                    let job = SkinJob {
                        m: rows(&p.shape_matrices[si]),
                        vertex_base: s.vertex_base,
                        vertex_count: s.vertex_count,
                        skin_base: s.skin_base.unwrap_or(u32::MAX),
                        joint_base: armature.map(|a| p.joint_base[a]).unwrap_or(0),
                        morph_base: morphs.as_ref().map(|x| x.delta_base).unwrap_or(0),
                        morph_first: p.morph_runs[si].0,
                        morph_count: p.morph_runs[si].1,
                        pad: 0,
                    };
                    pass.set_immediates(0, bytemuck::bytes_of(&job));
                    pass.dispatch_workgroups(s.vertex_count.div_ceil(64), 1, 1);
                }
            }
        }
        // the frame's normals as Blender rebuilds them on the posed shape (the shutter's ends shade nothing)
        if let Some(b) = &self.bend {
            b.dispatch(enc);
        }
        if shadows {
            self.build_rays(enc, cur, &shown)?;
        }
        let boxes: Vec<Option<(Vec3, Vec3)>> = (0..self.geo.meshes.len())
            .map(|mi| {
                poses.iter().flatten().map(|p| self.world_box(mi, p)).try_fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                    |(lo, hi), b| b.map(|(l, h)| (lo.min(l), hi.max(h))))
            })
            .collect();
        let cur = poses.swap_remove(1).unwrap();
        Ok(Staged { shown, cur, boxes })
    }

    /// The shadow rays' structures: rigid shapes' BLASes once, the others' on every frame, the TLAS of the shown meshes.
    pub(crate) fn build_rays(&mut self, enc: &mut wgpu::CommandEncoder, cur: &Pose, shown: &[bool]) -> Result<()> {
        let mut entries = Vec::new();
        for (si, s) in self.geo.shapes.iter().enumerate() {
            let Some(b) = &self.blas[si] else { continue };
            let rigid = matches!(s.kind, ShapeKind::Rigid);
            if rigid && b.built {
                continue;
            }
            let src = if rigid { &self.rest } else { &self.posed[1] };
            let geoms: Vec<wgpu::BlasTriangleGeometry<'_>> = b.runs.iter().zip(&b.sizes).map(|(r, size)| {
                wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: src,
                    first_vertex: s.vertex_base,
                    vertex_stride: 32,
                    index_buffer: Some(&self.indices),
                    first_index: Some(s.index_base + r.0 * 3),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }
            }).collect();
            entries.push(wgpu::BlasBuildEntry { blas: &b.blas, geometry: wgpu::BlasGeometries::TriangleGeometries(geoms) });
        }
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            let inst = match &self.blas[m.shape] {
                Some(b) if shown[mi] && m.visible.shadow => {
                    let t = rows(&cur.matrices[mi]);
                    let flat = [t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1], t[1][2], t[1][3], t[2][0], t[2][1], t[2][2], t[2][3]];
                    Some(wgpu::TlasInstance::new(&b.blas, flat, mi as u32, 0xff))
                }
                _ => None,
            };
            *self.tlas.get_mut_single(mi).context("TLAS instance")? = inst;
        }
        enc.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas));
        drop(entries);
        for b in self.blas.iter_mut().flatten() {
            b.built = true;
        }
        Ok(())
    }

    /// The world box of a rigid mesh on this frame (None: it deforms, so no box is known without posing it).
    fn world_box(&self, mi: usize, cur: &Pose) -> Option<(Vec3, Vec3)> {
        let m = &self.geo.meshes[mi];
        let shape = &self.geo.shapes[m.shape];
        if !matches!(shape.kind, ShapeKind::Rigid) {
            return None;
        }
        let mx = cur.matrices[mi];
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { shape.lo.x } else { shape.hi.x },
                if i & 2 == 0 { shape.lo.y } else { shape.hi.y },
                if i & 4 == 0 { shape.lo.z } else { shape.hi.z },
            );
            let p = mx.transform_point3(c);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        Some((lo, hi))
    }

    /// The lights' shadow maps on bake frame `k`: every face's matrix written, and per layer its multi-draws planned
    /// into `args`, the opaque casters' and then the hashed ones' (casters whose box misses a face are left out of it;
    /// a cube light's windows draw every caster of its faces, the setup passes placing them; a mesh's opaque runs
    /// merged where they touch). Also returns, per light, a sun map's depth span and texel size (see `lights_at`). A
    /// sun's finer maps cover slices of the view of `cam` (`w` x `h`: shadows.rs SUN_CASCADES).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn plan_shadow_maps(&self, k: usize, cur: &Pose, shown: &[bool], args: &mut Vec<IndirectArgs>, cam: &Cam, w: u32,
                        h: u32) -> Result<(Vec<(usize, [Vec<Group>; 2])>, Vec<Option<[f32; 2]>>)> {
        let sm = &self.shadow_maps;
        let mut plan = Vec::new();
        let mut suns = vec![None; self.lights.len()];
        if sm.face_count() == 0 {
            return Ok((plan, suns));
        }
        let casters: Vec<usize> = (0..self.geo.meshes.len()).filter(|&mi| shown[mi] && self.geo.meshes[mi].visible.shadow).collect();
        let boxes: Vec<Option<(Vec3, Vec3)>> = (0..self.geo.meshes.len()).map(|mi| self.world_box(mi, cur)).collect();
        // the scene's box (suns fit their map to it): rigid boxes, deforming shapes' rest boxes moved by their matrix
        let (mut slo, mut shi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &mi in &casters {
            let (lo, hi) = boxes[mi].unwrap_or_else(|| {
                let s = &self.geo.shapes[self.geo.meshes[mi].shape];
                let c = cur.shape_matrices[self.geo.meshes[mi].shape].transform_point3((s.lo + s.hi) * 0.5);
                let r = (s.hi - s.lo).length() * 0.5 + 0.5;
                (c - Vec3::splat(r), c + Vec3::splat(r))
            });
            slo = slo.min(lo);
            shi = shi.max(hi);
        }
        let mut faces = vec![ShadowFaceGpu::zeroed(); sm.face_count()];
        // per entry (cube faces, suns' maps): the casters drawn into it; and every cube light's, for the windows (which
        // the setup passes give to any light's faces)
        let mut drawn: Vec<Option<Vec<usize>>> = vec![None; sm.face_count()];
        let mut for_windows = Vec::new();
        for (li, l) in self.lights.iter().enumerate() {
            let Some(slot) = sm.slots[li] else { continue };
            let Some(d) = self.light_frame(l, k)? else { continue };
            let v3 = |i: usize| Vec3::new(d[i], d[i + 1], d[i + 2]);
            let (c, f, r, u) = (v3(0), v3(3).normalize(), v3(6).normalize_or_zero(), v3(9).normalize_or_zero());
            if slot.sun {
                if casters.is_empty() {
                    continue;
                }
                // the finer maps over the view from the camera to SUN_NEAR x 8^c metres, the last over the scene
                for c in 0..slot.faces {
                    let (m, span, texel) = if c + 1 == slot.faces {
                        shadows::sun_matrix(f, r, u, slo, shi, sm.fine.size)
                    } else {
                        let reach = shadows::SUN_NEAR * 8f32.powi(c as i32);
                        shadows::sun_cascade(f, r, u, slo, shi, &view_slice(cam, w as f32, h as f32, reach), sm.fine.size)
                    };
                    if c + 1 == slot.faces {
                        suns[li] = Some([span, texel]);
                    }
                    faces[(slot.first + c) as usize] = ShadowFaceGpu { vp: mat(&m) };
                    drawn[(slot.first + c) as usize] = Some(casters.clone());
                }
                continue;
            }
            let near = shadow_near(d[20]);
            for face in 0..slot.faces {
                let (s, t, a) = shadows::face_axes(face, r, u, f);
                let entry = (slot.first + face) as usize;
                faces[entry] = ShadowFaceGpu { vp: mat(&shadows::face_matrix(c, s, t, a, near)) };
                let list: Vec<usize> = casters.iter().copied()
                    .filter(|&mi| boxes[mi].is_none_or(|(lo, hi)| box_in_face(lo, hi, c, s, t, a, near)))
                    .collect();
                for_windows.extend_from_slice(&list);
                drawn[entry] = Some(list);
            }
        }
        self.gpu.queue.write_buffer(&sm.faces_buf, 0, bytemuck::cast_slice(&faces));
        for (entry, list) in drawn.iter().enumerate() {
            if let Some(list) = list {
                plan.push((entry, self.caster_groups(list, args)));
            }
        }
        if !sm.window_entries().is_empty() {
            for_windows.sort_unstable();
            for_windows.dedup();
            let groups = self.caster_groups(&for_windows, args);
            for entry in sm.window_entries() {
                plan.push((entry as usize, groups.clone()));
            }
        }
        Ok((plan, suns))
    }

    /// The shadow passes' multi-draws of `casters` (meshes) planned into `args`: [the opaque runs (a mesh's merged
    /// where they touch), the hashed casters (grouped by material)].
    fn caster_groups(&self, casters: &[usize], args: &mut Vec<IndirectArgs>) -> [Vec<Group>; 2] {
        let mut tested: Vec<(usize, IndirectArgs)> = Vec::new();
        let mut opaque: Vec<IndirectArgs> = Vec::new();
        for &mi in casters {
            let m = &self.geo.meshes[mi];
            let shape = &self.geo.shapes[m.shape];
            let base = self.draw_base[mi] as usize;
            let mut run: Option<(u32, u32, usize)> = None;
            for di in base..base + m.slots.len() {
                let d = &self.draws[di];
                if d.tri_count == 0 {
                    continue;
                }
                if self.pipes.shadow[d.material as usize].is_some() {
                    tested.push((d.material as usize, indirect(shape, d.first_tri, d.tri_count, di)));
                    continue;
                }
                run = match run {
                    Some((f0, n0, d0)) if f0 + n0 == d.first_tri => Some((f0, n0 + d.tri_count, d0)),
                    Some((f0, n0, d0)) => {
                        opaque.push(indirect(shape, f0, n0, d0));
                        Some((d.first_tri, d.tri_count, di))
                    }
                    None => Some((d.first_tri, d.tri_count, di)),
                };
            }
            if let Some((f0, n0, d0)) = run {
                opaque.push(indirect(shape, f0, n0, d0));
            }
        }
        tested.sort_by_key(|x| x.0);
        // the hashed casters' opaque parts go into the depth with the opaque casters, their partly transparent parts
        // into the colour layers after
        let mut solid = Vec::new();
        for a in opaque {
            push_group(&mut solid, args, Pipe::ShadowDepth, a);
        }
        for &(m, a) in &tested {
            push_group(&mut solid, args, Pipe::Shadow(m), a);
        }
        let mut hashed = Vec::new();
        for (m, a) in tested {
            push_group(&mut hashed, args, Pipe::ShadowHashed(m), a);
        }
        [solid, hashed]
    }

    /// A multi-draw's pipeline.
    fn pipe(&self, p: Pipe) -> &wgpu::RenderPipeline {
        match p {
            Pipe::Depth(cull) => &self.pipes.depth[cull as usize],
            Pipe::Prepass(m) => self.pipes.prepass[m].as_ref().unwrap(),
            Pipe::GBuffer(m) => self.pipes.gbuffer[m].as_ref().unwrap(),
            Pipe::ShadowDepth => &self.pipes.shadow_depth,
            Pipe::Shadow(m) => self.pipes.shadow[m].as_ref().unwrap(),
            Pipe::ShadowHashed(m) => self.pipes.shadow_hashed[m].as_ref().unwrap(),
        }
    }

    /// Issue planned multi-draws (`sample`: the sample, or the shadow face). Every drawing pipeline shares one layout,
    /// so the immediates set after the first pipeline hold for the rest.
    fn draw_groups(&self, pass: &mut wgpu::RenderPass<'_>, groups: &[Group], indirect: &wgpu::Buffer, sample: u32) {
        for (i, g) in groups.iter().enumerate() {
            pass.set_pipeline(self.pipe(g.pipe));
            if i == 0 {
                pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: 0, sample }));
            }
            pass.multi_draw_indexed_indirect(indirect, g.first as u64 * INDIRECT_STRIDE, g.count);
        }
    }

    /// The frame record the drawing shaders read: camera `cam` (the frame's own time) on a `w` x `h` picture, `lights`
    /// lights, `samples` samples, motion blur on or off, the frame's random seed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn frame_record(&self, cam: &Cam, w: u32, h: u32, lights: u32, samples: u32, motion: bool, seed: u32)
                               -> FrameGpu {
        let camera = orthonormal(&blender_matrix(&cam.world));
        let fwd = cam.forward();
        let p = &self.params;
        FrameGpu {
            view: mat(&cam.view()),
            camera: mat(&camera),
            camera_inv: mat(&camera.inverse()),
            cam_pos: [cam.position().x, cam.position().y, cam.position().z, if cam.ortho.is_some() { 1.0 } else { 0.0 }],
            cam_fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            size: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
            light_count: lights,
            sample_count: samples,
            flags: motion as u32,
            frame_seed: seed,
            // the spread is the map tracing's (measured); ray-traced shadows aim over the light's true shape
            shadow: [p.shadow_offset, p.shadow_offset_rel,
                     if p.shadow_mode == ShadowMode::Map { p.shadow_spread } else { 1.0 }, if p.shadows { 1.0 } else { 0.0 }],
            scale: [p.diffuse, p.glossy, p.mip_bias.exp2(), p.alpha_cell],
            trace: [self.shadow_steps as f32, p.shadow_trace, p.ltc_sphere_diffuse as u32 as f32,
                    p.ltc_sphere_glossy as u32 as f32],
            bias: [p.shadow_normal_bias, p.shadow_depth_bias, (p.ltc_glossy == crate::params::LtcFit::Eevee) as u32 as f32,
                   p.shadow_filter_offset],
            shadow_map: [p.shadow_step_power, (p.shadow_rule == crate::params::ShadowRule::V2) as u32 as f32,
                         p.shadow_filter, p.shadow_lod_reach],
            shadow_lod: [cam.pixel_footprint(w as f32, h as f32), p.shadow_lod_bias, p.shadow_limit_lod,
                         self.shadow_maps.base_lod as f32],
            shadow_pool: self.shadow_maps.pool_record(),
            // no world and no tracing unless the lit frame's encoding says (encode_frame: world.rs, raytrace.rs): the
            // looks' and layers' passes are EEVEE's with a black world and no ray tracing
            world: [0.0; 4],
            world_probe: [0.0; 4],
            rt: [0.0; 4],
            rt2: [0.0; 4],
            rt3: [0.0; 4],
            rt4: [0.0; 4],
            sampling: [self.shadow_rays as f32, (p.shadow_sequence == ShadowSequence::Sobol) as u32 as f32, 0.0, 0.0],
        }
    }

    /// Record and submit one frame. Returns the submission, and the CPU milliseconds spent posing the frame (`stage`)
    /// and finishing and submitting its commands.
    #[allow(clippy::too_many_arguments)]
    fn encode_frame(&mut self, output: &str, job: &Job, samples: &[Sample], opts: &Options, w: u32, h: u32, slot: usize)
                    -> Result<(wgpu::SubmissionIndex, f64, f64)> {
        let lit = self.bake.lit()?;
        let k = job.k;
        let kf = k as f64;
        // motion blur: mkr's own draws each sample at its time in the shutter (the poses of the shutter's open, middle
        // and close interpolated); EEVEE's draws them all at the middle and its post process smears each one along the
        // pixels' velocities to the open and the close
        let shutter = opts.shutter.map(|s| Shutter::new(s, &self.mb_position));
        let quality = self.time_sampled();
        let timed = shutter.filter(|_| quality && samples.len() > 1);
        let post = shutter.filter(|_| !quality);
        let times: [f64; 3] = match timed.or(post) {
            Some(s) => s.times(kf),
            None => [kf, kf, kf],
        };
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let mut marks: Vec<u8> = Vec::new();
        let stamp = |enc: &mut wgpu::CommandEncoder, marks: &mut Vec<u8>, phase: u8, timers: &Option<[Timer; 2]>| {
            if let Some(t) = timers
                && (marks.len() as u32) < t[slot].capacity
            {
                enc.write_timestamp(&t[slot].set, marks.len() as u32);
                marks.push(phase);
            }
        };
        stamp(&mut enc, &mut marks, 0, &self.timers);
        let rays = self.params.shadow_mode == ShadowMode::Rays && self.params.shadows;
        let t_stage = Instant::now();
        let Staged { shown, cur, boxes } = self.stage(&mut enc, output, k, times, timed.or(post).is_some(), rays, &job.hidden)?;
        let stage_ms = t_stage.elapsed().as_secs_f64() * 1e3;
        stamp(&mut enc, &mut marks, 0, &self.timers);
        // the camera: shading on the frame's time, each sample's raster on its own
        let cam = self.camera_at(&job.camera, times[1])?;
        let view = cam.view();
        // every indirect draw of the frame: the shadow faces', then the prepass's and the G-buffer's
        let mut args: Vec<IndirectArgs> = Vec::new();
        let (shadow_plan, sun_maps) = if self.params.shadows && !rays {
            self.plan_shadow_maps(k, &cur, &shown, &mut args, &cam, w, h)?
        } else {
            (Vec::new(), Vec::new())
        };
        let queue = &self.gpu.queue;
        let lights = self.lights_at(k, &sun_maps)?;
        let seed = (job.frame as u32).wrapping_mul(0x9e37_79b9);
        let mut frame = self.frame_record(&cam, w, h, lights.len() as u32, samples.len() as u32, timed.is_some(), seed);
        if post.is_some() {
            frame.flags |= 2;
        }
        if let Some(world) = &self.world {
            (frame.world, frame.world_probe) = world.record(&self.params);
        }
        if let Some(rt) = &self.raytrace {
            let zw;
            (frame.rt, frame.rt2, frame.rt3, frame.rt4, zw) = rt.record(&cam, &self.params);
            frame.sampling[2..].copy_from_slice(&zw);
        }
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        if !lights.is_empty() {
            queue.write_buffer(&self.lights_buf, 0, bytemuck::cast_slice(&lights));
        }
        // the world's probe, drawn again when its animated values change (its time counts as the pose's)
        if let (Some(world), Some(bg)) = (self.world.as_mut(), self.world_bg.as_ref()) {
            world.update(&mut enc, bg, &self.mats.params_at(self.bake, k)?);
            stamp(&mut enc, &mut marks, 0, &self.timers);
        }
        // depth of field: the samples' share of the aperture through the lens, the post process's on the film after
        // them (its pipelines and targets made when first needed)
        let (lens_share, post_share) = dof::split(self.params.dof, &self.dof_scene, samples.len() as u32);
        let dof_rec = dof::record(&cam, w, h, post_share, &self.dof_scene, &self.params);
        if dof_rec.is_some() {
            let device = &self.gpu.device;
            let post = self.dof_post.get_or_insert_with(|| DofPost::new(device, self.gpu.cache.as_ref()));
            if !self.dof_targets.as_ref().is_some_and(|d| d.w == w && d.h == h) {
                let depth = &self.targets.as_ref().unwrap().depth_read;
                self.dof_targets = Some(post.targets(device, w, h, self.film.as_ref().unwrap(), depth));
            }
        }
        // the post motion blur's cameras: the shutter's middle, open and close, without jitter or lens
        let plain = |c: &Cam| mat(&(c.projection(w as f32, h as f32, glam::Vec2::ZERO) * c.view()));
        let mb_cams = match post {
            Some(_) => [plain(&cam), plain(&self.camera_at(&job.camera, times[0])?), plain(&self.camera_at(&job.camera, times[2])?)],
            None => [[[0.0; 4]; 4]; 3],
        };
        let mb_rec = post.map(|_| motion_blur::record(&cam, w, h, self.mb_depth_scale));
        if mb_rec.is_some() {
            let device = &self.gpu.device;
            let taps = self.params.mb_taps;
            let mbp = self.mb_post.get_or_insert_with(|| MotionPost::new(device, self.gpu.cache.as_ref(), taps));
            if !self.mb_targets.as_ref().is_some_and(|d| d.w == w && d.h == h) {
                let t = self.targets.as_ref().unwrap();
                self.mb_targets = Some(mbp.targets(device, w, h, &self.samples_buf, &t.depth_read, &t.gb_motion, &t.colors,
                                                   &self.accumulate_bgl, self.film.as_ref().unwrap(), &t.taps));
            }
        }
        let mut sgpu = Vec::with_capacity(samples.len());
        for (si, s) in samples.iter().enumerate() {
            let (pair, t, cam_s) = match timed {
                Some(sh) => {
                    let half = sh.length * 0.5;
                    let c = self.camera_at(&job.camera, times[1] + s.time as f64)?;
                    if s.time < 0.0 { (0u32, 1.0 + s.time / half, c) } else { (1u32, s.time / half, c) }
                }
                None => (1u32, 0.0, cam),
            };
            let lens = cam_s.lens_offset(s.lens, samples.len() as u32) * lens_share;
            let vp = cam_s.view_projection(w as f32, h as f32, s.jitter, lens);
            let n = si as u32 + 1;
            sgpu.push(SampleGpu {
                vp: mat(&vp),
                vp_inv: mat(&vp.as_dmat4().inverse().as_mat4()),
                info: [pair, si as u32, 0, 0],
                t: [t, s.jitter.x, s.jitter.y, 0.0],
                rand: [film::halton(n, 7), film::halton(n, 11), film::halton(n, 13), film::halton(n, 17)],
                vp_cur: mb_cams[0],
                vp_prev: mb_cams[1],
                vp_next: mb_cams[2],
            });
        }
        queue.write_buffer(&self.samples_buf, 0, bytemuck::cast_slice(&sgpu));

        // the view volumes of the shutter's cameras (the frame's own alone without motion blur), a few pixels wider: a
        // rigid mesh whose box (over the shutter) misses every one of them is not drawn
        let margin = (4.0 / w as f32, 4.0 / h as f32);
        let mut volumes = vec![view_volume(&cam, w as f32, h as f32, margin, lens_share)];
        if timed.or(post).is_some() {
            for t in [times[0], times[2]] {
                volumes.push(view_volume(&self.camera_at(&job.camera, t)?, w as f32, h as f32, margin, lens_share));
            }
        }
        let in_view: Vec<bool> = boxes.iter().map(|b| match b {
            Some((lo, hi)) => volumes.iter().any(|v| box_in_volume(*lo, *hi, v)),
            None => true,
        }).collect();

        // this frame's draws: opaque and hashed ones, then the blended ones back to front
        let mut solid = Vec::new();
        let mut blended = Vec::new();
        for (di, d) in self.draws.iter().enumerate() {
            let mi = d.mesh as usize;
            let m = &self.geo.meshes[mi];
            if !shown[mi] || !in_view[mi] || !m.visible.camera || d.tri_count == 0 {
                continue;
            }
            if self.mats.list[d.material as usize].blended {
                let shape = &self.geo.shapes[m.shape];
                let c = cur.matrices[mi] * ((shape.lo + shape.hi) * 0.5).extend(1.0);
                blended.push(((view * c).z, di));
            } else {
                solid.push(di);
            }
        }
        // farthest first: most negative camera z
        blended.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // the solid draws as multi-draws grouped by pipeline. Every one goes through the depth prepass (opaque ones
        // depth alone), so the G-buffer pass runs each pixel's kept surface once: its extras are storage writes,
        // which only that single writer keeps consistent.
        let mut pre_groups = Vec::new();
        let mut pre_list: Vec<(Pipe, usize)> = solid.iter().map(|&di| {
            let mi = self.draws[di].material as usize;
            let p = if self.pipes.prepass[mi].is_some() { Pipe::Prepass(mi) } else { Pipe::Depth(self.mats.list[mi].cull) };
            (p, di)
        }).collect();
        pre_list.sort_by_key(|&(p, di)| (match p { Pipe::Depth(c) => (0, c as usize), Pipe::Prepass(m) => (1, m), _ => (2, 0) }, di));
        for (p, di) in pre_list {
            let d = &self.draws[di];
            let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
            push_group(&mut pre_groups, &mut args, p, indirect(shape, d.first_tri, d.tri_count, di));
        }
        let mut gb_groups = Vec::new();
        solid.sort_by_key(|&di| (self.draws[di].material, di));
        for &di in &solid {
            let d = &self.draws[di];
            if self.pipes.gbuffer[d.material as usize].is_none() {
                continue;
            }
            let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
            push_group(&mut gb_groups, &mut args, Pipe::GBuffer(d.material as usize), indirect(shape, d.first_tri, d.tri_count, di));
        }
        let need = (args.len().max(1) as u64) * INDIRECT_STRIDE;
        if self.indirect.size() < need {
            self.indirect = sized(&self.gpu.device, "indirect draws", need.next_power_of_two(),
                                  wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST);
        }
        if !args.is_empty() {
            self.gpu.queue.write_buffer(&self.indirect, 0, bytemuck::cast_slice(&args));
        }
        // the shadow windows' setup reads the first sample's depth and the frame's records
        if !shadow_plan.is_empty() && self.shadow_setup_bg.is_none() {
            let depth = &self.targets.as_ref().unwrap().depth_read;
            self.shadow_setup_bg = self.shadow_maps.setup_bind_group(&self.gpu.device, &self.frame_buf, &self.samples_buf,
                                                                     &self.lights_buf, depth);
        }
        let t = self.targets.as_ref().unwrap();
        let bg = self.render_bg.as_ref().unwrap();
        let (gw, gh) = (w.div_ceil(8), h.div_ceil(8));
        // many lights: each tile's culled first (the same picture)
        let light_pipe = match &self.pipes.light_tiled {
            Some(p) if lights.len() > pipelines::TILED_FROM => p,
            _ => &self.pipes.light,
        };
        for si in 0..samples.len() {
            let imm = |di: usize| Immediate { draw: di as u32, sample: si as u32 };
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("prepass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &t.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                        // the G-buffer pass's first-surface test starts from 0 (pipelines.rs first_surface)
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                self.draw_groups(&mut pass, &pre_groups, &self.indirect, si as u32);
            }
            stamp(&mut enc, &mut marks, 2, &self.timers);
            // the shadow maps, once the first sample's depth places their windows
            if si == 0 && !shadow_plan.is_empty() {
                if let Some(setup_bg) = &self.shadow_setup_bg {
                    self.shadow_maps.encode_setup(&mut enc, setup_bg, w, h);
                }
                let sbg = self.shadow_bg.as_ref().unwrap();
                let sm = &self.shadow_maps;
                for (entry, [solid, hashed]) in &shadow_plan {
                    let views = sm.entry_views(*entry as u32);
                    // the opaque casters (and the hashed ones' opaque parts): the depth and the depths' R; then the hashed
                    // casters' partly transparent parts in front of them: the depths' G and the transparency (cleared
                    // even when the entry has none: the lookups read them)
                    let colour = |view, load: wgpu::LoadOp<wgpu::Color>| Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    });
                    let clear = |value: f64| wgpu::LoadOp::Clear(wgpu::Color { r: value, g: value, b: value, a: value });
                    {
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("shadow map"),
                            color_attachments: &[colour(&views[1], clear(0.0))],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &views[0],
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_bind_group(0, sbg, &[]);
                        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                        self.draw_groups(&mut pass, solid, &self.indirect, *entry as u32);
                    }
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow map, hashed casters"),
                        color_attachments: &[colour(&views[1], wgpu::LoadOp::Load), colour(&views[2], clear(1.0))],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &views[0],
                            depth_ops: None,
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if !hashed.is_empty() {
                        pass.set_bind_group(0, sbg, &[]);
                        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                        self.draw_groups(&mut pass, hashed, &self.indirect, *entry as u32);
                    }
                }
                stamp(&mut enc, &mut marks, 1, &self.timers);
            }
            {
                let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gbuffer"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment { view: &t.gb_base, depth_slice: None, resolve_target: None, ops: clear }),
                        Some(wgpu::RenderPassColorAttachment { view: &t.gb_motion, depth_slice: None, resolve_target: None, ops: clear }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &t.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Discard }),
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_stencil_reference(0);
                pass.set_bind_group(1, &t.gbuffer_bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                self.draw_groups(&mut pass, &gb_groups, &self.indirect, si as u32);
            }
            stamp(&mut enc, &mut marks, 3, &self.timers);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lighting"), timestamp_writes: None });
                pass.set_pipeline(light_pipe);
                pass.set_bind_group(0, bg, &[]);
                // the screen tracing's first rays find the light of a pass of their own sample, its lobes reading the
                // probe (light_bg[1] writes the picture sample 0 reads)
                if si == 0 && self.raytrace.is_some() {
                    pass.set_bind_group(1, &t.light_bg[1], &[]);
                    pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: 1, sample: 0 }));
                    pass.dispatch_workgroups(gw, gh, 1);
                }
                pass.set_bind_group(1, &t.light_bg[si % SAMPLES_PER_FILM_PASS], &[]);
                pass.set_immediates(0, bytemuck::bytes_of(&imm(0)));
                pass.dispatch_workgroups(gw, gh, 1);
            }
            // the world where the camera sees no surface (the lighting left those pixels black)
            if let Some(world) = self.world.as_ref().filter(|w| w.seen) {
                world.background(&mut enc, bg, &t.colors[si % SAMPLES_PER_FILM_PASS], &t.depth, si as u32);
            }
            stamp(&mut enc, &mut marks, 4, &self.timers);
            if !blended.is_empty() {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blended"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &t.colors[si % SAMPLES_PER_FILM_PASS],
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &t.depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &(_, di) in &blended {
                    let d = &self.draws[di];
                    let mi = d.material as usize;
                    let Some(pipe) = &self.pipes.blend[mi] else { continue };
                    let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
                    let first = shape.index_base + d.first_tri * 3;
                    // a pipeline must be set before its immediates
                    if !self.mats.list[mi].show_back {
                        pass.set_pipeline(&self.pipes.depth_blend[self.mats.list[mi].cull as usize]);
                        pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                        pass.draw_indexed(first..first + d.tri_count * 3, shape.vertex_base as i32, di as u32..di as u32 + 1);
                    }
                    pass.set_pipeline(pipe);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    pass.draw_indexed(first..first + d.tri_count * 3, shape.vertex_base as i32, di as u32..di as u32 + 1);
                }
            }
            stamp(&mut enc, &mut marks, 5, &self.timers);
            // EEVEE's motion blur: the sample's picture smeared along its pixels' velocities (into the post's own
            // pictures, which the film then takes)
            let k = si % SAMPLES_PER_FILM_PASS;
            let mut film_bg = &t.accumulate_bg;
            if let (Some(rec), Some(mbp), Some(mt)) = (&mb_rec, &self.mb_post, &self.mb_targets) {
                mbp.encode(&mut enc, mt, k, rec, si as u32, film::halton(si as u32 + 1, 5));
                stamp(&mut enc, &mut marks, 6, &self.timers);
                film_bg = &mt.accumulate;
            }
            // the film takes the pictures four samples at a time
            if k == SAMPLES_PER_FILM_PASS - 1 || si + 1 == samples.len() {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("accumulate"), timestamp_writes: None });
                pass.set_pipeline(&self.accumulate_pipe);
                pass.set_bind_group(0, film_bg, &[]);
                let first_sample = (si - k) as u32;
                let a = AccumulateGpu { width: w, height: h, first_sample, count: k as u32 + 1, first: (first_sample == 0) as u32, pad: [0; 3] };
                pass.set_immediates(0, bytemuck::bytes_of(&a));
                pass.dispatch_workgroups(gw, gh, 1);
            }
            stamp(&mut enc, &mut marks, 7, &self.timers);
        }
        // the depth of field's post process on the film, then the view transform
        if let (Some(rec), Some(post), Some(dt)) = (&dof_rec, &self.dof_post, &self.dof_targets) {
            post.encode(&mut enc, dt, rec);
        }
        stamp(&mut enc, &mut marks, 8, &self.timers);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("resolve"), timestamp_writes: None });
            pass.set_pipeline(&self.resolve_pipe);
            pass.set_bind_group(0, &t.resolve_bg, &[]);
            let r = ResolveGpu {
                width: w,
                height: h,
                lut_n: self.lut.n as u32,
                per_channel: (lit.view.transform == "Standard") as u32,
                lo: self.lut.lo,
                hi: self.lut.hi,
                dither: (lit.view.dither as f32) * self.params.dither,
                bias: self.params.dither_bias,
                zero: [self.lut.zero[0], self.lut.zero[1], self.lut.zero[2], 0.0],
                exposure: self.params.exposure.exp2(),
                frame: job.frame as u32,
                pad2: [0; 2],
            };
            pass.set_immediates(0, bytemuck::bytes_of(&r));
            pass.dispatch_workgroups(gw, gh, 1);
        }
        enc.copy_buffer_to_buffer(&t.pixels, 0, &t.readback[slot], 0, (w * h * 4) as u64);
        if let Some(fr) = &t.film_read {
            enc.copy_buffer_to_buffer(self.film.as_ref().unwrap(), 0, &fr[slot], 0, (w * h * 16) as u64);
        }
        stamp(&mut enc, &mut marks, 9, &self.timers);
        if let Some(q) = &mut self.timers {
            let n = marks.len() as u32;
            let timer = &mut q[slot];
            enc.resolve_query_set(&timer.set, 0..n, &timer.resolve, 0);
            enc.copy_buffer_to_buffer(&timer.resolve, 0, &timer.read, 0, n as u64 * 8);
            timer.marks = marks;
        }
        let t_submit = Instant::now();
        let index = self.gpu.queue.submit(Some(enc.finish()));
        Ok((index, stage_ms, t_submit.elapsed().as_secs_f64() * 1e3))
    }
}

/// The planes (a.p + d >= 0 inside) of a reverse-Z view-projection's frustum, its sides pushed out by `margin` (fractions
/// of the half width and height): left, right, bottom, top, near.
fn frustum_planes(vp: &Mat4, margin: (f32, f32)) -> [glam::Vec4; 5] {
    let r = |i: usize| glam::Vec4::new(vp.x_axis[i], vp.y_axis[i], vp.z_axis[i], vp.w_axis[i]);
    let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
    let (mx, my) = (1.0 + margin.0, 1.0 + margin.1);
    [r0 + r3 * mx, r3 * mx - r0, r1 + r3 * my, r3 * my - r1, r3 - r2]
}

/// The planes (a.p + d >= 0 inside) bounding what a camera's samples see on a W x H picture, its sides pushed out by
/// `margin` (fractions of the half width and height): per side (left, right, bottom, top, near) two planes, a point
/// inside either being inside the side. Without depth of field the frustum's plane twice; with it, the frustums' of
/// the extremes across that side of the `share` of the aperture the samples look through (a point's place on the
/// picture moves linearly with the lens offset, so a point some point of the aperture sees is inside one of them).
fn view_volume(cam: &Cam, w: f32, h: f32, margin: (f32, f32), share: f32) -> [[glam::Vec4; 2]; 5] {
    let planes = |lens: glam::Vec2| frustum_planes(&cam.view_projection(w, h, glam::Vec2::ZERO, lens), margin);
    let f = planes(glam::Vec2::ZERO);
    let e = cam.aperture_extent() * share;
    if e == glam::Vec2::ZERO {
        return f.map(|p| [p, p]);
    }
    let (l, r) = (planes(glam::Vec2::new(-e.x, 0.0)), planes(glam::Vec2::new(e.x, 0.0)));
    let (b, t) = (planes(glam::Vec2::new(0.0, -e.y)), planes(glam::Vec2::new(0.0, e.y)));
    [[l[0], r[0]], [l[1], r[1]], [b[2], t[2]], [b[3], t[3]], [f[4], f[4]]]
}

/// The corners of what a camera sees on a W x H picture from its near plane to `reach` metres away (its far plane at
/// most), a tenth wider each way for the samples' jitter, lens and shutter: where a sun's finer map holds receivers.
fn view_slice(cam: &Cam, w: f32, h: f32, reach: f32) -> [Vec3; 8] {
    let inv = cam.view_projection(w, h, glam::Vec2::ZERO, glam::Vec2::ZERO).inverse();
    let (n, f) = (cam.near, cam.far);
    // the reversed depth at d metres along the view axis
    let depth = |d: f32| {
        let d = d.clamp(n, f);
        if cam.ortho.is_some() { (f - d) / (f - n) } else { n * (f - d) / ((f - n) * d) }
    };
    std::array::from_fn(|i| {
        let x = if i & 1 == 0 { -1.1 } else { 1.1 };
        let y = if i & 2 == 0 { -1.1 } else { 1.1 };
        inv.project_point3(Vec3::new(x, y, depth(if i & 4 == 0 { n } else { reach })))
    })
}

/// Whether a world box reaches inside a view volume (conservative: a box crossing a corner region counts as in).
fn box_in_volume(lo: Vec3, hi: Vec3, volume: &[[glam::Vec4; 2]; 5]) -> bool {
    volume.iter().all(|side| side.iter().any(|p| {
        let n = p.truncate();
        let far = Vec3::new(if n.x >= 0.0 { hi.x } else { lo.x }, if n.y >= 0.0 { hi.y } else { lo.y },
                            if n.z >= 0.0 { hi.z } else { lo.z });
        n.dot(far) + p.w >= 0.0
    }))
}

/// Whether a world box can show in a cube face (origin c, uv axes s, t, depth axis a) at `near` and beyond.
fn box_in_face(lo: Vec3, hi: Vec3, c: Vec3, s: Vec3, t: Vec3, a: Vec3, near: f32) -> bool {
    let mut out = [true; 5];
    for i in 0..8 {
        let p = Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let d = p - c;
        let (x, y, z) = (s.dot(d), t.dot(d), a.dot(d));
        out[0] &= x > z;
        out[1] &= -x > z;
        out[2] &= y > z;
        out[3] &= -y > z;
        out[4] &= z < near;
    }
    !out.iter().any(|&o| o)
}

/// The near plane of a light's shadow cube: casters inside a sphere light's radius cast nothing (EEVEE clips them).
fn shadow_near(radius: f32) -> f32 {
    radius.max(1e-3)
}

/// 1 / R² of a light's influence radius R, where EEVEE's fade ends (0: no fade): its own (`custom`: cutoff_distance,
/// when use_custom_distance is on), else where its `power` (energy max(colour) max(diffuse, specular, transmission
/// factor)) falls under the scene's light `threshold`, measured on probes of every type, size, colour and factor:
/// R² = power / (100 threshold).
fn influence_inv_r2(custom: Option<f32>, power: f32, threshold: f32) -> f32 {
    let r2 = match custom {
        Some(r) => r * r,
        None if threshold > 0.0 => power.abs() / (100.0 * threshold),
        None => 0.0,
    };
    if r2 > 0.0 { 1.0 / r2 } else { 0.0 }
}

pub(crate) fn b(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn shape_label(s: &crate::scene::Shape) -> String {
    format!("shape at vertex {}", s.vertex_base)
}

fn make_blas(device: &wgpu::Device, label: &str, vertices: u32, runs: &[(u32, u32)], opaque: &[bool], rigid: bool)
             -> ShapeBlas {
    let sizes: Vec<_> = runs.iter().zip(opaque).map(|(r, &o)| wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertices,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(r.1 * 3),
        flags: if o { wgpu::AccelerationStructureGeometryFlags::OPAQUE } else { wgpu::AccelerationStructureGeometryFlags::empty() },
    }).collect();
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some(label),
            flags: if rigid { wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE } else { wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD },
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: sizes.clone() },
    );
    ShapeBlas { blas, sizes, runs: runs.to_vec(), built: false }
}

/// The LUT layers as given, padded to two layers (a one-layer view would not be an array).
fn pad_layers(data: &[f32], w: u32, h: u32, layers: u32) -> Vec<f32> {
    let mut out = data.to_vec();
    if layers < 2 {
        out.resize((w * h * 4 * 2) as usize, 0.0);
    }
    out
}

/// The rotation and translation of a matrix, its scale removed.
pub(crate) fn orthonormal(m: &Mat4) -> Mat4 {
    let x = m.x_axis.truncate().normalize_or_zero();
    let y = m.y_axis.truncate().normalize_or_zero();
    let z = m.z_axis.truncate().normalize_or_zero();
    Mat4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), m.w_axis)
}

/// A stable number in [0, 1) from a name (Object Info's Random, standing in for Blender's per-object one).
pub(crate) fn unit_hash(name: &str) -> f32 {
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    (h >> 8) as f32 / 16777216.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lights_custom_distance_replaces_its_threshold_radius() {
        // 1000 W of white light under the default threshold (0.01) reaches sqrt(1000) m
        assert!((influence_inv_r2(None, 1000.0, 0.01) - 1.0 / 1000.0).abs() < 1e-9);
        // its own cutoff distance whatever its power and the threshold
        assert_eq!(influence_inv_r2(Some(90.0), 1000.0, 0.01), 1.0 / 8100.0);
        assert_eq!(influence_inv_r2(Some(90.0), 1.0, 0.0), 1.0 / 8100.0);
        // no threshold and no distance of its own: it never fades
        assert_eq!(influence_inv_r2(None, 1000.0, 0.0), 0.0);
    }
}
