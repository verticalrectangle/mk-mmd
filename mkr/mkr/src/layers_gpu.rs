//! The layers' flat passes on the GPU (layers.rs names the files), drawn on the scene `Renderer::stage` posed on the
//! frame's own time (no motion blur, as Blender draws them: styles.Looks._film turns it off), each sample into a
//! transparent picture of premultiplied RGBA (a layer of its own), the pass's samples spread with the pass's filter
//! into its film and resolved to the 8 bits Blender writes in one compute pass (shaders/layer_film.wgsl), read back
//! into a ring of buffers and encoded by the writers while the GPU draws the next passes:
//!
//!   figure  the selected meshes' triangles, every one opaque (Workbench's object colour ignores the materials), through
//!           Workbench's antialiasing as measured (below); its film alpha in 8 bits, divided back by a whole pixel's 254
//!           (mkmmd.core.shotstyle.ALPHA_FULL) and written as grey RGB
//!   keep    every shown mesh with a material, depth-tested as the lit pass draws them: MMD materials by their own alpha
//!           (hashed ones alpha-tested by their prepass, blended ones blended), the rest opaque; white where a kept
//!           material is seen, black elsewhere; EEVEE's 16 samples and filter; R x A of the 8-bit straight RGBA
//!   screen  the screen type's meshes (buffers of their own) through the output's screen camera, lights off: their
//!           emission and transparency as the lit pass draws them; Standard view (the bake's screen LUT) on the
//!           straight colour, straight RGBA; with negative type on screen, first that alone the same way into the
//!           negative layer, whose file is written before the screen layer's (Writer::push_after)
//!
//! Workbench's antialiasing (render_aa "16"), measured on Blender 4.2's own renders of probe scenes (tiny squares that
//! only one sample point can see, a slanted half-plane, a turned square; 16-bit alpha): its 16 sample points sit at
//! fixed offsets from the pixel centre (WORKBENCH_JITTER, +-0.005 px), and the sample a pixel's neighbour d draws counts
//! for the pixel with the Blackman-Harris weight (2 px wide) of d + 2 x its offset, twice the distance its point is
//! from the pixel; the weighted mean is then bent by WORKBENCH_TRANSFER (a whole pixel 1, half the weight ~0.4: the
//! film loses ~0.13 px of coverage along every edge) and laid between the film's alpha for nothing and for a whole
//! pixel (WORKBENCH_BLACK, WORKBENCH_WHITE: 8 bits make them 0 and 254).

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use rayon::prelude::*;

use crate::camera::{Cam, Fit, affine, render_size};
use crate::film::{self, Sample, TAP_COUNT};
use crate::layers::{Counts, Flat, ScreenPart, live_screen, mmd_alpha};
use crate::lut::ViewLut;
use crate::output::{Pixels, Writer};
use crate::pipelines::{self, COLOR_FORMAT, DEPTH_FORMAT, Immediate};
use crate::render::{DrawGpu, MeshGpu, Renderer, SampleGpu, SceneBuffers, mat, sized, storage, unit_hash};
use crate::scene::{Geometry, ShapeKind, rows};

const LAYER_DRAW_WGSL: &str = include_str!("shaders/layer_draw.wgsl");
const LAYER_FILM_WGSL: &str = include_str!("shaders/layer_film.wgsl");

/// Samples of the keep mask (styles.Looks._film: EEVEE, 16).
const FLAT_SAMPLES: u32 = 16;
/// Workbench's 16 sample points (pixels from the pixel centre, x right, y down), as measured.
const WORKBENCH_JITTER: [[f32; 2]; 16] = [
    [-0.045, -0.485], [0.415, -0.405], [-0.325, -0.345], [0.175, -0.315], [-0.065, -0.235], [0.475, -0.155],
    [-0.275, -0.065], [0.235, -0.065], [0.0, 0.0], [-0.465, 0.095], [-0.215, 0.165], [0.275, 0.185], [0.025, 0.275],
    [0.455, 0.355], [-0.275, 0.405], [0.215, 0.445],
];
/// Workbench's alpha for the weighted mean coverage a = i / 16, i = 0..16 (least squares over the probes' edges).
const WORKBENCH_TRANSFER: [f32; 17] = [
    0.0, 0.0377, 0.0744, 0.1113, 0.1601, 0.2043, 0.2562, 0.3067, 0.3418, 0.4416, 0.4882, 0.5669, 0.6368, 0.7243, 0.8139,
    0.8917, 1.0,
];
/// Workbench's film alpha over nothing and over a whole pixel (16 samples).
const WORKBENCH_BLACK: f32 = 0.00148;
const WORKBENCH_WHITE: f32 = 0.99707;
/// Readback buffers in flight: the passes the GPU may run ahead of the collecting.
const RING: usize = 8;
/// Pictures waiting for the writers before the passes wait for them.
const WRITING: usize = 16;
/// Pictures of a pass's samples the film reads in one go (a pass of more samples runs in batches of them).
const LAYERS: usize = 16;
/// Film flags: the batch starts the pass's film; it ends it (the resolve).
const STARTS: u32 = 1;
const ENDS: u32 = 2;

/// The film pass's record (shaders/layer_film.wgsl: Film).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FilmGpu {
    width: u32,
    height: u32,
    mode: u32,
    count: u32,
    taps: u32,
    flags: u32,
    stride: u32,
    lut_n: u32,
    lo: f32,
    hi: f32,
    black: f32,
    white: f32,
    zero: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Figure = 0,
    Keep = 1,
    Screen = 2,
}

impl Mode {
    /// How the film's bytes make the file: a figure's grey as RGB, a keep mask's as RGBA (alpha 1), screen type RGBA.
    fn pixels(self, stride: usize, data: Vec<u8>) -> Pixels {
        match self {
            Mode::Figure => Pixels::Grey { stride, channels: 3, data },
            Mode::Keep => Pixels::Grey { stride, channels: 4, data },
            Mode::Screen => Pixels::Packed { channels: 4, data },
        }
    }
}

/// How the keep mask draws a material: opaque, by its MMD alpha alpha-tested (hashed) or blended.
#[derive(Clone, Copy, PartialEq, Debug)]
enum KeepKind {
    Opaque,
    Hashed,
    Blended,
}

/// A Blackman-Harris window `width` wide at `x` from its centre.
fn blackman_harris(x: f32, width: f32) -> f32 {
    if x.abs() >= width * 0.5 {
        return 0.0;
    }
    let u = std::f32::consts::TAU * (x / width + 0.5);
    0.35875 - 0.48829 * u.cos() + 0.14128 * (2.0 * u).cos() - 0.01168 * (3.0 * u).cos()
}

/// Workbench's samples: each at its offset j, counting for the pixel d away with the weight of d + 2 j.
fn workbench_samples() -> Vec<Sample> {
    WORKBENCH_JITTER
        .iter()
        .map(|&[x, y]| {
            let j = Vec2::new(x, y);
            let mut taps = [0.0; TAP_COUNT];
            let side = 2 * film::TAPS + 1;
            for dy in -film::TAPS..=film::TAPS {
                for dx in -film::TAPS..=film::TAPS {
                    let d = Vec2::new(dx as f32, dy as f32);
                    taps[((dy + film::TAPS) * side + dx + film::TAPS) as usize] = blackman_harris((d + 2.0 * j).length(), 2.0);
                }
            }
            Sample { jitter: j, time: 0.0, lens: 0, taps }
        })
        .collect()
}

/// The layer passes' "over": premultiplied colour plus the destination times the transmittance (dual source), the
/// alpha likewise.
const OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::Src1, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::Src1Alpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Render pipelines of one shader module with the scene's bind group layout (through the driver's pipeline cache).
struct Maker<'a> {
    device: &'a wgpu::Device,
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    cache: Option<&'a wgpu::PipelineCache>,
}

impl Maker<'_> {
    /// `fs`: (entry, blend, writes colour); `depth`: (writes, compare), None without a depth attachment.
    fn pipe(&self, label: &str, vs: &str, fs: Option<(&str, Option<wgpu::BlendState>)>, colour: bool, cull: bool,
            depth: Option<(bool, wgpu::CompareFunction)>) -> wgpu::RenderPipeline {
        let targets: Vec<Option<wgpu::ColorTargetState>> = if colour {
            let blend = fs.and_then(|f| f.1);
            vec![Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })]
        } else {
            Vec::new()
        };
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState { module: &self.module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull.then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: depth.map(|(write, compare)| wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: fs.map(|f| wgpu::FragmentState {
                module: &self.module,
                entry_point: Some(f.0),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: self.cache,
        })
    }
}

/// The screen type's meshes on the GPU: their own geometry buffers and draws, drawn with the scene's materials. It
/// holds the records a pass needs (`screen_cover`): a project's whole screen type can run to gigabytes (jwin_mv's 180
/// words, 88 million vertices).
struct ScreenKit {
    geo: Geometry,
    /// Index into lit.screen.meshes of each mesh's record.
    recs: Vec<usize>,
    posed: wgpu::Buffer,
    attrs: wgpu::Buffer,
    indices: wgpu::Buffer,
    meshes_buf: wgpu::Buffer,
    draws_buf: wgpu::Buffer,
    draws: Vec<DrawGpu>,
    draw_base: Vec<u32>,
    /// Per shape: the triangles (lit.bin offset) its index range holds now.
    loaded: Vec<Option<usize>>,
    /// Made by layer_targets (a lit frame between may have grown the scene's buffers) and on loading other records.
    bg: Option<wgpu::BindGroup>,
    /// Per material (MaterialSet index): the prepass of a hashed one, the lit pass, the blended pass (alpha kept);
    /// every material of the output's screen type has them.
    pre: BTreeMap<usize, wgpu::RenderPipeline>,
    main: BTreeMap<usize, wgpu::RenderPipeline>,
    blend: BTreeMap<usize, wgpu::RenderPipeline>,
    lut: ViewLut,
    lut_buf: wgpu::Buffer,
    samples: Vec<Sample>,
}

/// Screen type records on the GPU (ScreenKit's geometry part, `screen_geo`).
struct ScreenGeo {
    geo: Geometry,
    recs: Vec<usize>,
    posed: wgpu::Buffer,
    attrs: wgpu::Buffer,
    indices: wgpu::Buffer,
    meshes_buf: wgpu::Buffer,
    draws_buf: wgpu::Buffer,
    draws: Vec<DrawGpu>,
    draw_base: Vec<u32>,
}

/// The layer passes' pictures, depth, film and readbacks at one size.
struct Targets {
    /// A pass's samples' pictures, one layer each (LAYERS at most: more samples run in batches).
    layers: Vec<wgpu::TextureView>,
    depth: wgpu::TextureView,
    /// The film's bytes: a grey byte per pixel (figure, keep; rows `stride` u32 words apart) or RGBA8 (screen).
    pixels: wgpu::Buffer,
    stride: u32,
    readback: Vec<wgpu::Buffer>,
    /// The film reads the layers, the samples' taps and a LUT (the screen type's when there is one, else the view's,
    /// unread), writes the pixels, and keeps its sums in a film buffer between batches.
    film_bg: wgpu::BindGroup,
    /// The bound LUT's size, log2 range and black (the screen type's; unused without it).
    lut: (u32, f32, f32, [f32; 4]),
}

/// A pass waiting for its pixels.
struct Pending {
    slot: usize,
    path: PathBuf,
    mode: Mode,
    /// Its file goes to disk before the next pass's (a frame's negative layer, before its screen layer).
    before: bool,
    index: wgpu::SubmissionIndex,
    bytes: u64,
    mapped: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// Everything the flat passes need beside the Renderer's own state (Renderer::flat_kit makes it once for a run's flats).
pub struct LayerKit {
    figure: wgpu::RenderPipeline,
    /// [black, white][culled]
    keep_flat: [[wgpu::RenderPipeline; 2]; 2],
    /// Per blended MMD material: [black, white].
    keep_blend: BTreeMap<usize, [wgpu::RenderPipeline; 2]>,
    keep_kind: Vec<KeepKind>,
    kept: Vec<bool>,
    screen: Option<ScreenKit>,
    /// Per mode (figure, keep, screen): its samples, and where their taps start in `consts` (in samples).
    sets: [Vec<Sample>; 3],
    tap_base: [u32; 3],
    /// The figure's transfer, then every sample set's taps (shaders/layer_film.wgsl: consts).
    consts: wgpu::Buffer,
    film: wgpu::ComputePipeline,
    film_bgl: wgpu::BindGroupLayout,
    targets: Option<Targets>,
    pending: std::collections::VecDeque<Pending>,
    /// The file and picture of a pass written before the next, waiting for that pass's.
    held: Option<(PathBuf, Pixels)>,
    next_slot: usize,
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

impl Renderer<'_> {
    /// Draw the frames' flat layer passes (figures, keep masks with the materials `kept`, screen type) into their files.
    pub fn draw_flat(&mut self, output: &str, size: [u32; 2], percent: u32, flats: &[Flat], kept: &[String],
                     writer: &mut Writer) -> Result<Counts> {
        match self.flat_kit(output, flats, kept)? {
            Some(mut kit) => self.draw_flats(&mut kit, output, size, percent, flats, writer),
            None => Ok(Counts::default()),
        }
    }

    /// The pipelines, film and screen buffers the flat passes of `flats` of `output` need (None: they need nothing),
    /// made once for any of those flats drawn in any number of `draw_flats` calls, lit frames drawn between them.
    pub fn flat_kit(&self, output: &str, flats: &[Flat], kept: &[String]) -> Result<Option<LayerKit>> {
        self.layer_kit(output, flats, kept)
    }

    /// Draw `flats` (any of the flats `kit` was made for) into their files: every picture handed to the writers when
    /// it returns.
    pub fn draw_flats(&mut self, kit: &mut LayerKit, output: &str, size: [u32; 2], percent: u32, flats: &[Flat],
                      writer: &mut Writer) -> Result<Counts> {
        let [w, h] = render_size(size, percent);
        if w == 0 || h == 0 {
            anyhow::bail!("--percent {percent} leaves no pixels");
        }
        let mut counts = Counts::default();
        if flats.is_empty() {
            return Ok(counts);
        }
        let most = kit.sets.iter().map(Vec::len).max().unwrap_or(1).max(1);
        self.render_bind_group(most);
        self.layer_targets(kit, w, h, most);           // again each call: a lit frame between may have grown the buffers
        for flat in flats {
            let kf = flat.k as f64;
            let seed = (flat.frame as u32).wrapping_mul(0x9e37_79b9);
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("layers") });
            // the bind group holds the shadow rays' TLAS, which a submit may use only once built: no lit frame of this
            // run may have built it (the flat passes trace no rays, so it is built once and left as it is)
            let rays = !self.blas.iter().flatten().any(|b| b.built);
            let staged = self.stage(&mut enc, output, flat.k, [kf, kf, kf], false, rays, &[])?;
            let mut first = Some(enc);
            for (path, cam, meshes) in &flat.figures {
                let cam = self.camera_at(cam, kf)?;
                self.layer_frame(&cam, w, h, &kit.sets[Mode::Figure as usize], seed);
                let mut enc = self.layer_encoder(&mut first);
                self.figure_pass(kit, &mut enc, &staged.shown, meshes, w, h);
                self.finish_pass(kit, enc, path, Mode::Figure, false, w, h, writer)?;
                counts.figure += 1;
            }
            if let Some((path, cam)) = &flat.keep {
                let cam = self.camera_at(cam, kf)?;
                self.layer_frame(&cam, w, h, &kit.sets[Mode::Keep as usize], seed);
                let mut enc = self.layer_encoder(&mut first);
                self.keep_pass(kit, &mut enc, &staged, &cam, w, h);
                self.finish_pass(kit, enc, path, Mode::Keep, false, w, h, writer)?;
                counts.keep += 1;
            }
            if let Some((path, silhouette)) = &flat.screen {
                let cam = screen_camera(self.bake, output)?;
                self.screen_cover(kit, output, flat.frame)?;
                self.layer_frame(&cam, w, h, &kit.sets[Mode::Screen as usize], seed);
                // the negative type alone into its layer, then the rest into the screen layer (nothing in it when only
                // negative type is on screen), written after the negative layer: its presence says that one is there
                let negative = flat.negative.iter().map(|p| (p, ScreenPart::Negative, true));
                let over = (path, ScreenPart::Over { silhouette: *silhouette }, false);
                for (file, part, before) in negative.chain([over]) {
                    self.load_screen(kit, output, flat.frame, flat.k, part)?;
                    let mut enc = self.layer_encoder(&mut first);
                    self.screen_pass(kit, &mut enc, &cam, w, h);
                    self.finish_pass(kit, enc, file, Mode::Screen, before, w, h, writer)?;
                }
                counts.screen += 1;
                counts.negative += usize::from(flat.negative.is_some());
            }
            if let Some(enc) = first {
                self.gpu.queue.submit(Some(enc.finish()));
            }
        }
        while let Some(p) = kit.pending.pop_front() {
            self.collect_layer(kit, p, w, h, writer)?;
        }
        Ok(counts)
    }

    /// The staging encoder for a frame's first pass, a fresh one for the others.
    fn layer_encoder(&self, first: &mut Option<wgpu::CommandEncoder>) -> wgpu::CommandEncoder {
        first.take().unwrap_or_else(|| {
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("layers") })
        })
    }

    /// The frame record and samples of a flat pass through `cam` (lights off, no motion blur).
    fn layer_frame(&self, cam: &Cam, w: u32, h: u32, samples: &[Sample], seed: u32) {
        let frame = self.frame_record(cam, w, h, 0, samples.len() as u32, false, seed);
        let queue = &self.gpu.queue;
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        let view = cam.view();
        let sgpu: Vec<SampleGpu> = samples.iter().enumerate().map(|(si, s)| {
            let n = si as u32 + 1;
            let vp = cam.projection(w as f32, h as f32, s.jitter) * view;
            SampleGpu {
                vp: mat(&vp),
                vp_inv: mat(&vp.as_dmat4().inverse().as_mat4()),
                info: [1, si as u32, 0, 0],
                t: [0.0, s.jitter.x, s.jitter.y, 0.0],
                rand: [film::halton(n, 7), film::halton(n, 11), film::halton(n, 13), film::halton(n, 17)],
                ..bytemuck::Zeroable::zeroed()
            }
        }).collect();
        queue.write_buffer(&self.samples_buf, 0, bytemuck::cast_slice(&sgpu));
    }

    /// The pipelines, film and screen buffers the flats of `output` need (None: they need nothing).
    fn layer_kit(&self, output: &str, flats: &[Flat], kept_names: &[String]) -> Result<Option<LayerKit>> {
        if flats.is_empty() {
            return Ok(None);
        }
        let device = &self.gpu.device;
        let bake = self.bake;
        let lit = bake.lit()?;
        let mats = &self.mats;
        let n = mats.list.len();
        // the keep mask's materials: MMD ones by their alpha
        let keep_wanted = flats.iter().any(|f| f.keep.is_some());
        let mut keep_kind = vec![KeepKind::Opaque; n];
        let mut used = vec![false; n];
        if keep_wanted {
            for (i, m) in mats.list.iter().enumerate() {
                if m.transparent && mmd_alpha(bake, &m.name).unwrap_or(false) {
                    keep_kind[i] = if m.blended { KeepKind::Blended } else { KeepKind::Hashed };
                    used[i] |= m.blended;
                }
            }
        }
        let mut kept = vec![false; n];
        // the screen type: passes for every material of the output's, the records on screen on the first frame that
        // shows some (screen_cover loads each frame's)
        let screen_first = match (&lit.screen, flats.iter().find(|f| f.screen.is_some())) {
            (Some(s), Some(f)) => Some((s, crate::layers::screen_records(s, output, &[f.frame]))),
            _ => None,
        };
        let mut screen_mats = Vec::new();
        if let Some((s, _)) = &screen_first {
            let names: std::collections::BTreeSet<&str> = s.meshes.iter()
                .filter(|m| m.aspect.as_deref().is_none_or(|a| a == output))
                .flat_map(|m| m.materials.iter().flatten().map(String::as_str))
                .collect();
            for name in names {
                let i = mats.index(name)?;
                used[i] = true;
                screen_mats.push(i);
            }
        }
        // the module: the drawing module of those materials, the flat surfaces, their own entry points
        let args = "in: VsOut, @builtin(front_facing) front: bool, @builtin(primitive_index) prim: u32";
        let mut src = pipelines::module_source(mats, &used);
        src.push('\n');
        src.push_str(LAYER_DRAW_WGSL);
        for (i, m) in mats.list.iter().enumerate() {
            let id = &m.id;
            if keep_kind[i] == KeepKind::Blended {
                for v in [0, 1] {
                    src.push_str(&format!(
                        "\n@fragment\nfn flk{v}_{id}({args}) -> Blended {{\n    let s = fragment_surface(in, front, prim);\n    let a = clamp(mat_{id}_alpha(s), 0.0, 1.0);\n    return Blended(vec4<f32>(vec3<f32>({v}.0 * a), a), vec4<f32>(1.0 - a));\n}}\n"
                    ));
                }
            }
            if screen_mats.contains(&i) && m.blended {
                src.push_str(&format!(
                    "\n@fragment\nfn fls_{id}({args}) -> Blended {{\n    let s = fragment_surface(in, front, prim);\n    let c = mat_{id}(s);\n    let t = clamp(c.transmittance, vec3<f32>(0.0), vec3<f32>(1.0));\n    let o = 1.0 - mks_avg3(t);\n    return Blended(vec4<f32>(light_closures(c, s), o), vec4<f32>(t, 1.0 - o));\n}}\n"
                ));
            }
        }
        if let Ok(path) = std::env::var("MKR_DUMP_LAYER_WGSL") {
            std::fs::write(&path, &src).with_context(|| format!("cannot write {path}"))?;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layers"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            anyhow::bail!("the layer shaders do not compile: {e}");
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layers"),
            bind_group_layouts: &[Some(&self.render_bgl)],
            immediate_size: std::mem::size_of::<Immediate>() as u32,
        });
        let mk = Maker { device, module, layout, cache: self.gpu.cache.as_ref() };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let gt = wgpu::CompareFunction::Greater;
        let eq = wgpu::CompareFunction::Equal;
        let figure = mk.pipe("figure", "vs_depth", Some(("fl_cover", None)), true, false, None);
        let keep_flat = ["fl_black", "fl_white"]
            .map(|fs| [false, true].map(|cull| mk.pipe(fs, "vs_depth", Some((fs, None)), true, cull, Some((false, eq)))));
        let mut keep_blend = BTreeMap::new();
        for (i, m) in mats.list.iter().enumerate() {
            if keep_kind[i] == KeepKind::Blended {
                let cmp = if m.show_back { gt } else { eq };
                keep_blend.insert(i, [0, 1].map(|v| {
                    let fs = format!("flk{v}_{}", m.id);
                    mk.pipe(&fs, "vs_main", Some((&fs, Some(OVER))), true, m.cull, Some((false, cmp)))
                }));
            }
        }
        let screen = match screen_first {
            Some((s, recs)) => {
                let mut pre = BTreeMap::new();
                let mut main = BTreeMap::new();
                let mut blend = BTreeMap::new();
                for &i in &screen_mats {
                    let m = &mats.list[i];
                    if m.blended {
                        let cmp = if m.show_back { gt } else { eq };
                        let fs = format!("fls_{}", m.id);
                        blend.insert(i, mk.pipe(&fs, "vs_main", Some((&fs, Some(OVER))), true, m.cull, Some((false, cmp))));
                        continue;
                    }
                    if m.transparent {
                        let fa = format!("fa_{}", m.id);
                        pre.insert(i, mk.pipe(&fa, "vs_main", Some((&fa, None)), false, m.cull, Some((true, gt))));
                    }
                    let fs = format!("fs_{}", m.id);
                    main.insert(i, mk.pipe(&fs, "vs_main", Some((&fs, None)), true, m.cull, Some((false, eq))));
                }
                Some(self.screen_kit(s, &recs, pre, main, blend)?)
            }
            None => None,
        };
        if let Some(e) = pollster::block_on(scope.pop()) {
            anyhow::bail!("the layer pipelines cannot be made: {e}");
        }
        // the film: every sample set's taps after the figure's transfer
        let film_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer film"),
            source: wgpu::ShaderSource::Wgsl(LAYER_FILM_WGSL.into()),
        });
        let film_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer film"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let cache = self.gpu.cache.as_ref();
        let film = compute(device, &film_module, &film_bgl, "film_pass", std::mem::size_of::<FilmGpu>() as u32, cache);
        if let Err(e) = self.gpu.save_cache() {
            eprintln!("mkr: the pipeline cache is not kept: {e:#}");
        }
        let sets = [
            workbench_samples(),
            film::samples(FLAT_SAMPLES, &self.filter, self.params.jitter, None),
            screen.as_ref().map(|s| s.samples.clone()).unwrap_or_default(),
        ];
        let tap_base = [0, sets[0].len() as u32, (sets[0].len() + sets[1].len()) as u32];
        let mut consts = vec![0.0f32; 20];
        consts[..WORKBENCH_TRANSFER.len()].copy_from_slice(&WORKBENCH_TRANSFER);
        for s in sets.iter().flatten() {
            let mut taps = [0.0f32; 28];
            taps[..TAP_COUNT].copy_from_slice(&s.taps);
            consts.extend_from_slice(&taps);
        }
        let consts = storage(device, "layer film consts", bytemuck::cast_slice(&consts), wgpu::BufferUsages::empty());
        // which materials are kept: the request's
        for name in kept_names {
            kept[mats.index(name)?] = true;
        }
        Ok(Some(LayerKit {
            figure,
            keep_flat,
            keep_blend,
            keep_kind,
            kept,
            screen,
            sets,
            tap_base,
            consts,
            film,
            film_bgl,
            targets: None,
            pending: Default::default(),
            held: None,
            next_slot: 0,
        }))
    }

    /// The screen kit: the passes of its materials, its view LUT and samples, the records `recs` of `rec.meshes`.
    fn screen_kit(&self, rec: &crate::bake::ScreenRec, recs: &[usize], pre: BTreeMap<usize, wgpu::RenderPipeline>,
                  main: BTreeMap<usize, wgpu::RenderPipeline>, blend: BTreeMap<usize, wgpu::RenderPipeline>)
                  -> Result<ScreenKit> {
        let g = self.screen_geo(rec, recs)?;
        let lut = ViewLut::load(&self.bake.dir.join(&rec.lut.file), rec.lut.n, rec.lut.lo as f32, rec.lut.hi as f32,
                                rec.lut.zero.or(rec.zero))?;
        let lut_buf = storage(&self.gpu.device, "screen lut", bytemuck::cast_slice(&lut.data), wgpu::BufferUsages::empty());
        let samples = film::samples(rec.pass.samples.default.max(1), &self.filter, self.params.jitter, None);
        Ok(ScreenKit {
            loaded: vec![None; g.geo.shapes.len()],
            geo: g.geo,
            recs: g.recs,
            posed: g.posed,
            attrs: g.attrs,
            indices: g.indices,
            meshes_buf: g.meshes_buf,
            draws_buf: g.draws_buf,
            draws: g.draws,
            draw_base: g.draw_base,
            bg: None,
            pre,
            main,
            blend,
            lut,
            lut_buf,
            samples,
        })
    }

    /// The records `recs` (indices into `rec.meshes`) of the screen type on the GPU: their geometry (sized for the
    /// largest of each one's frames), draws and mesh records.
    fn screen_geo(&self, rec: &crate::bake::ScreenRec, recs: &[usize]) -> Result<ScreenGeo> {
        let device = &self.gpu.device;
        let mats = &self.mats;
        let chosen: Vec<crate::bake::MeshRec> = recs.iter().map(|&i| rec.meshes[i].clone()).collect();
        let geo = Geometry::from_records(self.bake, &chosen, &|m: &str| mats.attributes(m))?;
        let mat_of: Vec<usize> = geo.materials.iter().map(|n| mats.index(n)).collect::<Result<_>>()?;
        let mut draws = Vec::new();
        let mut draw_base = Vec::new();
        for (mi, m) in geo.meshes.iter().enumerate() {
            draw_base.push(draws.len() as u32);
            let shape = &geo.shapes[m.shape];
            for s in &m.slots {
                let mat = s.material.map(|i| mat_of[i]).unwrap_or(mats.default_index());
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
                    first_tri: 0,
                    tri_count: 0,
                    remap,
                    param_base: ml.param_base,
                    image_base: ml.image_base,
                    pad0: 0,
                    pad1: 0,
                });
            }
        }
        let of_shape: Vec<usize> = geo.shapes.iter().map(|s| match s.kind {
            ShapeKind::ByFrame { rec } => Ok(recs[rec]),
            _ => anyhow::bail!("screen type that is not frame by frame"),
        }).collect::<Result<_>>()?;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Ok(ScreenGeo {
            recs: geo.meshes.iter().map(|m| of_shape[m.shape]).collect(),
            posed: storage(device, "screen posed", bytemuck::cast_slice(&geo.rest), wgpu::BufferUsages::empty()),
            attrs: storage(device, "screen attrs", bytemuck::cast_slice(&geo.attrs), wgpu::BufferUsages::empty()),
            indices: storage(device, "screen indices", bytemuck::cast_slice(&geo.indices), wgpu::BufferUsages::INDEX),
            meshes_buf: sized(device, "screen meshes", (geo.meshes.len() * std::mem::size_of::<MeshGpu>()) as u64, rw),
            draws_buf: storage(device, "screen draws", bytemuck::cast_slice(&draws), wgpu::BufferUsages::empty()),
            draws,
            draw_base,
            geo,
        })
    }

    /// Make `kit`'s screen type hold every record on screen at `frame` in `output`: when one is missing, the kit holds
    /// those records alone from then on.
    fn screen_cover(&self, kit: &mut LayerKit, output: &str, frame: i64) -> Result<()> {
        let rec = self.bake.lit()?.screen.as_ref().context("the bake holds no screen type")?;
        let want = crate::layers::screen_records(rec, output, &[frame]);
        let s = kit.screen.as_mut().context("no screen kit")?;
        if want.iter().all(|i| s.recs.contains(i)) {
            return Ok(());
        }
        let g = self.screen_geo(rec, &want)?;
        s.loaded = vec![None; g.geo.shapes.len()];
        (s.geo, s.recs, s.posed, s.attrs, s.indices) = (g.geo, g.recs, g.posed, g.attrs, g.indices);
        (s.meshes_buf, s.draws_buf, s.draws, s.draw_base) = (g.meshes_buf, g.draws_buf, g.draws, g.draw_base);
        s.bg = Some(self.scene_bind_group("screen", SceneBuffers {
            posed: [&s.posed, &s.posed, &s.posed],
            attrs: &s.attrs,
            indices: &s.indices,
            meshes: &s.meshes_buf,
            draws: &s.draws_buf,
        }));
        Ok(())
    }

    /// The layer passes' pictures (a layer per sample, for passes of up to `most` samples), depth, film and readbacks
    /// at w x h, and the screen kit's bind group.
    fn layer_targets(&self, kit: &mut LayerKit, w: u32, h: u32, most: usize) {
        let device = &self.gpu.device;
        if let Some(s) = &mut kit.screen {
            s.bg = Some(self.scene_bind_group("screen", SceneBuffers {
                posed: [&s.posed, &s.posed, &s.posed],
                attrs: &s.attrs,
                indices: &s.indices,
                meshes: &s.meshes_buf,
                draws: &s.draws_buf,
            }));
        }
        let tex = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, layers: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let count = most.clamp(1, LAYERS) as u32;
        let pictures = tex("layer samples", COLOR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                           count);
        let layers = (0..count)
            .map(|i| pictures.create_view(&wgpu::TextureViewDescriptor {
                label: Some("layer sample"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i,
                array_layer_count: Some(1),
                ..Default::default()
            }))
            .collect();
        let all = pictures.create_view(&wgpu::TextureViewDescriptor {
            label: Some("layer samples"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let depth = tex("layer depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT, 1).create_view(&Default::default());
        let n = (w * h) as u64;
        // the sums between batches only when a pass has more samples than the layers
        let film = sized(device, "layer film", if most > LAYERS { n * 32 } else { 16 }, wgpu::BufferUsages::STORAGE);
        let stride = w.div_ceil(4);
        let bytes = n * 4;
        let pixels = sized(device, "layer pixels", bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = (0..RING)
            .map(|_| sized(device, "layer readback", bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST))
            .collect();
        let lut_buf = kit.screen.as_ref().map(|s| &s.lut_buf).unwrap_or(&self.lut_buf);
        let film_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer film"),
            layout: &kit.film_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&all) },
                wgpu::BindGroupEntry { binding: 1, resource: kit.consts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lut_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pixels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: film.as_entire_binding() },
            ],
        });
        let lut = match &kit.screen {
            Some(s) => (s.lut.n as u32, s.lut.lo, s.lut.hi, [s.lut.zero[0], s.lut.zero[1], s.lut.zero[2], 0.0]),
            None => (self.lut.n as u32, self.lut.lo, self.lut.hi, [0.0; 4]),
        };
        kit.targets = Some(Targets { layers, depth, pixels, stride, readback, film_bg, lut });
    }

    /// A figure: every triangle of `meshes` shown on the frame, opaque, no depth.
    fn figure_pass(&self, kit: &LayerKit, enc: &mut wgpu::CommandEncoder, shown: &[bool], meshes: &[usize], w: u32, h: u32) {
        let t = kit.targets.as_ref().unwrap();
        let bg = self.render_bg.as_ref().unwrap();
        let mut draws = Vec::new();
        for &mi in meshes {
            let m = &self.geo.meshes[mi];
            if !shown.get(mi).copied().unwrap_or(false) || !m.visible.camera {
                continue;
            }
            let base = self.draw_base[mi] as usize;
            for di in base..base + m.slots.len() {
                if self.draws[di].tri_count > 0 {
                    draws.push(di);
                }
            }
        }
        let n = kit.sets[Mode::Figure as usize].len();
        for si in 0..n {
            {
                let mut pass = begin_colour(enc, &t.layers[si % LAYERS], None);
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_pipeline(&kit.figure);
                for &di in &draws {
                    let d = &self.draws[di];
                    let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
                    pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: di as u32, sample: si as u32 }));
                    let first = shape.index_base + d.first_tri * 3;
                    let inst = di as u32;
                    pass.draw_indexed(first..first + d.tri_count * 3, shape.vertex_base as i32, inst..inst + 1);
                }
            }
            film_batch(kit, enc, Mode::Figure, si, n, w, h);
        }
    }

    /// The keep mask: the lit pass's visibility (opaque, hashed, blended back to front), flat white or black.
    fn keep_pass(&self, kit: &LayerKit, enc: &mut wgpu::CommandEncoder, staged: &crate::render::Staged, cam: &Cam,
                 w: u32, h: u32) {
        let t = kit.targets.as_ref().unwrap();
        let bg = self.render_bg.as_ref().unwrap();
        let view = cam.view();
        let mut solid = Vec::new();
        let mut blended = Vec::new();
        for (mi, m) in self.geo.meshes.iter().enumerate() {
            // an object without any material is left out (Blender would draw it lit grey)
            if !staged.shown[mi] || !m.visible.camera || m.slots.iter().all(|s| s.material.is_none()) {
                continue;
            }
            let base = self.draw_base[mi] as usize;
            for (si, s) in m.slots.iter().enumerate() {
                let di = base + si;
                let d = &self.draws[di];
                if d.tri_count == 0 {
                    continue;
                }
                let mat = d.material as usize;
                let kept = s.material.is_some() && kit.kept[mat];
                if kit.keep_kind[mat] == KeepKind::Blended {
                    let shape = &self.geo.shapes[m.shape];
                    let c = staged.cur.matrices[mi] * ((shape.lo + shape.hi) * 0.5).extend(1.0);
                    blended.push(((view * c).z, di, kept));
                } else {
                    solid.push((di, kept));
                }
            }
        }
        blended.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let n = kit.sets[Mode::Keep as usize].len();
        for si in 0..n {
            let imm = |di: usize| Immediate { draw: di as u32, sample: si as u32 };
            {
                let mut pass = begin_depth(enc, &t.depth);
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &(di, _) in &solid {
                    let d = &self.draws[di];
                    let mat = d.material as usize;
                    let pipe = match (kit.keep_kind[mat], &self.pipes.prepass[mat]) {
                        (KeepKind::Hashed, Some(p)) => p,
                        _ => &self.pipes.depth[self.mats.list[mat].cull as usize],
                    };
                    pass.set_pipeline(pipe);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    self.draw_main(&mut pass, di);
                }
            }
            {
                let mut pass = begin_colour(enc, &t.layers[si % LAYERS], Some(&t.depth));
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &(di, kept) in &solid {
                    let mat = self.draws[di].material as usize;
                    pass.set_pipeline(&kit.keep_flat[kept as usize][self.mats.list[mat].cull as usize]);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    self.draw_main(&mut pass, di);
                }
                for &(_, di, kept) in &blended {
                    let mat = self.draws[di].material as usize;
                    let m = &self.mats.list[mat];
                    if !m.show_back {
                        pass.set_pipeline(&self.pipes.depth_blend[m.cull as usize]);
                        pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                        self.draw_main(&mut pass, di);
                    }
                    pass.set_pipeline(&kit.keep_blend[&mat][kept as usize]);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    self.draw_main(&mut pass, di);
                }
            }
            film_batch(kit, enc, Mode::Keep, si, n, w, h);
        }
    }

    /// Draw `di` of the scene's draws (the draw index rides as the instance: the vertex shaders read it there).
    fn draw_main(&self, pass: &mut wgpu::RenderPass<'_>, di: usize) {
        let d = &self.draws[di];
        let shape = &self.geo.shapes[self.geo.meshes[d.mesh as usize].shape];
        let first = shape.index_base + d.first_tri * 3;
        pass.draw_indexed(first..first + d.tri_count * 3, shape.vertex_base as i32, di as u32..di as u32 + 1);
    }

    /// Upload the `part` of the screen type on screen at `frame` (bake frame `k`): its geometry there (the words
    /// worked out in parallel), its mesh records, its draws' runs (none for the type not on screen).
    fn load_screen(&self, kit: &mut LayerKit, output: &str, frame: i64, k: usize, part: ScreenPart) -> Result<()> {
        let bake = self.bake;
        let rec = bake.lit()?.screen.as_ref().context("the bake holds no screen type")?;
        let live: Vec<&str> = live_screen(rec, output, frame, part).map(|m| m.name.as_str()).collect();
        let s = kit.screen.as_mut().context("no screen kit")?;
        let (geo, recs) = (&s.geo, &s.recs);
        let geoms: Vec<Option<crate::scene::FrameGeom>> = geo.meshes.par_iter().enumerate()
            .map(|(mi, m)| {
                let r = &rec.meshes[recs[mi]];
                if live.contains(&r.name.as_str()) && m.visible.camera {
                    crate::screen_geo::frame_geom(bake, r, &geo.shapes[m.shape], k, k as f64, true)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<_>>()?;
        let queue = &self.gpu.queue;
        let mut meshes = Vec::with_capacity(s.geo.meshes.len());
        for ((mi, m), geom) in s.geo.meshes.iter().enumerate().zip(&geoms) {
            let shape = &s.geo.shapes[m.shape];
            let base = s.draw_base[mi] as usize;
            match geom {
                Some(g) => {
                    queue.write_buffer(&s.posed, shape.vertex_base as u64 * 32, bytemuck::cast_slice(&g.posed));
                    if !g.attrs.is_empty() {
                        queue.write_buffer(&s.attrs, shape.attr_base as u64 * 16, bytemuck::cast_slice(&g.attrs));
                    }
                    if s.loaded[m.shape] != Some(g.triangles_offset) {
                        queue.write_buffer(&s.indices, shape.index_base as u64 * 4, bytemuck::cast_slice(&g.indices));
                        s.loaded[m.shape] = Some(g.triangles_offset);
                    }
                    for (di, slot) in m.slots.iter().enumerate() {
                        let run = g.slots.iter().find(|x| x[0] == slot.slot).map(|x| (x[1] as u32, x[2] as u32)).unwrap_or((0, 0));
                        s.draws[base + di].first_tri = run.0;
                        s.draws[base + di].tri_count = run.1;
                    }
                }
                None => {
                    for di in 0..m.slots.len() {
                        s.draws[base + di].tri_count = 0;
                    }
                }
            }
            let id = Mat4::IDENTITY;
            meshes.push(MeshGpu {
                object: mat(&id),
                object_inv: mat(&id),
                m_prev: rows(&id),
                m_cur: rows(&id),
                m_next: rows(&id),
                n_cur: rows(&id),
                object_color: m.color,
                object_info: [m.pass_index, unit_hash(&m.name), 0.0, 0.0],
                vertex_base: shape.vertex_base,
                attr_base: shape.attr_base,
                attr_stride: shape.attr_stride,
                index_base: shape.index_base,
                draw_base: s.draw_base[mi],
                generated: shape.generated.unwrap_or(u32::MAX),
                flags: 0,
                pad: 0,
            });
        }
        if !meshes.is_empty() {
            queue.write_buffer(&s.meshes_buf, 0, bytemuck::cast_slice(&meshes));
            queue.write_buffer(&s.draws_buf, 0, bytemuck::cast_slice(&s.draws));
        }
        Ok(())
    }

    /// The screen type through the screen camera, as the lit pass draws its materials (no light: their emission).
    fn screen_pass(&self, kit: &LayerKit, enc: &mut wgpu::CommandEncoder, cam: &Cam, w: u32, h: u32) {
        let t = kit.targets.as_ref().unwrap();
        let s = kit.screen.as_ref().unwrap();
        let bg = s.bg.as_ref().unwrap();
        let view = cam.view();
        let mut solid = Vec::new();
        let mut blended = Vec::new();
        for (di, d) in s.draws.iter().enumerate() {
            if d.tri_count == 0 {
                continue;
            }
            let mat = d.material as usize;
            if s.blend.contains_key(&mat) {
                let shape = &s.geo.shapes[s.geo.meshes[d.mesh as usize].shape];
                blended.push(((view * ((shape.lo + shape.hi) * 0.5).extend(1.0)).z, di));
            } else {
                solid.push(di);
            }
        }
        blended.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let draw = |pass: &mut wgpu::RenderPass<'_>, di: usize| {
            let d = &s.draws[di];
            let shape = &s.geo.shapes[s.geo.meshes[d.mesh as usize].shape];
            let first = shape.index_base + d.first_tri * 3;
            pass.draw_indexed(first..first + d.tri_count * 3, shape.vertex_base as i32, di as u32..di as u32 + 1);
        };
        let n = kit.sets[Mode::Screen as usize].len();
        for si in 0..n {
            let imm = |di: usize| Immediate { draw: di as u32, sample: si as u32 };
            {
                let mut pass = begin_depth(enc, &t.depth);
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(s.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &di in &solid {
                    let mat = s.draws[di].material as usize;
                    let pipe = s.pre.get(&mat).unwrap_or(&self.pipes.depth[self.mats.list[mat].cull as usize]);
                    pass.set_pipeline(pipe);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    draw(&mut pass, di);
                }
            }
            {
                let mut pass = begin_colour(enc, &t.layers[si % LAYERS], Some(&t.depth));
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(s.indices.slice(..), wgpu::IndexFormat::Uint32);
                for &di in &solid {
                    let Some(pipe) = s.main.get(&(s.draws[di].material as usize)) else { continue };
                    pass.set_pipeline(pipe);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    draw(&mut pass, di);
                }
                for &(_, di) in &blended {
                    let mat = s.draws[di].material as usize;
                    let m = &self.mats.list[mat];
                    if !m.show_back {
                        pass.set_pipeline(&self.pipes.depth_blend[m.cull as usize]);
                        pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                        draw(&mut pass, di);
                    }
                    pass.set_pipeline(&s.blend[&mat]);
                    pass.set_immediates(0, bytemuck::bytes_of(&imm(di)));
                    draw(&mut pass, di);
                }
            }
            film_batch(kit, enc, Mode::Screen, si, n, w, h);
        }
    }

    /// Read the pass's film back (a ring of buffers) and submit it; the oldest pass is collected first when the ring is
    /// full. `before`: its file goes to disk before the next pass's (a frame's negative layer, then its screen layer).
    #[allow(clippy::too_many_arguments)]
    fn finish_pass(&mut self, kit: &mut LayerKit, mut enc: wgpu::CommandEncoder, path: &std::path::Path, mode: Mode,
                   before: bool, w: u32, h: u32, writer: &mut Writer) -> Result<()> {
        while kit.pending.len() >= RING {
            let p = kit.pending.pop_front().context("checked")?;
            self.collect_layer(kit, p, w, h, writer)?;
        }
        let slot = kit.next_slot % RING;
        kit.next_slot += 1;
        let t = kit.targets.as_ref().context("no layer targets")?;
        let bytes = match mode {
            Mode::Screen => (w * h * 4) as u64,
            _ => (t.stride * 4 * h) as u64,
        };
        enc.copy_buffer_to_buffer(&t.pixels, 0, &t.readback[slot], 0, bytes);
        let index = self.gpu.queue.submit(Some(enc.finish()));
        let (tx, mapped) = std::sync::mpsc::channel();
        t.readback[slot].slice(..bytes).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        kit.pending.push_back(Pending { slot, path: path.to_path_buf(), mode, before, index, bytes, mapped });
        Ok(())
    }

    /// Wait for a pass and hand its bytes to the writers (which lay them out as the file's channels); a pass written
    /// `before` the next waits in the kit for that one, the two then written in order (Writer::push_after).
    fn collect_layer(&self, kit: &mut LayerKit, p: Pending, w: u32, h: u32, writer: &mut Writer) -> Result<()> {
        let t = kit.targets.as_ref().context("no layer targets")?;
        let buf = &t.readback[p.slot];
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(p.index), timeout: None })
            .map_err(|e| anyhow::anyhow!("GPU wait: {e}"))?;
        p.mapped.recv()?.map_err(|e| anyhow::anyhow!("readback: {e}"))?;
        let data = buf.slice(..p.bytes).get_mapped_range().map_err(|e| anyhow::anyhow!("readback: {e:?}"))?.to_vec();
        buf.unmap();
        let pixels = p.mode.pixels(t.stride as usize * 4, data);
        if p.before {
            kit.held = Some((p.path, pixels));
            return Ok(());
        }
        writer.push_after(kit.held.take(), p.path, w, h, pixels, WRITING)
    }

    /// A silhouette frame's knock pass (styles.Looks: its knock-out screen type, reversed out of the figure in the
    /// picture): that type alone, on screen at `frame` (bake frame `k`) of `output`, drawn as the screen layer draws the
    /// rest (`kit`: `flat_kit` of a frame of `output` with screen type), as straight RGBA bytes (h, w, 4), top row first.
    pub(crate) fn screen_knock(&mut self, kit: &mut LayerKit, output: &str, w: u32, h: u32, frame: i64, k: usize)
                               -> Result<Vec<u8>> {
        let most = kit.sets.iter().map(Vec::len).max().unwrap_or(1).max(1);
        self.render_bind_group(most);
        self.screen_cover(kit, output, frame)?;
        self.layer_targets(kit, w, h, most);            // each time: a pass between may have grown the shared buffers
        let cam = screen_camera(self.bake, output)?;
        self.load_screen(kit, output, frame, k, ScreenPart::Knock)?;
        self.layer_frame(&cam, w, h, &kit.sets[Mode::Screen as usize], (frame as u32).wrapping_mul(0x9e37_79b9));
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("screen knock") });
        self.screen_pass(kit, &mut enc, &cam, w, h);
        let t = kit.targets.as_ref().context("no layer targets")?;
        let (buf, bytes) = (&t.readback[0], (w * h * 4) as u64);
        enc.copy_buffer_to_buffer(&t.pixels, 0, buf, 0, bytes);
        let index = self.gpu.queue.submit(Some(enc.finish()));
        let (tx, mapped) = std::sync::mpsc::channel();
        buf.slice(..bytes).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .map_err(|e| anyhow::anyhow!("GPU wait: {e}"))?;
        mapped.recv()?.map_err(|e| anyhow::anyhow!("readback: {e}"))?;
        let data = buf.slice(..bytes).get_mapped_range().map_err(|e| anyhow::anyhow!("readback: {e:?}"))?.to_vec();
        buf.unmap();
        Ok(data)
    }
}

/// The screen camera of `output`: orthographic, one frame height tall, over the plane z = 0.
fn screen_camera(bake: &crate::bake::Bake, output: &str) -> Result<Cam> {
    let rec = bake.lit()?.screen.as_ref().context("the bake holds no screen type")?;
    let c = rec.cameras.get(output).with_context(|| format!("the screen type has no camera for output {output}"))?;
    Ok(Cam {
        world: affine(&c.matrix),
        lens: 50.0,
        shift: Vec2::ZERO,
        near: c.clip[0],
        far: c.clip[1],
        sensor: 36.0,
        sensor_height: 36.0,
        fit: match c.sensor_fit.as_str() {
            "HORIZONTAL" => Fit::Horizontal,
            "VERTICAL" => Fit::Vertical,
            _ => Fit::Auto,
        },
        ortho: Some(c.ortho_scale),
        dof: None,
    })
}

/// A depth-only pass, cleared (reversed depth: 0 is far).
fn begin_depth<'e>(enc: &'e mut wgpu::CommandEncoder, depth: &wgpu::TextureView) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("layer prepass"),
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

/// The sample's picture, cleared transparent, over the prepass's depth when there is one.
fn begin_colour<'e>(enc: &'e mut wgpu::CommandEncoder, color: &wgpu::TextureView, depth: Option<&wgpu::TextureView>)
                    -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("layer colour"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: depth.map(|d| wgpu::RenderPassDepthStencilAttachment {
            view: d,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// The samples of a pass of `n` its film takes in once sample `si` is drawn: (first, count, flags) when `si` fills
/// the layers or ends the pass, None while the batch goes on.
fn batch(si: usize, n: usize) -> Option<(usize, usize, u32)> {
    let done = si + 1;
    if done % LAYERS != 0 && done != n {
        return None;
    }
    let first = (done - 1) / LAYERS * LAYERS;
    let flags = if first == 0 { STARTS } else { 0 } | if done == n { ENDS } else { 0 };
    Some((first, done - first, flags))
}

/// Once sample `si` of a pass of `n` (`mode`'s samples) is drawn: its batch of pictures spread into the film when it
/// is full or the pass's last, the last also resolved into the pixels.
fn film_batch(kit: &LayerKit, enc: &mut wgpu::CommandEncoder, mode: Mode, si: usize, n: usize, w: u32, h: u32) {
    let Some((first, count, flags)) = batch(si, n) else { return };
    let t = kit.targets.as_ref().unwrap();
    let (lut_n, lo, hi, zero) = t.lut;
    let f = FilmGpu {
        width: w,
        height: h,
        mode: mode as u32,
        count: count as u32,
        taps: kit.tap_base[mode as usize] + first as u32,
        flags,
        stride: t.stride,
        lut_n,
        lo,
        hi,
        black: WORKBENCH_BLACK,
        white: WORKBENCH_WHITE,
        zero,
    };
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("layer film"), timestamp_writes: None });
    pass.set_pipeline(&kit.film);
    pass.set_bind_group(0, &t.film_bg, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&f));
    pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_counts_where_twice_its_offset_points() {
        let s = workbench_samples();
        assert_eq!(s.len(), 16);
        let side = (2 * film::TAPS + 1) as usize;
        let tap = |s: &Sample, dx: i32, dy: i32| s.taps[(dy + film::TAPS) as usize * side + (dx + film::TAPS) as usize];
        // the centre sample counts for its own pixel only (the window is 0 a pixel away)
        let c = s.iter().find(|x| x.jitter == Vec2::ZERO).unwrap();
        assert!((tap(c, 0, 0) - 1.0).abs() < 1e-6 && tap(c, 1, 0) == 0.0 && tap(c, 0, 1) == 0.0);
        // the sample near the top edge of its pixel, (-0.045, -0.485): the pixel above takes the one drawn below it
        // (d = (0, 1)) at nearly full weight, its own pixel almost nothing
        let t = s.iter().find(|x| (x.jitter - Vec2::new(-0.045, -0.485)).length() < 1e-6).unwrap();
        assert!(tap(t, 0, 1) > 0.9 && tap(t, 0, 0) < 0.01, "{:?}", t.taps);
        // no weight reaches two pixels away
        for x in &s {
            for d in -film::TAPS..=film::TAPS {
                assert_eq!(tap(x, 2, d) + tap(x, -2, d) + tap(x, d, 2) + tap(x, d, -2), 0.0);
            }
        }
    }

    #[test]
    fn the_transfer_rises_from_none_to_a_whole_pixel() {
        assert_eq!(WORKBENCH_TRANSFER[0], 0.0);
        assert_eq!(WORKBENCH_TRANSFER[16], 1.0);
        assert!(WORKBENCH_TRANSFER.windows(2).all(|w| w[1] > w[0]));
        // 8 bits: nothing is 0 and a whole pixel 254, which mk divides back to 255 (shotstyle.ALPHA_FULL)
        let a8 = |v: f32| (v * 255.0 + 0.5).floor();
        assert_eq!(a8(WORKBENCH_BLACK), 0.0);
        assert_eq!(a8(WORKBENCH_WHITE), 254.0);
    }

    #[test]
    fn the_film_takes_every_sample_once_in_order() {
        for n in [1, 15, 16, 17, 32, 33, 64] {
            let batches: Vec<_> = (0..n).filter_map(|si| batch(si, n)).collect();
            // the batches cover 0..n in order, none wider than the layers
            let mut next = 0;
            for &(first, count, _) in &batches {
                assert_eq!(first, next, "n {n}: {batches:?}");
                assert!(count >= 1 && count <= LAYERS);
                next += count;
            }
            assert_eq!(next, n);
            // the first starts the film, the last resolves it, the others carry it
            assert!(batches.iter().enumerate().all(|(i, b)| (b.2 & STARTS != 0) == (i == 0)));
            assert!(batches.iter().enumerate().all(|(i, b)| (b.2 & ENDS != 0) == (i + 1 == batches.len())));
        }
        assert_eq!(batch(15, 16), Some((0, 16, STARTS | ENDS)));
        assert_eq!(batch(3, 16), None);
    }
}
