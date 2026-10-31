//! The drawing modules and their pipelines. Every material is compiled into a small module of its own (the drawing core
//! of render.wgsl, the material, its entry points), so changing one material recompiles one module, and the driver's
//! pipeline cache (gpu.rs) keeps every compile across runs:
//!
//! - `fa_<id>`: the depth prepass of a transparent material (hashed alpha).
//! - `fg_<id>`: the G-buffer pass of an opaque or dithered material (its closures for the deferred lighting).
//! - `fb_<id>`: a blended material, lit forward and blended over the picture.
//! - `fz_<id>`: a shadow map's pass of a material whose shadow follows its opacity.
//!
//! The lighting of every opaque pixel is one compute pass compiled once (deferred.wgsl); blended materials and those
//! calling Shader to RGB carry the light code in their own module. `shadow_mode: "rays"` traces shadow rays instead of
//! shadow maps: its modules hold every alpha-casting material's opacity for the rays to test.

use std::num::NonZeroU32;

use anyhow::{Context, Result, bail};
use wgpu::util::DeviceExt;

use crate::gpu::Gpu;
use crate::materials::{Material, MaterialSet};
use crate::params::ShadowMode;

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The lit frame's depth (and of the layers and looks drawn with its pipelines); its stencil keeps the G-buffer pass to
/// one surface per pixel (`first_surface`).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;
pub const GB_BASE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
pub const GB_MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// The first pass of a shadow map (opaque casters and the hashed ones' opaque parts): their depth into the depths' R.
const SHADOW_SOLID_TARGET: [Option<wgpu::ColorTargetState>; 1] =
    [Some(wgpu::ColorTargetState { format: crate::shadows::DEPTHS_FORMAT, blend: None, write_mask: wgpu::ColorWrites::RED })];

const ENABLES: &str = "enable wgpu_ray_query;\nenable wgpu_binding_array;\nenable dual_source_blending;\nenable primitive_index;\n\n";
pub(crate) const RECORDS_WGSL: &str = include_str!("shaders/records.wgsl");
const RENDER_WGSL: &str = include_str!("shaders/render.wgsl");
const GBUFFER_WGSL: &str = include_str!("shaders/gbuffer.wgsl");
const GBUFFER_WRITE_WGSL: &str = include_str!("shaders/gbuffer_write.wgsl");
const LTC_WGSL: &str = include_str!("shaders/ltc.wgsl");
const LIGHT_WGSL: &str = include_str!("shaders/light.wgsl");
pub(crate) const SHADOW_LOD_WGSL: &str = include_str!("shaders/shadow_lod.wgsl");
const SHADOW_MAP_WGSL: &str = include_str!("shaders/shadow_map.wgsl");
const SHADOW_RAYS_WGSL: &str = include_str!("shaders/shadow_rays.wgsl");
const DEFERRED_WGSL: &str = include_str!("shaders/deferred.wgsl");
const RAYTRACE_WGSL: &str = include_str!("shaders/raytrace.wgsl");
const WORLD_LIGHT_WGSL: &str = include_str!("shaders/world_light.wgsl");
const WORLD_WGSL: &str = include_str!("shaders/world.wgsl");
pub(crate) const CUBE_WGSL: &str = include_str!("shaders/cube.wgsl");

/// Per material (None where it has no such pass), and the shared pipelines.
pub struct Pipelines {
    pub prepass: Vec<Option<wgpu::RenderPipeline>>,
    pub gbuffer: Vec<Option<wgpu::RenderPipeline>>,
    pub blend: Vec<Option<wgpu::RenderPipeline>>,
    /// Shadow maps' casters whose shadow follows their opacity: where they are opaque (into the depth), and where they
    /// are partly transparent (into the colour layers: shadows.rs).
    pub shadow: Vec<Option<wgpu::RenderPipeline>>,
    pub shadow_hashed: Vec<Option<wgpu::RenderPipeline>>,
    /// Depth only, [no culling, back faces culled].
    pub depth: [wgpu::RenderPipeline; 2],
    /// Depth only for blended surfaces drawn without their back layers.
    pub depth_blend: [wgpu::RenderPipeline; 2],
    /// Shadow maps: casters that are opaque everywhere.
    pub shadow_depth: wgpu::RenderPipeline,
    /// The deferred lighting pass: every light on every pixel, and (scenes of more than TILED_FROM lights) the lights
    /// culled per tile first (deferred.wgsl light_pass_tiled), which draws the same picture.
    pub light: wgpu::ComputePipeline,
    pub light_tiled: Option<wgpu::ComputePipeline>,
}

/// Per-draw data pushed as immediates (`sample`: the shadow face in shadow passes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Immediate {
    pub draw: u32,
    pub sample: u32,
}

/// A frame of more lights than this culls them per tile (deferred.wgsl light_pass_tiled); fewer, each pixel runs them
/// all, which costs less than the culling.
pub const TILED_FROM: usize = 8;

/// The bind group layouts: group 0 of every drawing pipeline, group 1 of the G-buffer pass and of the lighting pass.
pub struct Layouts {
    pub render: wgpu::BindGroupLayout,
    pub gbuffer: wgpu::BindGroupLayout,
    pub light: wgpu::BindGroupLayout,
}

fn buffer_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn render_layout(device: &wgpu::Device, images: u32) -> wgpu::BindGroupLayout {
    let all = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    let shade = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    let buf = |binding: u32| buffer_entry(binding, all, true);
    let sampler = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: shade,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let tex = |binding: u32, dim: wgpu::TextureViewDimension, count: Option<NonZeroU32>| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: shade,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dim,
            multisampled: false,
        },
        count,
    };
    // the shadow maps' colour layers (shadows.rs), read texel by texel
    let tex_raw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: shade,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render"),
        entries: &[
            buf(0),
            buf(1),
            buf(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: shade,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            buf(4),
            buf(5),
            buf(6),
            buf(7),
            buf(8),
            buf(9),
            buf(10),
            buf(11),
            tex(12, wgpu::TextureViewDimension::D2, NonZeroU32::new(images)),
            sampler(13),
            sampler(14),
            sampler(15),
            sampler(16),
            sampler(17),
            sampler(18),
            tex(19, wgpu::TextureViewDimension::D2Array, None),
            sampler(20),
            tex(21, wgpu::TextureViewDimension::D2, None),
            buf(22),
            tex_raw(23),
            buf(24),
            buf(25),
            tex_raw(26),
            tex_raw(28),
            tex_raw(29),
            tex(30, wgpu::TextureViewDimension::Cube, None),
            buf(31),
        ],
    })
}

pub fn layouts(device: &wgpu::Device, images: u32) -> Layouts {
    let render = render_layout(device, images);
    let gbuffer = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gbuffer"),
        entries: &[buffer_entry(0, wgpu::ShaderStages::FRAGMENT, false)],
    });
    let cs = wgpu::ShaderStages::COMPUTE;
    let texture = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: cs,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    };
    let light = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting"),
        entries: &[
            texture(0, wgpu::TextureSampleType::Uint),
            texture(1, wgpu::TextureSampleType::Float { filterable: false }),
            texture(2, wgpu::TextureSampleType::Depth),
            buffer_entry(3, cs, true),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: cs,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: COLOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // the screen tracing's light (raytrace.wgsl): the previous sample's lit picture, and this one's
            texture(5, wgpu::TextureSampleType::Float { filterable: false }),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: cs,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: COLOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // the traced reflections' history (raytrace.wgsl rt_temporal): the previous sample's, and this one's
            texture(7, wgpu::TextureSampleType::Float { filterable: false }),
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: cs,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: COLOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    Layouts { render, gbuffer, light }
}

// ================================================================================================ the module sources
/// The drawing core every module starts with.
fn core(mats: &MaterialSet) -> String {
    let mut s = String::with_capacity(mats.prelude.len() + RECORDS_WGSL.len() + RENDER_WGSL.len() + GBUFFER_WGSL.len() + 256);
    s.push_str(ENABLES);
    s.push_str(RECORDS_WGSL);
    s.push('\n');
    s.push_str(&mats.prelude);
    s.push('\n');
    s.push_str(RENDER_WGSL);
    s.push('\n');
    s.push_str(GBUFFER_WGSL);
    s.push('\n');
    s
}

/// The light code (and its shadows), for modules that light closures themselves; `have`: the materials whose code the
/// module holds already (rays mode adds every other alpha-casting material, which shadow rays test).
fn lighting(mats: &MaterialSet, mode: ShadowMode, have: &[usize]) -> String {
    let mut s = String::new();
    s.push_str(LTC_WGSL);
    s.push('\n');
    s.push_str(LIGHT_WGSL);
    s.push('\n');
    s.push_str(WORLD_LIGHT_WGSL);
    s.push('\n');
    // Shader to RGB: the closures lit as the surface would be (the world's light too)
    let to_rgb = |light: &str| format!("\nfn mkr_shader_to_rgb(c: Closures, s: Surface) -> vec4<f32> {{\n    let h = shading_of(c, s);\n    var o = c.emission + {light}(h);\n    if (frame.world.x > 0.5) {{\n        o = o + world_light(h, c.reflection_roughness, c.coat_roughness);\n    }}\n    return vec4<f32>(o, 1.0);\n}}\n\n");
    match mode {
        ShadowMode::Map => {
            s.push_str(SHADOW_LOD_WGSL);
            s.push('\n');
            s.push_str(&format!("const SUN_CASCADES: u32 = {}u;\n", mats.sun_cascades));
            s.push_str(SHADOW_MAP_WGSL);
            s.push_str(&to_rgb("light_shading"));
        }
        ShadowMode::Rays => {
            s.push_str(SHADOW_RAYS_WGSL);
            // a shadow ray may test an opacity that reads Shader to RGB: WGSL cannot recurse, so this one is unshadowed
            s.push_str(&to_rgb("light_shading_unshadowed"));
            s.push_str("fn material_alpha_lod(m: u32, s: Surface) -> f32 {\n    switch m {\n");
            for (i, m) in mats.list.iter().enumerate() {
                if m.casts_alpha_shadow() {
                    s.push_str(&format!("        case {i}u: {{ return mat_{}_alpha_lod(s); }}\n", m.id));
                }
            }
            s.push_str("        default: { return 1.0; }\n    }\n}\n\n");
            for (i, m) in mats.list.iter().enumerate() {
                if m.casts_alpha_shadow() && !have.contains(&i) {
                    s.push_str(&m.wgsl);
                    s.push('\n');
                }
            }
        }
    }
    s
}

const ARGS: &str = "in: VsOut, @builtin(front_facing) front: bool, @builtin(primitive_index) prim: u32";

fn entry_prepass(m: &Material) -> String {
    let id = &m.id;
    format!("@fragment\nfn fa_{id}({ARGS}) {{\n    let threshold = alpha_threshold(in.p, imm.sample);\n    let s = fragment_surface(in, front, prim);\n    if (mat_{id}_alpha(s) <= threshold) {{\n        discard;\n    }}\n}}\n\n")
}

fn entry_gbuffer(m: &Material) -> String {
    let id = &m.id;
    format!("@fragment @early_depth_test(force)\nfn fg_{id}({ARGS}) -> GOut {{\n    let s = fragment_surface(in, front, prim);\n    return gbuffer_write(mat_{id}(s), s, in);\n}}\n\n")
}

/// A hashed caster in a shadow map (shadows.rs): `fz_` keeps its opaque fragments (alpha 1, as an opaque caster's,
/// into the depth and the depths' R), `fh_` its partly transparent ones (their depth into the depths' G and their
/// transparency).
fn entry_shadow(m: &Material) -> String {
    let id = &m.id;
    format!("@fragment\nfn fz_{id}({ARGS}) -> @location(0) vec4<f32> {{\n    let s = shadow_surface(in, front, prim);\n    if (mat_{id}_alpha(s) < SHADOW_OPAQUE) {{\n        discard;\n    }}\n    return vec4<f32>(in.pos.z, 0.0, 0.0, 0.0);\n}}\n\n\
             @fragment\nfn fh_{id}({ARGS}) -> ShadowOut {{\n    let s = shadow_surface(in, front, prim);\n    let a = clamp(mat_{id}_alpha(s), 0.0, 1.0);\n    if (a <= 0.0 || a >= SHADOW_OPAQUE) {{\n        discard;\n    }}\n    return ShadowOut(vec4<f32>(0.0, in.pos.z, 0.0, 0.0), 1.0 - a);\n}}\n\n")
}

const BLENDED_STRUCT: &str = "struct Blended {\n    @location(0) @blend_src(0) color: vec4<f32>,\n    @location(0) @blend_src(1) transmittance: vec4<f32>,\n}\n\n";

fn entry_blended(m: &Material) -> String {
    let id = &m.id;
    format!("@fragment\nfn fb_{id}({ARGS}) -> Blended {{\n    let s = fragment_surface(in, front, prim);\n    let c = mat_{id}(s);\n    return Blended(vec4<f32>(light_closures(c, s), 1.0), vec4<f32>(clamp(c.transmittance, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));\n}}\n\n")
}

fn entry_forward(m: &Material) -> String {
    let id = &m.id;
    format!("@fragment\nfn fs_{id}({ARGS}) -> @location(0) vec4<f32> {{\n    let s = fragment_surface(in, front, prim);\n    return vec4<f32>(shade_dithered(mat_{id}(s), s), 1.0);\n}}\n\n")
}

/// The module of material `i` alone: its passes' entry points (and the light code when it lights closures itself).
pub fn material_module(mats: &MaterialSet, i: usize, mode: ShadowMode) -> String {
    let m = &mats.list[i];
    let lit = m.blended || m.uses_shader_to_rgb;
    let mut s = core(mats);
    if lit {
        s.push_str(&lighting(mats, mode, &[i]));
    }
    s.push_str(&m.wgsl);
    s.push('\n');
    if m.blended {
        s.push_str(BLENDED_STRUCT);
        s.push_str(&entry_blended(m));
    } else {
        s.push_str(GBUFFER_WRITE_WGSL);
        s.push('\n');
        s.push_str(&entry_gbuffer(m));
        if m.transparent {
            s.push_str(&entry_prepass(m));
        }
    }
    if m.casts_alpha_shadow() {
        s.push_str(&entry_shadow(m));
    }
    s
}

/// The shared module: depth-only passes, shadow maps' opaque casters.
fn shared_module(mats: &MaterialSet) -> String {
    core(mats)
}

/// The deferred lighting pass's module: the screen tracing's code (raytrace.wgsl) used only when the scene traces
/// (RT_TRACED), so a scene without ray tracing compiles the same code whatever the tracing does.
pub fn light_module(mats: &MaterialSet, mode: ShadowMode) -> String {
    let mut s = core(mats);
    s.push_str(&lighting(mats, mode, &[]));
    s.push_str(&format!("const RT_TRACED: bool = {};\n", mats.traced));
    s.push_str(RAYTRACE_WGSL);
    s.push('\n');
    s.push_str(DEFERRED_WGSL);
    s
}

/// The world's module (world.rs): the drawing core, the world's code (its parameters and images at the world's own
/// bases) and its entry points, the background and the probe's faces, drawn as Light Path's `probe_ray` sees them.
pub fn world_module(mats: &MaterialSet, probe_ray: &str) -> Option<String> {
    let w = mats.world.as_ref()?;
    let mut s = core(mats);
    s.push_str(&format!("const WORLD_PARAM_BASE: u32 = {}u;\nconst WORLD_IMAGE_BASE: u32 = {}u;\n\
                         const WORLD_PROBE_RAY: u32 = {probe_ray};\n\n", w.param_base, w.image_base));
    // a world's Shader to RGB sees no lights
    s.push_str("fn mkr_shader_to_rgb(c: Closures, s: Surface) -> vec4<f32> {\n    return vec4<f32>(c.emission, 1.0);\n}\n\n");
    s.push_str(&w.wgsl);
    s.push('\n');
    s.push_str(CUBE_WGSL);
    s.push('\n');
    s.push_str(WORLD_WGSL);
    Some(s)
}

/// One module holding the `used` materials with forward-lit entry points (`fs_` lit, `fa_` prepass, `fb_` blended)
/// and the light code with shadow maps: what the layer passes draw with.
pub fn module_source(mats: &MaterialSet, used: &[bool]) -> String {
    let have: Vec<usize> = (0..mats.list.len()).filter(|&i| used[i]).collect();
    let mut s = core(mats);
    s.push_str(&lighting(mats, ShadowMode::Map, &have));
    s.push_str(BLENDED_STRUCT);
    for &i in &have {
        let m = &mats.list[i];
        s.push_str(&m.wgsl);
        s.push('\n');
        if m.blended {
            s.push_str(&entry_blended(m));
        } else {
            s.push_str(&entry_forward(m));
            if m.transparent {
                s.push_str(&entry_prepass(m));
            }
        }
    }
    s
}

// ====================================================================================================== pipelines
fn dump(name: &str, source: &str) -> Result<()> {
    if let Ok(dir) = std::env::var("MKR_DUMP_WGSL") {
        std::fs::create_dir_all(&dir).with_context(|| format!("cannot make {dir}"))?;
        let path = std::path::Path::new(&dir).join(format!("{name}.wgsl"));
        std::fs::write(&path, source).with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

pub(crate) fn module(device: &wgpu::Device, label: &str, source: String) -> Result<wgpu::ShaderModule> {
    dump(label, &source)?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        bail!("the {label} shaders do not compile: {e}");
    }
    Ok(module)
}

/// What a material pipeline is: which pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pass {
    Prepass,
    GBuffer,
    Blend,
    Shadow,
    ShadowHashed,
}

struct Builder<'a> {
    device: &'a wgpu::Device,
    cache: Option<&'a wgpu::PipelineCache>,
    main: wgpu::PipelineLayout,
    gbuffer: wgpu::PipelineLayout,
}

fn depth_state(format: wgpu::TextureFormat, write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

/// `d` with a stencil test that passes a pixel's first fragment alone (the stencil cleared to 0 by the prepass, the
/// pass's reference 0): coplanar surfaces at the prepass's depth all pass its Equal test, and fragments' storage writes
/// have no order, so without it which surface's extras a pixel keeps would change from run to run. Stencil tests run
/// in primitive order, so the first surface drawn wins, every time.
fn first_surface(d: wgpu::DepthStencilState) -> wgpu::DepthStencilState {
    let face = wgpu::StencilFaceState {
        compare: wgpu::CompareFunction::Equal,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: wgpu::StencilOperation::IncrementClamp,
    };
    wgpu::DepthStencilState { stencil: wgpu::StencilState { front: face, back: face, read_mask: 0xff, write_mask: 0xff }, ..d }
}

fn primitive(cull: bool) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: cull.then_some(wgpu::Face::Back),
        ..Default::default()
    }
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn render(&self, label: &str, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, vs: &str,
              fs: Option<&str>, targets: &[Option<wgpu::ColorTargetState>], cull: bool, depth: wgpu::DepthStencilState)
              -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
            primitive: primitive(cull),
            depth_stencil: Some(depth),
            multisample: Default::default(),
            fragment: fs.map(|entry| wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets,
            }),
            multiview_mask: None,
            cache: self.cache,
        })
    }

    fn material(&self, mats: &MaterialSet, i: usize, pass: Pass, module: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
        let m = &mats.list[i];
        let gt = wgpu::CompareFunction::Greater;
        let id = &m.id;
        match pass {
            Pass::Prepass => {
                let fa = format!("fa_{id}");
                self.render(&fa, &self.main, module, "vs_main", Some(&fa), &[], m.cull, depth_state(DEPTH_FORMAT, true, gt))
            }
            Pass::GBuffer => {
                let fg = format!("fg_{id}");
                let targets = [
                    Some(wgpu::ColorTargetState { format: GB_BASE_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: GB_MOTION_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ];
                // the prepass left every pixel's kept surface: it alone runs here (the extras are storage writes,
                // which a second surface at the very same depth would race: the stencil lets only the first through)
                self.render(&fg, &self.gbuffer, module, "vs_main", Some(&fg), &targets, m.cull,
                            first_surface(depth_state(DEPTH_FORMAT, false, wgpu::CompareFunction::Equal)))
            }
            Pass::Blend => {
                // colour + destination x transmittance (dual source)
                let blend = wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::Src1, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                };
                let fb = format!("fb_{id}");
                let targets = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })];
                let cmp = if m.show_back { gt } else { wgpu::CompareFunction::Equal };
                self.render(&fb, &self.main, module, "vs_main", Some(&fb), &targets, m.cull, depth_state(DEPTH_FORMAT, false, cmp))
            }
            Pass::Shadow => {
                let fz = format!("fz_{id}");
                self.render(&fz, &self.main, module, "vs_shadow_full", Some(&fz), &SHADOW_SOLID_TARGET, false,
                            depth_state(SHADOW_FORMAT, true, gt))
            }
            Pass::ShadowHashed => {
                // in front of the opaque casters (drawn first, depth read only): the nearest depth by max into the
                // depths' G, the transparencies multiplied
                let fh = format!("fh_{id}");
                let max = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One,
                                                 operation: wgpu::BlendOperation::Max };
                let product = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::Src,
                                                     operation: wgpu::BlendOperation::Add };
                let targets = [
                    Some(wgpu::ColorTargetState { format: crate::shadows::DEPTHS_FORMAT,
                                                  blend: Some(wgpu::BlendState { color: max, alpha: max }),
                                                  write_mask: wgpu::ColorWrites::GREEN }),
                    Some(wgpu::ColorTargetState { format: crate::shadows::CLEAR_FORMAT,
                                                  blend: Some(wgpu::BlendState { color: product, alpha: product }),
                                                  write_mask: wgpu::ColorWrites::ALL }),
                ];
                self.render(&fh, &self.main, module, "vs_shadow_full", Some(&fh), &targets, false, depth_state(SHADOW_FORMAT, false, gt))
            }
        }
    }
}

/// Every pipeline the frames need: the `used` materials' passes (compiled `threads` at a time) and the shared ones (the
/// tiled lighting pass when the scene has more than TILED_FROM `lights`).
pub fn make(gpu: &Gpu, layouts: &Layouts, mats: &MaterialSet, used: &[bool], mode: ShadowMode, threads: usize,
            lights: usize) -> Result<Pipelines> {
    let device = &gpu.device;
    let imm = std::mem::size_of::<Immediate>() as u32;
    let pipeline_layout = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: groups, immediate_size: imm })
    };
    let b = Builder {
        device,
        cache: gpu.cache.as_ref(),
        main: pipeline_layout("render", &[Some(&layouts.render)]),
        gbuffer: pipeline_layout("gbuffer", &[Some(&layouts.render), Some(&layouts.gbuffer)]),
    };
    let light_layout = pipeline_layout("lighting", &[Some(&layouts.render), Some(&layouts.light)]);

    // the shared module and the lighting module first: their errors are the likeliest
    let shared = module(device, "shared", shared_module(mats))?;
    let light_mod = module(device, "lighting", light_module(mats, mode))?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let gt = wgpu::CompareFunction::Greater;
    let ge = wgpu::CompareFunction::GreaterEqual;
    let none_target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty() })];
    let depth = [false, true].map(|c| b.render("depth", &b.main, &shared, "vs_depth", None, &[], c, depth_state(DEPTH_FORMAT, true, gt)));
    let depth_blend = [false, true].map(|c| {
        b.render("depth blend", &b.main, &shared, "vs_depth", Some("fs_none"), &none_target, c, depth_state(DEPTH_FORMAT, true, ge))
    });
    let shadow_depth = b.render("shadow depth", &b.main, &shared, "vs_shadow", Some("fs_shadow_depth"), &SHADOW_SOLID_TARGET,
                                false, depth_state(SHADOW_FORMAT, true, gt));
    let light_entry = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&light_layout),
        module: &light_mod,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: gpu.cache.as_ref(),
    });
    let light = light_entry("light_pass");
    let light_tiled = (lights > TILED_FROM).then(|| light_entry("light_pass_tiled"));
    if let Some(e) = pollster::block_on(scope.pop()) {
        bail!("the shared pipelines cannot be made: {e}");
    }

    // the materials: one module each, their pipelines spread over threads (the driver compiles inside the calls)
    let n = mats.list.len();
    let jobs: Vec<usize> = (0..n).filter(|&i| used[i]).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Made = (usize, Pass, wgpu::RenderPipeline);
    let made: Vec<Result<Vec<Made>>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads.max(1))
            .map(|_| {
                sc.spawn(|| -> Result<Vec<Made>> {
                    let mut done = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&i) = jobs.get(j) else { break };
                        let m = &mats.list[i];
                        let module = module(device, &format!("material {} ({})", m.name, m.id), material_module(mats, i, mode))?;
                        let mut passes = Vec::new();
                        if m.blended {
                            passes.push(Pass::Blend);
                        } else {
                            passes.push(Pass::GBuffer);
                            if m.transparent {
                                passes.push(Pass::Prepass);
                            }
                        }
                        if m.casts_alpha_shadow() && mode == ShadowMode::Map {
                            passes.push(Pass::Shadow);
                            passes.push(Pass::ShadowHashed);
                        }
                        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                        for p in passes {
                            done.push((i, p, b.material(mats, i, p, &module)));
                        }
                        if let Some(e) = pollster::block_on(scope.pop()) {
                            bail!("material {}: its pipelines cannot be made: {e}", m.name);
                        }
                    }
                    Ok(done)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a pipeline thread panicked")).collect()
    });
    let mut out = Pipelines {
        prepass: (0..n).map(|_| None).collect(),
        gbuffer: (0..n).map(|_| None).collect(),
        blend: (0..n).map(|_| None).collect(),
        shadow: (0..n).map(|_| None).collect(),
        shadow_hashed: (0..n).map(|_| None).collect(),
        depth,
        depth_blend,
        shadow_depth,
        light,
        light_tiled,
    };
    for r in made {
        for (i, pass, p) in r? {
            match pass {
                Pass::Prepass => out.prepass[i] = Some(p),
                Pass::GBuffer => out.gbuffer[i] = Some(p),
                Pass::Blend => out.blend[i] = Some(p),
                Pass::Shadow => out.shadow[i] = Some(p),
                Pass::ShadowHashed => out.shadow_hashed[i] = Some(p),
            }
        }
    }
    Ok(out)
}

pub fn upload_texture(gpu: &Gpu, name: &str, px: &crate::images::Pixels) -> wgpu::TextureView {
    let mut data = Vec::new();
    for l in &px.levels {
        data.extend_from_slice(l);
    }
    let tex = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some(name),
            size: wgpu::Extent3d { width: px.width, height: px.height, depth_or_array_layers: 1 },
            mip_level_count: px.levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: px.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &data,
    );
    tex.create_view(&Default::default())
}

/// A float texture (rgba32float, `layers` layers) for lookup tables.
pub fn table_texture(gpu: &Gpu, label: &str, w: u32, h: u32, layers: u32, data: &[f32]) -> wgpu::TextureView {
    let tex = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(data),
    );
    let dim = if layers > 1 { wgpu::TextureViewDimension::D2Array } else { wgpu::TextureViewDimension::D2 };
    tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dim), ..Default::default() })
}

/// Linear (mipmapped, anisotropic) and nearest samplers for each extension (repeat, extend, mirror), then the tables'.
pub fn samplers(device: &wgpu::Device, anisotropy: u16) -> Vec<wgpu::Sampler> {
    let modes = [wgpu::AddressMode::Repeat, wgpu::AddressMode::ClampToEdge, wgpu::AddressMode::MirrorRepeat];
    let mut out = Vec::new();
    for linear in [true, false] {
        for &m in &modes {
            let f = if linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
            out.push(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("image"),
                address_mode_u: m,
                address_mode_v: m,
                address_mode_w: m,
                mag_filter: f,
                min_filter: f,
                mipmap_filter: if linear { wgpu::MipmapFilterMode::Linear } else { wgpu::MipmapFilterMode::Nearest },
                lod_min_clamp: 0.0,
                lod_max_clamp: if linear { 32.0 } else { 0.0 },
                anisotropy_clamp: if linear { anisotropy.max(1) } else { 1 },
                ..Default::default()
            }));
        }
    }
    out.push(device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("table"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    }));
    out
}
