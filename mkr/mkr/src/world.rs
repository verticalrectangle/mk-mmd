//! The world (bake.json `lit.world`) as EEVEE Next draws it when it traces no rays (measured on the lab's world_*
//! cases): the camera sees the world's shader itself wherever no surface is (`background`, a pass after the lighting,
//! per sample), and surfaces are lit by its probe: the world's radiance drawn into a cube map once per frame (again only
//! when its animated values change), its mips the 2 x 2 average of the one above for glossy lobes to read at a level by
//! roughness, and its L1 spherical harmonics, projected from a small mip, for diffuse light (shaders/world_light.wgsl).
//! EEVEE leaves world light unshadowed without ray tracing: no occlusion here either.

use anyhow::{Result, bail};

use crate::gpu::Gpu;
use crate::materials::MaterialSet;
use crate::params::Params;
use crate::pipelines::{self, COLOR_FORMAT, DEPTH_FORMAT, Immediate};

const PROBE_WGSL: &str = include_str!("shaders/world_probe.wgsl");
/// The probe's mip the harmonics are projected from: the first at most this many texels across.
const HARMONICS_SIZE: u32 = 32;
/// The Light Path ray type the probe's texels are drawn as (mkr-shade's MKR_RAY_*): EEVEE's probe is not a camera ray
/// (lab world_lightpath).
const PROBE_RAY: &str = "MKR_RAY_GLOSSY";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeImm {
    size: u32,
    pad: [u32; 3],
}

/// What the drawing bind group (bindings 30, 31) holds of the world: its probe's cube and harmonics, or stand-ins.
pub struct WorldViews {
    pub cube: wgpu::TextureView,
    pub harmonics: wgpu::Buffer,
}

fn cube_texture(device: &wgpu::Device, size: u32, mips: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world probe"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

fn cube_view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() })
}

fn harmonics_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    crate::render::sized(device, "world harmonics", 64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)
}

/// Stand-ins for a scene without a world (and for the probe's own passes, which draw into the probe): a black cube
/// and zero harmonics.
pub fn stand_ins(device: &wgpu::Device) -> WorldViews {
    let tex = cube_texture(device, 1, 1);
    WorldViews { cube: cube_view(&tex), harmonics: harmonics_buffer(device) }
}

pub struct WorldGpu {
    pub views: WorldViews,
    /// Mip 0's faces, drawn into.
    faces: Vec<wgpu::TextureView>,
    size: u32,
    mips: u32,
    probe: wgpu::RenderPipeline,
    background: wgpu::RenderPipeline,
    downsample: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    /// Per mip from 1: its average of the one above; then the harmonics' projection.
    down_bgs: Vec<wgpu::BindGroup>,
    project_bg: wgpu::BindGroup,
    harmonics_mip: u32,
    /// The world's parameter slots, and their values the probe was last drawn with (None: not drawn yet).
    slots: std::ops::Range<usize>,
    drawn: Option<Vec<[f32; 4]>>,
    /// The camera sees it: the film is not transparent (EEVEE ignores the world's ray visibility flags, Cycles' own:
    /// lab world_visibility).
    pub seen: bool,
}

impl WorldGpu {
    /// The world's passes, or None when the scene has none (`mats.world`). `render_layout`: group 0 of the drawing
    /// pipelines.
    pub fn new(gpu: &Gpu, mats: &MaterialSet, render_layout: &wgpu::BindGroupLayout, film_transparent: bool)
               -> Result<Option<WorldGpu>> {
        let Some(world) = &mats.world else { return Ok(None) };
        let Some(source) = pipelines::world_module(mats, PROBE_RAY) else { return Ok(None) };
        let device = &gpu.device;
        let resolution = world.prop("probe_resolution").unwrap_or(1024.0) as u32;
        let size = (resolution / 2).clamp(16, 2048).next_power_of_two();
        let mips = size.trailing_zeros() + 1;
        let tex = cube_texture(device, size, mips);
        let faces = (0..6).map(|f| tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: f,
            array_layer_count: Some(1),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        })).collect();
        let level = |m: u32| tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_mip_level: m,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let views = WorldViews { cube: cube_view(&tex), harmonics: harmonics_buffer(device) };

        let cache = gpu.cache.as_ref();
        let module = pipelines::module(device, "world", source)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world"),
            bind_group_layouts: &[Some(render_layout)],
            immediate_size: std::mem::size_of::<Immediate>() as u32,
        });
        let pipe = |label: &str, fs: &str, depth: Option<wgpu::DepthStencilState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_world"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: depth,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache,
            })
        };
        let probe = pipe("world probe", "fs_world_probe", None);
        // where the depth is still the cleared far plane: no surface
        let background = pipe("world background", "fs_world_background", Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Equal),
            stencil: Default::default(),
            bias: Default::default(),
        }));

        let probe_module = pipelines::module(device, "world probe", format!("{}\n{PROBE_WGSL}", pipelines::CUBE_WGSL))?;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world probe"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: COLOR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let downsample = crate::render::compute_pipe(device, &probe_module, &bgl, "downsample",
                                                     std::mem::size_of::<ProbeImm>() as u32, cache);
        let project = crate::render::compute_pipe(device, &probe_module, &bgl, "project",
                                                  std::mem::size_of::<ProbeImm>() as u32, cache);
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("the world's pipelines cannot be made: {e}");
        }
        let group = |src: &wgpu::TextureView, dst: &wgpu::TextureView| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world probe"),
            layout: &bgl,
            entries: &[
                crate::render::b(0, wgpu::BindingResource::TextureView(src)),
                crate::render::b(1, wgpu::BindingResource::TextureView(dst)),
                crate::render::b(2, views.harmonics.as_entire_binding()),
            ],
        });
        let down_bgs = (1..mips).map(|m| group(&level(m - 1), &level(m))).collect();
        let harmonics_mip = size.trailing_zeros().saturating_sub(HARMONICS_SIZE.trailing_zeros());
        // the projection writes no texel: any mip but the one it reads stands for its target
        let project_bg = group(&level(harmonics_mip), &level(if harmonics_mip == 0 { mips - 1 } else { 0 }));
        let n = world.params.iter().map(|p| p.slot as usize + 1).max().unwrap_or(0);
        let base = world.param_base as usize;
        Ok(Some(WorldGpu {
            views,
            faces,
            size,
            mips,
            probe,
            background,
            downsample,
            project,
            down_bgs,
            project_bg,
            harmonics_mip,
            slots: base..base + n,
            drawn: None,
            seen: !film_transparent,
        }))
    }

    /// Draw the probe for the frame whose parameter values are `params` (the scene's, materials' first), unless it
    /// shows the world as it is already: its faces, its mips and its harmonics. `bg`: group 0 with the world's
    /// stand-ins (the probe cannot be read while it is drawn).
    pub fn update(&mut self, enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, params: &[[f32; 4]]) {
        let own = params.get(self.slots.clone()).unwrap_or(&[]).to_vec();
        if self.drawn.as_ref() == Some(&own) {
            return;
        }
        for (f, face) in self.faces.iter().enumerate() {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world probe face"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: face,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.probe);
            pass.set_bind_group(0, bg, &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: f as u32, sample: self.size }));
            pass.draw(0..3, 0..1);
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("world probe mips"), timestamp_writes: None });
        pass.set_pipeline(&self.downsample);
        for (i, bg) in self.down_bgs.iter().enumerate() {
            let s = (self.size >> (i + 1)).max(1);
            pass.set_bind_group(0, bg, &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&ProbeImm { size: s, pad: [0; 3] }));
            pass.dispatch_workgroups(s.div_ceil(8), s.div_ceil(8), 6);
        }
        pass.set_pipeline(&self.project);
        pass.set_bind_group(0, &self.project_bg, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&ProbeImm { size: (self.size >> self.harmonics_mip).max(1), pad: [0; 3] }));
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        self.drawn = Some(own);
    }

    /// The frame record's world fields: [1 (it lights), the last mip, the harmonics' band weights], [glossy blur, the
    /// harmonics blend's roughness from and to, the probe's texels across].
    pub fn record(&self, p: &Params) -> ([f32; 4], [f32; 4]) {
        ([1.0, (self.mips - 1) as f32, p.world_band0, p.world_band1],
         [p.world_glossy_blur, p.world_glossy_mix_from, p.world_glossy_mix_to, self.size as f32])
    }

    /// The background of sample `sample` into `color` where `depth` holds no surface.
    pub fn background(&self, enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, color: &wgpu::TextureView,
                      depth: &wgpu::TextureView, sample: u32) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world background"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.background);
        pass.set_bind_group(0, bg, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&Immediate { draw: 0, sample }));
        pass.draw(0..3, 0..1);
    }
}
