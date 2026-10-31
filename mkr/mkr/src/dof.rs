//! Depth of field. A camera with it (camera.rs `Dof`) is drawn through a lens by default (`dof: "lens"`): every
//! sample through its own point of the whole aperture, the focus plane kept (camera.rs `view_projection`), what EEVEE's
//! jittered camera and Cycles converge on. `dof: "eevee"` follows the scene's EEVEE instead: its default is a post
//! process over each sample's picture (the scene's `use_bokeh_jittered` off), emulated here once on the film after the
//! samples (shaders/dof.wgsl: the half-resolution copy that favours the widest blur and dark colours, its mips, the
//! tiles' widest blurs, the background and foreground layers gathered from rings of samples whose blur covers the
//! pixel within the aperture's shape, the slight focus at full resolution and the layers laid over each other); with
//! `use_bokeh_jittered` on EEVEE moves the camera over most of the aperture and leaves the post process the rest (its
//! over-blur).

use bytemuck::{Pod, Zeroable};
use serde_json::Value;

use crate::camera::Cam;
use crate::params::{DofMode, Params};
use crate::render::{b, compute_pipe};

const DOF_WGSL: &str = include_str!("shaders/dof.wgsl");
/// Full-resolution pixels across a tile of the blurs' extents (shaders/dof.wgsl TILE).
const TILE: u32 = 16;
/// Levels of the half-resolution copy the rings sample (the outer rings of wide blurs read the coarser ones).
const MIPS: u32 = 4;
const HALF_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The scene's EEVEE depth of field settings (bake `lit.eevee`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneDof {
    /// use_bokeh_jittered: the camera jittered over the aperture each sample, the post process blurring the rest.
    pub jittered: bool,
    /// bokeh_overblur (a fraction, not Blender's percentage): the post process's share of the aperture beyond the
    /// least that fills the gaps between the samples' points.
    pub overblur: f32,
    /// bokeh_max_size: the widest circle of confusion, radius in pixels.
    pub max_size: f32,
}

impl SceneDof {
    pub fn of(eevee: &Value) -> SceneDof {
        let f = |key: &str, d: f64| eevee.get(key).and_then(Value::as_f64).unwrap_or(d) as f32;
        SceneDof {
            jittered: eevee.get("use_bokeh_jittered").and_then(Value::as_bool).unwrap_or(false),
            overblur: f("bokeh_overblur", 5.0) / 100.0,
            max_size: f("bokeh_max_size", 100.0),
        }
    }
}

/// The shares of the aperture a frame's depth of field takes from the samples' lens (camera.rs `lens_offset`) and from
/// the post process (`n` samples): EEVEE's post process alone; its jittered camera with the post process filling
/// `1 / sqrt(n)` of the aperture between the samples' points plus the over-blur; the lens alone; or none.
pub fn split(mode: DofMode, scene: &SceneDof, n: u32) -> (f32, f32) {
    match mode {
        DofMode::Off => (0.0, 0.0),
        DofMode::Lens => (1.0, 0.0),
        DofMode::Eevee if scene.jittered && n > 1 => {
            let post = (1.0 / (n as f32).sqrt() + scene.overblur).min(1.0);
            (1.0 - post, post)
        }
        DofMode::Eevee => (0.0, 1.0),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DofGpu {
    size: [u32; 2],
    half: [u32; 2],
    tiles: [u32; 2],
    level: u32,
    blades: u32,
    coc: [f32; 4],
    shape: [f32; 4],
    weights: [f32; 4],
}

/// The post process's record for a W x H frame through `cam` with `share` of its aperture (None: nothing to blur).
pub(crate) fn record(cam: &Cam, w: u32, h: u32, share: f32, scene: &SceneDof, p: &Params) -> Option<DofGpu> {
    let d = cam.dof?;
    let scale = cam.coc_scale(w as f32, h as f32) * share;
    if scale <= 0.0 {
        return None;
    }
    // an anamorphic aperture: the circle's radius its wider axis, the shape squeezed along the other
    let ratio = d.ratio.max(1e-3);
    let scale = scale * (1.0 / ratio).max(1.0);
    let aniso = [(1.0 / ratio).min(1.0), ratio.min(1.0)];
    // CoC = scale (1 / focus - 1 / depth), 1 / depth = 1 / far + z (1 / near - 1 / far) for the reversed depth z
    let (n, f) = (cam.near, cam.far);
    let s = d.focus_distance.max(1e-4);
    let half = [w.div_ceil(2), h.div_ceil(2)];
    Some(DofGpu {
        size: [w, h],
        half,
        tiles: [w.div_ceil(TILE), h.div_ceil(TILE)],
        level: 0,
        blades: if d.blades >= 3.0 { d.blades.round() as u32 } else { 0 },
        coc: [scale * (1.0 / s - 1.0 / f), -scale * (1.0 / n - 1.0 / f), scene.max_size, 0.0],
        shape: [d.rotation, aniso[0], aniso[1], p.dof_rings.max(1) as f32],
        weights: [p.dof_color, p.dof_coc, p.dof_slight, p.dof_floor],
    })
}

/// The post process's pipelines (made the first time a frame needs them).
pub(crate) struct DofPost {
    pipes: [wgpu::ComputePipeline; 6],
    layouts: [wgpu::BindGroupLayout; 6],
    sampler: wgpu::Sampler,
}

/// Its textures and bind groups for one picture size, over the film and the depth of the targets.
pub(crate) struct DofTargets {
    pub(crate) w: u32,
    pub(crate) h: u32,
    mips: Vec<[u32; 2]>,
    tiles: [u32; 2],
    groups: Vec<wgpu::BindGroup>,
}

const SETUP: usize = 0;
const MIP: usize = 1;
const TILES: usize = 2;
const DILATE: usize = 3;
const GATHER: usize = 4;
const RESOLVE: usize = 5;
const ENTRIES: [&str; 6] = ["setup", "mip", "tile", "dilate", "gather", "resolve"];

fn storage_buf(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn store(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HALF_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl DofPost {
    pub(crate) fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> DofPost {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof"),
            source: wgpu::ShaderSource::Wgsl(DOF_WGSL.into()),
        });
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let entries: [Vec<wgpu::BindGroupLayoutEntry>; 6] = [
            vec![storage_buf(0), texture(1, wgpu::TextureSampleType::Depth), store(2), store(4)],
            vec![store(4), texture(5, float)],
            vec![texture(3, float), storage_buf(6)],
            vec![storage_buf(6), storage_buf(7)],
            vec![texture(5, float), storage_buf(7), store(8), store(9), sampler_entry(12)],
            vec![storage_buf(0), texture(3, float), storage_buf(7), texture(10, float), texture(11, float), sampler_entry(12)],
        ];
        let layouts = std::array::from_fn(|i| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(ENTRIES[i]), entries: &entries[i] })
        });
        let size = std::mem::size_of::<DofGpu>() as u32;
        let pipes = std::array::from_fn(|i| compute_pipe(device, &module, &layouts[i], ENTRIES[i], size, cache));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        DofPost { pipes, layouts, sampler }
    }

    /// The textures and bind groups of a W x H picture whose samples are summed in `film` (rgb x weight, weight) and
    /// whose depth (the last sample's) is `depth`.
    pub(crate) fn targets(&self, device: &wgpu::Device, w: u32, h: u32, film: &wgpu::Buffer, depth: &wgpu::TextureView)
                          -> DofTargets {
        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let tex = |label: &str, size: [u32; 2], mips: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HALF_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let half = [w.div_ceil(2), h.div_ceil(2)];
        let mips: Vec<[u32; 2]> = (0..MIPS).map(|l| [(half[0] >> l).max(1), (half[1] >> l).max(1)]).collect();
        let tiles = [w.div_ceil(TILE), h.div_ceil(TILE)];
        let full = tex("dof full", [w, h], 1).create_view(&Default::default());
        let half_tex = tex("dof half", half, MIPS);
        let levels: Vec<wgpu::TextureView> = (0..MIPS)
            .map(|l| half_tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            }))
            .collect();
        let half_all = half_tex.create_view(&Default::default());
        let bg = tex("dof background", half, 1).create_view(&Default::default());
        let fg = tex("dof foreground", half, 1).create_view(&Default::default());
        let buf = |label: &str| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (tiles[0] * tiles[1]) as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let (tile_buf, dilated) = (buf("dof tiles"), buf("dof dilated"));
        let view = wgpu::BindingResource::TextureView;
        let group = |i: usize, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(ENTRIES[i]), layout: &self.layouts[i], entries })
        };
        let mut groups = vec![group(SETUP, &[b(0, film.as_entire_binding()), b(1, view(depth)), b(2, view(&full)),
                                            b(4, view(&levels[0]))])];
        for l in 1..MIPS as usize {
            groups.push(group(MIP, &[b(4, view(&levels[l])), b(5, view(&levels[l - 1]))]));
        }
        let sampler = wgpu::BindingResource::Sampler(&self.sampler);
        groups.push(group(TILES, &[b(3, view(&full)), b(6, tile_buf.as_entire_binding())]));
        groups.push(group(DILATE, &[b(6, tile_buf.as_entire_binding()), b(7, dilated.as_entire_binding())]));
        groups.push(group(GATHER, &[b(5, view(&half_all)), b(7, dilated.as_entire_binding()), b(8, view(&bg)),
                                    b(9, view(&fg)), b(12, sampler.clone())]));
        groups.push(group(RESOLVE, &[b(0, film.as_entire_binding()), b(3, view(&full)), b(7, dilated.as_entire_binding()),
                                     b(10, view(&bg)), b(11, view(&fg)), b(12, sampler)]));
        DofTargets { w, h, mips, tiles, groups }
    }

    /// Blur the film of `t` in place by `rec` (DofTargets' passes in order: setup, mips, tiles, dilate, gather,
    /// resolve).
    pub(crate) fn encode(&self, enc: &mut wgpu::CommandEncoder, t: &DofTargets, rec: &DofGpu) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dof"), timestamp_writes: None });
        let mut run = |pipe: usize, group: usize, level: u32, size: [u32; 2]| {
            pass.set_pipeline(&self.pipes[pipe]);
            pass.set_bind_group(0, &t.groups[group], &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&DofGpu { level, ..*rec }));
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        };
        run(SETUP, 0, 0, t.mips[0]);
        for l in 1..MIPS as usize {
            run(MIP, l, l as u32, t.mips[l]);
        }
        let g = MIPS as usize;
        run(TILES, g, 0, t.tiles);
        run(DILATE, g + 1, 0, t.tiles);
        run(GATHER, g + 2, 0, t.mips[0]);
        run(RESOLVE, g + 3, 0, [t.w, t.h]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_follows_the_scene_and_the_mode() {
        let off = SceneDof { jittered: false, overblur: 0.05, max_size: 100.0 };
        let on = SceneDof { jittered: true, ..off };
        assert_eq!(split(DofMode::Eevee, &off, 64), (0.0, 1.0));
        let (lens, post) = split(DofMode::Eevee, &on, 64);
        assert!((post - 0.175).abs() < 1e-6 && (lens + post - 1.0).abs() < 1e-6);
        assert_eq!(split(DofMode::Eevee, &on, 1), (0.0, 1.0));
        assert_eq!(split(DofMode::Lens, &on, 64), (1.0, 0.0));
        let s = SceneDof::of(&serde_json::json!({"use_bokeh_jittered": true, "bokeh_overblur": 10.0, "bokeh_max_size": 50.0}));
        assert_eq!(s, SceneDof { jittered: true, overblur: 0.1, max_size: 50.0 });
    }
}
