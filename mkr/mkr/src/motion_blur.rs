//! Motion blur. With `--motion-blur` mkr blurs motion as the scene's EEVEE Next does unless `--params` asks for
//! mkr's own (`motion_blur: "quality"`): EEVEE (motion_blur_steps 1) draws every sample at the shutter's middle and
//! smears it in a post process along each pixel's velocity to where the shutter's open and its close show it
//! (shaders/motion_blur.wgsl: the tiles' longest velocities, their dilation along the blur, the gather), run here on
//! each sample's picture before the film takes it, as EEVEE runs it; `"quality"` draws each sample at its own time in
//! the shutter (true motion blur: what Cycles draws).

use bytemuck::{Pod, Zeroable};

use crate::camera::Cam;
use crate::render::{b, compute_pipe, sized};

const MOTION_BLUR_WGSL: &str = include_str!("shaders/motion_blur.wgsl");
const RECORDS_WGSL: &str = include_str!("shaders/records.wgsl");
/// Pixels across a tile (shaders/motion_blur.wgsl TILE).
pub const TILE: u32 = 32;

/// Where the frame's time sits in the shutter (Blender's motion_blur_position) and the shutter in frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shutter {
    pub length: f32,
    /// Frames from the frame's own time to the shutter's middle: 0 centred, +length/2 opening on the frame (START),
    /// -length/2 closing on it (END).
    pub middle: f32,
}

impl Shutter {
    pub fn new(length: f32, position: &str) -> Shutter {
        let middle = match position {
            "START" => 0.5 * length,
            "END" => -0.5 * length,
            _ => 0.0,
        };
        Shutter { length, middle }
    }

    /// The shutter's open, middle and close, in fractional frames, around frame `k`.
    pub fn times(&self, k: f64) -> [f64; 3] {
        let m = k + self.middle as f64;
        let h = 0.5 * self.length as f64;
        [m - h, m, m + h]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MbGpu {
    size: [u32; 2],
    tiles: [u32; 2],
    sample: u32,
    ortho: u32,
    near: f32,
    far: f32,
    depth_scale: f32,
    noise: f32,
    pad: [u32; 2],
}

/// The post process's record for a W x H frame through `cam` (the shutter's middle).
pub(crate) fn record(cam: &Cam, w: u32, h: u32, depth_scale: f32) -> MbGpu {
    MbGpu {
        size: [w, h],
        tiles: [w.div_ceil(TILE), h.div_ceil(TILE)],
        sample: 0,
        ortho: cam.ortho.is_some() as u32,
        near: cam.near,
        far: cam.far,
        depth_scale,
        noise: 0.0,
        pad: [0; 2],
    }
}

const TILES: usize = 0;
const DILATE: usize = 1;
const GATHER: usize = 2;
const ENTRIES: [&str; 3] = ["tile_max", "dilate", "gather"];

/// The post process's pipelines (made the first time a frame needs them).
pub(crate) struct MotionPost {
    pipes: [wgpu::ComputePipeline; 3],
    layout: wgpu::BindGroupLayout,
}

/// Its buffers and bind groups for one picture size: over the samples' records, the targets' depth and motion G-buffer,
/// and each of the film's sample pictures, blurred into pictures of its own that the film takes instead (`accumulate`:
/// the film's bind group over them).
pub(crate) struct MotionTargets {
    pub(crate) w: u32,
    pub(crate) h: u32,
    tiles: [u32; 2],
    dilated: wgpu::Buffer,
    groups: Vec<wgpu::BindGroup>,
    pub(crate) accumulate: wgpu::BindGroup,
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

fn buffer(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }
}

fn texture(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }
}

impl MotionPost {
    /// The post process's pipelines, gathering `taps` samples per pixel, velocity and half of the shutter.
    pub(crate) fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>, taps: u32) -> MotionPost {
        let source = format!("const TAPS: i32 = {taps};\n{RECORDS_WGSL}\n{MOTION_BLUR_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion blur"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion blur"),
            entries: &[
                entry(0, buffer(true)),
                entry(1, texture(wgpu::TextureSampleType::Depth)),
                entry(2, texture(float)),
                entry(4, texture(float)),
                entry(5, buffer(false)),
                entry(6, buffer(false)),
                entry(7, wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: crate::pipelines::COLOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                entry(8, buffer(false)),
            ],
        });
        let size = std::mem::size_of::<MbGpu>() as u32;
        let pipes = std::array::from_fn(|i| compute_pipe(device, &module, &layout, ENTRIES[i], size, cache));
        MotionPost { pipes, layout }
    }

    /// The buffers and bind groups of a W x H picture: `samples` the samples' records, `depth` and `motion` the targets'
    /// depth and motion G-buffer, `colors` the film's sample pictures; the film's bind group (`accumulate_bgl`: four
    /// pictures, the film, its taps) over the blurred ones.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn targets(&self, device: &wgpu::Device, w: u32, h: u32, samples: &wgpu::Buffer, depth: &wgpu::TextureView,
                          motion: &wgpu::TextureView, colors: &[wgpu::TextureView], accumulate_bgl: &wgpu::BindGroupLayout,
                          film: &wgpu::Buffer, taps: &wgpu::Buffer) -> MotionTargets {
        let tiles = [w.div_ceil(TILE), h.div_ceil(TILE)];
        let n = (tiles[0] * tiles[1]) as u64;
        let tile_buf = sized(device, "motion blur tiles", n * 16, wgpu::BufferUsages::STORAGE);
        let dilated = sized(device, "motion blur dilated", n * 8, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let pixels = sized(device, "motion blur pixels", (w * h) as u64 * 8, wgpu::BufferUsages::STORAGE);
        let outs: Vec<wgpu::TextureView> = colors.iter().map(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("motion blur out"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::pipelines::COLOR_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }).create_view(&Default::default())
        }).collect();
        let view = wgpu::BindingResource::TextureView;
        let groups = colors.iter().zip(&outs).map(|(c, o)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("motion blur"),
                layout: &self.layout,
                entries: &[
                    b(0, samples.as_entire_binding()),
                    b(1, view(depth)),
                    b(2, view(motion)),
                    b(4, view(c)),
                    b(5, tile_buf.as_entire_binding()),
                    b(6, dilated.as_entire_binding()),
                    b(7, view(o)),
                    b(8, pixels.as_entire_binding()),
                ],
            })
        }).collect();
        let accumulate = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate, motion blurred"),
            layout: accumulate_bgl,
            entries: &[b(0, view(&outs[0])), b(1, view(&outs[1])), b(2, view(&outs[2])), b(3, view(&outs[3])),
                       b(4, film.as_entire_binding()), b(5, taps.as_entire_binding())],
        });
        MotionTargets { w, h, tiles, dilated, groups, accumulate }
    }

    /// Blur sample `sample`'s picture (`colors[k]` of `targets`) into the post's own picture `k` by `rec`, `noise` the
    /// sample's random offset.
    pub(crate) fn encode(&self, enc: &mut wgpu::CommandEncoder, t: &MotionTargets, k: usize, rec: &MbGpu, sample: u32,
                         noise: f32) {
        enc.clear_buffer(&t.dilated, 0, None);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("motion blur"), timestamp_writes: None });
        let r = MbGpu { sample, noise, ..*rec };
        pass.set_bind_group(0, &t.groups[k], &[]);
        pass.set_pipeline(&self.pipes[TILES]);
        pass.set_immediates(0, bytemuck::bytes_of(&r));
        pass.dispatch_workgroups(t.tiles[0], t.tiles[1], 1);
        pass.set_pipeline(&self.pipes[DILATE]);
        pass.set_immediates(0, bytemuck::bytes_of(&r));
        pass.dispatch_workgroups(t.tiles[0].div_ceil(8), t.tiles[1].div_ceil(8), 1);
        pass.set_pipeline(&self.pipes[GATHER]);
        pass.set_immediates(0, bytemuck::bytes_of(&r));
        pass.dispatch_workgroups(t.w.div_ceil(8), t.h.div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shutter's open, middle and close around a frame by Blender's motion_blur_position.
    #[test]
    fn the_position_moves_the_shutter() {
        assert_eq!(Shutter::new(0.5, "CENTER").times(10.0), [9.75, 10.0, 10.25]);
        assert_eq!(Shutter::new(0.5, "START").times(10.0), [10.0, 10.25, 10.5]);
        assert_eq!(Shutter::new(0.5, "END").times(10.0), [9.5, 9.75, 10.0]);
    }
}
