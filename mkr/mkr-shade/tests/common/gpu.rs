//! A small wgpu harness that evaluates compiled materials over an n x n grid of shading points, as the lab's Blender
//! scenes lay them out (`lab/blender_lab.py`): quad (i, j) is pixel (i, j). Shared by the lab example and the tests.
#![allow(dead_code)]

use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;
/// What the fragment returns for each quad.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The material's emission (an Emission shader of strength 1 shows its colour); shading normal +Z.
    Field = 0,
    /// Every closure lit by a uniform white world as EEVEE lights it: diffuse times the measured diffuse response of
    /// that row, reflection times the measured specular response of that quad (rgb), coat, emission, transmittance, and
    /// translucent and refraction squared (EEVEE's world light through them picks the colour up twice). The shading
    /// normal tilts per row (N.I = rows[j]).
    Closure = 1,
    /// The emission, with the shading normal tilted per row as for closures.
    Shade = 2,
    /// Closure flags: (translucent_slab, holdout, opacity); shading normal +Z.
    Flags = 3,
}

/// Render pixels per quad side, as the lab's Blender scenes render them.
pub const K: u32 = 5;

const HARNESS: &str = r#"
struct LabParams {
  n: u32,
  kind: u32,
  k: u32,
  pad1: u32,
  attr_src: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read> lab_in: array<f32>;
@group(0) @binding(1) var<uniform> lab: LabParams;
@group(0) @binding(2) var lut_tex: texture_2d_array<f32>;
@group(0) @binding(3) var lut_samp: sampler;
@group(0) @binding(4) var<storage, read> lab_rows: array<f32>;
@group(0) @binding(5) var lab_img: texture_2d<f32>;

fn mkr_lut(uv: vec2f, layer: u32) -> vec4f {
  return textureSampleLevel(lut_tex, lut_samp, uv, i32(layer), 0.0);
}

// The lab's one image, texel (x, y) with rows from the top of the file, under extension `ext` (0 REPEAT, 1 EXTEND,
// 2 CLIP: transparent black outside, 3 MIRROR).
fn lab_texel(ix: i32, iy: i32, ext: u32) -> vec4f {
  let size = vec2i(textureDimensions(lab_img));
  var x = ix;
  var y = iy;
  if (ext == 0u) {
    x = ((x % size.x) + size.x) % size.x;
    y = ((y % size.y) + size.y) % size.y;
  } else if (ext == 1u) {
    x = clamp(x, 0, size.x - 1);
    y = clamp(y, 0, size.y - 1);
  } else if (ext == 2u) {
    if (x < 0 || y < 0 || x >= size.x || y >= size.y) {
      return vec4f(0.0);
    }
  } else {
    let px = 2 * size.x;
    let py = 2 * size.y;
    x = ((x % px) + px) % px;
    y = ((y % py) + py) % py;
    if (x >= size.x) {
      x = px - 1 - x;
    }
    if (y >= size.y) {
      y = py - 1 - y;
    }
  }
  return textureLoad(lab_img, vec2i(x, y), 0);
}

// Blender's uv (v up) on the image: Closest takes the texel under it, Linear blends the four nearest (the GPU's
// bilinear filter, extension applied per texel).
fn mkr_image(index: u32, uv: vec2f, interp: u32, ext: u32) -> vec4f {
  let size = vec2f(textureDimensions(lab_img));
  let t = vec2f(uv.x, 1.0 - uv.y) * size;
  if (interp == 1u) {
    return lab_texel(i32(floor(t.x)), i32(floor(t.y)), ext);
  }
  let p = t - 0.5;
  let i0 = floor(p);
  let f = p - i0;
  let x = i32(i0.x);
  let y = i32(i0.y);
  let a = mix(lab_texel(x, y, ext), lab_texel(x + 1, y, ext), f.x);
  let b = mix(lab_texel(x, y + 1, ext), lab_texel(x + 1, y + 1, ext), f.x);
  return mix(a, b, f.y);
}
fn mkr_image_lod(index: u32, uv: vec2f, interp: u32, ext: u32) -> vec4f { return mkr_image(index, uv, interp, ext); }
fn mkr_param(slot: u32) -> vec4f { return vec4f(0.0); }
fn mkr_shader_to_rgb(c: Closures, s: Surface) -> vec4f { return vec4f(c.emission, 1.0); }

// src: 0..2 the quad's lab_a..lab_c, 10 the grid's UV map at the pixel (u, v in 0..1 over the grid).
fn lab_attr(src: u32, q: u32, uv: vec2f) -> vec4f {
  if (src < 3u) {
    let o = q * 9u + src * 3u;
    return vec4f(lab_in[o], lab_in[o + 1u], lab_in[o + 2u], 1.0);
  }
  if (src == 10u) {
    return vec4f(uv, 0.0, 1.0);
  }
  return vec4f(0.0);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

// Pixel (x, y) of an (n k) x (n k) image, rows from the grid's bottom: quad (x / k, y / k).
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  let n = lab.n;
  let nf = f32(n);
  let kf = f32(lab.k);
  let qi = x / lab.k;
  let qj = y / lab.k;
  let q = qj * n + qi;
  let gx = (f32(x) + 0.5) / kf;
  let gy = (f32(y) + 0.5) / kf;
  let uv = vec2f(gx / nf, gy / nf);
  var s: Surface;
  s.P = vec3f(gx - nf * 0.5, gy - nf * 0.5, 0.0);
  s.Ng = vec3f(0.0, 0.0, 1.0);
  s.N = s.Ng;
  if (lab.kind == 1u || lab.kind == 2u) {
    let c = lab_rows[qj];
    s.N = vec3f(sqrt(max(0.0, 1.0 - c * c)), 0.0, c);
  }
  s.I = vec3f(0.0, 0.0, 1.0);
  s.backfacing = 0.0;
  s.generated = vec3f(uv, 0.5);
  for (var k = 0u; k < 8u; k++) {
    s.attrs[k] = lab_attr(lab.attr_src[k / 4u][k % 4u], q, uv);
  }
  s.object = mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0),
                     vec4f(0.0, 0.0, 0.0, 1.0));
  s.object_inv = s.object;
  s.camera = mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0),
                     vec4f(0.0, 0.0, 10.0, 1.0));
  s.camera_inv = mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0),
                         vec4f(0.0, 0.0, -10.0, 1.0));
  s.ortho = 1.0;
  s.screen = uv;
  s.rand = vec4f(0.5);
  s.ray_type = 0u;
  s.object_color = vec4f(1.0);
  s.object_info = vec4f(0.0);
  let c = mat_lab(s);
  if (lab.kind == 1u) {
    let go = n + qj * 3u;
    let gain = vec3f(lab_rows[go], lab_rows[go + 1u], lab_rows[go + 2u]);
    let so = 4u * n + q * 3u;
    let spec = vec3f(lab_rows[so], lab_rows[so + 1u], lab_rows[so + 2u]);
    let lit = c.diffuse * gain + c.translucent * c.translucent + c.reflection * spec + c.coat +
              c.refraction * c.refraction + c.emission + c.transmittance;
    return vec4f(lit, 1.0);
  }
  if (lab.kind == 3u) {
    return vec4f(c.translucent_slab, c.holdout, mks_opacity(c), 1.0);
  }
  return vec4f(c.emission, 1.0);
}
"#;
/// An 8-bit RGBA image, rows from the top, uploaded as sRGB (hardware decode) or as is.
pub struct LabImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub srgb: bool,
}


pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    lut_view: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pub adapter: String,
}

/// Which lab input feeds each `Surface.attrs` slot of `m` (0..2: lab_a..lab_c, 10: the grid's UV map, 99: none).
pub fn attr_sources(m: &mkr_shade::CompiledMaterial) -> [u32; 8] {
    let mut src = [99u32; 8];
    for (k, a) in m.attributes.iter().enumerate().take(8) {
        src[k] = match a.as_str() {
            "attr:lab_a" => 0,
            "attr:lab_b" => 1,
            "attr:lab_c" => 2,
            "uv:" | "uv:UVMap" => 10,
            _ => 99,
        };
    }
    src
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no Vulkan GPU adapter: {e}"))?;
        let features = adapter.features() & wgpu::Features::FLOAT32_FILTERABLE;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mkr-shade lab"),
            required_features: features,
            ..Default::default()
        }))?;
        let lut = mkr_shade::lut();
        let size = wgpu::Extent3d { width: lut.size, height: lut.size, depth_or_array_layers: lut.layers };
        let tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("lut"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&lut.data),
        );
        let lut_view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let filterable = features.contains(wgpu::Features::FLOAT32_FILTERABLE);
        let filter = if filterable { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lut"),
            mag_filter: filter,
            min_filter: filter,
            ..Default::default()
        });
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lab"),
            entries: &[
                storage(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(if filterable {
                        wgpu::SamplerBindingType::Filtering
                    } else {
                        wgpu::SamplerBindingType::NonFiltering
                    }),
                    count: None,
                },
                storage(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lab"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let adapter_name = adapter.get_info().name;
        Ok(Gpu { device, queue, lut_view, lut_sampler, layout, pipeline_layout, adapter: adapter_name })
    }

    /// Evaluate material `m` (compiled with id "lab") on an n x n grid. `inputs`: n * n * 9 floats (lab_a, lab_b,
    /// lab_c per quad); `rows`: per row N.I (closure kind), per row the world's diffuse response (rgb), per quad the
    /// world's specular response (rgb); missing values are 1. `image`: the material's image 0, if any.
    /// Returns n * n * 4 floats, quad (i, j) at (j * n + i) * 4.
    pub fn eval(
        &self,
        m: &mkr_shade::CompiledMaterial,
        n: u32,
        kind: Kind,
        inputs: &[f32],
        rows: &[f32],
        image: Option<&LabImage>,
    ) -> Result<Vec<f32>> {
        let wgsl = format!("{}{}{}", mkr_shade::PRELUDE, m.wgsl, HARNESS);
        let d = &self.device;
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lab"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let format = wgpu::TextureFormat::Rgba32Float;
        let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lab"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(anyhow!("pipeline: {e}"));
        }
        let mut params = vec![n, kind as u32, K, 0];
        params.extend_from_slice(&attr_sources(m));
        let pbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut inputs = inputs.to_vec();
        inputs.resize((n * n * 9) as usize, 0.0);
        let ibuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inputs"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut rows = rows.to_vec();
        rows.resize((4 * n + 3 * n * n) as usize, 1.0);
        let rbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rows"),
            contents: bytemuck::cast_slice(&rows),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy = LabImage { width: 1, height: 1, rgba: vec![255, 0, 255, 255], srgb: false };
        let img = image.unwrap_or(&dummy);
        let img_tex = d.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("image"),
                size: wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if img.srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &img.rgba,
        );
        let img_view = img_tex.create_view(&Default::default());
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lab"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ibuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.lut_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: rbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&img_view) },
            ],
        });
        let size = n * K;
        let target = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let row_bytes = (size * 16).div_ceil(256) * 256;
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row_bytes * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lab"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(size) },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        d.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("poll: {e}"))?;
        let data = slice.get_mapped_range().map_err(|e| anyhow!("map: {e:?}"))?;
        // Each quad's centre pixel, as the lab samples Blender's renders.
        let mut out = Vec::with_capacity((n * n * 4) as usize);
        for j in 0..n {
            let y = j * K + K / 2;
            let row: &[f32] = bytemuck::cast_slice(&data[(y * row_bytes) as usize..(y * row_bytes + size * 16) as usize]);
            for i in 0..n {
                let x = (i * K + K / 2) as usize;
                out.extend_from_slice(&row[x * 4..x * 4 + 4]);
            }
        }
        drop(data);
        readback.unmap();
        Ok(out)
    }
}

/// EEVEE's film: negative and NaN radiance land as 0.
pub fn film(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.max(0.0) }
}

/// Error of `ours` against a reference read from EEVEE's half-float film: absolute below 1, relative above.
pub fn film_error(ours: f32, reference: f32) -> f64 {
    let a = film(ours) as f64;
    let b = reference as f64;
    if !a.is_finite() {
        return f64::INFINITY;
    }
    (a - b).abs() / b.abs().max(1.0)
}
