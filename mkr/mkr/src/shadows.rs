//! Shadow maps (EEVEE Next's default shadows, `shadow_mode: "map"`): per light that casts shadows, a cube of 90 degree
//! faces around its centre in its own frame (five for area and spot lights: none behind them), or one orthographic
//! face along a sun's direction fitted to the scene; drawn once a frame on the frame's own time. EEVEE draws each cube
//! at the level its receivers on screen ask for (8192 texels across a face at the finest; shaders/shadow_lod.wgsl has
//! the rules measured): mkr keeps every cube face whole at a coarse level (the coarse maps: `face_size` texels across,
//! level log2(8192 / face_size)) and shares a pool of finer windows among the lights (the fine maps, with the suns'
//! maps: `window_size` texels across), which the setup passes (shaders/shadow_setup.wgsl) place over the receivers of
//! the frame's first sample, after its depth prepass and before the maps are drawn. shaders/shadow_map.wgsl reads them
//! back and traces shadow rays through them; its face table is `face_axes` here.
//!
//! The maps' entries (`faces_buf`, a draw's `imm.sample`): the cube faces (coarse layers 0..), then the suns' maps and
//! the windows (fine layers 0..).
//!
//! Casters whose shadow follows their opacity (hashed: DITHERED materials with transparent shadows) cast, in EEVEE,
//! the shadow of a hashed alpha test drawn again for every sample: per texel and sample a caster stops the light with
//! the probability of its alpha (measured: a plate of alpha 0.5 lets through 0.495 of the light at 64 samples, with a
//! texel's noise). mkr draws its maps once a frame, so it keeps that probability instead: each map's depths hold the
//! opaque casters and the hashed casters' opaque parts (R), and the nearest of the hashed casters' partly transparent
//! parts in front of them (G, blended by max), its transparency layer the product of their transparencies (1 - alpha,
//! blended by multiplication). A lookup sees G with the probability of their coverage, drawn per texel and sample.

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// EEVEE's texels across a cube face at its finest level.
pub const LOD0: u32 = 8192;
/// What the lighting reads of a map, per texel: the depths (R the opaque casters' and the hashed casters' opaque
/// parts', G the nearest partly transparent hashed caster's in front of it, blended by max; reverse depth) and the
/// product of those hashed casters' transparencies (blended by multiplication). The depth attachment beside them only
/// tests depth while the maps are drawn.
pub const DEPTHS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
pub const CLEAR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// Bytes per texel of a map: the depth attachment, the depths, the transparency.
const TEXEL_BYTES: u64 = 14;
/// The cube faces' maps never take more than this (their size halves until they fit).
const FACES_BUDGET: u64 = 1 << 30;
/// Per cube face and level, the box of its receivers' texels (shadow_setup.wgsl `tags`).
const TAG_WORDS: u64 = 24;
const SETUP_WGSL: &str = include_str!("shaders/shadow_setup.wgsl");

/// One entry of the maps: world -> clip (cubes: reverse-Z with an infinite far plane; suns: reverse linear depth).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowFaceGpu {
    pub vp: [[f32; 4]; 4],
}

/// A light's place in the maps: its first entry and how many (cube faces in the coarse maps, or a sun's map in the
/// fine maps).
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub first: u32,
    pub faces: u32,
    pub sun: bool,
}

/// The passes that place the windows.
struct Setup {
    layout: wgpu::BindGroupLayout,
    tag: wgpu::ComputePipeline,
    place: wgpu::ComputePipeline,
}

/// One array of maps: per layer the depth attachment, the depths and the transparency (DEPTHS_FORMAT); the arrays
/// the lighting reads and one view per layer (what each map's passes draw into).
pub struct MapArray {
    pub size: u32,
    pub depths: wgpu::TextureView,
    pub clear: wgpu::TextureView,
    pub layers: Vec<[wgpu::TextureView; 3]>,
}

pub struct ShadowMaps {
    /// The coarse maps (the cube faces whole) and the fine ones (the suns' maps, then the windows).
    pub coarse: MapArray,
    pub fine: MapArray,
    /// The level of the whole faces' maps (EEVEE's levels: 8192 texels across at 0).
    pub base_lod: u32,
    /// Per light of the bake (its order): where its maps are, if it casts shadows.
    pub slots: Vec<Option<Slot>>,
    /// Entries: cube faces [0, faces), suns' maps [faces, faces + suns), windows [faces + suns, + windows).
    pub faces: u32,
    pub suns: u32,
    pub windows: u32,
    /// The entries' matrices: the faces' and suns' written every frame by render.rs, the windows' by the setup passes.
    pub faces_buf: wgpu::Buffer,
    /// Per cube face: its window (origin in its level's texels, level, fine layer; -1: none), written by the setup.
    pub windows_buf: wgpu::Buffer,
    /// Per face and level: the box of its receivers, cleared every frame.
    tags_buf: wgpu::Buffer,
    setup: Option<Setup>,
    /// 1 x 1 stand-ins bound while the maps themselves are drawn into.
    pub dummy: MapArray,
}

/// A face's uv and depth axes in the light's frame (right, up, fwd), as shaders/shadow_map.wgsl numbers them.
pub fn face_axes(face: u32, r: Vec3, u: Vec3, f: Vec3) -> (Vec3, Vec3, Vec3) {
    match face {
        0 => (r, u, f),
        1 => (-f, u, r),
        2 => (f, u, -r),
        3 => (r, -f, u),
        4 => (r, f, -u),
        _ => (-r, u, -f),
    }
}

/// A cube face's world -> clip: x = s.d, y = t.d, z = near, w = a.d for d = p - c: uv = (s.d, t.d) / a.d, depth = near /
/// a.d (reverse-Z, infinite far).
pub fn face_matrix(c: Vec3, s: Vec3, t: Vec3, a: Vec3, near: f32) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(s.x, t.x, 0.0, a.x),
        Vec4::new(s.y, t.y, 0.0, a.y),
        Vec4::new(s.z, t.z, 0.0, a.z),
        Vec4::new(-s.dot(c), -t.dot(c), near, -a.dot(c)),
    )
}

/// A sun's maps: the finest first, each over the slice of the view from the camera to SUN_NEAR x 8^k metres (k < the
/// last; sun_cascade), the last over the whole scene (sun_matrix). A receiver reads the finest one around it
/// (shaders/shadow_map.wgsl), as EEVEE's sun reads its clipmap's level around the camera at the receiver's distance: a
/// map of the whole scene alone keeps no shadow of a figure in a scene a kilometre across.
pub const SUN_CASCADES: u32 = 4;
pub const SUN_NEAR: f32 = 3.0;

/// A sun's map: orthographic along `f` over the box `lo`..`hi` (world), x and y in [-1, 1] across `extent`, depth 1 at
/// the side toward the sun falling linearly to 0 at `span` past it. Returns (matrix, span, texel size for `size`).
pub fn sun_matrix(f: Vec3, r: Vec3, u: Vec3, lo: Vec3, hi: Vec3, size: u32) -> (Mat4, f32, f32) {
    let centre = (lo + hi) * 0.5;
    let half = (hi - lo).length() * 0.5 + 1e-3;
    sun_square(f, r, u, centre, half, r.dot(centre), u.dot(centre), half, size)
}

/// A finer map of a sun over the points `pts` (a slice of the view): sun_matrix's depth (the box lo..hi along `f`, so
/// every caster between the sun and them is drawn and the depth span is the same), x and y over the square around the
/// points' extent across `f` (no more than the box's).
pub fn sun_cascade(f: Vec3, r: Vec3, u: Vec3, lo: Vec3, hi: Vec3, pts: &[Vec3], size: u32) -> (Mat4, f32, f32) {
    let centre = (lo + hi) * 0.5;
    let half = (hi - lo).length() * 0.5 + 1e-3;
    let (bx, by) = (r.dot(centre), u.dot(centre));
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in pts {
        (x0, x1) = (x0.min(r.dot(*p)), x1.max(r.dot(*p)));
        (y0, y1) = (y0.min(u.dot(*p)), y1.max(u.dot(*p)));
    }
    (x0, x1, y0, y1) = (x0.max(bx - half), x1.min(bx + half), y0.max(by - half), y1.min(by + half));
    if !(x0 < x1 && y0 < y1) {
        return sun_matrix(f, r, u, lo, hi, size);
    }
    let ext = 0.5 * (x1 - x0).max(y1 - y0) + 1e-3;
    sun_square(f, r, u, centre, half, 0.5 * (x0 + x1), 0.5 * (y0 + y1), ext, size)
}

/// The orthographic map along `f` whose depth runs `half` either side of `centre` and whose square is `ext` either side
/// of (cx, cy) on the (r, u) axes.
#[allow(clippy::too_many_arguments)]
fn sun_square(f: Vec3, r: Vec3, u: Vec3, centre: Vec3, half: f32, cx: f32, cy: f32, ext: f32, size: u32)
              -> (Mat4, f32, f32) {
    // the near plane sits `half` before the centre (toward the sun), the far one `half` after
    let span = 2.0 * half;
    let near_c = centre - f * half;
    // depth = 1 - f.(p - near_c) / span
    let m = Mat4::from_cols(
        Vec4::new(r.x / ext, u.x / ext, -f.x / span, 0.0),
        Vec4::new(r.y / ext, u.y / ext, -f.y / span, 0.0),
        Vec4::new(r.z / ext, u.z / ext, -f.z / span, 0.0),
        Vec4::new(-cx / ext, -cy / ext, 1.0 + f.dot(near_c) / span, 1.0),
    );
    (m, span, 2.0 * ext / size as f32)
}

/// The cube faces' size: `size`, halved while `faces` of them would take more than FACES_BUDGET (16 at least).
pub fn fitted_face_size(faces: u32, size: u32) -> u32 {
    let mut s = size;
    while s > 16 && faces as u64 * (s as u64).pow(2) * TEXEL_BYTES > FACES_BUDGET {
        s /= 2;
    }
    s
}

impl MapArray {
    fn new(device: &wgpu::Device, label: &str, size: u32, layers_n: u32) -> MapArray {
        let tex = |what: &str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{label}: {what}")),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers_n.max(1) },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let texs = [tex("depth", crate::pipelines::SHADOW_FORMAT), tex("depths", DEPTHS_FORMAT), tex("transparency", CLEAR_FORMAT)];
        let array = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
        };
        let layer = |t: &wgpu::Texture, l: u32| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: l,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        MapArray {
            size,
            depths: array(&texs[1]),
            clear: array(&texs[2]),
            layers: (0..layers_n).map(|l| [layer(&texs[0], l), layer(&texs[1], l), layer(&texs[2], l)]).collect(),
        }
    }

    /// The (depths, transparency) array views the lighting reads.
    pub fn views(&self) -> [&wgpu::TextureView; 2] {
        [&self.depths, &self.clear]
    }
}

impl ShadowMaps {
    /// Room for every shadow-casting light of `kinds` (the bake's light types, in order; None: casts no shadow): cube
    /// faces of `face_size` texels across (halved while they would take more than FACES_BUDGET), suns' maps and a pool
    /// of `windows` windows of `window_size` (powers of two up to 8192; the windows finer than the faces).
    pub fn new(device: &wgpu::Device, kinds: &[Option<&str>], window_size: u32, face_size: u32, windows: u32,
               cache: Option<&wgpu::PipelineCache>) -> Result<ShadowMaps> {
        for (what, s) in [("shadow_map_size", window_size), ("shadow_face_size", face_size)] {
            if !s.is_power_of_two() || !(16..=LOD0).contains(&s) {
                bail!("{what} {s}: a power of two from 16 to {LOD0}");
            }
        }
        let mut slots = Vec::with_capacity(kinds.len());
        let (mut faces_n, mut suns_n) = (0u32, 0u32);
        for k in kinds {
            slots.push(k.map(|kind| match kind {
                "SUN" => {
                    suns_n += SUN_CASCADES;
                    Slot { first: suns_n - SUN_CASCADES, faces: SUN_CASCADES, sun: true }
                }
                _ => {
                    let faces = if kind == "POINT" { 6 } else { 5 };
                    faces_n += faces;
                    Slot { first: faces_n - faces, faces, sun: false }
                }
            }));
        }
        for s in slots.iter_mut().flatten().filter(|s| s.sun) {
            s.first += faces_n;
        }
        let face_size = fitted_face_size(faces_n, face_size);
        let base_lod = (LOD0 / face_size).trailing_zeros();
        // windows only where they are finer than the faces, and only for cube lights
        let windows_n = if faces_n == 0 || window_size <= face_size { 0 } else { windows };
        let coarse = MapArray::new(device, "shadow maps (cube faces)", if faces_n == 0 { 1 } else { face_size }, faces_n);
        let fine = MapArray::new(device, "shadow maps (suns, windows)", if suns_n + windows_n == 0 { 1 } else { window_size },
                                 suns_n + windows_n);
        let dummy = MapArray::new(device, "no shadow maps", 1, 1);
        let entries = faces_n + suns_n + windows_n;
        let buffer = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage, mapped_at_creation: false })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let faces_buf = buffer("shadow faces", entries.max(1) as u64 * std::mem::size_of::<ShadowFaceGpu>() as u64, storage);
        // every face starts without a window (layer -1) until the setup passes place them
        let windows_buf = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
            label: Some("shadow windows"),
            contents: &vec![0xff; faces_n.max(1) as usize * 16],
            usage: storage,
        });
        let tags_buf = buffer("shadow tags", faces_n.max(1) as u64 * TAG_WORDS * 4, storage);
        let setup = (windows_n > 0).then(|| Setup::new(device, cache));
        Ok(ShadowMaps { coarse, fine, base_lod, slots, faces: faces_n, suns: suns_n, windows: windows_n, faces_buf,
                        windows_buf, tags_buf, setup, dummy })
    }

    /// The views an entry's passes draw into: (depth, hashed, transparency).
    pub fn entry_views(&self, entry: u32) -> &[wgpu::TextureView; 3] {
        if entry < self.faces { &self.coarse.layers[entry as usize] } else { &self.fine.layers[(entry - self.faces) as usize] }
    }

    /// The windows' entries.
    pub fn window_entries(&self) -> std::ops::Range<u32> {
        self.faces + self.suns..self.faces + self.suns + self.windows
    }

    /// The frame record's `shadow_pool`: the first fine entry, the first window, how many, a window's texels across.
    pub fn pool_record(&self) -> [f32; 4] {
        [self.faces as f32, (self.faces + self.suns) as f32, self.windows as f32, self.fine.size as f32]
    }

    /// The setup passes' bind group over the frame's records and its first sample's depth (None: no windows).
    pub fn setup_bind_group(&self, device: &wgpu::Device, frame: &wgpu::Buffer, samples: &wgpu::Buffer, lights: &wgpu::Buffer,
                            depth: &wgpu::TextureView) -> Option<wgpu::BindGroup> {
        let s = self.setup.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow setup"),
            layout: &s.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: samples.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 4, resource: self.tags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.windows_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.faces_buf.as_entire_binding() },
            ],
        }))
    }

    /// Place the windows for a `w` x `h` frame (its first sample's depth drawn, its faces' matrices and lights written).
    pub fn encode_setup(&self, enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, w: u32, h: u32) {
        let Some(s) = &self.setup else { return };
        enc.clear_buffer(&self.tags_buf, 0, None);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("shadow windows"), timestamp_writes: None });
        pass.set_bind_group(0, bg, &[]);
        pass.set_pipeline(&s.tag);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        pass.set_pipeline(&s.place);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// The cube faces' and suns' entries (what render.rs fills each frame).
    pub fn face_count(&self) -> usize {
        (self.faces + self.suns) as usize
    }

    /// Every entry, windows included.
    pub fn entry_count(&self) -> usize {
        (self.faces + self.suns + self.windows) as usize
    }
}

impl Setup {
    fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> Setup {
        let source = format!("{}\n{}\n{}", crate::pipelines::RECORDS_WGSL, crate::pipelines::SHADOW_LOD_WGSL, SETUP_WGSL);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow setup"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let cs = wgpu::ShaderStages::COMPUTE;
        let buf = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow setup"),
            entries: &[
                buf(0, true),
                buf(1, true),
                buf(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: cs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buf(4, false),
                buf(5, false),
                buf(6, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow setup"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache,
            })
        };
        Setup { tag: pipe("shadow_tag"), place: pipe("shadow_windows"), layout }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's lookup (shaders/shadow_map.wgsl shadow_face_of, in Rust): face, uv, distance along its axis.
    fn lookup(d: Vec3, r: Vec3, u: Vec3, f: Vec3) -> (u32, f32, f32, f32) {
        let (x, y, z) = (d.dot(r), d.dot(u), d.dot(f));
        let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
        let (face, uv, a) = if az >= ax && az >= ay {
            if z > 0.0 { (0, (x, y), az) } else { (5, (-x, y), az) }
        } else if ax >= ay {
            if x > 0.0 { (1, (-z, y), ax) } else { (2, (z, y), ax) }
        } else if y > 0.0 {
            (3, (x, -z), ay)
        } else {
            (4, (x, z), ay)
        };
        (face, uv.0 / a, uv.1 / a, a)
    }

    /// The faces' size halves while they would take more than the budget.
    #[test]
    fn many_faces_get_smaller() {
        assert_eq!(fitted_face_size(30, 1024), 1024);          // 30 x 1024² x 14 B = 0.44 GB
        assert_eq!(fitted_face_size(200, 1024), 512);          // 2.9 GB at 1024, 0.73 GB at 512
        assert_eq!(fitted_face_size(300, 1024), 256);
        assert_eq!(fitted_face_size(1_000_000, 64), 16);       // never below 16
    }

    #[test]
    fn faces_draw_where_the_lookup_reads() {
        // a tilted light frame
        let f = Vec3::new(0.3, -0.9, 0.2).normalize();
        let r = f.cross(Vec3::Y).normalize();
        let u = r.cross(f);
        let c = Vec3::new(1.0, 3.0, -2.0);
        let near = 0.01;
        for (i, d) in [Vec3::new(0.2, -1.0, 0.1), Vec3::new(1.0, 0.3, 0.2), Vec3::new(-0.7, 0.1, 0.9),
                       Vec3::new(0.1, 1.0, -0.3), Vec3::new(0.05, 0.02, -1.0), Vec3::new(-0.4, -0.5, -0.45)]
            .iter().enumerate() {
            let p = c + *d * 2.5;
            let (face, u_, v_, a) = lookup(p - c, r, u, f);
            let (s, t, ax) = face_axes(face, r, u, f);
            let clip = face_matrix(c, s, t, ax, near) * p.extend(1.0);
            assert!((clip.x / clip.w - u_).abs() < 1e-5, "case {i}: x");
            assert!((clip.y / clip.w - v_).abs() < 1e-5, "case {i}: y");
            assert!((clip.z / clip.w - near / a).abs() < 1e-6, "case {i}: depth");
            assert!(u_.abs() <= 1.0 + 1e-6 && v_.abs() <= 1.0 + 1e-6, "case {i}: inside its face");
        }
    }

    #[test]
    fn sun_depth_falls_away_from_the_sun() {
        let f = Vec3::new(0.0, -1.0, 0.0);
        let (r, u) = (Vec3::X, Vec3::Z);
        let (m, span, texel) = sun_matrix(f, r, u, Vec3::splat(-1.0), Vec3::splat(1.0), 1024);
        let top = m * Vec3::new(0.0, 1.0, 0.0).extend(1.0);
        let bottom = m * Vec3::new(0.0, -1.0, 0.0).extend(1.0);
        assert!(top.z > bottom.z, "nearer the sun is deeper (reverse)");
        assert!(top.z <= 1.0 + 1e-6 && bottom.z >= 0.0);
        assert!(span > 3.4 && texel > 0.0);
    }

    #[test]
    fn a_sun_cascade_covers_its_slice_finer_than_the_scene_map() {
        let (f, r, u) = (Vec3::new(0.0, -1.0, 0.0), Vec3::X, Vec3::Z);
        let (lo, hi) = (Vec3::splat(-1000.0), Vec3::splat(1000.0));
        let pts = [Vec3::new(1.0, 0.0, 1.0), Vec3::new(3.0, 0.5, 2.0), Vec3::new(2.0, 1.0, 4.0)];
        let (whole, span0, texel0) = sun_matrix(f, r, u, lo, hi, 2048);
        let (m, span, texel) = sun_cascade(f, r, u, lo, hi, &pts, 2048);
        assert_eq!(span, span0, "every map of a sun spans the same depth");
        assert!(texel < texel0 / 100.0, "{texel} against the scene map's {texel0}");
        for p in pts {
            let c = m * p.extend(1.0);
            assert!(c.x.abs() <= 1.0 && c.y.abs() <= 1.0, "{p} falls outside its cascade: {c}");
            // the depth runs as the scene map's does
            assert!((c.z - (whole * p.extend(1.0)).z).abs() < 1e-6);
        }
    }
}
