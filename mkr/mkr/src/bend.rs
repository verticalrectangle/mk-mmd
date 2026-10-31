//! Normals that bend with a deforming mesh as Blender's do. Blender gives every corner of a mesh the normal of its fan
//! (the corners round a vertex joined across its smooth edges) and rebuilds it as the mesh deforms; the bake carries,
//! for every mesh that bends (skinned or shape-keyed), what that rebuild needs (bake_lit's MESH `bend`: the docstring of
//! mkmmd/blender/build/bake_lit.py), and mkmmd/core/bend.py's `normals` is the reference this module follows:
//!
//! - per face n_f = the unit sum of its triangles' cross products;
//! - per fan lnor = the unit sum over its corners of n_f times the face's angle at the corner, ra = the mean angle
//!   between lnor and its edges, ref = its reference edge flattened onto lnor's plane, ortho = lnor x ref, span = the
//!   angle round lnor from ref to its other edge flattened alike (a whole turn when the two are one edge);
//! - per vertex a = alpha ra (alpha < 0: 2 pi + alpha (2 pi - ra)), b = beta span and
//!   normal = cos(a) lnor + sin(a) (cos(b) ref + sin(b) ortho).
//!
//! `Bend` does it on the GPU on every frame (shaders/bend.wgsl): after the skin pass, two compute passes over every
//! bending mesh at once rewrite the normals of the frame's posed vertices, one per face, one per fan (which writes its
//! vertices' normals). `normals` is the same rebuild on the CPU in f64, bend.py step by step: the reference the GPU's
//! f32 one is tested against. The GPU evaluates the same maths in forms f32 keeps exact (the shader's header: angles
//! as atan2, a triangle's cross product from its two shortest edges); measured on the lab's MMD model and on Crimson's,
//! it lands within 0.01 deg of `normals` on every vertex.
//!
//! The posed vertices are in the world where bend.py's are in the object's space: the rebuild reads angles, which the
//! world matrix keeps (rotation, uniform scale and translation: no non-uniform scale in MMD rigs, as for skin.wgsl's
//! normals). A mirroring matrix (negative determinant) keeps them too but turns every cross product around: such a
//! mesh turns its lnor and its ortho axis back (`mirrored`), so its normals are the mirror image of the object-space
//! ones (bend.py's meshes are never mirrored).

use std::ops::Range;

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat3};
use serde::Deserialize;
use wgpu::util::DeviceExt;

use crate::bake::{AttrRec, Bake, MeshRec};
use crate::camera::affine;
use crate::render::{sized, storage};
use crate::scene::{Geometry, Shape, ShapeKind};

const BEND_WGSL: &str = include_str!("shaders/bend.wgsl");
const TURN: f64 = std::f64::consts::TAU;
/// Invocations per workgroup (the shaders' @workgroup_size), and the workgroups a dispatch dimension surely may have.
const GROUP: u32 = 64;
const MAX_GROUPS: u32 = 65535;
/// FanGpu::flags: the mesh's matrix mirrors.
const MIRRORED: u32 = 1;

// =================================================================================================== the bake's record
/// A list per key in lit.bin: offsets BYTE u32 [count + 1], items BYTE u32 [size].
#[derive(Deserialize, Debug, Clone)]
struct ListRec {
    count: usize,
    offsets: usize,
    items: usize,
    size: usize,
}

/// bake_lit's MESH `bend`: per fan (BYTE u32 [fans]) a vertex at its point, at the far end of its reference edge and of
/// its other edge; per fan its corners and its edges, per face its triangles.
#[derive(Deserialize, Debug, Clone)]
struct BendRec {
    fans: usize,
    center: usize,
    #[serde(rename = "ref")]
    reference: usize,
    other: usize,
    corners: ListRec,
    edges: ListRec,
    faces: ListRec,
}

/// Items by key: key i's run is `offsets[i]..offsets[i + 1]` (of entries that may be several items each).
#[derive(Clone, Debug, Default)]
pub struct Lists {
    pub offsets: Vec<u32>,
    pub items: Vec<u32>,
}

impl Lists {
    pub fn keys(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn run(&self, key: usize) -> Range<usize> {
        self.offsets[key] as usize..self.offsets[key + 1] as usize
    }
}

/// A bending mesh's rebuild data; vertices, faces and triangles are numbered within the mesh.
#[derive(Clone, Debug, Default)]
pub struct MeshBend {
    /// Per vertex: its fan, alpha (its angle from the fan's lnor in units of ra; negative: past it, counted from a
    /// whole turn) and beta (its angle round lnor from ref in units of span).
    pub fan: Vec<u32>,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    /// Per fan: a vertex at its point, at the far end of its reference edge, at the far end of its other edge.
    pub center: Vec<u32>,
    pub reference: Vec<u32>,
    pub other: Vec<u32>,
    /// Per fan its corners, three items each: the face, the previous vertex and the next one (runs count corners).
    pub corners: Lists,
    /// Per fan its edges' far-end vertices.
    pub edges: Lists,
    /// Per face its triangles (indices into the mesh's triangle list).
    pub faces: Lists,
}

fn words(bake: &Bake, offset: usize, n: usize) -> Result<Vec<u32>> {
    Ok(bake.bytes(offset, n * 4)?.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn floats(bake: &Bake, offset: usize, n: usize) -> Result<Vec<f32>> {
    Ok(bake.bytes(offset, n * 4)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn attribute<'a>(rec: &'a MeshRec, name: &str, ty: &str, comp: usize) -> Result<&'a AttrRec> {
    let a = rec.attributes.get(name).with_context(|| format!("no {name} attribute"))?;
    if a.ty != ty || a.comp != comp {
        bail!("its {name} attribute is {} x {}, not {comp} x {ty}", a.comp, a.ty);
    }
    Ok(a)
}

/// A list of `rec.count` keys, each entry `per` items, checked to run over its items in order.
fn lists(bake: &Bake, rec: &ListRec, per: usize, what: &str) -> Result<Lists> {
    let offsets = words(bake, rec.offsets, rec.count + 1)?;
    let items = words(bake, rec.items, rec.size)?;
    if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] > w[1]) || offsets[rec.count] as usize * per != rec.size {
        bail!("its {what} do not run over their {} items", rec.size);
    }
    Ok(Lists { offsets, items })
}

fn below<'a>(items: impl IntoIterator<Item = &'a u32>, n: usize, what: &str) -> Result<()> {
    match items.into_iter().find(|&&i| i as usize >= n) {
        Some(i) => bail!("{what} {i} of {n}"),
        None => Ok(()),
    }
}

impl MeshBend {
    /// The bend data of the bake's mesh `rec` (None: it does not bend), checked against its vertices and triangles.
    pub fn read(bake: &Bake, rec: &MeshRec) -> Result<Option<MeshBend>> {
        let Some(raw) = rec.bend.as_ref().filter(|v| !v.is_null()) else { return Ok(None) };
        let older = || format!("mesh {}: its bend record is from an older bake: rebake (mk build --bake)", rec.name);
        let b = BendRec::deserialize(raw).with_context(older)?;
        if !rec.attributes.contains_key("fan") || !rec.attributes.contains_key("bend") {
            bail!(older());
        }
        Self::load(bake, rec, &b).with_context(|| format!("mesh {}: its bend record", rec.name)).map(Some)
    }

    fn load(bake: &Bake, rec: &MeshRec, b: &BendRec) -> Result<MeshBend> {
        let nv = rec.vertices;
        let fan = words(bake, attribute(rec, "fan", "u32", 1)?.offset, nv)?;
        let ab = floats(bake, attribute(rec, "bend", "f32", 2)?.offset, 2 * nv)?;
        let m = MeshBend {
            fan,
            alpha: ab.iter().step_by(2).copied().collect(),
            beta: ab.iter().skip(1).step_by(2).copied().collect(),
            center: words(bake, b.center, b.fans)?,
            reference: words(bake, b.reference, b.fans)?,
            other: words(bake, b.other, b.fans)?,
            corners: lists(bake, &b.corners, 3, "corners")?,
            edges: lists(bake, &b.edges, 1, "edges")?,
            faces: lists(bake, &b.faces, 1, "faces")?,
        };
        if m.corners.keys() != b.fans || m.edges.keys() != b.fans {
            bail!("{} fans, with corners for {} and edges for {}", b.fans, m.corners.keys(), m.edges.keys());
        }
        below(&m.fan, b.fans, "fan")?;
        below(m.center.iter().chain(&m.reference).chain(&m.other), nv, "fan vertex")?;
        below(m.corners.items.iter().step_by(3), m.faces.keys(), "corner face")?;
        below(m.corners.items.chunks_exact(3).flat_map(|c| &c[1..]), nv, "corner vertex")?;
        below(&m.edges.items, nv, "edge vertex")?;
        below(&m.faces.items, rec.triangles.as_ref().map_or(0, |t| t.count), "face triangle")?;
        Ok(m)
    }
}

/// Whether the mesh's world matrix mirrors (a negative determinant), read on every frame of its record; a mesh that
/// mirrors on some frames and not on others bends as on its first frame, with a warning.
fn mirrored(bake: &Bake, rec: &MeshRec) -> Result<bool> {
    let Some(m) = &rec.matrix else { return Ok(false) };
    let mut first = None;
    let mut flips = false;
    for k in 0..m.frames.max(1) {
        let det = Mat3::from_mat4(affine(bake.f(m.offset + 12 * k, 12)?)).determinant();
        if det == 0.0 {
            continue;
        }
        match first {
            None => first = Some(det < 0.0),
            Some(f) => flips |= f != (det < 0.0),
        }
    }
    if flips {
        eprintln!("mkr: mesh {} mirrors on some frames and not on others: its normals bend as on its first frame", rec.name);
    }
    Ok(first.unwrap_or(false))
}

/// A mesh whose normals bend: its record in the bake's `lit.meshes`, its shape, its bend data and whether its matrix
/// mirrors.
pub struct Bending<'g> {
    pub record: usize,
    pub shape: &'g Shape,
    pub data: MeshBend,
    pub mirrored: bool,
}

/// The meshes of the bake (`geo` gathered from its `lit.meshes`) whose normals bend: deforming shapes with a bend record.
pub fn bending<'g>(bake: &Bake, geo: &'g Geometry) -> Result<Vec<Bending<'g>>> {
    let mut out = Vec::new();
    for (ri, rec) in bake.lit()?.meshes.iter().enumerate() {
        let Some(mesh) = geo.meshes.get(ri) else { break };
        let shape = &geo.shapes[mesh.shape];
        if !matches!(shape.kind, ShapeKind::Deforming { .. }) {
            continue;
        }
        let Some(data) = MeshBend::read(bake, rec)? else { continue };
        if mesh.name != rec.name || data.fan.len() != shape.vertex_count as usize {
            bail!("mesh {}: the geometry's mesh {} ({} vertices) is not its record", rec.name, mesh.name, shape.vertex_count);
        }
        out.push(Bending { record: ri, shape, data, mirrored: mirrored(bake, rec)? });
    }
    Ok(out)
}

// ================================================================================================= the rebuild, on the CPU
fn unit(v: DVec3) -> DVec3 {
    v / v.length().max(1e-30)
}

/// bend.py's `_angle`: the arccos of the unit vectors' dot product (pi / 2 when one of them is zero).
fn angle(a: DVec3, b: DVec3) -> f64 {
    unit(a).dot(unit(b)).clamp(-1.0, 1.0).acos()
}

/// A fan's frame on one shape: lnor, ra, ref, ortho, span.
#[derive(Clone, Copy, Debug)]
struct Frame {
    ln: DVec3,
    ra: f64,
    r: DVec3,
    o: DVec3,
    span: f64,
}

impl Frame {
    fn normal(&self, alpha: f64, beta: f64) -> DVec3 {
        let a = if alpha >= 0.0 { alpha * self.ra } else { TURN + alpha * (TURN - self.ra) };
        let b = beta * self.span;
        unit(a.cos() * self.ln + a.sin() * (b.cos() * self.r + b.sin() * self.o))
    }
}

/// bend.py's `normals` in f64: the normal of each vertex of a mesh whose vertices are at `p`, with triangles `tris`
/// (vertices numbered within the mesh), rebuilt from its bend data; `mirrored`: `p` went through a mirroring matrix
/// (the module's doc).
pub fn normals(p: &[DVec3], tris: &[[u32; 3]], b: &MeshBend, mirrored: bool) -> Vec<DVec3> {
    let sign = if mirrored { -1.0 } else { 1.0 };
    let at = |v: u32| p[v as usize];
    let tri_n: Vec<DVec3> = tris.iter().map(|t| (at(t[1]) - at(t[0])).cross(at(t[2]) - at(t[0]))).collect();
    let face_n: Vec<DVec3> = (0..b.faces.keys())
        .map(|f| unit(b.faces.run(f).map(|i| tri_n[b.faces.items[i] as usize]).sum()))
        .collect();
    let frames: Vec<Frame> = (0..b.center.len())
        .map(|f| {
            let c = at(b.center[f]);
            let sum: DVec3 = b.corners.run(f)
                .map(|k| {
                    let it = &b.corners.items[3 * k..3 * k + 3];
                    face_n[it[0] as usize] * angle(at(it[1]) - c, at(it[2]) - c)
                })
                .sum();
            let ln = sign * unit(sum);
            let edges = b.edges.run(f);
            let n = edges.len().max(1) as f64;
            let ra = edges.map(|k| angle(at(b.edges.items[k]) - c, ln)).sum::<f64>() / n;
            let flat = |w: u32| {
                let d = at(w) - c;
                unit(d - ln * d.dot(ln))
            };
            let (r, oth) = (flat(b.reference[f]), flat(b.other[f]));
            let o = sign * ln.cross(r);
            let d = r.dot(oth).clamp(-1.0, 1.0);
            let span = if d >= 1.0 - 1e-6 {
                TURN
            } else if o.dot(oth) < 0.0 {
                TURN - d.acos()
            } else {
                d.acos()
            };
            Frame { ln, ra, r, o, span }
        })
        .collect();
    b.fan.iter().zip(b.alpha.iter().zip(&b.beta))
        .map(|(&f, (&alpha, &beta))| frames[f as usize].normal(alpha as f64, beta as f64))
        .collect()
}

// ================================================================================================= the rebuild, on the GPU
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct FanGpu {
    center: u32,
    reference: u32,
    other: u32,
    flags: u32,
    /// [first, end) of its corners, edges and vertices.
    corners: [u32; 2],
    edges: [u32; 2],
    verts: [u32; 2],
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct VertGpu {
    vertex: u32,
    alpha: f32,
    beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SizesGpu {
    faces: u32,
    fans: u32,
    pad: [u32; 2],
}

fn index32(n: usize) -> Result<u32> {
    u32::try_from(n).context("the bending meshes are too big to index with u32")
}

/// The bending meshes' data as the shaders read it: vertices of the posed buffer, faces and fans numbered across meshes.
struct Packed {
    face_first: Vec<u32>,
    tri_verts: Vec<u32>,
    fans: Vec<FanGpu>,
    corners: Vec<u32>,
    edges: Vec<u32>,
    verts: Vec<VertGpu>,
}

impl Packed {
    fn new() -> Packed {
        Packed { face_first: vec![0], tri_verts: Vec::new(), fans: Vec::new(), corners: Vec::new(), edges: Vec::new(),
                 verts: Vec::new() }
    }

    fn faces(&self) -> usize {
        self.face_first.len() - 1
    }

    /// Add a mesh whose vertices start at `vertex_base` of the posed buffer; `tris`: its triangles, three vertices each
    /// (numbered within the mesh).
    fn add(&mut self, b: &MeshBend, vertex_base: u32, tris: &[u32], mirrored: bool) -> Result<()> {
        let g = |v: u32| vertex_base + v;
        let face_base = index32(self.faces())?;
        for f in 0..b.faces.keys() {
            for i in b.faces.run(f) {
                let t = b.faces.items[i] as usize;
                self.tri_verts.extend(tris[3 * t..3 * t + 3].iter().map(|&v| g(v)));
            }
            self.face_first.push(index32(self.tri_verts.len() / 3)?);
        }
        // its vertices by fan
        let nf = b.center.len();
        let mut first = vec![0usize; nf + 1];
        for &f in &b.fan {
            first[f as usize + 1] += 1;
        }
        for f in 0..nf {
            first[f + 1] += first[f];
        }
        let mut next = first.clone();
        let mut by_fan = vec![0u32; b.fan.len()];
        for (v, &f) in b.fan.iter().enumerate() {
            by_fan[next[f as usize]] = v as u32;
            next[f as usize] += 1;
        }
        let flags = if mirrored { MIRRORED } else { 0 };
        for f in 0..nf {
            let c0 = index32(self.corners.len() / 3)?;
            for k in b.corners.run(f) {
                let it = &b.corners.items[3 * k..3 * k + 3];
                self.corners.extend([face_base + it[0], g(it[1]), g(it[2])]);
            }
            let e0 = index32(self.edges.len())?;
            self.edges.extend(b.edges.run(f).map(|k| g(b.edges.items[k])));
            let v0 = index32(self.verts.len())?;
            self.verts.extend(by_fan[first[f]..first[f + 1]].iter().map(|&v| {
                VertGpu { vertex: g(v), alpha: b.alpha[v as usize], beta: b.beta[v as usize] }
            }));
            self.fans.push(FanGpu {
                center: g(b.center[f]),
                reference: g(b.reference[f]),
                other: g(b.other[f]),
                flags,
                corners: [c0, index32(self.corners.len() / 3)?],
                edges: [e0, index32(self.edges.len())?],
                verts: [v0, index32(self.verts.len())?],
                pad: [0; 2],
            });
        }
        index32(self.fans.len())?;
        Ok(())
    }
}

/// Workgroups (x, y) covering `n` items, GROUP to a workgroup and at most MAX_GROUPS along x; the shaders number an
/// invocation's item x + y * (workgroups along x) * GROUP.
fn groups(n: u32) -> (u32, u32) {
    let g = n.div_ceil(GROUP);
    let y = g.div_ceil(MAX_GROUPS).max(1);
    (g.div_ceil(y), y)
}

/// The bending meshes' normals rebuilt on the GPU: their data, the two passes and their bindings (to the posed buffer of
/// the frame's own time).
pub struct Bend {
    faces_pipe: wgpu::ComputePipeline,
    fans_pipe: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    faces: u32,
    fans: u32,
    /// The buffers the bind group reads.
    _buffers: Vec<wgpu::Buffer>,
}

impl Bend {
    /// The bending meshes' rebuild data on the GPU; None when no mesh of the bake bends. `geo` is the bake's `lit.meshes`
    /// gathered (Geometry::load), `posed` the posed-vertex buffer `dispatch` rewrites the normals of.
    pub fn new(device: &wgpu::Device, bake: &Bake, geo: &Geometry, posed: &wgpu::Buffer) -> Result<Option<Bend>> {
        let mut packed = Packed::new();
        for m in bending(bake, geo)? {
            let s = m.shape;
            let tris = &geo.indices[s.index_base as usize..(s.index_base + s.index_count) as usize];
            packed.add(&m.data, s.vertex_base, tris, m.mirrored)?;
        }
        if packed.fans.is_empty() {
            return Ok(None);
        }
        Bend::build(device, &packed, posed).map(Some)
    }

    fn build(device: &wgpu::Device, p: &Packed, posed: &wgpu::Buffer) -> Result<Bend> {
        let faces = index32(p.faces())?;
        let fans = index32(p.fans.len())?;
        let read = |label: &str, bytes: &[u8]| storage(device, label, bytes, wgpu::BufferUsages::empty());
        let buffers = vec![
            read("bend face first", bytemuck::cast_slice(&p.face_first)),
            read("bend triangles", bytemuck::cast_slice(&p.tri_verts)),
            sized(device, "bend face normals", faces as u64 * 16, wgpu::BufferUsages::STORAGE),
            read("bend fans", bytemuck::cast_slice(&p.fans)),
            read("bend corners", bytemuck::cast_slice(&p.corners)),
            read("bend edges", bytemuck::cast_slice(&p.edges)),
            read("bend vertices", bytemuck::cast_slice(&p.verts)),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bend sizes"),
                contents: bytemuck::bytes_of(&SizesGpu { faces, fans, pad: [0; 2] }),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        ];
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let (ro, rw) = (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false });
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bend"),
            entries: &[entry(0, rw), entry(1, ro), entry(2, ro), entry(3, rw), entry(4, ro), entry(5, ro), entry(6, ro),
                       entry(7, ro), entry(8, wgpu::BufferBindingType::Uniform)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bend"),
            source: wgpu::ShaderSource::Wgsl(BEND_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bend"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (faces_pipe, fans_pipe) = (pipe("faces"), pipe("fans"));
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: posed.as_entire_binding() }];
        entries.extend(buffers.iter().enumerate().map(|(i, b)| {
            wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }
        }));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bend"), layout: &bgl, entries: &entries });
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("the bend passes cannot be made: {e}");
        }
        Ok(Bend { faces_pipe, fans_pipe, bind, faces, fans, _buffers: buffers })
    }

    /// Rebuild the normals of every bending mesh in the posed buffer given to `new` from its positions (the skin pass
    /// has posed them for this frame).
    pub fn dispatch(&self, enc: &mut wgpu::CommandEncoder) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bend"), timestamp_writes: None });
        for (pipe, n) in [(&self.faces_pipe, self.faces), (&self.fans_pipe, self.fans)] {
            if n == 0 {
                continue;
            }
            let (x, y) = groups(n);
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, PI};
    use std::path::{Path, PathBuf};

    use glam::{Mat4, Vec3};

    use super::*;
    use crate::gpu::Gpu;
    use crate::scene::Pose;

    /// Lists from runs of entries of `per` items each.
    fn lists(runs: &[Vec<u32>], per: usize) -> Lists {
        let mut offsets = vec![0u32];
        let mut items = Vec::new();
        for r in runs {
            items.extend(r);
            offsets.push((items.len() / per) as u32);
        }
        Lists { offsets, items }
    }

    /// The angle between two vectors in degrees (atan2: exact for small ones).
    fn deg(a: DVec3, b: DVec3) -> f64 {
        a.cross(b).length().atan2(a.dot(b)).to_degrees()
    }

    /// Within 1e-5 deg (alpha and beta are f32: 1/3 is off by 3e-8 of itself) and of unit length.
    fn assert_near(got: DVec3, want: DVec3, what: &str) {
        let d = deg(got, want);
        assert!(d < 1e-5 && (got.length() - 1.0).abs() < 1e-9, "{what}: {got} is {d} deg from {want}");
    }

    type Mesh = (Vec<DVec3>, Vec<[u32; 3]>, MeshBend);

    /// One fan round vertex 0 from its faces, each a triangle (0, a, b) whose corner at vertex 0 is (face, b, a), plus
    /// copies of vertex 0 carrying (alpha, beta); every vertex is in the fan (alpha and beta 0 but the copies').
    fn fan_mesh(mut p: Vec<DVec3>, rims: &[[u32; 2]], reference: u32, other: u32, copies: &[(f32, f32)]) -> Mesh {
        let n0 = p.len();
        let centre = p[0];
        p.extend(copies.iter().map(|_| centre));
        let tris: Vec<[u32; 3]> = rims.iter().map(|&[a, b]| [0, a, b]).collect();
        let mut far: Vec<u32> = rims.iter().flatten().copied().collect();
        far.sort();
        far.dedup();
        let mut alpha = vec![0.0; n0];
        let mut beta = vec![0.0; n0];
        alpha.extend(copies.iter().map(|c| c.0));
        beta.extend(copies.iter().map(|c| c.1));
        let b = MeshBend {
            fan: vec![0; p.len()],
            alpha,
            beta,
            center: vec![0],
            reference: vec![reference],
            other: vec![other],
            corners: lists(&[rims.iter().enumerate().flat_map(|(f, &[a, b])| [f as u32, b, a]).collect()], 3),
            edges: lists(&[far], 1),
            faces: lists(&(0..rims.len() as u32).map(|t| vec![t]).collect::<Vec<_>>(), 1),
        };
        (p, tris, b)
    }

    /// A square pyramid's apex (vertex 0, its base round it 1 below): a closed fan of four faces, lnor +z, every edge
    /// 135 deg from it, ref +x; copies 5.. of the apex.
    fn pyramid() -> Mesh {
        let p = vec![DVec3::Z, DVec3::X, DVec3::Y, DVec3::NEG_X, DVec3::NEG_Y];
        let copies = [(1.0 / 3.0, 0.0), (1.0 / 3.0, 0.25), (-0.5, 0.0), (1.0, 0.5)];
        fan_mesh(p, &[[1, 2], [2, 3], [3, 4], [4, 1]], 1, 1, &copies)
    }

    /// Three quarter-turn faces in the plane z = 0 from +x round to -y: an open fan, ref +x, other -y, span 3 pi / 2;
    /// copies 5.. of its centre.
    fn open_fan() -> Mesh {
        let p = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::NEG_X, DVec3::NEG_Y];
        fan_mesh(p, &[[1, 2], [2, 3], [3, 4]], 1, 4, &[(1.0, 0.5), (1.0, 1.0 / 3.0), (0.5, 0.0)])
    }

    #[test]
    fn rebuilds_a_closed_fan() {
        let (p, tris, b) = pyramid();
        let n = normals(&p, &tris, &b, false);
        let s = FRAC_1_SQRT_2;
        assert_near(n[0], DVec3::Z, "alpha 0: lnor");
        // a = ra / 3 = 45 deg toward ref (+x), then a quarter turn round lnor (toward ortho = lnor x ref = +y)
        assert_near(n[5], DVec3::new(s, 0.0, s), "alpha 1/3");
        assert_near(n[6], DVec3::new(0.0, s, s), "alpha 1/3, beta 1/4");
        // alpha -1/2: half way from a whole turn back to ra: 2 pi - (2 pi - 3 pi / 4) / 2 = 11 pi / 8
        let a = 11.0 * PI / 8.0;
        assert_near(n[7], DVec3::new(a.sin(), 0.0, a.cos()), "alpha -1/2");
        // alpha 1, beta 1/2: along the edge opposite ref's
        assert_near(n[8], DVec3::new(-s, 0.0, -s), "alpha 1, beta 1/2");
    }

    #[test]
    fn spans_an_open_fan() {
        let (p, tris, b) = open_fan();
        let n = normals(&p, &tris, &b, false);
        let s = FRAC_1_SQRT_2;
        assert_near(n[0], DVec3::Z, "lnor");
        assert_near(n[5], DVec3::new(-s, s, 0.0), "beta 1/2: the fan's bisector");
        assert_near(n[6], DVec3::Y, "beta 1/3");
        assert_near(n[7], DVec3::new(s, 0.0, s), "alpha 1/2 toward ref");
    }

    #[test]
    fn weights_faces_by_their_angle_at_the_corner() {
        // a flat face (normal +z, 90 deg at the corner) and a tilted one (normal (1, 0, 1) / sqrt 2, acos(1 / sqrt 3))
        let p = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::new(-1.0, 1.0, 1.0)];
        let (p, tris, b) = fan_mesh(p, &[[1, 2], [2, 3]], 1, 3, &[]);
        let n = normals(&p, &tris, &b, false);
        let want = DVec3::Z * (PI / 2.0) + DVec3::new(1.0, 0.0, 1.0).normalize() * (1.0 / 3f64.sqrt()).acos();
        assert_near(n[0], want.normalize(), "lnor");
    }

    #[test]
    fn a_mirrored_mesh_bends_as_the_mirror_image() {
        let flip = DVec3::new(-1.0, 1.0, 1.0);
        for (name, (p, tris, b)) in [("closed", pyramid()), ("open", open_fan())] {
            let want = normals(&p, &tris, &b, false);
            let mirror: Vec<DVec3> = p.iter().map(|&v| v * flip).collect();
            let got = normals(&mirror, &tris, &b, true);
            for (v, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_near(*g, *w * flip, &format!("{name} fan, vertex {v}"));
            }
        }
    }

    #[test]
    fn groups_cover_every_item() {
        assert_eq!(groups(1), (1, 1));
        assert_eq!(groups(64 * 65535), (65535, 1));
        let (x, y) = groups(64 * 65535 + 1);
        assert_eq!(y, 2);
        assert!(x <= MAX_GROUPS && x * y * GROUP > 64 * 65535);
    }

    // ------------------------------------------------------------------------------------------------- GPU against CPU
    fn gpu() -> Option<Gpu> {
        match Gpu::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("bend: no GPU ({e}): skipped");
                None
            }
        }
    }

    fn posed_buffer(gpu: &Gpu, vertices: usize) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bend test posed"),
            size: (vertices.max(1) * 32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// The normals in `posed` after `bend` ran on vertices at `positions` (normals zero before).
    fn run(gpu: &Gpu, bend: &Bend, posed: &wgpu::Buffer, positions: &[Vec3]) -> Vec<DVec3> {
        let data: Vec<[f32; 8]> = positions.iter().map(|p| [p.x, p.y, p.z, 1.0, 0.0, 0.0, 0.0, 0.0]).collect();
        gpu.queue.write_buffer(posed, 0, bytemuck::cast_slice(&data));
        let size = (data.len() * 32) as u64;
        let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bend test read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        bend.dispatch(&mut enc);
        enc.copy_buffer_to_buffer(posed, 0, &read, 0, size);
        gpu.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.wait().unwrap();
        rx.recv().unwrap().unwrap();
        let out = {
            let view = read.slice(..).get_mapped_range().unwrap();
            let px: &[[f32; 8]] = bytemuck::cast_slice(&view);
            px.iter().map(|x| DVec3::new(x[4] as f64, x[5] as f64, x[6] as f64)).collect()
        };
        read.unmap();
        out
    }

    /// (max, 99th percentile) as numpy's percentile has it (linear between the ranks).
    fn stats(mut v: Vec<f64>) -> (f64, f64) {
        v.sort_by(f64::total_cmp);
        let at = 0.99 * (v.len() - 1) as f64;
        let (lo, t) = (at.floor() as usize, at.fract());
        let p99 = v[lo] + (v[(lo + 1).min(v.len() - 1)] - v[lo]) * t;
        (v[v.len() - 1], p99)
    }

    /// A tube of `rings` rings of `segs` vertices with quads between the rings (two triangles each), a fan per vertex
    /// (closed inside, open on the end rings), one more vertex in every seventh vertex's fan, alpha and beta from a
    /// fixed sequence.
    fn tube(rings: usize, segs: usize) -> Mesh {
        let n = rings * segs;
        let id = |i: usize, j: usize| (i * segs + j % segs) as u32;
        let mut p: Vec<DVec3> = (0..n)
            .map(|v| {
                let t = TURN * (v % segs) as f64 / segs as f64;
                DVec3::new(0.3 * t.cos(), 0.08 * (v / segs) as f64, 0.3 * t.sin())
            })
            .collect();
        let mut tris = Vec::new();
        let mut faces = Vec::new();
        let mut corners = vec![Vec::new(); n];
        for i in 0..rings - 1 {
            for j in 0..segs {
                let q = [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)];
                let f = faces.len() as u32;
                faces.push(vec![tris.len() as u32, tris.len() as u32 + 1]);
                tris.push([q[0], q[1], q[2]]);
                tris.push([q[0], q[2], q[3]]);
                for k in 0..4 {
                    corners[q[k] as usize].extend([f, q[(k + 3) % 4], q[(k + 1) % 4]]);
                }
            }
        }
        let edges: Vec<Vec<u32>> = corners.iter()
            .map(|c| {
                let mut e: Vec<u32> = c.chunks_exact(3).flat_map(|x| [x[1], x[2]]).collect();
                e.sort();
                e.dedup();
                e
            })
            .collect();
        let (mut reference, mut other) = (Vec::new(), Vec::new());
        for (v, c) in corners.iter().enumerate() {
            let (i, j) = (v / segs, v % segs);
            if i == 0 || i == rings - 1 {
                reference.push(id(i, j + 1));
                other.push(id(i, j + segs - 1));
            } else {
                reference.push(c[2]);
                other.push(c[2]);
            }
        }
        let mut fan: Vec<u32> = (0..n as u32).collect();
        for v in (0..n).step_by(7) {
            p.push(p[v]);
            fan.push(v as u32);
        }
        let mut seed = 0x2545_f491u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let alpha = (0..p.len()).map(|_| next() * 1.6 - 0.6).collect();
        let beta = (0..p.len()).map(|_| next()).collect();
        let b = MeshBend {
            fan,
            alpha,
            beta,
            center: (0..n as u32).collect(),
            reference,
            other,
            corners: lists(&corners, 3),
            edges: lists(&edges, 1),
            faces: lists(&faces, 1),
        };
        (p, tris, b)
    }

    /// The tube twisted, bent round the x axis by `bend` radians per unit of height, moved, and mirrored in x (`mirror`).
    fn deform(p: DVec3, bend: f64, mirror: bool) -> DVec3 {
        let (a, twist) = (bend * p.y, 0.7 * p.y);
        let q = DVec3::new(p.x * twist.cos() - p.z * twist.sin(), p.y, p.x * twist.sin() + p.z * twist.cos());
        let r = DVec3::new(q.x, q.y * a.cos() - q.z * a.sin(), q.y * a.sin() + q.z * a.cos());
        let w = r + DVec3::new(4.0, 1.5, -3.0);
        if mirror { w * DVec3::new(-1.0, 1.0, 1.0) } else { w }
    }

    #[test]
    fn gpu_matches_cpu_on_a_bent_tube() {
        let Some(gpu) = gpu() else { return };
        let (rest, tris, b) = tube(24, 32);
        let flat: Vec<u32> = tris.iter().flatten().copied().collect();
        // two meshes in one posed buffer after other vertices, the second mirrored
        let n = rest.len();
        let bases = [5usize, 5 + n + 3];
        let total = bases[1] + n;
        let posed = posed_buffer(&gpu, total);
        let mut packed = Packed::new();
        for (i, &base) in bases.iter().enumerate() {
            packed.add(&b, base as u32, &flat, i == 1).unwrap();
        }
        let bend = Bend::build(&gpu.device, &packed, &posed).unwrap();
        let mut all = Vec::new();
        for bent in [0.0, 1.5, 4.0] {
            let mut pos = vec![Vec3::ZERO; total];
            let mut cpu = Vec::new();
            for (i, &base) in bases.iter().enumerate() {
                let world: Vec<Vec3> = rest.iter().map(|&v| deform(v, bent, i == 1).as_vec3()).collect();
                pos[base..base + n].copy_from_slice(&world);
                let p: Vec<DVec3> = world.iter().map(|v| v.as_dvec3()).collect();
                cpu.push(normals(&p, &tris, &b, i == 1));
            }
            let got = run(&gpu, &bend, &posed, &pos);
            for (i, &base) in bases.iter().enumerate() {
                let angles: Vec<f64> = (0..n).map(|v| deg(got[base + v], cpu[i][v])).collect();
                let (max, p99) = stats(angles.clone());
                eprintln!("bend: tube bent {bent} rad{}: GPU vs CPU max {max:.5} p99 {p99:.5} deg",
                          if i == 1 { ", mirrored" } else { "" });
                assert!(got[base..base + n].iter().all(|v| (v.length() - 1.0).abs() < 1e-5), "a unit normal each");
                all.extend(angles);
            }
            assert!(got[..bases[0]].iter().chain(&got[bases[0] + n..bases[1]]).all(|v| *v == DVec3::ZERO),
                    "the vertices of no bending mesh keep their normals");
        }
        let (max, _) = stats(all);
        assert!(max < 0.01, "GPU vs CPU: max {max} deg");
    }

    /// Every vertex posed on the CPU as skin.wgsl poses it (f32): (position, its rest normal turned); rigid shapes stay
    /// at rest.
    fn pose_cpu(geo: &Geometry, pose: &Pose) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut pos: Vec<Vec3> = geo.rest.iter().map(|r| Vec3::from_slice(&r.pos[..3])).collect();
        let mut nrm: Vec<Vec3> = geo.rest.iter().map(|r| Vec3::from_slice(&r.nrm[..3])).collect();
        for (si, s) in geo.shapes.iter().enumerate() {
            let ShapeKind::Deforming { armature, morphs } = &s.kind else { continue };
            let m: Mat4 = pose.shape_matrices[si];
            let (first, count) = pose.morph_runs[si];
            let joint_base = armature.map(|a| pose.joint_base[a] as usize).unwrap_or(0);
            for local in 0..s.vertex_count as usize {
                let v = s.vertex_base as usize + local;
                let mut p = pos[v];
                for k in 0..count as usize {
                    let [key, w] = pose.morphs[first as usize + k];
                    let base = morphs.as_ref().unwrap().delta_base as usize;
                    let at = base + (key.to_bits() as usize * s.vertex_count as usize + local) * 3;
                    p += w * Vec3::from_slice(&geo.deltas[at..at + 3]);
                }
                let mut wp = m.transform_point3(p);
                let mut wn = m.transform_vector3(nrm[v]);
                if let Some(sb) = s.skin_base {
                    let sk = geo.skin[sb as usize + local];
                    if sk.weights.iter().sum::<f32>() > 0.0 {
                        let mut rows = [[0.0f32; 4]; 3];
                        for (i, &w) in sk.weights.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let pair = sk.joints[i / 2];
                            let j = (if i % 2 == 0 { pair & 0xffff } else { pair >> 16 }) as usize;
                            for (r, row) in rows.iter_mut().enumerate() {
                                let bone = pose.joints[(joint_base + j) * 3 + r];
                                for c in 0..4 {
                                    row[c] += w * bone[c];
                                }
                            }
                        }
                        let apply = |q: Vec3, h: f32| {
                            Vec3::from_array(rows.map(|r| r[0] * q.x + r[1] * q.y + r[2] * q.z + r[3] * h))
                        };
                        wp = apply(wp, 1.0);
                        wn = apply(wn, 0.0);
                    }
                }
                pos[v] = wp;
                nrm[v] = wn.normalize();
            }
        }
        (pos, nrm)
    }

    /// The GPU's normals against the CPU port's on bake frames `ks` (None: every frame) of the bake in `dir`, from the
    /// same posed vertices, printed per bending mesh and frame with how far both are from the baked rest normals turned
    /// with the vertices (Blender's own normals where the pose is the rest pose); the GPU-CPU angles (deg) of them all.
    fn against_cpu(gpu: &Gpu, dir: &Path, ks: Option<Vec<usize>>) -> Vec<f64> {
        let bake = Bake::open(dir).unwrap();
        let geo = Geometry::load(&bake, &|_: &str| Vec::new()).unwrap();
        let posed = posed_buffer(gpu, geo.rest.len());
        let bend = Bend::new(&gpu.device, &bake, &geo, &posed).unwrap().expect("a mesh of the bake bends");
        let meshes = bending(&bake, &geo).unwrap();
        let mut all = Vec::new();
        for k in ks.unwrap_or_else(|| (0..bake.doc.frames).collect()) {
            let pose = Pose::at(&bake, &geo, k as f64).unwrap();
            let (pos, turned) = pose_cpu(&geo, &pose);
            let got = run(gpu, &bend, &posed, &pos);
            for m in &meshes {
                let (base, n) = (m.shape.vertex_base as usize, m.shape.vertex_count as usize);
                let p: Vec<DVec3> = pos[base..base + n].iter().map(|v| v.as_dvec3()).collect();
                let idx = &geo.indices[m.shape.index_base as usize..(m.shape.index_base + m.shape.index_count) as usize];
                let tris: Vec<[u32; 3]> = idx.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
                let cpu = normals(&p, &tris, &m.data, m.mirrored);
                let vs_cpu: Vec<f64> = (0..n).map(|v| deg(got[base + v], cpu[v])).collect();
                let gpu_turned: Vec<f64> = (0..n).map(|v| deg(got[base + v], turned[base + v].as_dvec3())).collect();
                let cpu_turned: Vec<f64> = (0..n).map(|v| deg(cpu[v], turned[base + v].as_dvec3())).collect();
                let backward = gpu_turned.iter().filter(|&&d| d > 90.0).count();
                let ((a, b), (c, d), (e, f)) = (stats(vs_cpu.clone()), stats(gpu_turned), stats(cpu_turned));
                eprintln!("bend: {} frame {}: GPU vs CPU max {a:.4} p99 {b:.4}; vs the rest normals turned: GPU max \
                           {c:.3} p99 {d:.4} ({backward} over 90), CPU max {e:.3} p99 {f:.4} deg ({n} vertices)",
                          bake.lit().unwrap().meshes[m.record].name, bake.doc.frame0 + k as i64);
                all.extend(vs_cpu);
            }
        }
        all
    }

    /// The lab folder as `mk parity lab` finds it: $MK_LAB, else <$MK_CACHE or ~/.cache/mk>/lab.
    fn lab_root() -> Option<PathBuf> {
        let dir = match std::env::var_os("MK_LAB") {
            Some(d) => PathBuf::from(d),
            None => match std::env::var_os("MK_CACHE") {
                Some(c) => PathBuf::from(c).join("lab"),
                None => PathBuf::from(std::env::var_os("HOME")?).join(".cache/mk/lab"),
            },
        };
        dir.is_dir().then_some(dir)
    }

    /// The lab's bending cases (`mk parity lab`), every frame: the GPU within 0.05 deg of the CPU port at p99.
    #[test]
    fn gpu_matches_cpu_on_the_lab_bakes() {
        let Some(lab) = lab_root() else {
            eprintln!("bend: no lab folder ($MK_LAB): skipped");
            return;
        };
        let Some(gpu) = gpu() else { return };
        for case in ["skinned_bend", "mmd_model"] {
            let dir = lab.join(case).join("scene.bake");
            if !dir.join("bake.json").is_file() {
                eprintln!("bend: no {case} bake in {}: skipped", lab.display());
                continue;
            }
            let (max, p99) = stats(against_cpu(&gpu, &dir, None));
            eprintln!("bend: {case}: GPU vs CPU over every frame max {max:.4} p99 {p99:.4} deg");
            assert!(p99 < 0.05, "{case}: GPU vs CPU p99 {p99} deg");
        }
    }

    /// Any bake: MKR_BEND_BAKE=<bake folder> [MKR_BEND_FRAMES=<Blender frames, comma separated>] cargo test --release
    /// -p mkr bend -- --ignored --nocapture.
    #[test]
    #[ignore]
    fn gpu_matches_cpu_on_a_bake() {
        let dir = PathBuf::from(std::env::var_os("MKR_BEND_BAKE").expect("MKR_BEND_BAKE: a bake folder"));
        let Some(gpu) = gpu() else { return };
        let frame0 = Bake::open(&dir).unwrap().doc.frame0;
        let ks = std::env::var("MKR_BEND_FRAMES").ok().map(|s| {
            s.split(',').map(|f| (f.trim().parse::<i64>().expect("a Blender frame") - frame0) as usize).collect()
        });
        let (max, p99) = stats(against_cpu(&gpu, &dir, ks));
        eprintln!("bend: {}: GPU vs CPU max {max:.4} p99 {p99:.4} deg", dir.display());
        assert!(p99 < 0.05, "GPU vs CPU p99 {p99} deg");
    }
}
