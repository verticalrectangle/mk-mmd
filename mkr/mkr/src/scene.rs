//! The lit scene on the GPU: the bake's `lit.meshes` as shapes (vertex data: rest vertices, skins, shape keys, the
//! attributes their materials read, triangles by material slot) and meshes (an object showing a shape: its matrix on
//! each frame, its materials, when it shows). Instances share their source's shape.
//!
//! Every shape owns a range of the posed-vertex buffers (one per time of a frame: the shutter's open, the frame, the
//! shutter's close): a rigid shape keeps its own-space vertices there (its meshes' matrices place them), a deforming one
//! is posed into the world by a compute pass (shaders/skin.wgsl), a frame-by-frame one is uploaded from the bake.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::bake::{AttrRec, Bake, Geom, MatrixRec, MeshRec, MeshVisible};
use crate::camera::affine;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RestVertex {
    pub pos: [f32; 4],
    pub nrm: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SkinVertex {
    pub joints: [u32; 4],
    pub weights: [f32; 8],
}

/// A material slot's run of triangles in its shape.
#[derive(Clone, Debug)]
pub struct Slot {
    pub slot: usize,
    pub first: u32,
    pub count: u32,
    /// Index into the scene's material list (None: an empty slot, drawn with Blender's default surface).
    pub material: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Morphs {
    /// f32 index of the deltas in the scene's delta buffer (key-major: key, vertex, xyz).
    pub delta_base: u32,
    pub count: usize,
    /// FLOAT offset of the per-frame values in bake.bin.
    pub values: usize,
}

#[derive(Clone, Debug)]
pub enum ShapeKind {
    /// Its own-space vertices never change.
    Rigid,
    /// Posed per time: shape keys, then its mesh's matrix, then its armature.
    Deforming { armature: Option<usize>, morphs: Option<Morphs> },
    /// A mesh the bake holds frame by frame: a geometry per frame in its object's space, placed by its mesh's matrix
    /// (the identity for screen type, whose words are in the world).
    ByFrame { rec: usize },
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub vertex_base: u32,
    /// Vertices (a frame-by-frame shape: the most any of its geometries has).
    pub vertex_count: u32,
    /// First index (u32) of its triangles; room for `index_count`.
    pub index_base: u32,
    pub index_count: u32,
    /// vec4 index of its first vertex's attributes, and how many vec4s each vertex has.
    pub attr_base: u32,
    pub attr_stride: u32,
    pub attr_names: Vec<String>,
    pub generated: Option<u32>,
    pub skin_base: Option<u32>,
    pub kind: ShapeKind,
    /// Own-space bounds (sorting blended surfaces).
    pub lo: Vec3,
    pub hi: Vec3,
}

/// An object showing a shape.
#[derive(Clone, Debug)]
pub struct Mesh {
    pub name: String,
    pub shape: usize,
    pub slots: Vec<Slot>,
    /// What places its shape's posed vertices in the world per frame (None: they are in the world already).
    pub matrix: Option<MatrixRec>,
    /// The object's own world matrix per frame (Object coordinates and Object Info read it).
    pub object_matrix: Option<MatrixRec>,
    /// FLOAT offset of its per-frame shown flags.
    pub shown: Option<usize>,
    pub visible: MeshVisible,
    /// The only output it renders for (mk's per-output objects), if any.
    pub aspect: Option<String>,
    pub color: [f32; 4],
    pub pass_index: f32,
}

impl Mesh {
    pub fn tri_count(&self) -> u32 {
        self.slots.iter().map(|s| s.count).sum()
    }
}

/// The scene's geometry as CPU arrays, ready to upload.
pub struct Geometry {
    pub shapes: Vec<Shape>,
    pub meshes: Vec<Mesh>,
    pub rest: Vec<RestVertex>,
    pub skin: Vec<SkinVertex>,
    pub deltas: Vec<f32>,
    pub attrs: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    /// Material names in index order (Slot::material).
    pub materials: Vec<String>,
}

fn f32s(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn attr_bytes(rec: &AttrRec, vertices: usize) -> usize {
    let size = match rec.ty.as_str() {
        "u8" | "i8" => 1,
        "u16" | "i16" | "f16" => 2,
        _ => 4,
    };
    rec.comp * vertices * size
}

/// One attribute of `vertices` vertices as f32, `rec.comp` per vertex.
pub fn read_attr(bake: &Bake, rec: &AttrRec, vertices: usize) -> Result<Vec<f32>> {
    let bytes = bake.bytes(rec.offset, attr_bytes(rec, vertices))?;
    Ok(match rec.ty.as_str() {
        "f32" => f32s(bytes).collect(),
        "u16" => bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as f32).collect(),
        "u8" => bytes.iter().map(|&b| b as f32).collect(),
        "u32" => bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32).collect(),
        other => bail!("attribute type {other} is not one mkr reads"),
    })
}

fn read_u16(bake: &Bake, rec: &AttrRec, vertices: usize) -> Result<Vec<u16>> {
    if rec.ty != "u16" {
        bail!("joints of type {} (mkr reads u16)", rec.ty);
    }
    let bytes = bake.bytes(rec.offset, attr_bytes(rec, vertices))?;
    Ok(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
}

/// Blender's vec4 of an attribute value: float f -> (f, f, f, 1), float2 -> (x, y, 0, 1), float3 -> (x, y, z, 1),
/// four components as they are.
pub(crate) fn widen(v: &[f32]) -> [f32; 4] {
    match v.len() {
        1 => [v[0], v[0], v[0], 1.0],
        2 => [v[0], v[1], 0.0, 1.0],
        3 => [v[0], v[1], v[2], 1.0],
        _ => [v[0], v[1], v[2], v[3]],
    }
}

/// The attribute a material's name asks for (`uv:` alone is the active render UV map; the bake stores that alias).
fn find_attr<'a>(attrs: &'a BTreeMap<String, AttrRec>, name: &str) -> Option<&'a AttrRec> {
    attrs.get(name)
}

/// The attributes `materials` read (`reads(material)`), in first-seen order.
fn attribute_names(materials: &[Option<String>], reads: &Reads<'_>) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for m in materials.iter().flatten() {
        for a in reads(m) {
            if !names.contains(&a) {
                names.push(a);
            }
        }
    }
    names
}

/// What each material reads besides position and normal (mkr-shade's CompiledMaterial attributes).
pub type Reads<'a> = dyn Fn(&str) -> Vec<String> + 'a;

impl Geometry {
    /// Gather every mesh of the lit bake. `reads(material)` names the attributes a material reads.
    pub fn load(bake: &Bake, reads: &Reads<'_>) -> Result<Geometry> {
        Self::from_records(bake, &bake.lit()?.meshes, reads)
    }

    /// Gather the meshes `recs` (records of the lit bake: `lit.meshes`, or another list of them); a frame-by-frame
    /// shape's `ShapeKind::ByFrame { rec }` indexes `recs`.
    pub fn from_records(bake: &Bake, recs: &[MeshRec], reads: &Reads<'_>) -> Result<Geometry> {
        let mut g = Geometry {
            shapes: Vec::new(),
            meshes: Vec::new(),
            rest: Vec::new(),
            skin: Vec::new(),
            deltas: Vec::new(),
            attrs: Vec::new(),
            indices: Vec::new(),
            materials: Vec::new(),
        };
        let mut mat_index: BTreeMap<String, usize> = BTreeMap::new();
        // instances share a shape: same vertices, same triangles, same attributes read
        let mut shared: BTreeMap<(usize, usize, Vec<String>), usize> = BTreeMap::new();
        for (ri, rec) in recs.iter().enumerate() {
            let attr_names = attribute_names(&rec.materials, reads);
            let shape = if let Some(bf) = &rec.by_frame {
                g.add_by_frame(bake, ri, rec, &bf.geoms, attr_names).with_context(|| format!("mesh {}", rec.name))?
            } else {
                let tris = rec.triangles.as_ref().with_context(|| format!("mesh {}: no triangles", rec.name))?;
                let pos = rec.attributes.get("position").with_context(|| format!("mesh {}: no position", rec.name))?;
                let rigid = rec.kind != "skinned" && rec.morphs.as_ref().is_none_or(|m| m.names.is_empty());
                let key = (pos.offset, tris.offset, attr_names.clone());
                match shared.get(&key) {
                    Some(&s) if rigid => s,
                    _ => {
                        let s = g.add_shape(bake, rec, attr_names).with_context(|| format!("mesh {}", rec.name))?;
                        if rigid {
                            shared.insert(key, s);
                        }
                        s
                    }
                }
            };
            let slots_rec: Vec<[usize; 3]> = match &rec.by_frame {
                // a frame-by-frame mesh: one draw per material slot, its run set each frame
                Some(_) => (0..rec.materials.len().max(1)).map(|s| [s, 0, 0]).collect(),
                None => rec.slots.clone(),
            };
            let mut slots = Vec::new();
            for [slot, first, count] in slots_rec {
                let material = match rec.materials.get(slot).cloned().flatten() {
                    Some(m) => {
                        let next = mat_index.len();
                        let i = *mat_index.entry(m.clone()).or_insert(next);
                        if i == g.materials.len() {
                            g.materials.push(m);
                        }
                        Some(i)
                    }
                    None => None,
                };
                slots.push(Slot { slot, first: first as u32, count: count as u32, material });
            }
            let object = rec.object.as_ref();
            g.meshes.push(Mesh {
                name: rec.name.clone(),
                shape,
                slots,
                matrix: match &g.shapes[shape].kind {
                    ShapeKind::Rigid | ShapeKind::ByFrame { .. } => rec.matrix.clone(),
                    ShapeKind::Deforming { .. } => None,
                },
                object_matrix: rec.matrix.clone(),
                shown: rec.shown,
                visible: rec.visible.clone(),
                aspect: rec.aspect.clone(),
                color: object.and_then(|o| o.color).unwrap_or([1.0; 4]),
                pass_index: object.map(|o| o.pass_index as f32).unwrap_or(0.0),
            });
        }
        Ok(g)
    }

    fn add_shape(&mut self, bake: &Bake, rec: &MeshRec, attr_names: Vec<String>) -> Result<usize> {
        let vertices = rec.vertices;
        let attributes = &rec.attributes;
        let pos = read_attr(bake, attributes.get("position").context("no position attribute")?, vertices)?;
        let nrm = read_attr(bake, attributes.get("normal").context("no normal attribute")?, vertices)?;
        if pos.len() != vertices * 3 || nrm.len() != vertices * 3 {
            bail!("position / normal are not 3 floats per vertex");
        }
        let vertex_base = self.rest.len() as u32;
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in 0..vertices {
            let p = Vec3::new(pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]);
            lo = lo.min(p);
            hi = hi.max(p);
            self.rest.push(RestVertex { pos: [p.x, p.y, p.z, 1.0], nrm: [nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2], 0.0] });
        }

        let armature = match &rec.armature {
            Some(a) if rec.kind == "skinned" => Some(
                bake.doc.armatures.iter().position(|x| &x.name == a)
                    .with_context(|| format!("armature {a} is not in the bake"))?,
            ),
            _ => None,
        };
        let mut skin_base = None;
        if armature.is_some() {
            let j0 = read_u16(bake, attributes.get("joints").context("skinned without joints")?, vertices)?;
            let w0 = read_attr(bake, attributes.get("weights").context("skinned without weights")?, vertices)?;
            let j1 = attributes.get("joints2").map(|r| read_u16(bake, r, vertices)).transpose()?;
            let w1 = attributes.get("weights2").map(|r| read_attr(bake, r, vertices)).transpose()?;
            skin_base = Some(self.skin.len() as u32);
            for v in 0..vertices {
                let mut j = [0u16; 8];
                let mut w = [0.0f32; 8];
                for i in 0..4 {
                    j[i] = j0[4 * v + i];
                    w[i] = w0[4 * v + i];
                    if let (Some(j1), Some(w1)) = (&j1, &w1) {
                        j[4 + i] = j1[4 * v + i];
                        w[4 + i] = w1[4 * v + i];
                    }
                }
                let pack = |a: u16, b: u16| a as u32 | ((b as u32) << 16);
                self.skin.push(SkinVertex {
                    joints: [pack(j[0], j[1]), pack(j[2], j[3]), pack(j[4], j[5]), pack(j[6], j[7])],
                    weights: w,
                });
            }
        }
        let mut morphs = None;
        if let Some(m) = &rec.morphs {
            if !m.names.is_empty() {
                let n = m.names.len() * vertices * 3;
                let delta_base = self.deltas.len() as u32;
                self.deltas.extend(f32s(bake.bytes(m.deltas, n * 4)?));
                morphs = Some(Morphs { delta_base, count: m.names.len(), values: m.values });
            }
        }
        let kind = if armature.is_some() || morphs.is_some() {
            ShapeKind::Deforming { armature, morphs }
        } else {
            ShapeKind::Rigid
        };
        let (attr_base, stride, generated, attr_names) = self.add_attrs(bake, attributes, vertices, attr_names)?;
        let tris = rec.triangles.as_ref().context("no triangles")?;
        let index_base = self.indices.len() as u32;
        self.read_indices(bake, tris.offset, tris.count, vertices)?;
        self.shapes.push(Shape {
            vertex_base,
            vertex_count: vertices as u32,
            index_base,
            index_count: (tris.count * 3) as u32,
            attr_base,
            attr_stride: stride,
            attr_names,
            generated,
            skin_base,
            kind,
            lo,
            hi,
        });
        Ok(self.shapes.len() - 1)
    }

    /// A frame-by-frame shape: room for its largest geometry; the frame's one is uploaded when drawn.
    fn add_by_frame(&mut self, bake: &Bake, ri: usize, rec: &MeshRec, geoms: &[Geom], attr_names: Vec<String>)
                    -> Result<usize> {
        let vertices = geoms.iter().map(|g| g.vertices).max().unwrap_or(0);
        let tris = geoms.iter().map(|g| g.triangles.count).max().unwrap_or(0);
        let vertex_base = self.rest.len() as u32;
        self.rest.resize(self.rest.len() + vertices, RestVertex::zeroed());
        let has_generated = geoms.iter().any(|g| g.attributes.contains_key("generated"));
        let mut names = attr_names;
        if has_generated && !names.iter().any(|a| a == "generated") {
            names.push("generated".into());
        }
        let generated = names.iter().position(|a| a == "generated").map(|p| p as u32);
        let attr_base = self.attrs.len() as u32;
        self.attrs.resize(self.attrs.len() + names.len() * vertices, [0.0; 4]);
        let index_base = self.indices.len() as u32;
        self.indices.resize(self.indices.len() + tris * 3, 0);
        let _ = (bake, rec);
        let (mut lo, mut hi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        if let Some(g) = geoms.first() {
            if let Some(p) = g.attributes.get("position") {
                let pos = read_attr(bake, p, g.vertices)?;
                lo = Vec3::splat(f32::MAX);
                hi = Vec3::splat(f32::MIN);
                for v in pos.chunks_exact(3) {
                    lo = lo.min(Vec3::new(v[0], v[1], v[2]));
                    hi = hi.max(Vec3::new(v[0], v[1], v[2]));
                }
            }
        }
        self.shapes.push(Shape {
            vertex_base,
            vertex_count: vertices as u32,
            index_base,
            index_count: (tris * 3) as u32,
            attr_base,
            attr_stride: names.len() as u32,
            attr_names: names,
            generated,
            skin_base: None,
            kind: ShapeKind::ByFrame { rec: ri },
            lo,
            hi,
        });
        Ok(self.shapes.len() - 1)
    }

    /// Append the attributes `names` (plus Blender's Generated coordinates when the shape has them) of `vertices`
    /// vertices: (first vec4, vec4s per vertex, the Generated slot, the slots' names).
    fn add_attrs(&mut self, bake: &Bake, attributes: &BTreeMap<String, AttrRec>, vertices: usize, mut names: Vec<String>)
                 -> Result<(u32, u32, Option<u32>, Vec<String>)> {
        if attributes.contains_key("generated") && !names.iter().any(|a| a == "generated") {
            names.push("generated".into());
        }
        let generated = names.iter().position(|a| a == "generated").map(|p| p as u32);
        let base = self.attrs.len() as u32;
        let stride = names.len();
        self.attrs.resize(self.attrs.len() + stride * vertices, [0.0; 4]);
        for (si, an) in names.iter().enumerate() {
            // attributes a shape lacks (and "tangent:", which the bake does not hold) read as 0, mkr-shade's rule
            let Some(r) = find_attr(attributes, an) else { continue };
            let data = read_attr(bake, r, vertices)?;
            for v in 0..vertices {
                self.attrs[base as usize + v * stride + si] = widen(&data[v * r.comp..(v + 1) * r.comp]);
            }
        }
        Ok((base, stride as u32, generated, names))
    }

    fn read_indices(&mut self, bake: &Bake, offset: usize, count: usize, vertices: usize) -> Result<()> {
        let ids = bake.bytes(offset, count * 12)?;
        for b in ids.chunks_exact(4) {
            let i = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if i as usize >= vertices {
                bail!("triangle vertex {i} of {vertices}");
            }
            self.indices.push(i);
        }
        Ok(())
    }
}

/// One frame of a frame-by-frame shape, ready to upload into its ranges.
pub struct FrameGeom {
    pub geom: usize,
    pub posed: Vec<[f32; 8]>,
    pub attrs: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub triangles_offset: usize,
    /// (slot, first triangle, count)
    pub slots: Vec<[usize; 3]>,
}

/// The geometry a frame-by-frame mesh shows on bake frame `k` (None: hidden), with its posed vertices on time `time`
/// (positions between the frames around it when their geometries match), attributes as `shape` reads them.
pub fn by_frame_geom(bake: &Bake, rec: &MeshRec, shape: &Shape, k: usize, time: f64, with_data: bool)
                     -> Result<Option<FrameGeom>> {
    let bf = rec.by_frame.as_ref().context("not a frame-by-frame mesh")?;
    let gi = bf.frames.get(k).copied().unwrap_or(-1);
    if gi < 0 {
        return Ok(None);
    }
    let g = bf.geoms.get(gi as usize).context("frame geometry out of range")?;
    let read_pn = |g: &Geom| -> Result<(Vec<f32>, Vec<f32>)> {
        let p = read_attr(bake, g.attributes.get("position").context("geometry without position")?, g.vertices)?;
        let n = read_attr(bake, g.attributes.get("normal").context("geometry without normal")?, g.vertices)?;
        Ok((p, n))
    };
    let (mut p, mut n) = read_pn(g)?;
    let (k0, k1, t) = bracket(time, bake.doc.frames);
    if (k0 != k || t != 0.0) && (k0 != k1) {
        // the shutter's end: between the geometries of the frames around it, when they are this one's topology
        let at = |kk: usize| bf.frames.get(kk).copied().unwrap_or(-1);
        let (g0, g1) = (at(k0), at(k1));
        if g0 >= 0 && g1 >= 0 {
            let (a, b) = (&bf.geoms[g0 as usize], &bf.geoms[g1 as usize]);
            if a.vertices == g.vertices && b.vertices == g.vertices && a.triangles.offset == g.triangles.offset
                && b.triangles.offset == g.triangles.offset {
                let (pa, na) = read_pn(a)?;
                let (pb, nb) = read_pn(b)?;
                for i in 0..p.len() {
                    p[i] = pa[i] + (pb[i] - pa[i]) * t;
                    n[i] = na[i] + (nb[i] - na[i]) * t;
                }
            }
        }
    }
    let posed = (0..g.vertices)
        .map(|v| [p[3 * v], p[3 * v + 1], p[3 * v + 2], 1.0, n[3 * v], n[3 * v + 1], n[3 * v + 2], 0.0])
        .collect();
    let mut attrs = Vec::new();
    let mut indices = Vec::new();
    if with_data {
        let stride = shape.attr_names.len();
        attrs = vec![[0.0f32; 4]; stride * g.vertices];
        for (si, an) in shape.attr_names.iter().enumerate() {
            let Some(r) = find_attr(&g.attributes, an) else { continue };
            let data = read_attr(bake, r, g.vertices)?;
            for v in 0..g.vertices {
                attrs[v * stride + si] = widen(&data[v * r.comp..(v + 1) * r.comp]);
            }
        }
        let ids = bake.bytes(g.triangles.offset, g.triangles.count * 12)?;
        indices = ids.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    }
    Ok(Some(FrameGeom { geom: gi as usize, posed, attrs, indices, triangles_offset: g.triangles.offset, slots: g.slots.clone() }))
}

// ================================================================================================= one time of a frame
/// A 3x4 affine matrix as three rows (what the shaders apply).
pub fn rows(m: &Mat4) -> [[f32; 4]; 3] {
    let r = |i: usize| [m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]];
    [r(0), r(1), r(2)]
}

/// Linear blend of two 3x4 matrices.
pub fn mix(a: &Mat4, b: &Mat4, t: f32) -> Mat4 {
    if t == 0.0 {
        return *a;
    }
    Mat4::from_cols(
        a.x_axis + (b.x_axis - a.x_axis) * t,
        a.y_axis + (b.y_axis - a.y_axis) * t,
        a.z_axis + (b.z_axis - a.z_axis) * t,
        a.w_axis + (b.w_axis - a.w_axis) * t,
    )
}

/// A fractional bake frame as the two frames around it and how far between them (clamped to the clip).
pub fn bracket(time: f64, frames: usize) -> (usize, usize, f32) {
    let last = frames.saturating_sub(1) as f64;
    let t = time.clamp(0.0, last);
    let k0 = t.floor() as usize;
    let k1 = (k0 + 1).min(frames.saturating_sub(1));
    (k0, k1, (t - k0 as f64) as f32)
}

/// A per-frame matrix record on a fractional frame.
pub fn matrix_at(bake: &Bake, m: &MatrixRec, time: f64) -> Result<Mat4> {
    let (k0, k1, t) = bracket(time, bake.doc.frames);
    let at = |k: usize| -> Result<Mat4> {
        let k = if m.frames == 1 { 0 } else { k };
        Ok(affine(bake.f(m.offset + 12 * k, 12)?))
    };
    Ok(mix(&at(k0)?, &at(k1)?, t))
}

/// The scene at one time: joint rows per bone (all armatures), each mesh's matrix, each deforming shape's posing
/// matrix and active shape keys.
pub struct Pose {
    pub joints: Vec<[f32; 4]>,
    pub joint_base: Vec<u32>,
    /// Per mesh: its world matrix (identity when its shape is posed into the world).
    pub matrices: Vec<Mat4>,
    /// Per shape: the matrix its posing applies (the world matrix of the mesh that shows it).
    pub shape_matrices: Vec<Mat4>,
    /// (key, weight) pairs; each shape's run is morph_runs[shape] = (first, count).
    pub morphs: Vec<[f32; 2]>,
    pub morph_runs: Vec<(u32, u32)>,
}

impl Pose {
    pub fn at(bake: &Bake, geo: &Geometry, time: f64) -> Result<Pose> {
        let lit = bake.lit()?;
        let (k0, k1, t) = bracket(time, bake.doc.frames);
        let mut joints = Vec::new();
        let mut joint_base = Vec::new();
        for a in &bake.doc.armatures {
            joint_base.push((joints.len() / 3) as u32);
            let n = a.bones.len();
            let j0 = bake.f(a.offset + k0 * n * 12, n * 12)?;
            let j1 = bake.f(a.offset + k1 * n * 12, n * 12)?;
            for b in 0..n {
                let m = mix(&affine(&j0[b * 12..b * 12 + 12]), &affine(&j1[b * 12..b * 12 + 12]), t);
                joints.extend(rows(&m));
            }
        }
        let mut matrices = Vec::with_capacity(geo.meshes.len());
        let mut shape_matrices = vec![Mat4::IDENTITY; geo.shapes.len()];
        for m in &geo.meshes {
            matrices.push(match &m.matrix {
                Some(mr) => matrix_at(bake, mr, time)?,
                None => Mat4::IDENTITY,
            });
        }
        // a deforming shape belongs to one mesh: its record's matrix poses it
        for (ri, rec) in lit.meshes.iter().enumerate() {
            let Some(mesh) = geo.meshes.get(ri) else { continue };
            if let ShapeKind::Deforming { .. } = geo.shapes[mesh.shape].kind {
                if let Some(mr) = &rec.matrix {
                    shape_matrices[mesh.shape] = matrix_at(bake, mr, time)?;
                }
            }
        }
        let mut morphs = Vec::new();
        let mut morph_runs = Vec::with_capacity(geo.shapes.len());
        for s in &geo.shapes {
            let first = morphs.len() as u32;
            if let ShapeKind::Deforming { morphs: Some(keys), .. } = &s.kind {
                let v0 = bake.f(keys.values + k0 * keys.count, keys.count)?;
                let v1 = bake.f(keys.values + k1 * keys.count, keys.count)?;
                for i in 0..keys.count {
                    let w = v0[i] + (v1[i] - v0[i]) * t;
                    if w != 0.0 {
                        morphs.push([f32::from_bits(i as u32), w]);
                    }
                }
            }
            morph_runs.push((first, morphs.len() as u32 - first));
        }
        Ok(Pose { joints, joint_base, matrices, shape_matrices, morphs, morph_runs })
    }
}

/// Whether mesh `m` (the bake's record `rec`) shows on bake frame `k` for `output`.
pub fn shown(bake: &Bake, rec: &MeshRec, m: &Mesh, k: usize, output: &str) -> Result<bool> {
    if m.aspect.as_deref().is_some_and(|a| a != output) {
        return Ok(false);
    }
    if let Some(bf) = &rec.by_frame {
        if bf.frames.get(k).copied().unwrap_or(-1) < 0 {
            return Ok(false);
        }
    }
    match m.shown {
        Some(off) => Ok(bake.f(off + k, 1)?[0] > 0.5),
        None => Ok(true),
    }
}

/// glTF's axes to Blender's as a matrix acting on points: Blender (x, y, z) = glTF (x, -z, y).
pub const TO_BLENDER: Mat4 = crate::camera::TO_BLENDER;

/// A glTF-axes object matrix as Blender's (A^T M A).
pub fn blender_matrix(m: &Mat4) -> Mat4 {
    TO_BLENDER * *m * TO_BLENDER.inverse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brackets_clamp_to_the_clip() {
        assert_eq!(bracket(3.0, 10), (3, 4, 0.0));
        assert_eq!(bracket(3.25, 10), (3, 4, 0.25));
        assert_eq!(bracket(-0.2, 10), (0, 1, 0.0));
        assert_eq!(bracket(9.5, 10), (9, 9, 0.0));
        assert_eq!(bracket(0.0, 1), (0, 0, 0.0));
    }

    #[test]
    fn attributes_widen_like_blender() {
        assert_eq!(widen(&[0.5]), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(widen(&[0.1, 0.2]), [0.1, 0.2, 0.0, 1.0]);
        assert_eq!(widen(&[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(widen(&[1.0, 2.0, 3.0, 0.5]), [1.0, 2.0, 3.0, 0.5]);
    }

    #[test]
    fn blender_axes() {
        // a glTF translation along +Y (up) is Blender's +Z
        let m = blender_matrix(&Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)));
        assert!((m.w_axis.truncate() - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
    }

    /// A frame-by-frame mesh whose one geometry (in its own space) rides its per-frame matrix: the bake stores an
    /// object that only moves once, and the matrix places it, between frames too (motion blur's shutter).
    #[test]
    fn a_frame_by_frame_mesh_is_placed_by_its_matrix() {
        let dir = std::env::temp_dir().join(format!("mkr_scene_byframe_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pos: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let nrm: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut lit = Vec::new();
        lit.extend_from_slice(bytemuck::cast_slice(&pos));
        lit.extend_from_slice(bytemuck::cast_slice(&nrm));
        lit.extend_from_slice(bytemuck::cast_slice(&[0u32, 1, 2]));
        std::fs::write(dir.join("lit.bin"), &lit).unwrap();
        // two frames of a 3x4 matrix, column by column: the identity moved 1, then 2 along x
        let mut m = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        m.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
        std::fs::write(dir.join("bake.bin"), bytemuck::cast_slice(&m)).unwrap();
        let doc = serde_json::json!({"version": 5, "fps": 30.0, "frame0": 1, "frames": 2,
            "outputs": [{"name": "o", "size": [4, 4]}],
            "lit": {"view": {"lut": {"file": "view_lut.bin", "n": 2, "lo": -1.0, "hi": 1.0}},
                    "meshes": [{"name": "type", "kind": "by_frame", "materials": [null],
                                "matrix": {"offset": 0, "frames": 2},
                                "by_frame": {"frames": [0, 0], "geoms": [{"vertices": 3, "slots": [[0, 0, 1]],
                                    "triangles": {"offset": 72, "count": 1},
                                    "attributes": {"position": {"offset": 0, "comp": 3, "type": "f32"},
                                                   "normal": {"offset": 36, "comp": 3, "type": "f32"}}}]}}]}});
        std::fs::write(dir.join("bake.json"), doc.to_string()).unwrap();
        let bake = Bake::open(&dir).unwrap();
        let geo = Geometry::load(&bake, &|_: &str| Vec::new()).unwrap();
        assert!(matches!(geo.shapes[0].kind, ShapeKind::ByFrame { .. }) && geo.meshes[0].matrix.is_some());
        for (time, x) in [(0.0, 1.0f32), (1.0, 2.0), (0.5, 1.5)] {
            let pose = Pose::at(&bake, &geo, time).unwrap();
            assert!((pose.matrices[0].w_axis.x - x).abs() < 1e-6, "time {time}: {:?}", pose.matrices[0]);
        }
        // its geometry stays in its own space: the same on both frames
        let rec = &bake.lit().unwrap().meshes[0];
        for k in 0..2 {
            let g = by_frame_geom(&bake, rec, &geo.shapes[0], k, k as f64, false).unwrap().unwrap();
            assert_eq!(g.posed[1][..3], [1.0, 0.0, 0.0]);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
