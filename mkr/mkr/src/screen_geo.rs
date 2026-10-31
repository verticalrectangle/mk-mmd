//! Screen type as fitted geometry (bake_lit.py's docstring: SCREEN; mkmmd/core/fitgeo.py writes it and its `decode` is
//! the reference this follows): a word's frame rebuilt on the CPU from its geometry's prototype shapes, its islands
//! (each a prototype through a map) and the frame's pose, handed over as `scene::by_frame_geom` hands over a frame of a
//! frame-by-frame mesh, for the layers to upload.
//!
//! A pose (16-byte aligned in lit.bin): u32 kind (0: a map per island; 1: one map of the whole word, after each
//! island's map in the pose of bake frame `ref`; 2: a bent map per island), u32 attrs (1: island attribute values
//! follow; 0: those of the pose it follows), u32 ref, u32 0; f32 maps (kinds 0 and 2: islands x 16, kind 1: 16), each a
//! 4x4 M row-major on q = (x, y, z, 1): x' = (M0 . q) / (M3 . q), y' = M1 . q, z' = (M2 . q) / (M3 . q); kind 2: then
//! islands x 9 f32, each Q (3x3, row-major) adding Q . (u^2, u v, v^2) to (x', y', z'), u = x - ox and v = z - oz about
//! the island's prototype's first vertex; then with attrs, per `fit.attrs` entry islands x comp f32. A normal turns
//! with the cofactor det(J) J^-T of the map's Jacobian at the point, then is normalised.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::bake::{Bake, ByFrame, FitRec, MeshRec};
use crate::scene::{self, FrameGeom, Shape};

/// Pose kinds.
const ISLANDS: u32 = 0;
const WORD: u32 = 1;
const BENT: u32 = 2;

/// A decoded frame of a fitted word, its vertices island by island (its geometry's order).
pub struct Frame {
    /// The geometry (index into `by_frame.geoms`) that holds it.
    pub geom: usize,
    pub position: Vec<[f32; 3]>,
    /// Unit normals.
    pub normal: Vec<[f32; 3]>,
    /// Per attribute key its components and values (vertices x components), when asked for.
    pub attrs: BTreeMap<String, (usize, Vec<f32>)>,
    /// Three vertex numbers per triangle, grouped by slot as the geometry's `slots` say, when asked for.
    pub triangles: Vec<u32>,
}

/// The frame a word of screen type shows on bake frame `k`, ready to upload: `scene::by_frame_geom`'s answer for a
/// fitted record (one without poses is handed to it). Positions and normals on `time`: between the frames around it
/// when they hold the same geometry, as by_frame_geom blends them; attributes in `shape.attr_names` order.
pub fn frame_geom(bake: &Bake, rec: &MeshRec, shape: &Shape, k: usize, time: f64, with_data: bool)
                  -> Result<Option<FrameGeom>> {
    let bf = rec.by_frame.as_ref().context("not a frame-by-frame mesh")?;
    if bf.poses.is_none() {
        return scene::by_frame_geom(bake, rec, shape, k, time, with_data);
    }
    let Some(cur) = decode(bake, bf, k, with_data).with_context(|| format!("screen type {}", rec.name))? else {
        return Ok(None);
    };
    let (mut p, mut n) = (cur.position, cur.normal);
    let (k0, k1, t) = scene::bracket(time, bake.doc.frames);
    if (k0 != k || t != 0.0) && k0 != k1 {
        // the shutter's end: between the frames around it, when they hold this geometry
        let held = |kk: usize| bf.frames.get(kk).copied().unwrap_or(-1) == cur.geom as i64;
        if held(k0) && held(k1) {
            let a = decode(bake, bf, k0, false)?.context("a held frame decodes to nothing")?;
            let b = decode(bake, bf, k1, false)?.context("a held frame decodes to nothing")?;
            for v in 0..p.len() {
                for c in 0..3 {
                    p[v][c] = a.position[v][c] + (b.position[v][c] - a.position[v][c]) * t;
                    n[v][c] = a.normal[v][c] + (b.normal[v][c] - a.normal[v][c]) * t;
                }
            }
        }
    }
    let posed = p.iter().zip(&n).map(|(p, n)| [p[0], p[1], p[2], 1.0, n[0], n[1], n[2], 0.0]).collect();
    let mut attrs = Vec::new();
    if with_data {
        let stride = shape.attr_names.len();
        attrs = vec![[0.0f32; 4]; stride * p.len()];
        for (si, name) in shape.attr_names.iter().enumerate() {
            // attributes a word lacks read as 0, as by_frame_geom leaves them
            let Some((comp, vals)) = cur.attrs.get(name) else { continue };
            for v in 0..p.len() {
                attrs[v * stride + si] = scene::widen(&vals[v * comp..(v + 1) * comp]);
            }
        }
    }
    let g = &bf.geoms[cur.geom];
    Ok(Some(FrameGeom {
        geom: cur.geom,
        posed,
        attrs,
        indices: cur.triangles,
        triangles_offset: g.triangles.offset,
        slots: g.slots.clone(),
    }))
}

/// Bake frame `k` of a fitted record (None: hidden): positions and normals, and with `with_data` its attributes and
/// triangles.
pub fn decode(bake: &Bake, bf: &ByFrame, k: usize, with_data: bool) -> Result<Option<Frame>> {
    let poses = bf.poses.as_ref().context("not a fitted record (no poses)")?;
    let gi = bf.frames.get(k).copied().unwrap_or(-1);
    let at = poses.get(k).copied().unwrap_or(-1);
    if gi < 0 || at < 0 {
        return Ok(None);
    }
    let gi = gi as usize;
    let g = bf.geoms.get(gi).context("frame geometry out of range")?;
    let fit = g.fit.as_ref().context("a fitted record's geometry without its fit")?;
    let n = fit.islands.count;
    let islands = u32s(bake, fit.islands.table, n * 4)?;
    let protos = u32s(bake, fit.protos.table, fit.protos.count * 4)?;
    let pa = &fit.protos.attributes;
    let pv = fit.protos.vertices;
    let ppos = f32s(bake, pa.get("position").context("prototypes without positions")?.offset, pv * 3)?;
    let pnrm = f32s(bake, pa.get("normal").context("prototypes without normals")?.offset, pv * 3)?;
    let pose = Pose::read(bake, at as usize, n, fit)?;
    // the islands' maps (and bent terms), and the whole word's map after them for a word pose
    let (islands_pose, word_map, vals) = match pose.kind {
        ISLANDS | BENT => {
            let vals = pose.attrs.clone();
            (pose, None, vals)
        }
        WORD => {
            let ra = poses.get(pose.reference).copied().unwrap_or(-1);
            if ra < 0 {
                bail!("frame {k}'s pose follows frame {}, which has none", pose.reference);
            }
            let r = Pose::read(bake, ra as usize, n, fit)?;
            if r.kind == WORD {
                bail!("frame {k}'s pose follows frame {}, whose pose is not one of islands", pose.reference);
            }
            let vals = pose.attrs.or_else(|| r.attrs.clone());
            (r, Some(pose.maps[0]), vals)
        }
        other => bail!("frame {k}: a pose of kind {other}"),
    };
    let vals = vals.context("a pose without island attributes, nor one it follows")?;
    let mut position = Vec::with_capacity(g.vertices);
    let mut normal = Vec::with_capacity(g.vertices);
    let mut owner = Vec::with_capacity(g.vertices);
    for i in 0..n {
        let (p, first) = (islands[4 * i] as usize, islands[4 * i + 2] as usize);
        if p >= fit.protos.count || first != position.len() {
            bail!("island {i}: prototype {p}, first vertex {first} (vertex {} expected)", position.len());
        }
        let (v0, count) = (protos[4 * p] as usize, protos[4 * p + 1] as usize);
        if v0 + count > pv {
            bail!("prototype {p}: vertices {v0}..{} of {pv}", v0 + count);
        }
        let m = &islands_pose.maps[i];
        // a bent map's quadratic terms are about its prototype's first vertex
        let bent = islands_pose.bent.as_ref().map(|q| (&q[i], [ppos[3 * v0] as f64, ppos[3 * v0 + 2] as f64]));
        for j in v0..v0 + count {
            let q = [ppos[3 * j] as f64, ppos[3 * j + 1] as f64, ppos[3 * j + 2] as f64];
            let nq = [pnrm[3 * j] as f64, pnrm[3 * j + 1] as f64, pnrm[3 * j + 2] as f64];
            let (mut x, mut nn) = map(m, bent, q, nq);
            if let Some(g) = &word_map {
                (x, nn) = map(g, None, x, nn);
            }
            position.push([x[0] as f32, x[1] as f32, x[2] as f32]);
            normal.push(unit(nn));
            owner.push((i, j));
        }
    }
    if position.len() != g.vertices {
        bail!("its islands make {} vertices, its geometry says {}", position.len(), g.vertices);
    }
    let mut attrs = BTreeMap::new();
    let mut triangles = Vec::new();
    if with_data {
        for (key, spec) in pa.iter().filter(|(key, _)| *key != "position" && *key != "normal") {
            let a = f32s(bake, spec.offset, pv * spec.comp)?;
            let c = spec.comp;
            let mut out = Vec::with_capacity(owner.len() * c);
            for &(_, j) in &owner {
                out.extend_from_slice(&a[j * c..(j + 1) * c]);
            }
            attrs.insert(key.clone(), (c, out));
        }
        let mut at = 0;
        for (key, c) in &fit.attrs {
            let island = &vals[at..at + n * c];
            at += n * c;
            let mut out = Vec::with_capacity(owner.len() * c);
            for &(i, _) in &owner {
                out.extend_from_slice(&island[i * c..(i + 1) * c]);
            }
            attrs.insert(key.clone(), (*c, out));
        }
        let ptri = u32s(bake, fit.protos.triangles.offset, fit.protos.triangles.count * 3)?;
        triangles.reserve(g.triangles.count * 3);
        for i in 0..n {
            let (p, first, t_first) = (islands[4 * i] as usize, islands[4 * i + 2], islands[4 * i + 3] as usize);
            let (t0, tc) = (protos[4 * p + 2] as usize, protos[4 * p + 3] as usize);
            if t_first * 3 != triangles.len() || t0 + tc > fit.protos.triangles.count {
                bail!("island {i}: first triangle {t_first}, prototype triangles {t0}..{}", t0 + tc);
            }
            triangles.extend(ptri[3 * t0..3 * (t0 + tc)].iter().map(|&v| v + first));
        }
        if triangles.len() != g.triangles.count * 3 {
            bail!("its islands make {} triangles, its geometry says {}", triangles.len() / 3, g.triangles.count);
        }
    }
    Ok(Some(Frame { geom: gi, position, normal, attrs, triangles }))
}

/// A pose read from lit.bin: its kind, the bake frame a word pose follows, its maps, a bent pose's quadratic terms
/// and its island attribute values (per `fit.attrs` entry islands x comp, concatenated).
struct Pose {
    kind: u32,
    reference: usize,
    maps: Vec<[f64; 16]>,
    /// Per island Q (3x3, row-major) on (u^2, u v, v^2), u and v about the island's prototype's first vertex.
    bent: Option<Vec<[f64; 9]>>,
    attrs: Option<Vec<f32>>,
}

impl Pose {
    fn read(bake: &Bake, at: usize, islands: usize, fit: &FitRec) -> Result<Pose> {
        let head = u32s(bake, at, 4)?;
        let (kind, has, reference) = (head[0], head[1], head[2] as usize);
        let count = match kind {
            ISLANDS | BENT => islands,
            WORD => 1,
            other => bail!("a pose of kind {other} at byte {at}"),
        };
        let mut o = at + 16;
        let maps = f32s(bake, o, count * 16)?.chunks_exact(16).map(|m| std::array::from_fn(|i| m[i] as f64)).collect();
        o += 64 * count;
        let mut bent = None;
        if kind == BENT {
            bent = Some(f32s(bake, o, islands * 9)?.chunks_exact(9).map(|m| std::array::from_fn(|i| m[i] as f64)).collect());
            o += 36 * islands;
        }
        let attrs = match has {
            0 => None,
            _ => Some(f32s(bake, o, islands * fit.attrs.iter().map(|(_, c)| c).sum::<usize>())?),
        };
        Ok(Pose { kind, reference, maps, bent, attrs })
    }
}

/// A point q and a normal nq at it through the map m, plus a bent map's quadratic terms (Q, the origin's x and z):
/// (q's image, the normal turned by the cofactor of the Jacobian at q, unnormalised; the cofactor of J with rows j0,
/// j1, j2 has rows j1 x j2, j2 x j0, j0 x j1).
fn map(m: &[f64; 16], bent: Option<(&[f64; 9], [f64; 2])>, q: [f64; 3], nq: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let h = |r: usize| m[4 * r] * q[0] + m[4 * r + 1] * q[1] + m[4 * r + 2] * q[2] + m[4 * r + 3];
    let (h0, h1, h2, w) = (h(0), h(1), h(2), h(3));
    let (x, z) = (h0 / w, h2 / w);
    let mut p = [x, h1, z];
    let mut j = [
        [(m[0] - x * m[12]) / w, (m[1] - x * m[13]) / w, (m[2] - x * m[14]) / w],
        [m[4], m[5], m[6]],
        [(m[8] - z * m[12]) / w, (m[9] - z * m[13]) / w, (m[10] - z * m[14]) / w],
    ];
    if let Some((qm, o)) = bent {
        let (u, v) = (q[0] - o[0], q[2] - o[1]);
        for r in 0..3 {
            let (a, b, c) = (qm[3 * r], qm[3 * r + 1], qm[3 * r + 2]);
            p[r] += a * u * u + b * u * v + c * v * v;
            j[r][0] += 2.0 * a * u + b * v;
            j[r][2] += b * u + 2.0 * c * v;
        }
    }
    let n = [dot(cross(j[1], j[2]), nq), dot(cross(j[2], j[0]), nq), dot(cross(j[0], j[1]), nq)];
    (p, n)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(n: [f64; 3]) -> [f32; 3] {
    let l = dot(n, n).sqrt();
    if l > 0.0 { [(n[0] / l) as f32, (n[1] / l) as f32, (n[2] / l) as f32] } else { [0.0; 3] }
}

fn u32s(bake: &Bake, offset: usize, n: usize) -> Result<Vec<u32>> {
    Ok(bake.bytes(offset, n * 4)?.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn f32s(bake: &Bake, offset: usize, n: usize) -> Result<Vec<f32>> {
    Ok(bake.bytes(offset, n * 4)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ShapeKind;
    use glam::Vec3;
    use serde_json::Value;
    use std::path::Path;

    /// tests/fixtures/screen_geo: a synthetic word baked as fitted geometry by mkmmd/core/fitgeo.py and the frames it
    /// was made of (`python tests/test_fitgeo.py` writes both).
    fn fixture() -> (Bake, Value) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/screen_geo");
        let bake = Bake::open(&dir).expect("the screen_geo fixture bake");
        let raw = serde_json::from_str(&std::fs::read_to_string(dir.join("raw.json")).unwrap()).unwrap();
        (bake, raw)
    }

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    }

    fn word(bake: &Bake) -> &MeshRec {
        &bake.lit().unwrap().screen.as_ref().unwrap().meshes[0]
    }

    #[test]
    fn a_fitted_word_decodes_to_the_raw_frames_it_was_made_of() {
        let (bake, raw) = fixture();
        let bf = word(&bake).by_frame.as_ref().unwrap();
        let tol = raw["tol"].as_f64().unwrap();
        let mut shown = 0;
        for (k, want) in raw["frames"].as_array().unwrap().iter().enumerate() {
            let got = decode(&bake, bf, k, true).unwrap();
            if want.is_null() {
                assert!(got.is_none(), "frame {k} is hidden");
                continue;
            }
            let got = got.unwrap();
            shown += 1;
            let p = floats(&want["position"]);
            let n = floats(&want["normal"]);
            assert_eq!(got.position.len() * 3, p.len(), "frame {k}: vertices");
            for v in 0..got.position.len() {
                let d: f64 = (0..3).map(|c| (got.position[v][c] as f64 - p[3 * v + c]).powi(2)).sum::<f64>().sqrt();
                assert!(d <= tol, "frame {k} vertex {v}: {d} from its raw position");
                let cos: f64 = (0..3).map(|c| got.normal[v][c] as f64 * n[3 * v + c]).sum();
                assert!(cos > 0.999, "frame {k} vertex {v}: normal off by acos {cos}");
            }
            for (key, a) in want["attrs"].as_object().unwrap() {
                let (_, vals) = got.attrs.get(key).unwrap_or_else(|| panic!("frame {k}: no attribute {key}"));
                let a = floats(a);
                assert_eq!(vals.len(), a.len(), "frame {k} {key}");
                assert!(vals.iter().zip(&a).all(|(x, y)| *x as f64 == *y), "frame {k}: {key} differs");
            }
            assert_eq!(got.triangles.len(), want["triangles"].as_u64().unwrap() as usize * 3, "frame {k}: triangles");
            assert!(got.triangles.iter().all(|&i| (i as usize) < got.position.len()));
        }
        assert_eq!(shown, 8);
    }

    fn shape(names: &[&str], vertices: u32) -> Shape {
        Shape {
            vertex_base: 0,
            vertex_count: vertices,
            index_base: 0,
            index_count: 0,
            attr_base: 0,
            attr_stride: names.len() as u32,
            attr_names: names.iter().map(|s| s.to_string()).collect(),
            generated: None,
            skin_base: None,
            kind: ShapeKind::ByFrame { rec: 0 },
            lo: Vec3::splat(-1.0),
            hi: Vec3::splat(1.0),
        }
    }

    #[test]
    fn frame_geom_hands_the_layers_a_frame_as_by_frame_geom_does() {
        let (bake, raw) = fixture();
        let rec = word(&bake);
        let bf = rec.by_frame.as_ref().unwrap();
        let k = 4; // the alpha fade: island attributes of its own
        let want = &raw["frames"][k];
        let s = shape(&["uv:", "attr:mk_alpha", "color:none"], 0);
        let g = frame_geom(&bake, rec, &s, k, k as f64, true).unwrap().unwrap();
        let geom = &bf.geoms[g.geom];
        assert_eq!(g.triangles_offset, geom.triangles.offset);
        assert_eq!(g.slots, geom.slots);
        assert_eq!(g.indices.len(), geom.triangles.count * 3);
        let (p, uv, alpha) = (floats(&want["position"]), floats(&want["attrs"]["uv:"]), floats(&want["attrs"]["attr:mk_alpha"]));
        assert_eq!(g.posed.len(), geom.vertices);
        for v in 0..g.posed.len() {
            assert!((0..3).all(|c| (g.posed[v][c] as f64 - p[3 * v + c]).abs() <= 1e-5));
            assert_eq!((g.posed[v][3], g.posed[v][7]), (1.0, 0.0));
            assert_eq!(g.attrs[3 * v], [uv[2 * v] as f32, uv[2 * v + 1] as f32, 0.0, 1.0]);
            let a = alpha[v] as f32;
            assert_eq!(g.attrs[3 * v + 1], [a, a, a, 1.0]);
            assert_eq!(g.attrs[3 * v + 2], [0.0; 4]);
        }
        assert!(frame_geom(&bake, rec, &s, 6, 6.0, true).unwrap().is_none(), "frame 6 is hidden");
    }

    #[test]
    fn the_shutter_blends_two_frames_of_one_geometry() {
        let (bake, raw) = fixture();
        let rec = word(&bake);
        let s = shape(&[], 0);
        let mid = frame_geom(&bake, rec, &s, 1, 1.5, false).unwrap().unwrap();
        let (a, b) = (floats(&raw["frames"][1]["position"]), floats(&raw["frames"][2]["position"]));
        for v in 0..mid.posed.len() {
            for c in 0..3 {
                let want = 0.5 * (a[3 * v + c] + b[3 * v + c]);
                assert!((mid.posed[v][c] as f64 - want).abs() <= 1e-5, "vertex {v}");
            }
        }
    }
}
