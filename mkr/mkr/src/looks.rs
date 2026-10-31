//! The render-time looks mkr draws (docs/design.md: Shots: Looks: silhouette and vector; Rendering: mkr's looks): a
//! frame or layer in a shot's or a [[medium]] window's silhouette or vector look is not lit; mkr draws the passes
//! mkmmd.blender.styles renders in Blender for it (looks_gpu.rs), writes them into the request's spool and mk render's
//! worker processes compose them with the same compose_silhouette / compose_vector (mkmmd/render/compose.py).
//!
//! The request's `looks` (mkmmd/render/mkr.py `look_jobs`, its object patterns sorted on the bake's records as
//! styles.Looks sorts the scene's objects):
//!
//!   {"spool": DIR | null, "limit": n, "outputs": {output: {
//!       "sets":   [{kind: "silhouette", hide, subject, hard, soft, type, knock: [object names], subject_color,
//!                   accent_color: [r, g, b] linear, samples} |
//!                  {kind: "vector", hide, subject, type, ids: {material: id}, tex: bool, light: [x, y, z], samples}],
//!       "jobs":   [{id, role: frame | plate | bare | matte, frame, shot, set, camera, look}],
//!       "frames": {frame: job id}}}}
//!
//! A frame of the cut in a look is drawn by its frame job; a layer item (layers.rs) seen through a look names its job
//! (`looks` {output: job id}). Each job's passes go to SPOOL/<job>.look (written as <job>.part, then renamed), once fewer
//! than `limit` such files wait there (the composer deletes each when it is done):
//!
//!   "MKRLOOK1", u32 header length, header JSON {job, passes: {name: {offset, shape, type}}}, then the arrays at their
//!   byte offsets from the file's start (u8 / u16 / f32, little-endian, top row first):
//!   silhouette  flat (h, w, 4) u8: Workbench, flat light, the subject and hard accents in their colours, straight RGBA
//!                                  through the Standard view, Workbench's antialiasing (looks_gpu.rs)
//!               soft (h, w, 2) u8: EEVEE, the soft accents' alpha and their opaque share (the AOV), the subject and hard
//!                                  accents holding them out
//!               type, knock, screen_knock (h, w, 4) u8: EEVEE with the lights off, the type (held out by the subject and
//!                                  hard accents), the knock-out objects, the screen type reversed out of the figure,
//!                                  straight RGBA through the Standard view
//!               matte (2h, 2w) u8: a matte's figure, Workbench's coverage of the subject at twice the frame's size
//!   vector      id (2h, 2w) u16, depth (2h, 2w) f32 m, shade (2h, 2w) u8, tex (2h, 2w, 3) u8, type (h, w, 4) u8,
//!               matte (2h, 2w) u8: a matte job's figure, as the silhouette's (styles.Looks.matte draws both alike)

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::bake::Bake;
use crate::frames::shows;
use crate::info::{lens, mesh_shown};
use crate::materials::Support;

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Silhouette,
    Vector,
}

/// A look's objects as its passes draw them (names of the bake's look records; a lit mesh is its record's object's).
#[derive(Deserialize, Debug, Clone)]
pub struct LookSet {
    pub kind: Kind,
    #[serde(default)]
    pub hide: Vec<String>,
    #[serde(default)]
    pub subject: Vec<String>,
    #[serde(default)]
    pub hard: Vec<String>,
    #[serde(default)]
    pub soft: Vec<String>,
    #[serde(default, rename = "type")]
    pub kind_type: Vec<String>,
    #[serde(default)]
    pub knock: Vec<String>,
    /// The silhouette's flat colours, linear (Object.color as the look sets it).
    #[serde(default)]
    pub subject_color: [f32; 3],
    #[serde(default)]
    pub accent_color: [f32; 3],
    /// EEVEE samples of the soft, type and knock passes.
    #[serde(default = "sixteen")]
    pub samples: u32,
    /// The vector look's material ids (vector_table's: 1..n).
    #[serde(default)]
    pub ids: BTreeMap<String, u32>,
    /// The vector look draws the texture pass (a drawn or cel tone).
    #[serde(default)]
    pub tex: bool,
    /// Workbench's light_direction (toward the light, world).
    #[serde(default)]
    pub light: [f32; 3],
}

fn sixteen() -> u32 {
    16
}

/// What a pass draws an object as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Subject,
    Hard,
    Soft,
    Type,
    Knock,
}

impl LookSet {
    /// The class of the object `name` in this look (None: hidden, or not drawn).
    pub fn class_of(&self, name: &str) -> Option<Class> {
        let has = |v: &Vec<String>| v.iter().any(|n| n == name);
        if has(&self.subject) {
            Some(Class::Subject)
        } else if has(&self.hard) {
            Some(Class::Hard)
        } else if has(&self.soft) {
            Some(Class::Soft)
        } else if has(&self.kind_type) {
            Some(Class::Type)
        } else if has(&self.knock) {
            Some(Class::Knock)
        } else {
            None
        }
    }

    /// Every object one of its passes draws.
    pub fn drawn(&self) -> BTreeSet<&str> {
        [&self.subject, &self.hard, &self.soft, &self.kind_type, &self.knock].into_iter().flatten().map(String::as_str).collect()
    }
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A frame of the cut.
    Frame,
    /// Another shot's frame (plate/), or a glitch's shot with its objects hidden (bare/).
    Plate,
    Bare,
    /// A matte's figure (matte/) and its frame without it (back/).
    Matte,
}

#[derive(Deserialize, Debug, Clone)]
pub struct LookJob {
    pub id: usize,
    pub role: Role,
    pub frame: i64,
    pub shot: String,
    pub set: usize,
    pub camera: String,
    #[serde(default)]
    pub look: String,
}

#[derive(Debug, Clone, Default)]
pub struct OutputLooks {
    pub sets: Vec<LookSet>,
    pub jobs: Vec<LookJob>,
    /// The cut's frames in a look: frame -> job.
    pub frames: BTreeMap<i64, usize>,
}

#[derive(Debug, Clone, Default)]
pub struct LookRequest {
    pub spool: Option<PathBuf>,
    pub limit: usize,
    pub outputs: BTreeMap<String, OutputLooks>,
}

#[derive(Deserialize)]
struct RawOutput {
    #[serde(default)]
    sets: Vec<LookSet>,
    #[serde(default)]
    jobs: Vec<LookJob>,
    #[serde(default)]
    frames: BTreeMap<String, usize>,
}

impl LookRequest {
    pub fn parse(v: &Value) -> Result<LookRequest> {
        let spool = v.get("spool").and_then(|s| s.as_str()).map(PathBuf::from);
        let limit = v.get("limit").and_then(|l| l.as_u64()).unwrap_or(8).max(1) as usize;
        let mut outputs = BTreeMap::new();
        if let Some(o) = v.get("outputs").and_then(|o| o.as_object()) {
            for (name, rec) in o {
                let raw: RawOutput = serde_json::from_value(rec.clone()).with_context(|| format!("looks of output {name}"))?;
                let mut frames = BTreeMap::new();
                for (f, j) in raw.frames {
                    frames.insert(f.parse::<i64>().with_context(|| format!("looks {name}: {f} is not a frame"))?, j);
                }
                for (i, j) in raw.jobs.iter().enumerate() {
                    if j.id != i {
                        bail!("looks {name}: job {i} is numbered {}", j.id);
                    }
                    if j.set >= raw.sets.len() {
                        bail!("looks {name}: job {i} names set {} of {}", j.set, raw.sets.len());
                    }
                }
                if let Some((f, j)) = frames.iter().find(|(_, j)| **j >= raw.jobs.len()) {
                    bail!("looks {name}: frame {f} names job {j} of {}", raw.jobs.len());
                }
                outputs.insert(name.clone(), OutputLooks { sets: raw.sets, jobs: raw.jobs, frames });
            }
        }
        Ok(LookRequest { spool, limit, outputs })
    }
}

impl OutputLooks {
    pub fn job(&self, id: usize) -> Result<&LookJob> {
        self.jobs.get(id).with_context(|| format!("no look job {id}"))
    }

    pub fn set_of(&self, job: &LookJob) -> &LookSet {
        &self.sets[job.set]
    }

    /// The job of a cut frame in a look.
    pub fn frame_job(&self, frame: i64) -> Option<&LookJob> {
        self.frames.get(&frame).and_then(|&j| self.jobs.get(j))
    }
}

/// The bake's Workbench record of a material (bake.json `materials`, the vector look's: viewport colour and alpha, the
/// texture Workbench's Texture colour draws).
#[derive(Deserialize, Debug, Clone, Default)]
pub struct WorkbenchMaterial {
    #[serde(default = "one")]
    pub alpha: f32,
    #[serde(default)]
    pub color: [f32; 3],
    #[serde(default)]
    pub texture: String,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub cutoff: bool,
    #[serde(default)]
    pub premul: bool,
    #[serde(default)]
    pub srgb: bool,
    #[serde(default)]
    pub closest: bool,
    #[serde(default)]
    pub extension: Option<String>,
}

fn one() -> f32 {
    1.0
}

/// The Workbench record of material `name`, if the bake has one.
pub fn workbench_material(bake: &Bake, name: &str) -> Option<WorkbenchMaterial> {
    let v = bake.raw.get("materials")?.get(name)?;
    serde_json::from_value(v.clone()).ok()
}

/// Whether an object casts Workbench's shadow (bake.json `objects`: its display shadow), true for one it does not know.
pub fn casts_shadow(bake: &Bake, name: &str) -> bool {
    bake.raw.get("objects").and_then(|o| o.as_array()).and_then(|objs| {
        objs.iter().find(|o| o.get("name").and_then(|n| n.as_str()) == Some(name)).and_then(|o| o.get("shadow")).and_then(|s| s.as_bool())
    }).unwrap_or(true)
}

/// Whether a material lets the soft pass weigh it: its BLENDED render method is drawn by EEVEE's forward pass, which
/// writes no AOV.
fn blended(bake: &Bake, name: &str) -> bool {
    bake.doc.lit.as_ref().and_then(|l| l.materials.get(name)).and_then(|m| m.settings.get("surface_render_method"))
        .and_then(|v| v.as_str()) == Some("BLENDED")
}

/// Why mkr cannot draw look job `job` of `output`, None when it can: a frame outside the clip, a camera it does not
/// project through, a camera with depth of field under an EEVEE pass, something the look draws that the lit bake could
/// not hold (or that no look hides: a collection instance, a volume), a mesh mkr cannot hold or a material it cannot
/// draw in a pass that evaluates materials, instances in a silhouette's flat colours, a soft accent drawn BLENDED, a
/// vector look's texture the bake does not hold.
pub fn job_blocker(bake: &Bake, support: &Support, output: &str, looks: &OutputLooks, job: &LookJob) -> Result<Option<String>> {
    let Some(k) = bake.k(job.frame) else {
        return Ok(Some(format!("frame {} is outside the clip", job.frame)));
    };
    let set = looks.set_of(job);
    let shot = match crate::layers::shot(bake, &job.shot) {
        Ok(s) => s,
        Err(why) => return Ok(Some(why)),
    };
    let cam = match lens(bake, output, shot) {
        Ok(c) => c,
        Err(why) => return Ok(Some(why)),
    };
    if cam != job.camera {
        bail!("look job {}: camera {} is not shot {}'s for {output} ({cam})", job.id, job.camera, job.shot);
    }
    let lit = bake.lit()?;
    let eevee = set.kind == Kind::Silhouette && !(set.soft.is_empty() && set.kind_type.is_empty() && set.knock.is_empty())
        || set.kind == Kind::Vector && !set.kind_type.is_empty();
    if eevee {
        if let Some(lc) = lit.cameras.get(&cam) {
            if bake.f(lc.offset + crate::bake::LIT_CAMERA_FLOATS * k, 1)?[0] > 0.5 {
                return Ok(Some(format!("{cam}: depth of field (the look's EEVEE passes)")));
            }
        }
    }
    let hidden: BTreeSet<&str> = set.hide.iter().map(String::as_str).collect();
    if job.role == Role::Frame && set.kind == Kind::Silhouette {
        // screen type the bake could not hold may be knock-out type, reversed out of the figure in the picture
        if let Some(why) = crate::layers::screen_left_out(bake, output, job.frame) {
            return Ok(Some(why));
        }
        // the knock-out type itself: drawn as the screen layer draws the rest (layers_gpu.rs), at its samples
        match crate::layers::screen_blocker(bake, support, output, job.frame, crate::layers::ScreenPart::Knock)? {
            Some(Err(why)) => return Ok(Some(why)),
            Some(Ok(())) => {
                let n = lit.screen.as_ref().map(|s| s.pass.samples.default).unwrap_or(16);
                if set.samples != n {
                    return Ok(Some(format!("knock-out screen type in a look of {} samples (mkr draws it at the screen \
                                            layer's {n})", set.samples)));
                }
            }
            None => {}
        }
    }
    for b in bake.doc.blocks.iter().chain(&lit.left_out) {
        if shows(&b.shows, k) && !hidden.contains(b.name.as_str()) {
            return Ok(Some(format!("{}: {}", b.name, b.why)));
        }
    }
    for m in &lit.meshes {
        let Some(class) = set.class_of(m.record_name()) else { continue };
        if m.aspect.as_deref().is_some_and(|a| a != output) || !mesh_shown(bake, m, k)? || !m.visible.camera {
            continue;
        }
        if let Some(why) = support.meshes.get(&m.name) {
            return Ok(Some(format!("{}: {why}", m.name)));
        }
        if set.kind == Kind::Silhouette && matches!(class, Class::Subject | Class::Hard) && m.instance.is_some() {
            return Ok(Some(format!("{}: instances of {} in a silhouette's flat colours", m.record_name(), m.name)));
        }
        let evaluated = matches!(class, Class::Soft | Class::Type | Class::Knock);
        for name in m.materials.iter().flatten() {
            if evaluated {
                if let Some(nodes) = support.materials.get(name) {
                    return Ok(Some(format!("{}: material {name} uses {}", m.name, nodes.join(", "))));
                }
            }
            if class == Class::Soft && blended(bake, name) {
                return Ok(Some(format!("{}: soft accent material {name} is drawn blended (EEVEE writes no AOV)", m.name)));
            }
            if set.kind == Kind::Vector && set.tex && class == Class::Subject {
                if let Some(why) = texture_blocker(bake, name) {
                    return Ok(Some(format!("{}: {why}", m.name)));
                }
            }
        }
    }
    Ok(None)
}

/// Why the vector look's texture pass cannot draw material `name`: its Workbench texture is not among the lit images.
fn texture_blocker(bake: &Bake, name: &str) -> Option<String> {
    let wb = workbench_material(bake, name)?;
    if wb.texture == "none" || wb.texture.is_empty() {
        return None;
    }
    let Some(image) = &wb.image else {
        return Some(format!("material {name}: the bake does not name its texture (`mk build --bake`)"));
    };
    let lit = bake.doc.lit.as_ref()?;
    match lit.images.get(image) {
        Some(rec) if !rec.missing && crate::images::supported_colorspace(&rec.colorspace) => None,
        Some(rec) if rec.missing => None,
        Some(rec) => Some(format!("material {name}: texture {image} in colour space {}", rec.colorspace)),
        None => Some(format!("material {name}: its texture {image} is not in the lit bake")),
    }
}

// ================================================================================================ the pass files
/// One pass of a job's file: its name, shape and bytes (u8, u16 or f32 elements, little-endian).
pub struct PassData {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub kind: &'static str,
    pub bytes: Vec<u8>,
}

/// Wait until fewer than `limit` pass files wait in the spool (the composer deletes each it finishes); a composer that
/// takes nothing for ten minutes has stopped.
pub fn wait_for_room(spool: &Path, limit: usize) -> Result<()> {
    let t0 = std::time::Instant::now();
    loop {
        let waiting = std::fs::read_dir(spool)
            .with_context(|| format!("cannot read the look spool {}", spool.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "look"))
            .count();
        if waiting < limit {
            return Ok(());
        }
        if t0.elapsed().as_secs() > 600 {
            bail!("the look composer has taken no passes for ten minutes ({waiting} wait in {})", spool.display());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Write a job's passes into SPOOL/<job>.look, whole or not at all (module docstring).
pub fn write_passes(spool: &Path, job: usize, passes: &[PassData]) -> Result<()> {
    // the arrays start after a header slot long enough for any header, each 16-byte aligned
    let slot = 64 + 160 * passes.len();
    let mut offsets = Vec::with_capacity(passes.len());
    let mut recs = serde_json::Map::new();
    let mut offset = (12 + slot).next_multiple_of(16);
    for p in passes {
        let size = match p.kind {
            "u8" => 1,
            "u16" => 2,
            "f32" => 4,
            other => bail!("pass {}: element type {other}", p.name),
        };
        if p.bytes.len() != p.shape.iter().product::<usize>() * size {
            bail!("pass {}: {} bytes for shape {:?} of {}", p.name, p.bytes.len(), p.shape, p.kind);
        }
        recs.insert(p.name.to_string(), serde_json::json!({"offset": offset, "shape": p.shape, "type": p.kind}));
        offsets.push(offset);
        offset = (offset + p.bytes.len()).next_multiple_of(16);
    }
    let text = serde_json::to_vec(&serde_json::json!({"job": job, "passes": Value::Object(recs)}))?;
    if text.len() > slot {
        bail!("pass file header of {} bytes overflows its slot of {slot}", text.len());
    }
    let mut buf = Vec::with_capacity(offset);
    buf.extend_from_slice(b"MKRLOOK1");
    buf.extend_from_slice(&(text.len() as u32).to_le_bytes());
    buf.extend_from_slice(&text);
    for (p, &at) in passes.iter().zip(&offsets) {
        buf.resize(at, 0);
        buf.extend_from_slice(&p.bytes);
    }
    let tmp = spool.join(format!("{job}.part"));
    std::fs::write(&tmp, &buf).with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, spool.join(format!("{job}.look"))).with_context(|| format!("cannot finish {}", tmp.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_naming_a_missing_set_is_refused() {
        let v: Value = serde_json::from_str(
            r#"{"outputs": {"o": {"sets": [], "jobs": [{"id": 0, "role": "frame", "frame": 1, "shot": "s", "set": 0,
                "camera": "c"}], "frames": {"1": 0}}}}"#,
        ).unwrap();
        assert!(LookRequest::parse(&v).is_err());
    }

    #[test]
    fn pass_files_read_back_as_written() {
        let dir = std::env::temp_dir().join(format!("mkr_looks_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let passes = vec![
            PassData { name: "flat", shape: vec![2, 3, 4], kind: "u8", bytes: (0..24).collect() },
            PassData { name: "depth", shape: vec![2, 3], kind: "f32", bytes: (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect() },
        ];
        write_passes(&dir, 7, &passes).unwrap();
        let buf = std::fs::read(dir.join("7.look")).unwrap();
        assert_eq!(&buf[..8], b"MKRLOOK1");
        let n = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        let head: Value = serde_json::from_slice(&buf[12..12 + n]).unwrap();
        assert_eq!(head["job"], 7);
        let flat = &head["passes"]["flat"];
        let at = flat["offset"].as_u64().unwrap() as usize;
        assert_eq!(at % 16, 0);
        assert_eq!(&buf[at..at + 24], &(0..24).collect::<Vec<u8>>()[..]);
        let d = head["passes"]["depth"]["offset"].as_u64().unwrap() as usize;
        assert_eq!(f32::from_le_bytes([buf[d + 20], buf[d + 21], buf[d + 22], buf[d + 23]]), 5.0);
        assert!(!dir.join("7.part").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
