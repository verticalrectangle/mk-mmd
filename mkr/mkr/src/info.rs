//! `mkr info`: per output and frame whether mkr draws it, and why Blender must when it cannot (a reflection or a look
//! mk render did not hand mkr, a fisheye or panoramic camera, an orthographic camera with depth of field, a world mkr
//! cannot draw or EEVEE's ray tracing by light probes, an object the lit bake could not hold, a material or mesh mkr
//! cannot draw shown on that frame; a frame in a silhouette or vector look is mkr's when its look job is, looks.rs);
//! with a layer request (`--layers`), the same for every layer of those frames (layers.rs); plus everything mkr cannot
//! draw at all.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use serde::Serialize;

use crate::bake::{Bake, Shot};
use crate::frames::{medium_at, shot_at, shows};
use crate::layers::{Request, ScreenSorted, Sorted};
use crate::looks::{OutputLooks, job_blocker};
use crate::materials::Support;

#[derive(Serialize, Default)]
pub struct OutputPlan {
    pub draw: Vec<i64>,
    pub blender: BTreeMap<String, String>,
    /// Per frame of the request, per item: null when mkr draws it, else why Blender must.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<BTreeMap<String, Vec<Option<String>>>>,
    /// The keep masks and screen type of the frames mkr draws: which mkr draws too, which Blender must (and why); the
    /// screen type's also which of mkr's have negative type on screen (their negative layer too).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep: Option<Sorted>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screen: Option<ScreenSorted>,
}

#[derive(Serialize, Default)]
pub struct Unsupported {
    pub materials: BTreeMap<String, Vec<String>>,
    pub meshes: BTreeMap<String, String>,
    /// The layers mk render writes that mkr leaves to Blender, and why.
    pub layers: BTreeMap<String, String>,
    /// Why mkr draws no lit frame of the scene (materials.rs `scene_problem`): a world it cannot draw, EEVEE's ray
    /// tracing by light probes.
    pub scene: Option<String>,
}

/// What mkr never draws of the layers (mkmmd.core.transition.demands, screen type, keep masks).
fn blender_layers() -> BTreeMap<String, String> {
    [
        ("look", "a plate, bare frame, matte or figure through a reflection, or a look mk render did not give mkr"),
        ("fisheye", "any layer through a fisheye or panoramic camera (Cycles)"),
        ("screen", "screen type the bake could not hold (lit.screen.left_out), on the frames it is on screen"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

#[derive(Serialize)]
pub struct Info {
    pub outputs: BTreeMap<String, OutputPlan>,
    pub unsupported: Unsupported,
}

/// Why mkr cannot draw Blender frame `frame` of `output`, or None when it can (and through which camera). A frame in a
/// render-time look is drawn by its look job when `looks` has one (looks.rs).
pub fn blocker(bake: &Bake, support: &Support, output: &str, frame: i64, looks: Option<&OutputLooks>)
               -> Result<Result<String, String>> {
    let doc = &bake.doc;
    if bake.k(frame).is_none() {
        return Ok(Err(format!("frame {frame} is outside the clip ({}..{})", doc.frame0, doc.frame0 + doc.frames as i64 - 1)));
    }
    if let Some(l) = looks {
        if let Some(job) = l.frame_job(frame) {
            return Ok(match job_blocker(bake, support, output, l, job)? {
                None => Ok(job.camera.clone()),
                Some(why) => Err(format!("its {:?} look: {why}", l.set_of(job).kind).to_lowercase()),
            });
        }
    }
    let Some(shot) = shot_at(&doc.shots, frame) else {
        return Ok(Err("no shot in the cut".into()));
    };
    through(bake, support, output, frame, shot, &[], &BTreeSet::new())
}

/// Why mkr cannot draw Blender frame `frame` of `output` lit through `shot`'s camera (the cut's, a plate's) with the
/// lit meshes `hidden` (indices into `lit.meshes`) and the objects `hidden_names` left out, or the camera's name when it
/// can: a [[medium]] window's look or the shot's own, a fisheye or panoramic camera, an orthographic one with depth of
/// field (a perspective camera's mkr draws: camera.rs), a world mkr cannot draw or EEVEE's ray tracing by light probes
/// (the whole scene's), an object the lit bake could not hold, a material or mesh mkr cannot draw shown on that frame.
pub fn through(bake: &Bake, support: &Support, output: &str, frame: i64, shot: &Shot, hidden: &[usize],
               hidden_names: &BTreeSet<&str>) -> Result<Result<String, String>> {
    let doc = &bake.doc;
    let Some(k) = bake.k(frame) else {
        return Ok(Err(format!("frame {frame} is outside the clip")));
    };
    let cam = match camera(bake, output, frame, shot) {
        Ok(c) => c,
        Err(why) => return Ok(Err(why)),
    };
    let Some(lit) = &doc.lit else {
        return Ok(Err(format!("the bake (version {}) has no lit scene: `mk build --bake`", doc.version)));
    };
    if let Some(lc) = lit.cameras.get(&cam) {
        let d = bake.f(lc.offset + crate::bake::LIT_CAMERA_FLOATS * k, 3)?;
        if lc.kind == "ORTHO" && d[0] > 0.5 && d[2] > 0.0 {
            return Ok(Err(format!("{cam}: an orthographic camera's depth of field")));
        }
    }
    if let Some(why) = &support.scene {
        return Ok(Err(why.clone()));
    }
    for o in &lit.left_out {
        if shows(&o.shows, k) && !hidden_names.contains(o.name.as_str()) {
            return Ok(Err(format!("{}: {}", o.name, o.why)));
        }
    }
    for (mi, m) in lit.meshes.iter().enumerate() {
        if hidden.contains(&mi) || m.aspect.as_deref().is_some_and(|a| a != output) || !mesh_shown(bake, m, k)? {
            continue;
        }
        if let Some(why) = support.meshes.get(&m.name) {
            return Ok(Err(format!("{}: {why}", m.name)));
        }
        for name in m.materials.iter().flatten() {
            if let Some(nodes) = support.materials.get(name) {
                return Ok(Err(format!("{}: material {name} uses {}", m.name, nodes.join(", "))));
            }
        }
    }
    Ok(Ok(cam))
}

/// The camera `shot` sees `frame` of `output` through as mkr can draw it (a perspective or orthographic one, its
/// look none), or why not: a [[medium]] window's look or the shot's own, a camera the bake lacks, a fisheye or panoramic
/// one.
pub fn camera(bake: &Bake, output: &str, frame: i64, shot: &Shot) -> Result<String, String> {
    let doc = &bake.doc;
    if let Some(w) = medium_at(&doc.medium, frame) {
        if w.look(output).is_some() {
            return Err(format!("a [[medium]] window's look (medium {})", w.index));
        }
    }
    if let Some(kind) = shot.look(output) {
        return Err(format!("shot {}: a {kind} look", shot.name));
    }
    lens(bake, output, shot)
}

/// The shot's camera for `output` when mkr projects through it (perspective or orthographic), else why not.
pub fn lens(bake: &Bake, output: &str, shot: &Shot) -> Result<String, String> {
    let Some(cam) = shot.cameras.get(output) else {
        return Err(format!("shot {} has no camera for output {output}", shot.name));
    };
    let Some(rec) = bake.camera(cam) else {
        return Err(format!("its camera {cam} is not in the bake"));
    };
    if rec.fisheye.is_some() {
        return Err(format!("{cam}: a fisheye camera"));
    }
    if let Some(lc) = bake.doc.lit.as_ref().and_then(|l| l.cameras.get(cam)) {
        if lc.kind != "PERSP" && lc.kind != "ORTHO" {
            return Err(format!("{cam}: a {} camera", lc.kind.to_lowercase()));
        }
    }
    Ok(cam.clone())
}

/// Whether a lit mesh record shows on bake frame `k` (its shown flags; a frame-by-frame mesh's geometry there).
pub fn mesh_shown(bake: &Bake, m: &crate::bake::MeshRec, k: usize) -> Result<bool> {
    if let Some(bf) = &m.by_frame {
        if bf.frames.get(k).copied().unwrap_or(-1) < 0 {
            return Ok(false);
        }
    }
    match m.shown {
        Some(off) => Ok(bake.f(off + k, 1)?[0] > 0.5),
        None => Ok(true),
    }
}

pub fn info(bake: &Bake, outputs: &[String], frames: &[i64], layers: Option<&Request>) -> Result<Info> {
    let support = Support::of(bake)?;
    let mut out = BTreeMap::new();
    for o in outputs {
        if bake.output(o).is_none() {
            anyhow::bail!("the bake has no output {o} (outputs: {})",
                          bake.doc.outputs.iter().map(|x| x.name.as_str()).collect::<Vec<_>>().join(", "));
        }
        let mut plan = OutputPlan::default();
        let looks = layers.and_then(|r| r.looks_of(o));
        for &f in frames {
            match blocker(bake, &support, o, f, looks)? {
                Ok(_) => plan.draw.push(f),
                Err(why) => {
                    plan.blender.insert(f.to_string(), why);
                }
            }
        }
        if let Some(req) = layers {
            let mine: BTreeSet<i64> = plan.draw.iter().copied().collect();
            let v = crate::layers::sort(bake, &support, o, req, &mine)?;
            plan.layers = Some(v.items.into_iter().map(|(f, w)| (f.to_string(), w)).collect());
            plan.keep = Some(v.keep);
            plan.screen = Some(v.screen);
        }
        out.insert(o.clone(), plan);
    }
    Ok(Info {
        outputs: out,
        unsupported: Unsupported {
            materials: support.materials.clone(),
            meshes: support.meshes.clone(),
            layers: blender_layers(),
            scene: support.scene.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// A two-frame bake of one shot through a perspective camera, nothing in it, with this world and ray tracing (its
    /// method; None: off).
    fn bake_with(tag: &str, world: Value, raytracing: Option<&str>) -> (std::path::PathBuf, Bake) {
        let dir = std::env::temp_dir().join(format!("mkr_info_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let doc = json!({"version": 5, "fps": 30.0, "frame0": 1, "frames": 2, "outputs": [{"name": "o", "size": [4, 4]}],
            "shots": [{"name": "s", "from": 1, "to": 2, "cameras": {"o": "cam"}}],
            "cameras": [{"name": "cam", "sensor": 36.0, "offset": 0, "clip_at": 30}],
            "lit": {"view": {"lut": {"file": "view_lut.bin", "n": 2, "lo": -1.0, "hi": 1.0}},
                    "eevee": {"use_raytracing": raytracing.is_some(), "ray_tracing_method": raytracing.unwrap_or("SCREEN")},
                    "world": world, "meshes": []}});
        std::fs::write(dir.join("bake.json"), doc.to_string()).unwrap();
        let mut floats = Vec::new();
        for _ in 0..2 {
            floats.extend([1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0]);
        }
        floats.extend([0.1f32, 100.0, 0.1, 100.0]);
        std::fs::write(dir.join("bake.bin"), bytemuck::cast_slice(&floats)).unwrap();
        std::fs::write(dir.join("lit.bin"), b"").unwrap();
        let bake = Bake::open(&dir).unwrap();
        (dir, bake)
    }

    /// A world drawn with nodes: a Background (colour, strength; `keyed` names an animated input) into the output, with
    /// its World settings `props` (null: a bake from before they were kept).
    fn noded(color: [f64; 3], strength: f64, keyed: Option<&str>, props: Value) -> Value {
        let sock = |id: &str, v: Value| json!({"identifier": id, "name": id, "type": "VALUE", "linked": false, "value": v});
        json!({"name": "sky", "use_nodes": true, "color": [0.05, 0.05, 0.05], "props": props,
               "anim": keyed.map(|s| vec![json!({"node": "bg", "socket": s, "size": 1, "offset": 0})]).unwrap_or_default(),
               "tree": {"nodes": [
                   {"name": "bg", "type": "BACKGROUND", "inputs": [sock("Color", json!([color[0], color[1], color[2], 1.0])),
                                                                  sock("Strength", json!(strength))]},
                   {"name": "out", "type": "OUTPUT_WORLD", "props": {"is_active_output": true, "target": "ALL"},
                    "inputs": [sock("Surface", Value::Null), sock("Volume", Value::Null)]}],
                        "links": [["bg", "Background", "out", "Surface"]]}})
    }

    /// mkr draws the world (world.rs) and traces the screen as EEVEE's SCREEN method does (raytrace.rs): a frame is
    /// Blender's, saying why, when its world needs what mkr lacks (settings the bake does not hold, EEVEE's sun taken
    /// out of light over the world's sun threshold) or EEVEE traces by light probes; a world that is black, or drawn,
    /// and screen tracing are not.
    #[test]
    fn a_world_mkr_cannot_draw_or_probe_tracing_leaves_the_frames_to_blender() {
        let flat = |c: f64, props: Value| json!({"name": "w", "use_nodes": false, "color": [c, c, c], "props": props});
        let under = json!({"sun_threshold": 10.0, "probe_resolution": "512"});
        let off = json!({"sun_threshold": 0.0, "probe_resolution": "512"});
        let cases: [(&str, Value, Option<&str>, Option<&str>); 11] = [
            ("none", Value::Null, None, None),
            ("flat_black", flat(0.0, Value::Null), None, None),
            ("flat_grey", flat(0.05, under.clone()), None, None),
            ("flat_old", flat(0.05, Value::Null), None, Some("world w: its settings, which the bake does not hold")),
            ("bg_dark", noded([0.8; 3], 0.0, None, Value::Null), None, None),
            ("bg_lit", noded([0.8; 3], 1.0, None, under.clone()), None, None),
            ("bg_keyed", noded([0.8; 3], 0.0, Some("Strength"), under), None, Some("world sky: EEVEE takes its light")),
            ("bg_keyed_no_sun", noded([0.8; 3], 0.0, Some("Strength"), off.clone()), None, None),
            ("raytraced", Value::Null, Some("SCREEN"), None),
            ("both", noded([0.8; 3], 1.0, None, off), Some("SCREEN"), None),
            ("probe_traced", Value::Null, Some("PROBE"), Some("EEVEE ray tracing by light probes")),
        ];
        for (tag, world, rt, want) in cases {
            let (dir, bake) = bake_with(tag, world, rt);
            let got = info(&bake, &["o".to_string()], &[1, 2], None).unwrap();
            let plan = &got.outputs["o"];
            match want {
                None => {
                    assert_eq!(plan.draw, [1, 2], "{tag}: {:?}", plan.blender);
                    assert!(got.unsupported.scene.is_none(), "{tag}: {:?}", got.unsupported.scene);
                }
                Some(why) => {
                    assert!(plan.draw.is_empty(), "{tag}: {:?}", plan.draw);
                    for f in ["1", "2"] {
                        assert!(plan.blender[f].starts_with(why), "{tag}: frame {f}: {}", plan.blender[f]);
                    }
                    assert!(got.unsupported.scene.as_deref().is_some_and(|s| s.starts_with(why)), "{tag}");
                }
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
