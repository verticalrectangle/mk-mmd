//! The layers `mk render` writes beside a frame (mkmmd.core.transition.demands; docs/design.md: Shots: Transitions and
//! inserts, Glitches; Text: Screen type; Post: keep), drawn by mkr so that a frame needs no Blender. Files, relative to
//! an output's frame folder, as Blender writes them (mkmmd/blender/transition.py, styles.py):
//!
//!   plate/<shot>/<f>.png  another shot seen through its own camera, lit: a frame like the cut's (same samples, motion
//!                         blur and PNG)
//!   bare/<key>/<f>.png    the shot with a glitch's objects hidden (from the camera and from the shadows)
//!   figure/<key>/<f>.png  the coverage of a collage's objects alone as Workbench antialiases it: grey RGB, Workbench's
//!                         film alpha over 254/255 (shotstyle.ALPHA_FULL), 16 samples (layers_gpu.rs)
//!   point/<key>/<f>.json  a world point (the bake's `lit.points`) through the shot's camera: {p: [x, 1 - y], depth: z,
//!                         m: hypot((qx - x) w / h, qy - y)} with (x, y, z) world_to_camera_view's and q the point one
//!                         metre up the camera's +Y
//!   keep/<f>.png          `[post] keep`: EEVEE's flat emission, kept materials white and the rest black, MMD materials
//!                         transparent by their own alpha; linear, 16 samples, R x A of the RGBA pass, as RGBA (alpha 1)
//!   screen/<f>.png        the screen type but the negative: its emission through the screen camera, lights off, film
//!                         transparent, Standard view, straight RGBA; a frame with negative type on screen has one too
//!                         (transparent when that is all it has)
//!   negative/<f>.png      the negative type (`negative = true`): the same pass of it alone, written before the
//!                         frame's screen/<f>.png (whose presence says it is there); mk post inverts the picture under
//!                         its alpha
//!
//! A request (`--layers FILE`, written by mk render: mkmmd/render/mkr.py) lists the layers: {"items": {frame: [item]},
//! "keep": {"materials": [bake material names], "frames": [...]} | null, "screen": [frames] | null}. `sort` says which
//! of them mkr draws and why Blender must draw the others: a matte (a silhouette's figure), anything seen through a
//! render-time look, a fisheye, what the lit bake cannot hold.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::Vec3;
use serde::Serialize;
use serde_json::Value;

use crate::bake::{Bake, Shot};
use crate::camera::Cam;
use crate::frames::shows;
use crate::info::{camera, lens, mesh_shown, through};
use crate::looks::{LookRequest, OutputLooks, job_blocker};
use crate::materials::Support;
use crate::render::{Job, orthonormal};

// ===================================================================================================== the request
/// One layer of a frame (mkmmd.core.transition.demands): a figure's objects and a bare frame's hidden ones are the
/// object names mk render resolved its patterns to on the bake (`names`). A plate, bare frame or matte seen through a
/// render-time look names its look job per output (`look`, looks.rs); a figure seen through one has its names less what
/// the look hides per output (`look_names`).
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Plate { shot: String, look: BTreeMap<String, usize> },
    Bare { shot: String, key: String, names: Vec<String>, look: BTreeMap<String, usize> },
    Figure { shot: String, key: String, names: Vec<String>, look_names: BTreeMap<String, Vec<String>> },
    Point { shot: String, key: String },
    /// A matte's figure (matte/) and its frame without it (back/): drawn in its shot's flat look.
    Matte { shot: String, look: BTreeMap<String, usize> },
    /// A layer mkr does not draw, by its kind.
    Other { kind: String },
}

fn text(v: &Value, key: &str) -> Result<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string).with_context(|| format!("layer item {v}: no {key}"))
}

fn names(v: &Value) -> Result<Vec<String>> {
    match v.get("names") {
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).context("names: not a string")).collect(),
        _ => bail!("layer item {v}: no `names` (mk render resolves its object patterns on the bake)"),
    }
}

/// An item's look jobs per output ({output: job}).
fn item_looks(v: &Value) -> Result<BTreeMap<String, usize>> {
    match v.get("looks") {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(x) => serde_json::from_value(x.clone()).with_context(|| format!("layer item {v}: `looks` is not {{output: job}}")),
    }
}

impl Item {
    pub fn parse(v: &Value) -> Result<Item> {
        Ok(match v.get("kind").and_then(|k| k.as_str()) {
            Some("plate") => Item::Plate { shot: text(v, "shot")?, look: item_looks(v)? },
            Some("bare") => Item::Bare { shot: text(v, "shot")?, key: text(v, "key")?, names: names(v)?, look: item_looks(v)? },
            Some("figure") => {
                let look_names = match v.get("look_names") {
                    None | Some(Value::Null) => BTreeMap::new(),
                    Some(x) => serde_json::from_value(x.clone())
                        .with_context(|| format!("layer item {v}: `look_names` is not {{output: [names]}}"))?,
                };
                Item::Figure { shot: text(v, "shot")?, key: text(v, "key")?, names: names(v)?, look_names }
            }
            Some("point") => Item::Point { shot: text(v, "shot")?, key: text(v, "key")? },
            Some("matte") => Item::Matte { shot: text(v, "shot")?, look: item_looks(v)? },
            Some(kind) => Item::Other { kind: kind.to_string() },
            None => bail!("layer item {v}: no kind"),
        })
    }

    /// Where it is stored, relative to an output's frame folder (the file that marks it finished).
    pub fn rel(&self, frame: i64) -> Option<String> {
        Some(match self {
            Item::Plate { shot, .. } => format!("plate/{shot}/{frame:05}.png"),
            Item::Bare { key, .. } => format!("bare/{key}/{frame:05}.png"),
            Item::Figure { key, .. } => format!("figure/{key}/{frame:05}.png"),
            Item::Point { key, .. } => format!("point/{key}/{frame:05}.json"),
            Item::Matte { shot, .. } => format!("matte/{shot}/{frame:05}.png"),
            Item::Other { .. } => return None,
        })
    }

    pub fn kind(&self) -> &str {
        match self {
            Item::Plate { .. } => "plate",
            Item::Bare { .. } => "bare",
            Item::Figure { .. } => "figure",
            Item::Point { .. } => "point",
            Item::Matte { .. } => "matte",
            Item::Other { kind } => kind,
        }
    }

    /// Its look job in `output`, when it is seen through a render-time look there.
    pub fn look_job(&self, output: &str) -> Option<usize> {
        match self {
            Item::Plate { look, .. } | Item::Bare { look, .. } | Item::Matte { look, .. } => look.get(output).copied(),
            _ => None,
        }
    }
}

pub fn keep_rel(frame: i64) -> String {
    format!("keep/{frame:05}.png")
}

pub fn screen_rel(frame: i64) -> String {
    format!("screen/{frame:05}.png")
}

pub fn negative_rel(frame: i64) -> String {
    format!("negative/{frame:05}.png")
}

#[derive(Clone, Debug, Default)]
pub struct KeepRequest {
    /// The bake's materials that keep their colour.
    pub materials: Vec<String>,
    pub frames: Vec<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct Request {
    pub items: BTreeMap<i64, Vec<Item>>,
    pub keep: Option<KeepRequest>,
    /// Frames whose screen type (if any is on screen) gets its layer.
    pub screen: Vec<i64>,
    /// The render-time looks of the frames and layers (looks.rs).
    pub looks: Option<LookRequest>,
}

fn frame_list(v: &Value, what: &str) -> Result<Vec<i64>> {
    match v {
        Value::Null => Ok(Vec::new()),
        Value::Array(a) => a.iter().map(|x| x.as_i64().with_context(|| format!("{what}: {x} is not a frame"))).collect(),
        _ => bail!("{what}: not a list of frames"),
    }
}

impl Request {
    pub fn load(path: &Path) -> Result<Request> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let v: Value = serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))?;
        Request::parse(&v).with_context(|| format!("{}", path.display()))
    }

    pub fn parse(v: &Value) -> Result<Request> {
        let mut req = Request::default();
        if let Some(items) = v.get("items").and_then(|x| x.as_object()) {
            for (f, list) in items {
                let frame: i64 = f.parse().with_context(|| format!("items: {f} is not a frame"))?;
                let list = list.as_array().with_context(|| format!("items {f}: not a list"))?;
                req.items.insert(frame, list.iter().map(Item::parse).collect::<Result<_>>()?);
            }
        }
        if let Some(k) = v.get("keep").filter(|k| !k.is_null()) {
            let materials = k.get("materials").and_then(|m| m.as_array()).context("keep: no materials")?
                .iter().map(|m| m.as_str().map(str::to_string).context("keep: a material is not a name"))
                .collect::<Result<_>>()?;
            req.keep = Some(KeepRequest { materials, frames: frame_list(k.get("frames").unwrap_or(&Value::Null), "keep")? });
        }
        req.screen = frame_list(v.get("screen").unwrap_or(&Value::Null), "screen")?;
        if let Some(l) = v.get("looks").filter(|l| !l.is_null()) {
            req.looks = Some(LookRequest::parse(l).context("looks")?);
        }
        Ok(req)
    }

    /// Whether it asks for anything mkr draws on the GPU besides lit frames (figures, keep masks, screen type).
    pub fn flat(&self) -> bool {
        self.keep.as_ref().is_some_and(|k| !k.frames.is_empty())
            || !self.screen.is_empty()
            || self.items.values().flatten().any(|i| matches!(i, Item::Figure { .. }))
    }

    /// The looks of `output`'s frames and layers, if the request has them.
    pub fn looks_of(&self, output: &str) -> Option<&OutputLooks> {
        self.looks.as_ref().and_then(|l| l.outputs.get(output))
    }
}

// ================================================================================================= who draws what
/// Frames sorted between mkr and Blender: those mkr draws, and why Blender must draw the others.
#[derive(Serialize, Default, Clone, Debug)]
pub struct Sorted {
    pub draw: Vec<i64>,
    pub blender: BTreeMap<String, String>,
}

/// The screen type's frames sorted as `Sorted` sorts them, and those of `draw` with negative type on screen: mkr draws
/// their negative layer too, before their screen layer.
#[derive(Serialize, Default, Clone, Debug)]
pub struct ScreenSorted {
    pub draw: Vec<i64>,
    pub blender: BTreeMap<String, String>,
    pub negative: Vec<i64>,
}

pub struct Verdicts {
    /// Per frame of the request, per item: None when mkr draws it, else why Blender must.
    pub items: BTreeMap<i64, Vec<Option<String>>>,
    /// Over the frames mkr draws (`mine`): their keep masks, and their screen type (frames with any on screen,
    /// ordinary or negative).
    pub keep: Sorted,
    pub screen: ScreenSorted,
}

/// Sort the request's layers of `output` between mkr and Blender; `mine` are the frames mkr draws (the keep mask and
/// screen type of the others are Blender's, drawn with them).
pub fn sort(bake: &Bake, support: &Support, output: &str, req: &Request, mine: &BTreeSet<i64>) -> Result<Verdicts> {
    let looks = req.looks_of(output);
    let mut items = BTreeMap::new();
    for (&f, list) in &req.items {
        let mut out = Vec::with_capacity(list.len());
        for item in list {
            out.push(item_blocker(bake, support, output, f, item, looks)?);
        }
        items.insert(f, out);
    }
    let mut keep = Sorted::default();
    if let Some(k) = &req.keep {
        for &f in k.frames.iter().filter(|f| mine.contains(f)) {
            match keep_blocker(bake, output, f, &k.materials)? {
                None => keep.draw.push(f),
                Some(why) => {
                    keep.blender.insert(f.to_string(), why);
                }
            }
        }
    }
    let mut screen = ScreenSorted::default();
    for &f in req.screen.iter().filter(|f| mine.contains(f)) {
        match screen_layer_blocker(bake, support, output, f, silhouette_frame(looks, f))? {
            None => {}
            Some(Ok(negative)) => {
                screen.draw.push(f);
                if negative {
                    screen.negative.push(f);
                }
            }
            Some(Err(why)) => {
                screen.blender.insert(f.to_string(), why);
            }
        }
    }
    Ok(Verdicts { items, keep, screen })
}

/// The shot table's entry named `name` (a plate shot too).
pub fn shot<'a>(bake: &'a Bake, name: &str) -> Result<&'a Shot, String> {
    bake.doc.shots.iter().find(|s| s.name == name).ok_or_else(|| format!("no shot {name} in the bake's shot table"))
}

/// Blender refuses a layer through a shot whose camera mk build did not key over the frame (transition.Layers._camera).
fn keyed(shot: &Shot, frame: i64) -> Option<String> {
    let [lo, hi] = shot.keyed?;
    (!(lo <= frame && frame <= hi)).then(|| {
        format!("shot {}'s camera is keyed over frames {lo}..{hi}, not {frame} (run mk build again)", shot.name)
    })
}

/// Why mkr cannot draw `item` on `frame` of `output`, None when it can (`looks`: the request's looks of the output).
pub fn item_blocker(bake: &Bake, support: &Support, output: &str, frame: i64, item: &Item, looks: Option<&OutputLooks>)
                    -> Result<Option<String>> {
    let Some(k) = bake.k(frame) else {
        return Ok(Some(format!("frame {frame} is outside the clip")));
    };
    let name = match item {
        Item::Other { kind } => return Ok(Some(format!("mkr does not draw a {kind}"))),
        Item::Plate { shot, .. } | Item::Bare { shot, .. } | Item::Figure { shot, .. } | Item::Point { shot, .. }
        | Item::Matte { shot, .. } => shot,
    };
    let s = match shot(bake, name) {
        Ok(s) => s,
        Err(why) => return Ok(Some(why)),
    };
    if let Some(why) = keyed(s, frame) {
        return Ok(Some(why));
    }
    // seen through a look: its job's passes
    match (item.look_job(output), looks) {
        (Some(j), Some(l)) => return job_blocker(bake, support, output, l, l.job(j)?),
        (Some(_), None) => bail!("layer item {} of frame {frame} names a look job, and the request has no looks of {output}",
                                 item.kind()),
        (None, _) if matches!(item, Item::Matte { .. }) => {
            return Ok(Some("a matte is drawn in its shot's flat look, which mk render did not give mkr".into()));
        }
        (None, _) => {}
    }
    Ok(match item {
        Item::Plate { .. } => through(bake, support, output, frame, s, &[], &BTreeSet::new())?.err(),
        Item::Bare { names, .. } => {
            let hide: BTreeSet<&str> = names.iter().map(String::as_str).collect();
            through(bake, support, output, frame, s, &selected(bake, names), &hide)?.err()
        }
        Item::Figure { names, look_names, .. } => {
            // through a look, the coverage of what the look leaves of them: the lens is all that counts
            let (names, cam) = match look_names.get(output) {
                Some(n) => (n, lens(bake, output, s)),
                None => (names, camera(bake, output, frame, s)),
            };
            if let Err(why) = cam {
                return Ok(Some(why));
            }
            figure_blocker(bake, support, output, k, names)?
        }
        Item::Point { key, .. } => {
            if let Err(why) = lens(bake, output, s) {
                return Ok(Some(why));
            }
            point_at(bake, key, k).err()
        }
        Item::Matte { .. } | Item::Other { .. } => unreachable!(),
    })
}

/// The names a figure item draws in `output`: what its look leaves of them when it is seen through one.
pub fn figure_names<'a>(item: &'a Item, output: &str) -> &'a [String] {
    match item {
        Item::Figure { names, look_names, .. } => look_names.get(output).unwrap_or(names),
        _ => &[],
    }
}

/// The lit meshes (indices into `lit.meshes`) of the objects `names`: a mesh is its record's object's (an instance its
/// instancer's).
pub fn selected(bake: &Bake, names: &[String]) -> Vec<usize> {
    let want: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    let Some(lit) = &bake.doc.lit else { return Vec::new() };
    lit.meshes.iter().enumerate().filter(|(_, m)| want.contains(m.record_name())).map(|(i, _)| i).collect()
}

/// Why mkr cannot draw the coverage of `names` on bake frame `k`: one of them the bake could not hold.
fn figure_blocker(bake: &Bake, support: &Support, output: &str, k: usize, names: &[String]) -> Result<Option<String>> {
    let Some(lit) = &bake.doc.lit else {
        return Ok(Some(format!("the bake (version {}) has no lit scene: `mk build --bake`", bake.doc.version)));
    };
    let want: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    for b in bake.doc.blocks.iter().chain(&lit.left_out) {
        if want.contains(b.name.as_str()) && shows(&b.shows, k) {
            return Ok(Some(format!("{}: {}", b.name, b.why)));
        }
    }
    for mi in selected(bake, names) {
        let m = &lit.meshes[mi];
        if m.aspect.as_deref().is_some_and(|a| a != output) || !mesh_shown(bake, m, k)? {
            continue;
        }
        if let Some(why) = support.meshes.get(&m.name) {
            return Ok(Some(format!("{}: {why}", m.name)));
        }
    }
    Ok(None)
}

/// Why mkr cannot draw the keep mask of `frame` (one it draws lit): a kept or MMD material whose surface is not its MMD
/// shader's (Blender takes the alpha from the shader node itself).
fn keep_blocker(bake: &Bake, output: &str, frame: i64, materials: &[String]) -> Result<Option<String>> {
    let lit = bake.lit()?;
    for m in materials {
        if !lit.materials.contains_key(m) {
            return Ok(Some(format!("keep: the bake has no material {m}")));
        }
    }
    let k = bake.k(frame).context("frame outside the clip")?;
    for m in &lit.meshes {
        if m.aspect.as_deref().is_some_and(|a| a != output) || !mesh_shown(bake, m, k)? {
            continue;
        }
        for name in m.materials.iter().flatten() {
            if let Err(why) = mmd_alpha(bake, name) {
                return Ok(Some(why));
            }
        }
    }
    Ok(None)
}

/// Whether a material is an MMD one (mmd_tools' `mmd_shader` node with an Alpha output) whose surface is that node's
/// shader: its opacity is then the node's alpha, what the keep mask is transparent by. Err when it is MMD but its
/// surface comes from elsewhere.
pub fn mmd_alpha(bake: &Bake, material: &str) -> Result<bool, String> {
    let Some(tree) = bake.doc.lit.as_ref().and_then(|l| l.materials.get(material)).and_then(|m| m.tree.as_ref()) else {
        return Ok(false);
    };
    let nodes = tree.get("nodes").and_then(|n| n.as_array()).map(Vec::as_slice).unwrap_or(&[]);
    let Some(mmd) = nodes.iter().find(|n| n.get("name").and_then(|x| x.as_str()) == Some("mmd_shader")) else {
        return Ok(false);
    };
    let socket = |name: &str| {
        mmd.get("outputs").and_then(|o| o.as_array()).and_then(|o| {
            o.iter().find(|s| s.get("name").and_then(|x| x.as_str()) == Some(name))
                .and_then(|s| s.get("identifier")).and_then(|x| x.as_str())
        })
    };
    if socket("Alpha").is_none() {
        return Ok(false);
    }
    let shader = socket("Shader");
    let outs: Vec<&Value> = nodes.iter().filter(|n| n.get("type").and_then(|x| x.as_str()) == Some("OUTPUT_MATERIAL")).collect();
    let active = outs.iter().find(|n| n.pointer("/props/is_active_output").and_then(|x| x.as_bool()) == Some(true))
        .or(outs.first());
    let out_name = active.and_then(|n| n.get("name")).and_then(|x| x.as_str());
    let links = tree.get("links").and_then(|l| l.as_array()).map(Vec::as_slice).unwrap_or(&[]);
    let direct = links.iter().any(|l| {
        let s = |i: usize| l.get(i).and_then(|x| x.as_str());
        s(0) == Some("mmd_shader") && s(1) == shader && s(2) == out_name && s(3) == Some("Surface")
    });
    if direct { Ok(true) } else { Err(format!("material {material}: its surface is not its MMD shader's (keep mask)")) }
}

/// Whether `frame` of `output` has screen type on screen in `part` (None: nothing; else Ok when mkr draws it, Err why
/// not). A silhouette's frame keeps its knock-out type off the layer: its look reverses it out of the figure (`Knock`).
pub fn screen_blocker(bake: &Bake, support: &Support, output: &str, frame: i64, part: ScreenPart)
                      -> Result<Option<Result<(), String>>> {
    let lit = bake.lit()?;
    let Some(screen) = &lit.screen else {
        return Ok((bake.doc.screen > 0).then(|| Err("the bake holds no screen type (`mk build --bake`)".to_string())));
    };
    if !screen.cameras.contains_key(output) {
        return Ok(Some(Err(format!("the bake's screen type has no camera for output {output}"))));
    }
    if let Some(o) = screen_left_out(bake, output, frame) {
        return Ok(Some(Err(o)));
    }
    let k = bake.k(frame).context("frame outside the clip")?;
    // Blender writes the layer whenever a word's [on, off) window holds the frame, its geometry there or not
    let mut live = false;
    for m in live_screen(screen, output, frame, part) {
        live = true;
        if !mesh_shown(bake, m, k)? {
            continue;
        }
        if m.by_frame.is_none() {
            return Ok(Some(Err(format!("screen type {}: the bake holds no geometry of it", m.name))));
        }
        for name in m.materials.iter().flatten() {
            if let Some(nodes) = support.materials.get(name) {
                return Ok(Some(Err(format!("screen type {}: material {name} uses {}", m.name, nodes.join(", ")))));
            }
        }
    }
    Ok(live.then_some(Ok(())))
}

/// Whether `frame` of `output` has a screen layer (None: neither ordinary nor negative type is on screen; else Ok when
/// mkr draws it, with whether negative type is on screen, whose layer it draws first; Err why not): its `Over` and
/// `Negative` parts together, as Blender writes the two files (styles.Looks.screen_layer), both mkr's or both
/// Blender's. `silhouette`: a silhouette's frame, whose knock-out type stays off the layer.
pub fn screen_layer_blocker(bake: &Bake, support: &Support, output: &str, frame: i64, silhouette: bool)
                            -> Result<Option<Result<bool, String>>> {
    let over = screen_blocker(bake, support, output, frame, ScreenPart::Over { silhouette })?;
    let negative = screen_blocker(bake, support, output, frame, ScreenPart::Negative)?;
    Ok(match (over, negative) {
        (Some(Err(why)), _) | (_, Some(Err(why))) => Some(Err(why)),
        (None, None) => None,
        (_, negative) => Some(Ok(negative.is_some())),
    })
}

/// Screen type on screen at `frame` in `output` that the bake could not hold, and why.
pub fn screen_left_out(bake: &Bake, output: &str, frame: i64) -> Option<String> {
    let screen = bake.doc.lit.as_ref()?.screen.as_ref()?;
    let on = |aspect: &Option<String>, show: &Option<[i64; 2]>| {
        aspect.as_deref().is_none_or(|a| a == output) && show.is_none_or(|[a, b]| a <= frame && frame < b)
    };
    screen.left_out.iter().find(|o| on(&o.aspect, &o.show)).map(|o| format!("screen type {}: {}", o.name, o.why))
}

/// Which of the screen type on screen a pass draws: the screen layer's (`Over`: all but the negative type, and in a
/// silhouette's frame but its knock-out type too), the negative layer's (`Negative`: the negative type alone) or a
/// silhouette's knock pass (`Knock`: its knock-out type alone).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScreenPart {
    Over { silhouette: bool },
    Negative,
    Knock,
}

impl ScreenPart {
    fn draws(self, m: &crate::bake::MeshRec) -> bool {
        match self {
            ScreenPart::Over { silhouette } => !m.negative && !(silhouette && m.knockout),
            ScreenPart::Negative => m.negative,
            ScreenPart::Knock => m.knockout,
        }
    }
}

/// Whether screen type record `m` is on screen at `frame` in `output` (styles.Looks.screen_live: made for that output,
/// its [on, off) window holding the frame).
fn on_screen(m: &crate::bake::MeshRec, output: &str, frame: i64) -> bool {
    m.aspect.as_deref().is_none_or(|a| a == output) && m.show.is_none_or(|[on, off]| on <= frame && frame < off)
}

/// The screen type on screen at `frame` in `output`, the `part` of it a pass draws.
pub fn live_screen<'a>(screen: &'a crate::bake::ScreenRec, output: &'a str, frame: i64, part: ScreenPart)
                       -> impl Iterator<Item = &'a crate::bake::MeshRec> + 'a {
    screen.meshes.iter().filter(move |m| on_screen(m, output, frame) && part.draws(m))
}

/// The screen type records (indices into `screen.meshes`) on screen in `output` on any of `frames`, every part: what a
/// kit drawing those frames' screen type loads (a project's whole screen type can run to gigabytes: jwin_mv's 180
/// words hold 88 million vertices).
pub fn screen_records(screen: &crate::bake::ScreenRec, output: &str, frames: &[i64]) -> Vec<usize> {
    screen.meshes.iter().enumerate()
        .filter(|(_, m)| frames.iter().any(|&f| on_screen(m, output, f)))
        .map(|(i, _)| i)
        .collect()
}

/// Whether knock-out screen type is on screen at `frame` in `output` (a silhouette reverses it out of its figure).
pub fn screen_knockout_live(bake: &Bake, output: &str, frame: i64) -> bool {
    bake.doc.lit.as_ref().and_then(|l| l.screen.as_ref())
        .is_some_and(|s| s.meshes.iter().any(|m| m.knockout && on_screen(m, output, frame)))
}

/// Whether `frame` of `output` is a frame of the cut drawn in a silhouette (the request's looks).
pub fn silhouette_frame(looks: Option<&OutputLooks>, frame: i64) -> bool {
    looks.and_then(|l| l.frame_job(frame).map(|j| l.set_of(j).kind == crate::looks::Kind::Silhouette)).unwrap_or(false)
}

// ================================================================================================= the points
/// A layer point's world position (glTF axes) on bake frame `k`, or why there is none.
pub fn point_at(bake: &Bake, key: &str, k: usize) -> Result<Vec3, String> {
    let lit = bake.doc.lit.as_ref().ok_or("the bake has no lit scene")?;
    let p = lit.points.get(key).ok_or_else(|| format!("the bake has no point {key} (`mk build --bake`)"))?;
    let v = bake.f(p.offset + 3 * k, 3).map_err(|e| format!("point {key}: {e}"))?;
    let xyz = Vec3::new(v[0], v[1], v[2]);
    if !xyz.is_finite() {
        let failed = p.errors.iter().find(|e| {
            e.get("frames").and_then(|f| f.as_array()).is_some_and(|f| f.iter().any(|x| x.as_i64() == Some(bake.doc.frame0 + k as i64)))
        });
        return Err(match failed.and_then(|e| e.get("error")).and_then(|e| e.as_str()) {
            Some(e) => format!("point {key}: {:?} failed on this frame: {e}", p.expr),
            None => format!("point {key}: the bake has no position for this frame (`mk build --bake`)"),
        });
    }
    Ok(xyz)
}

/// The point file of world point `co` seen through `cam` in a `size` picture (FE.view / FE.scale for a perspective or
/// orthographic camera): p = (x, 1 - y) with (x, y) world_to_camera_view's fractions from the bottom left, depth its
/// distance along the view axis, m = hypot((qx - x) w / h, qy - y) for q the point plus the camera's own +Y axis.
pub fn point_file(cam: &Cam, size: [u32; 2], co: Vec3) -> (f64, f64, f64, f64) {
    let unit = Cam { world: orthonormal(&cam.world), ..*cam };
    let (w, h) = (size[0] as f32, size[1] as f32);
    let view = |q: Vec3| -> (f64, f64, f64) {
        let local = unit.view() * q.extend(1.0);
        let z = -local.z;
        if z == 0.0 && unit.ortho.is_none() {
            return (0.5, 0.5, 0.0);
        }
        let clip = unit.projection(w, h, glam::Vec2::ZERO) * local;
        let ndc = clip.truncate() / clip.w;
        (((ndc.x + 1.0) * 0.5) as f64, ((ndc.y + 1.0) * 0.5) as f64, z as f64)
    };
    let (x, y, z) = view(co);
    // the camera's own +Y (Blender's) with its scale: glTF (0, 0, -1) in its local axes
    let up = -cam.world.z_axis.truncate();
    let (qx, qy, _) = view(co + up);
    let m = (((qx - x) * w as f64 / h as f64).powi(2) + (qy - y).powi(2)).sqrt();
    (x, 1.0 - y, z, m)
}

/// Python's json of round(v, places): the shortest decimal that reads back, with a point.
fn num(v: f64, places: i32) -> String {
    let s = 10f64.powi(places);
    let r = (v * s).round() / s;
    let r = if r == 0.0 { 0.0 } else { r };
    let t = format!("{r}");
    if t.contains(['.', 'e', 'E', 'N', 'i']) { t } else { format!("{t}.0") }
}

pub fn point_json(p: (f64, f64, f64, f64)) -> String {
    format!("{{\"p\": [{}, {}], \"depth\": {}, \"m\": {}}}", num(p.0, 6), num(p.1, 6), num(p.2, 4), num(p.3, 6))
}

/// Write a small file whole or not at all (its folder made).
pub fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot make {}", dir.display()))?;
    }
    let tmp = path.with_extension("json.part");
    std::fs::write(&tmp, text).with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))
}

// ================================================================================================= what mkr draws
/// The lit frames of the request: plates and bare frames, as jobs through their shot's camera (a bare frame with its
/// objects hidden); those seen through a render-time look are look jobs (looks_gpu.rs). Items mkr cannot draw are
/// skipped with a warning (mk render gives them to Blender).
pub fn lit_jobs(bake: &Bake, support: &Support, output: &str, req: &Request, out: &Path, warnings: &mut Vec<String>)
                -> Result<Vec<Job>> {
    let looks = req.looks_of(output);
    let mut jobs = Vec::new();
    for (&f, list) in &req.items {
        for item in list.iter().filter(|i| matches!(i, Item::Plate { .. } | Item::Bare { .. })) {
            if item.look_job(output).is_some() {
                continue;
            }
            if let Some(why) = item_blocker(bake, support, output, f, item, looks)? {
                warnings.push(format!("frame {f}: {} {}: {why}, skipped", item.kind(), item.rel(f).unwrap_or_default()));
                continue;
            }
            let (shot_name, hidden) = match item {
                Item::Plate { shot, .. } => (shot, Vec::new()),
                Item::Bare { shot, names, .. } => (shot, selected(bake, names)),
                _ => unreachable!(),
            };
            let s = shot(bake, shot_name).map_err(anyhow::Error::msg)?;
            let camera = s.cameras.get(output).cloned().context("checked")?;
            jobs.push(Job { frame: f, k: bake.k(f).context("checked")?, camera, path: out.join(item.rel(f).unwrap()), hidden });
        }
    }
    Ok(jobs)
}

/// The look jobs (looks.rs) of the request that mkr draws for `output`: those of `frames` (the cut's frames it draws)
/// and of the layer items seen through a look; a job mkr cannot draw is skipped with a warning.
pub fn look_jobs(bake: &Bake, support: &Support, output: &str, req: &Request, frames: &[i64], warnings: &mut Vec<String>)
                 -> Result<Vec<usize>> {
    let Some(looks) = req.looks_of(output) else { return Ok(Vec::new()) };
    let mut ids = BTreeSet::new();
    for &f in frames {
        if let Some(&j) = looks.frames.get(&f) {
            ids.insert(j);
        }
    }
    for (&f, list) in &req.items {
        for item in list {
            let Some(j) = item.look_job(output) else { continue };
            match item_blocker(bake, support, output, f, item, Some(looks))? {
                None => {
                    ids.insert(j);
                }
                Some(why) => warnings.push(format!("frame {f}: {} {}: {why}, skipped", item.kind(), item.rel(f).unwrap_or_default())),
            }
        }
    }
    Ok(ids.into_iter().collect())
}

#[derive(Serialize, Default, Debug)]
pub struct Counts {
    pub plate: usize,
    pub bare: usize,
    pub figure: usize,
    pub point: usize,
    pub keep: usize,
    pub screen: usize,
    pub negative: usize,
}

/// One frame's flat passes (layers_gpu.rs): its figures (file, camera, meshes), its keep mask (file, camera), its
/// screen type (file, and whether the frame is a silhouette's, whose knock-out type stays off the layer) and, with
/// negative type on screen, its negative layer's file (drawn first: a frame with one has its screen layer too).
pub struct Flat {
    pub frame: i64,
    pub k: usize,
    pub figures: Vec<(PathBuf, String, Vec<usize>)>,
    pub keep: Option<(PathBuf, String)>,
    pub screen: Option<(PathBuf, bool)>,
    pub negative: Option<PathBuf>,
}

fn flat_of<'a>(flats: &'a mut BTreeMap<i64, Flat>, bake: &Bake, f: i64) -> Result<&'a mut Flat> {
    let k = bake.k(f).with_context(|| format!("frame {f} is outside the clip"))?;
    Ok(flats.entry(f).or_insert_with(|| {
        Flat { frame: f, k, figures: Vec::new(), keep: None, screen: None, negative: None }
    }))
}

/// The points of the request, written now, and the frames' flat passes for the GPU; items mkr cannot draw are skipped
/// with a warning.
pub fn plan_flat(bake: &Bake, support: &Support, output: &str, req: &Request, out: &Path, counts: &mut Counts,
                 warnings: &mut Vec<String>) -> Result<Vec<Flat>> {
    let size = bake.output(output).context("no such output")?.size;
    let looks = req.looks_of(output);
    let mut flats: BTreeMap<i64, Flat> = BTreeMap::new();
    for (&f, list) in &req.items {
        for item in list {
            if !matches!(item, Item::Figure { .. } | Item::Point { .. }) {
                continue;
            }
            if let Some(why) = item_blocker(bake, support, output, f, item, looks)? {
                warnings.push(format!("frame {f}: {} {}: {why}, skipped", item.kind(), item.rel(f).unwrap_or_default()));
                continue;
            }
            let path = out.join(item.rel(f).unwrap());
            match item {
                Item::Figure { shot: s, .. } => {
                    let cam = shot(bake, s).map_err(anyhow::Error::msg)?.cameras[output].clone();
                    flat_of(&mut flats, bake, f)?.figures.push((path, cam, selected(bake, figure_names(item, output))));
                }
                Item::Point { shot: s, key } => {
                    let k = bake.k(f).context("checked")?;
                    let name = &shot(bake, s).map_err(anyhow::Error::msg)?.cameras[output];
                    let rec = bake.camera(name).context("checked")?;
                    let lc = bake.doc.lit.as_ref().and_then(|l| l.cameras.get(name));
                    let cam = Cam::at(bake, rec, lc, k)?;
                    let co = point_at(bake, key, k).map_err(anyhow::Error::msg)?;
                    write_text(&path, &point_json(point_file(&cam, size, co)))?;
                    counts.point += 1;
                }
                _ => unreachable!(),
            }
        }
    }
    let cut = |f: i64| -> Result<Option<String>> { Ok(crate::info::blocker(bake, support, output, f, looks)?.ok()) };
    if let Some(keep) = &req.keep {
        for &f in &keep.frames {
            match (cut(f)?, keep_blocker(bake, output, f, &keep.materials)?) {
                (Some(cam), None) => flat_of(&mut flats, bake, f)?.keep = Some((out.join(keep_rel(f)), cam)),
                (None, _) => warnings.push(format!("frame {f}: keep mask: mkr does not draw the frame, skipped")),
                (_, Some(why)) => warnings.push(format!("frame {f}: keep mask: {why}, skipped")),
            }
        }
    }
    for &f in &req.screen {
        let silhouette = silhouette_frame(looks, f);
        match screen_layer_blocker(bake, support, output, f, silhouette)? {
            None => {}
            Some(Ok(negative)) => {
                let flat = flat_of(&mut flats, bake, f)?;
                flat.screen = Some((out.join(screen_rel(f)), silhouette));
                flat.negative = negative.then(|| out.join(negative_rel(f)));
            }
            Some(Err(why)) => warnings.push(format!("frame {f}: screen type: {why}, skipped")),
        }
    }
    Ok(flats.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn items_parse_and_name_their_files() {
        let v: Value = serde_json::from_str(
            r#"{"items": {"120": [{"kind": "figure", "key": "f1", "objects": ["a*"], "shot": "s", "names": ["a1", "a2"],
                                   "look_names": {"16x9": ["a1"]}},
                                  {"kind": "matte", "shot": "s", "looks": {"16x9": 3}},
                                  {"kind": "point", "key": "x3", "expr": "1", "shot": "s"},
                                  {"kind": "plate", "shot": "p"}]},
                "keep": {"materials": ["eye"], "frames": [120, 121]}, "screen": [120]}"#,
        ).unwrap();
        let r = Request::parse(&v).unwrap();
        let items = &r.items[&120];
        assert!(matches!(&items[0], Item::Figure { shot, key, .. } if shot == "s" && key == "f1"));
        assert_eq!(items[0].rel(120).unwrap(), "figure/f1/00120.png");
        assert_eq!(figure_names(&items[0], "16x9"), &["a1".to_string()]);
        assert_eq!(figure_names(&items[0], "9x16"), &["a1".to_string(), "a2".to_string()]);
        assert_eq!(items[1].rel(120).unwrap(), "matte/s/00120.png");
        assert_eq!((items[1].look_job("16x9"), items[1].look_job("1x1")), (Some(3), None));
        assert_eq!(items[2].rel(7).unwrap(), "point/x3/00007.json");
        assert_eq!(items[3], Item::Plate { shot: "p".into(), look: BTreeMap::new() });
        assert_eq!(r.keep.as_ref().unwrap().frames, vec![120, 121]);
        assert_eq!(r.screen, vec![120]);
        assert!(r.flat() && r.looks.is_none());
    }

    #[test]
    fn a_point_on_the_view_axis_lands_mid_frame() {
        // a camera at the origin with the bake's identity matrix: its local axes are glTF's (Blender's local -Z, the way
        // a camera looks, is glTF's -Y; its local +Y, up the picture, is glTF's -Z)
        let cam = Cam {
            world: glam::Mat4::IDENTITY,
            lens: 50.0,
            shift: glam::Vec2::ZERO,
            near: 0.1,
            far: 100.0,
            sensor: 36.0,
            sensor_height: 24.0,
            fit: crate::camera::Fit::Auto,
            ortho: None,
            dof: None,
        };
        // 5 m ahead on the view axis
        let p = point_file(&cam, [1920, 1080], Vec3::new(0.0, -5.0, 0.0));
        assert!((p.0 - 0.5).abs() < 1e-6 && (p.1 - 0.5).abs() < 1e-6, "{p:?}");
        assert!((p.2 - 5.0).abs() < 1e-5, "{p:?}");
        // a metre up at 5 m with a 50 mm lens on a 36 mm sensor: 50 / 36 / 5 of the width, in frame heights
        let want = 50.0 / 36.0 / 5.0 * 1920.0 / 1080.0;
        assert!((p.3 - want).abs() < 1e-4, "{p:?} vs {want}");
    }

    /// A word of screen type on screen in Blender frames [on, off): `kind` "plain", "knockout" or "negative", its one
    /// material, its geometry on every frame of `screen_bake` (`held`; else the bake holds none of it).
    fn word(name: &str, show: [i64; 2], kind: &str, material: &str, held: bool) -> Value {
        let by_frame = if held { json!({"geoms": [], "frames": [0, 0, 0, 0, 0, 0]}) } else { Value::Null };
        json!({"name": name, "kind": "by_frame", "materials": [material], "show": show, "knockout": kind == "knockout",
               "negative": kind == "negative", "by_frame": by_frame})
    }

    /// A bake of Blender frames 1..=6 whose lit scene is the screen type `words` of output o.
    fn screen_bake(tag: &str, words: Vec<Value>) -> (PathBuf, Bake) {
        let dir = std::env::temp_dir().join(format!("mkr_layers_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lut = |file: &str| json!({"file": file, "n": 2, "lo": -1.0, "hi": 1.0});
        let camera = json!({"ortho_scale": 1.0, "clip": [0.1, 100.0],
                            "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0]});
        let doc = json!({"version": 5, "fps": 30.0, "frame0": 1, "frames": 6,
                         "outputs": [{"name": "o", "size": [4, 4]}], "screen": words.len(),
                         "lit": {"view": {"lut": lut("view_lut.bin")}, "meshes": [],
                                 "screen": {"meshes": words, "left_out": [], "cameras": {"o": camera},
                                            "lut": lut("screen_lut.bin")}}});
        std::fs::write(dir.join("bake.json"), doc.to_string()).unwrap();
        std::fs::write(dir.join("bake.bin"), b"").unwrap();
        std::fs::write(dir.join("lit.bin"), b"").unwrap();
        let bake = Bake::open(&dir).unwrap();
        (dir, bake)
    }

    /// Each pass draws its own of the screen type: the screen layer all but the negative type (in a silhouette's frame
    /// but its knock-out type too), the negative layer the negative type alone, a silhouette's knock pass its knock-out
    /// type; a record from before negative type is not.
    #[test]
    fn each_screen_pass_draws_its_own_type() {
        let rec = |v: Value| serde_json::from_value::<crate::bake::MeshRec>(v).unwrap();
        let parts = [ScreenPart::Over { silhouette: false }, ScreenPart::Over { silhouette: true },
                     ScreenPart::Negative, ScreenPart::Knock];
        let draws = |v: Value| {
            let m = rec(v);
            parts.map(|p| p.draws(&m))
        };
        assert_eq!(draws(word("aa", [1, 2], "plain", "m", true)), [true, true, false, false]);
        assert_eq!(draws(word("bb", [1, 2], "knockout", "m", true)), [true, false, false, true]);
        assert_eq!(draws(word("cc", [1, 2], "negative", "m", true)), [false, false, true, false]);
        assert!(!rec(json!({"name": "dd", "knockout": true})).negative);
        assert!(!rec(json!({"name": "ee", "negative": null})).negative);
    }

    /// A frame's screen layer is mkr's when its ordinary and its negative type both are, Blender's when either is not;
    /// negative type alone on screen makes it live (its negative layer and an empty screen layer). The plan lists mkr's
    /// frames with negative type, and their flats name the negative layer beside the screen layer.
    #[test]
    fn ordinary_and_negative_type_sort_together() {
        let (dir, bake) = screen_bake("negative", vec![
            word("aa", [1, 3], "plain", "m", true),
            word("bb", [2, 4], "negative", "m", true),
            word("cc", [4, 5], "knockout", "m", true),
            word("dd", [5, 6], "negative", "m", false),
            word("ee", [6, 7], "plain", "noded", true),
            word("ff", [6, 7], "negative", "m", true),
        ]);
        let support = Support { materials: BTreeMap::from([("noded".to_string(), vec!["NODE".to_string()])]),
                                meshes: BTreeMap::new(), scene: None };
        let verdict = |f: i64, silhouette: bool| screen_layer_blocker(&bake, &support, "o", f, silhouette).unwrap();
        assert_eq!(verdict(1, false), Some(Ok(false)));
        assert_eq!(verdict(2, false), Some(Ok(true)));
        assert_eq!(verdict(3, false), Some(Ok(true)));          // negative type alone
        assert_eq!(verdict(4, false), Some(Ok(false)));         // knock-out type is laid over outside a silhouette
        assert_eq!(verdict(4, true), None);                     // and is the figure's in one
        let why = |f: i64| verdict(f, false).and_then(Result::err).unwrap_or_default();
        assert!(why(5).starts_with("screen type dd: the bake holds no geometry"), "{}", why(5));    // the negative part
        assert!(why(6).starts_with("screen type ee: material noded"), "{}", why(6));               // the ordinary part
        let req = Request { screen: (1..=6).collect(), ..Request::default() };
        let mine: BTreeSet<i64> = (1..=6).collect();
        let plan = serde_json::to_value(sort(&bake, &support, "o", &req, &mine).unwrap().screen).unwrap();
        assert_eq!(plan["draw"], json!([1, 2, 3, 4]));
        assert_eq!(plan["negative"], json!([2, 3]));
        assert_eq!(plan["blender"].as_object().unwrap().keys().collect::<Vec<_>>(), ["5", "6"]);
        let out = dir.join("frames");
        let mut warnings = Vec::new();
        let flats = plan_flat(&bake, &support, "o", &req, &out, &mut Counts::default(), &mut warnings).unwrap();
        let files: Vec<_> = flats.iter().map(|f| (f.frame, f.screen.clone(), f.negative.clone())).collect();
        let screen = |f: i64| Some((out.join(format!("screen/{f:05}.png")), false));
        assert_eq!(files, [(1, screen(1), None), (2, screen(2), Some(out.join("negative/00002.png"))),
                           (3, screen(3), Some(out.join("negative/00003.png"))), (4, screen(4), None)]);
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
