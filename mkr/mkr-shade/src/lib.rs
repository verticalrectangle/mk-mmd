//! mkr-shade: Blender 4.2 shader node graphs (the scene bake's GRAPH records) compiled to WGSL closure functions that
//! reproduce EEVEE Next's evaluation.
//!
//! Contract with the renderer: `src/wgsl/prelude.wgsl` (Surface, Closures, the functions the renderer defines).
//! A module is [`PRELUDE`] + every [`CompiledMaterial::wgsl`] + the renderer's code.
//!
//! ```no_run
//! # use std::collections::BTreeMap;
//! let lit: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("bake.json").unwrap()).unwrap();
//! let lit = &lit["lit"];
//! let groups: BTreeMap<String, mkr_shade::GroupDesc> = serde_json::from_value(lit["groups"].clone()).unwrap();
//! let images = mkr_shade::images_from_lit(lit);
//! let mat: mkr_shade::MaterialDesc = serde_json::from_value(lit["materials"]["skin"].clone()).unwrap();
//! let m = mkr_shade::compile("m0", &mat, &groups, &images, 0).unwrap();
//! assert!(m.unsupported.is_empty());
//! ```

mod codegen;
pub mod graph;
pub mod lut;
mod nodes;
pub mod probe;
mod sheen_table;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::Value;

pub use graph::{Anim, AnimSource, GroupDesc, MaterialDesc, WorldDesc};
pub use lut::{Lut, lut};

use codegen::{Gen, Mode, Shared};
use graph::{Flat, Ty};

/// The WGSL every material module starts with (contract + mkr-shade's library).
pub const PRELUDE: &str = concat!(
    include_str!("wgsl/prelude.wgsl"),
    include_str!("wgsl/math.wgsl"),
    include_str!("wgsl/color.wgsl"),
    include_str!("wgsl/noise.wgsl"),
    include_str!("wgsl/bsdf.wgsl"),
    include_str!("wgsl/surface.wgsl"),
);

/// Size of `Surface.attrs`.
pub const MAX_ATTRS: usize = 8;

/// An image as `compile` needs it; the slice order is the index `mkr_image` receives.
#[derive(Clone, Debug, Default)]
pub struct ImageDesc {
    pub name: String,
    /// Blender's colour space name ("sRGB", "Non-Color", "Linear Rec.709", ...).
    pub colorspace: String,
    /// STRAIGHT | PREMUL | CHANNEL_PACKED | NONE
    pub alpha_mode: String,
    /// The GPU texture holds premultiplied colour (Blender: float buffer and alpha_mode != STRAIGHT, or byte buffer
    /// and alpha_mode == PREMUL).
    pub premultiplied: bool,
    pub missing: bool,
}

impl ImageDesc {
    /// From a bake.json `lit.images` entry.
    pub fn from_bake(name: &str, v: &Value) -> ImageDesc {
        let alpha_mode = v.get("alpha_mode").and_then(|x| x.as_str()).unwrap_or("STRAIGHT").to_string();
        let float = v.get("float").and_then(|x| x.as_bool()).unwrap_or(false);
        let premultiplied = if float { alpha_mode != "STRAIGHT" } else { alpha_mode == "PREMUL" };
        ImageDesc {
            name: name.to_string(),
            colorspace: v.get("colorspace").and_then(|x| x.as_str()).unwrap_or("sRGB").to_string(),
            alpha_mode,
            premultiplied,
            missing: v.get("missing").and_then(|x| x.as_bool()).unwrap_or(false),
        }
    }
}

/// bake.json `lit.images` as a list, in the bake's key order.
pub fn images_from_lit(lit: &Value) -> Vec<ImageDesc> {
    match lit.get("images").and_then(|v| v.as_object()) {
        Some(m) => m.iter().map(|(k, v)| ImageDesc::from_bake(k, v)).collect(),
        None => Vec::new(),
    }
}

/// Colour spaces Blender treats as data (alpha never touches colour).
pub fn is_data_colorspace(name: &str) -> bool {
    matches!(name, "Non-Color" | "Raw" | "Generic Data" | "data")
}

/// Colour spaces whose texels need no conversion beyond what the renderer's upload does (sRGB decode or none).
pub fn known_colorspace(name: &str) -> bool {
    matches!(
        name,
        "sRGB" | "Non-Color" | "Raw" | "Linear Rec.709" | "Linear" | "Linear CIE-XYZ D65" | "Generic Data" | "data"
            | "scene_linear" | "Utility - Linear - sRGB" | "Utility - sRGB - Texture"
    ) || name.is_empty()
}

/// An animated socket's slot in the renderer's per-frame parameter buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamSlot {
    /// Global slot: `mkr_param(slot)`.
    pub slot: u32,
    pub source: AnimSource,
}

/// An AOV Output node of the material.
#[derive(Clone, Debug)]
pub struct Aov {
    pub name: String,
    /// `fn <function>(s: Surface) -> vec4f`: rgb = its Color input, a = its Value input.
    pub function: String,
}

/// One compiled material.
#[derive(Clone, Debug)]
pub struct CompiledMaterial {
    pub id: String,
    /// `mat_<id>`, `mat_<id>_alpha`, `mat_<id>_alpha_lod` (+ AOV functions and table helpers).
    pub wgsl: String,
    /// What `Surface.attrs[i]` must hold: "uv:" / "uv:<map>", "color:" / "color:<name>", "attr:<name>",
    /// "tangent:" / "tangent:<map>".
    pub attributes: Vec<String>,
    /// Indices into the images list that the code samples.
    pub images: Vec<usize>,
    /// Animated sockets: slot -> source. Slots are param_base .. param_base + param_count.
    pub params: Vec<ParamSlot>,
    pub param_count: u32,
    /// Opacity can drop below 1 (a Transparent BSDF, a Principled alpha, ... feeds the surface).
    pub transparent: bool,
    /// mat_<id> uses dpdx / dpdy (Bump): fragment stage only.
    pub uses_derivatives: bool,
    /// mat_<id> calls mkr_shader_to_rgb.
    pub uses_shader_to_rgb: bool,
    pub aovs: Vec<Aov>,
    /// Node types and features mkr-shade does not draw (their outputs are zeros); the material is not 1:1.
    pub unsupported: Vec<String>,
    pub warnings: Vec<String>,
}

fn valid_id(id: &str) -> bool {
    let mut c = id.chars();
    matches!(c.next(), Some(ch) if ch.is_ascii_alphabetic())
        && id.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        && !id.contains("__")
}

/// Blender's choice of output node for EEVEE (exact target first, then ALL; the active one among equals).
fn output_node(flat: &Flat, kind: &str) -> Option<usize> {
    let mut best: Option<(usize, bool, bool)> = None; // (index, exact target, active)
    for (i, n) in flat.nodes.iter().enumerate() {
        if n.kind != kind || n.path.contains('/') {
            continue;
        }
        let target = n.prop_str("target").unwrap_or("ALL");
        let exact = match target {
            "EEVEE" => true,
            "ALL" => false,
            _ => continue,
        };
        let active = n.prop_bool("is_active_output").unwrap_or(false);
        best = match best {
            None => Some((i, exact, active)),
            Some((_, false, _)) if exact => Some((i, exact, active)),
            Some((j, e, a)) if e == exact && active && !a => Some((i, exact, active)).or(Some((j, e, a))),
            other => other,
        };
    }
    best.map(|(i, _, _)| i)
}

/// Nodes reachable from `roots` through links.
fn reachable(flat: &Flat, roots: &[usize]) -> Vec<bool> {
    let mut seen = vec![false; flat.nodes.len()];
    let mut stack: Vec<usize> = roots.to_vec();
    while let Some(n) = stack.pop() {
        if std::mem::replace(&mut seen[n], true) {
            continue;
        }
        for s in &flat.nodes[n].inputs {
            if let Some((src, _)) = s.link {
                if !seen[src] {
                    stack.push(src);
                }
            }
        }
    }
    seen
}

/// Image nodes whose Alpha output feeds a reachable node.
fn alpha_used(flat: &Flat, seen: &[bool]) -> HashSet<usize> {
    let mut used = HashSet::new();
    for (m, node) in flat.nodes.iter().enumerate() {
        if !seen[m] {
            continue;
        }
        for s in &node.inputs {
            if let Some((src, out)) = s.link {
                let sn = &flat.nodes[src];
                if matches!(sn.kind.as_str(), "TEX_IMAGE" | "TEX_ENVIRONMENT")
                    && sn.outputs.get(out).map(|o| o.identifier == "Alpha").unwrap_or(false)
                {
                    used.insert(src);
                }
            }
        }
    }
    used
}

/// (node, output) pairs linked into reachable nodes.
fn used_outputs(flat: &Flat, seen: &[bool]) -> HashSet<(usize, usize)> {
    let mut used = HashSet::new();
    for (m, node) in flat.nodes.iter().enumerate() {
        if seen[m] {
            used.extend(node.inputs.iter().filter_map(|s| s.link));
        }
    }
    used
}

struct Built {
    wgsl: String,
    sh_attrs: Vec<String>,
    unsupported: BTreeSet<String>,
    warnings: BTreeSet<String>,
    uses_derivatives: bool,
    uses_shader_to_rgb: bool,
    transparent: bool,
    images: BTreeSet<usize>,
    aovs: Vec<Aov>,
}

fn build(
    id: &str,
    label: &str,
    flat: &Flat,
    out_kind: &str,
    images: &[ImageDesc],
    param_base: u32,
    pass_index: f64,
    world: bool,
    translucent_slab: bool,
) -> Built {
    let out = output_node(flat, out_kind);
    let mut roots: Vec<usize> = out.into_iter().collect();
    let aov_nodes: Vec<usize> = flat
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == "OUTPUT_AOV" && !n.path.contains('/'))
        .map(|(i, _)| i)
        .collect();
    roots.extend(&aov_nodes);
    let seen = reachable(flat, &roots);
    let image_index: HashMap<&str, usize> = images.iter().enumerate().map(|(i, im)| (im.name.as_str(), i)).collect();
    let mut sh = Shared {
        flat,
        images,
        image_index,
        param_base,
        pass_index,
        attrs: Vec::new(),
        unsupported: flat.unsupported.iter().cloned().collect(),
        warnings: flat.warnings.iter().cloned().collect(),
        uses_derivatives: false,
        uses_shader_to_rgb: false,
        alpha_used: alpha_used(flat, &seen),
        helpers: Vec::new(),
        helper_names: HashMap::new(),
        id: id.to_string(),
        transparent: false,
        used_images: BTreeSet::new(),
        used_outputs: used_outputs(flat, &seen),
    };
    if out.is_none() {
        sh.warnings.insert(format!("no active {out_kind} node: the surface is black"));
    }
    if let Some(o) = out {
        let node = &flat.nodes[o];
        // EEVEE uses the thickness for translucency, refraction and subsurface: a linked one is not drawn
        for (key, what) in [("Volume", "volume"), ("Displacement", "displacement"), ("Thickness", "thickness")] {
            if let Some(i) = node.input_index(key) {
                if node.inputs[i].link.is_some() {
                    sh.unsupported.insert(format!("{out_kind} {what}"));
                }
            }
        }
    }

    let surface = |g: &mut Gen| -> String {
        match out.and_then(|o| g.node(o).input_index("Surface").map(|i| (o, i))) {
            Some((o, i)) => g.input_idx(o, i, Ty::Shader),
            None => "mks_closures_none()".into(),
        }
    };

    let mut fns = Vec::new();
    {
        let mut g = Gen::new(&mut sh, Mode::Surface);
        let ret = surface(&mut g);
        // the material's thickness mode decides how the renderer lights translucent closures (prelude: Closures)
        let ret = if translucent_slab {
            g.stmt(format!("var mks_out: Closures = {ret};"));
            g.stmt("mks_out.translucent_slab = 1.0;");
            "mks_out".to_string()
        } else {
            ret
        };
        fns.push(g.finish(&format!("mat_{id}"), "Closures", &ret));
    }
    if !world {
        sh.transparent = false;
        for lod in [false, true] {
            let mut g = Gen::new(&mut sh, Mode::Alpha { lod });
            let ret = surface(&mut g);
            let name = if lod { format!("mat_{id}_alpha_lod") } else { format!("mat_{id}_alpha") };
            fns.push(g.finish(&name, "f32", &format!("mks_opacity({ret})")));
        }
    }
    let mut aovs = Vec::new();
    for (k, &a) in aov_nodes.iter().enumerate() {
        let mut g = Gen::new(&mut sh, Mode::Surface);
        let c = g.input(a, "Color", Ty::Color);
        let v = g.input(a, "Value", Ty::Float);
        let name = format!("mat_{id}_aov{k}");
        fns.push(g.finish(&name, "vec4f", &format!("vec4f(({c}).rgb, {v})")));
        aovs.push(Aov { name: flat.nodes[a].prop_str("aov_name").unwrap_or("").to_string(), function: name });
    }

    let mut wgsl = format!("// mkr-shade: {label}\n");
    for h in &sh.helpers {
        wgsl.push_str(h);
    }
    for f in fns {
        wgsl.push_str(&f);
    }
    Built {
        wgsl,
        sh_attrs: sh.attrs,
        unsupported: sh.unsupported,
        warnings: sh.warnings,
        uses_derivatives: sh.uses_derivatives,
        uses_shader_to_rgb: sh.uses_shader_to_rgb,
        transparent: sh.transparent,
        images: sh.used_images,
        aovs,
    }
}

fn finish(id: &str, b: Built, flat: &Flat, param_base: u32) -> CompiledMaterial {
    let params = flat
        .anims
        .iter()
        .enumerate()
        .map(|(i, s)| ParamSlot { slot: param_base + i as u32, source: s.clone() })
        .collect();
    CompiledMaterial {
        id: id.to_string(),
        wgsl: b.wgsl,
        attributes: b.sh_attrs,
        images: b.images.into_iter().collect(),
        params,
        param_count: flat.anims.len() as u32,
        transparent: b.transparent,
        uses_derivatives: b.uses_derivatives,
        uses_shader_to_rgb: b.uses_shader_to_rgb,
        aovs: b.aovs,
        unsupported: b.unsupported.into_iter().collect(),
        warnings: b.warnings.into_iter().collect(),
    }
}

/// A material without a node tree, as EEVEE draws it: a Principled BSDF from the material's viewport settings.
fn nodeless(id: &str, settings: &serde_json::Map<String, Value>) -> CompiledMaterial {
    let get = |k: &str, d: f64| settings.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    let col: Vec<f64> = settings
        .get("diffuse_color")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.8)).collect())
        .unwrap_or_else(|| vec![0.8, 0.8, 0.8, 1.0]);
    let c = |i: usize| codegen::lit(col.get(i).copied().unwrap_or(0.8));
    let spec = codegen::lit(get("specular_intensity", 0.5));
    let p = format!(
        "MksPrincipled(vec3f({}, {}, {}), {}, {}, 1.5, 1.0, s.N, 0.0, {spec}, vec3f(1.0), 0.0, 0.0, 0.03, 1.5, vec3f(1.0), s.N, 0.0, 0.5, vec3f(1.0), vec3f(0.0), 0.0)",
        c(0),
        c(1),
        c(2),
        codegen::lit(get("metallic", 0.0)),
        codegen::lit(get("roughness", 0.4))
    );
    let wgsl = format!(
        "// mkr-shade: material without nodes\nfn mat_{id}(s: Surface) -> Closures {{\n  return mks_bsdf_principled({p}, s.I, true);\n}}\nfn mat_{id}_alpha(s: Surface) -> f32 {{\n  return 1.0;\n}}\nfn mat_{id}_alpha_lod(s: Surface) -> f32 {{\n  return 1.0;\n}}\n"
    );
    CompiledMaterial {
        id: id.to_string(),
        wgsl,
        attributes: Vec::new(),
        images: Vec::new(),
        params: Vec::new(),
        param_count: 0,
        transparent: false,
        uses_derivatives: false,
        uses_shader_to_rgb: false,
        aovs: Vec::new(),
        unsupported: Vec::new(),
        warnings: Vec::new(),
    }
}

/// Why mkr cannot follow a material or world over the frames as Blender draws it: what is keyed or driven on it (`own`)
/// or on a node group its tree uses (followed down) that the bake holds one value of (bake_lit `_keyed`: a colour ramp's
/// stops, a node's setting, a material's or a world's own settings when it draws without nodes), or None.
pub fn keyed_reason(own: &[String], tree: Option<&graph::Graph>, groups: &BTreeMap<String, GroupDesc>) -> Option<String> {
    let mut paths: Vec<String> = own.to_vec();
    let mut seen = std::collections::BTreeSet::new();
    let mut todo: Vec<&graph::Graph> = tree.into_iter().collect();
    while let Some(t) = todo.pop() {
        for g in t.nodes.iter().filter_map(|n| n.group.as_deref()) {
            if let (true, Some(gd)) = (seen.insert(g), groups.get(g)) {
                paths.extend(gd.keyed.iter().map(|p| format!("group {g}: {p}")));
                todo.push(&gd.tree);
            }
        }
    }
    (!paths.is_empty())
        .then(|| format!("values keyed or driven that the bake holds one of (EEVEE draws them changing): {}", paths.join(", ")))
}

/// Compile one material. `id`: a WGSL identifier fragment (functions are `mat_<id>...`); `images`: the renderer's
/// global images list; `param_base`: the first `mkr_param` slot this material may use.
pub fn compile(
    id: &str,
    material: &MaterialDesc,
    groups: &BTreeMap<String, GroupDesc>,
    images: &[ImageDesc],
    param_base: u32,
) -> Result<CompiledMaterial, String> {
    if !valid_id(id) {
        return Err(format!("material id {id:?} is not a WGSL identifier fragment"));
    }
    let keyed = keyed_reason(&material.keyed, material.tree.as_ref(), groups);
    let Some(tree) = &material.tree else {
        let mut c = nodeless(id, &material.settings);
        c.unsupported.extend(keyed);
        return Ok(c);
    };
    let flat = graph::flatten(tree, &material.anim, groups);
    let pass_index = material.settings.get("pass_index").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let slab = material.settings.get("thickness_mode").and_then(|v| v.as_str()) == Some("SLAB");
    let b = build(id, "material", &flat, "OUTPUT_MATERIAL", images, param_base, pass_index, false, slab);
    let mut c = finish(id, b, &flat, param_base);
    c.unsupported.extend(keyed);
    Ok(c)
}

/// Compile the world: `mat_<id>(s) -> Closures` whose emission is the background radiance. The renderer evaluates it
/// with Surface.P = the unit direction looked along (world space), I = N = Ng = -P, generated = P, object matrices
/// identity, backfacing 0.
pub fn compile_world(
    id: &str,
    world: &WorldDesc,
    groups: &BTreeMap<String, GroupDesc>,
    images: &[ImageDesc],
    param_base: u32,
) -> Result<CompiledMaterial, String> {
    if !valid_id(id) {
        return Err(format!("world id {id:?} is not a WGSL identifier fragment"));
    }
    let tree = world.tree.clone().unwrap_or_default();
    let flat = graph::flatten(&tree, &world.anim, groups);
    let b = build(id, "world", &flat, "OUTPUT_WORLD", images, param_base, 0.0, true, false);
    let mut c = finish(id, b, &flat, param_base);
    c.unsupported.extend(keyed_reason(&world.keyed, Some(&tree), groups));
    Ok(c)
}

/// Parse and validate a complete module with naga (all capabilities); Err holds the formatted diagnostic.
pub fn validate(module_source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(module_source).map_err(|e| e.emit_to_string(module_source))?;
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).map_err(|e| e.emit_to_string(module_source))?;
    Ok(())
}

/// The functions the renderer provides, as trivial stand-ins, for validating material modules without a renderer.
pub const RENDERER_STUBS: &str = "
fn mkr_image(index: u32, uv: vec2f, interp: u32, ext: u32) -> vec4f { return vec4f(uv, f32(index + interp + ext), 1.0); }
fn mkr_image_lod(index: u32, uv: vec2f, interp: u32, ext: u32) -> vec4f { return vec4f(uv, f32(index + interp + ext), 1.0); }
fn mkr_param(slot: u32) -> vec4f { return vec4f(f32(slot)); }
fn mkr_shader_to_rgb(c: Closures, s: Surface) -> vec4f { return vec4f(c.diffuse + c.emission, 1.0); }
fn mkr_lut(uv: vec2f, layer: u32) -> vec4f { return vec4f(uv, f32(layer), 1.0); }
";

/// Collected for a set of materials: unsupported node types with the materials using them.
pub fn unsupported_report(compiled: &[(String, CompiledMaterial)]) -> BTreeMap<String, Vec<String>> {
    let mut r: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, m) in compiled {
        for u in &m.unsupported {
            r.entry(u.clone()).or_default().push(name.clone());
        }
    }
    r
}
