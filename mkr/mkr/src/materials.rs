//! The scene's materials through mkr-shade (Blender node graphs to WGSL closures), with what mkr needs to draw each:
//! its render method (DITHERED: hashed coverage, BLENDED: sorted and blended), back-face culling, whether its shadow
//! is alpha-tested, and the per-frame values of its animated sockets.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use mkr_shade::graph::AnimSource;
use mkr_shade::{CompiledMaterial, GroupDesc, ImageDesc, MaterialDesc, WorldDesc};
use serde_json::Value;

use crate::bake::{Bake, ImageRec};

/// The id of the default surface (empty material slots).
const DEFAULT: &str = "\u{0}default";

pub struct Material {
    pub name: String,
    /// Its WGSL id: a hash of its name, so its code does not change when other materials come or go.
    pub id: String,
    pub wgsl: String,
    pub attributes: Vec<String>,
    /// Opacity can drop below 1.
    pub transparent: bool,
    /// BLENDED render method: sorted, blended over the rest.
    pub blended: bool,
    pub cull: bool,
    /// show_transparent_back: every blended layer (else only the nearest).
    pub show_back: bool,
    /// Its shadow follows its opacity (else it casts as opaque).
    pub transparent_shadow: bool,
    /// It calls Shader to RGB (its module lights closures itself).
    pub uses_shader_to_rgb: bool,
    /// Its animated sockets; their slots count from 0, at `param_base` in the scene's parameter buffer.
    pub params: Vec<mkr_shade::ParamSlot>,
    pub param_base: u32,
    /// The scene's images behind its own image indices (mkr_image's), at `image_base` in the image map.
    pub image_ids: Vec<usize>,
    pub image_base: u32,
    pub unsupported: Vec<String>,
}

impl Material {
    /// Its shadow follows its opacity: shadow maps test it, and so do shadow rays (its BLAS geometry is not opaque).
    pub fn casts_alpha_shadow(&self) -> bool {
        self.transparent && self.transparent_shadow
    }

    /// Its slots that hold a value per object (Attribute nodes of type Object, Instancer, View Layer): (slot, the
    /// mesh's `object_attrs` key "KIND:name").
    pub fn object_slots(&self) -> impl Iterator<Item = (u32, String)> + '_ {
        self.params.iter().filter_map(|p| match &p.source {
            AnimSource::Object { kind, name } => Some((p.slot, format!("{kind}:{name}"))),
            _ => None,
        })
    }

    fn param_count(&self) -> u32 {
        self.params.iter().map(|p| p.slot + 1).max().unwrap_or(0)
    }
}

/// The parameter buffer (`mkr_param`): every material's block of animated sockets, then a block of its own for each
/// draw of a lit mesh whose material reads values per object (the material's block copied, its object slots the
/// mesh's `object_attrs`); a material's own block holds zeros there (screen type, the default surface).
pub struct ParamLayout {
    /// (block start, material, lit mesh) of each draw's own block.
    own: Vec<(u32, usize, usize)>,
    pub count: u32,
}

impl ParamLayout {
    pub fn new(mats: &MaterialSet) -> ParamLayout {
        ParamLayout { own: Vec::new(), count: mats.param_count }
    }

    /// The `param_base` of a draw of lit mesh `mesh` (an index into `lit.meshes`) in material `mat`.
    pub fn draw_base(&mut self, mats: &MaterialSet, mat: usize, mesh: usize) -> u32 {
        let m = &mats.list[mat];
        if m.object_slots().next().is_none() {
            return m.param_base;
        }
        let base = self.count;
        self.own.push((base, mat, mesh));
        self.count += m.param_count();
        base
    }

    /// The buffer's values on bake frame `k`, one vec4 per slot. An object attribute the bake does not hold for the
    /// mesh reads zeros: `Support` gives the frames that show such a mesh to Blender.
    pub fn values(&self, mats: &MaterialSet, bake: &Bake, k: usize) -> Result<Vec<[f32; 4]>> {
        let mut out = mats.params_at(bake, k)?;
        out.resize(self.count as usize, [0.0; 4]);
        let meshes = &bake.lit()?.meshes;
        for &(base, mi, mesh) in &self.own {
            let m = &mats.list[mi];
            let (b, own) = (base as usize, m.param_base as usize);
            for j in 0..m.param_count() as usize {
                out[b + j] = out[own + j];
            }
            for (slot, key) in m.object_slots() {
                if let Some(rec) = meshes[mesh].object_attrs.get(&key) {
                    out[b + slot as usize] = rec.at(bake, k)?;
                }
            }
        }
        Ok(out)
    }
}

pub struct MaterialSet {
    pub list: Vec<Material>,
    pub by_name: BTreeMap<String, usize>,
    /// The scene's world when it shows or lights anything (`world_shows`): None draws EEVEE's black void.
    pub world: Option<World>,
    /// The scene's images (the image binding array's order).
    pub images: Vec<(String, ImageRec)>,
    pub prelude: String,
    /// The light code's maps per sun (shadows.rs SUN_CASCADES when a sun casts shadows, else 1: the code of one map).
    pub sun_cascades: u32,
    /// Whether the scene traces EEVEE's rays through the screen (raytrace.rs): the light code holds the tracing only
    /// then (pipelines.rs RT_TRACED), so a scene without ray tracing compiles the code it compiled before.
    pub traced: bool,
    pub param_count: u32,
    /// Every material's image indices in the scene's, material after material (Material::image_base), then the world's.
    pub image_map: Vec<u32>,
    /// Per material (index): its anim records; per group name: its anim records.
    mat_anim: Vec<Vec<crate::bake::AnimRec>>,
    group_anim: BTreeMap<String, Vec<crate::bake::AnimRec>>,
}

/// The scene's world (bake.json `lit.world`) compiled by mkr-shade: `mat_world(s) -> Closures`, whose emission is the
/// background radiance in the direction `s.P`, with its World settings. Its ray visibility flags are Cycles' alone:
/// EEVEE shows and lights with a world whatever they say (lab world_visibility).
pub struct World {
    pub name: String,
    pub wgsl: String,
    /// Its animated sockets (slots from 0, at `param_base`) and images (at `image_base` in the image map), as a
    /// material's.
    pub params: Vec<mkr_shade::ParamSlot>,
    pub param_base: u32,
    pub image_base: u32,
    /// What mkr cannot draw of it (node types mkr-shade lacks, a volume, settings the bake does not hold).
    pub unsupported: Vec<String>,
    /// Its World properties (probe_resolution, sun_threshold, sun_angle, use_sun_shadow, ...).
    pub props: serde_json::Map<String, Value>,
    anim: Vec<crate::bake::AnimRec>,
}

impl World {
    /// The world record `w` compiled as `mat_world`, its parameter slots and images placed at `param_base` and
    /// `image_base` of the scene's; with the scene's images it samples, in its own order (the image map's entries).
    fn compile(w: &Value, groups: &BTreeMap<String, GroupDesc>, descs: &[ImageDesc], recs: &[(String, ImageRec)],
               param_base: u32, image_base: u32) -> Result<(World, Vec<usize>)> {
        let name = w.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
        let anim: Vec<crate::bake::AnimRec> = match w.get("anim") {
            Some(a) if !a.is_null() => serde_json::from_value(a.clone()).with_context(|| format!("world {name} anim"))?,
            _ => Vec::new(),
        };
        let mut unsupported = Vec::new();
        let props = w.get("props").and_then(Value::as_object).cloned();
        if props.is_none() {
            unsupported.push("its settings, which the bake does not hold (`mk build --bake`)".to_string());
        }
        let nodes = w.get("use_nodes").and_then(Value::as_bool).unwrap_or(false) && w.get("tree").is_some_and(|t| !t.is_null());
        let (wgsl, params, image_ids) = if nodes {
            let desc: WorldDesc = serde_json::from_value(w.clone()).with_context(|| format!("world {name}"))?;
            let first = mkr_shade::compile_world("world", &desc, groups, descs, 0)
                .map_err(|e| anyhow::anyhow!("world {name}: {e}"))?;
            let image_ids = first.images.clone();
            let c = if image_ids.is_empty() {
                first
            } else {
                let own: Vec<ImageDesc> = image_ids.iter().map(|&i| descs[i].clone()).collect();
                mkr_shade::compile_world("world", &desc, groups, &own, 0).map_err(|e| anyhow::anyhow!("world {name}: {e}"))?
            };
            unsupported.extend(c.unsupported);
            for &i in &image_ids {
                let (iname, rec) = &recs[i];
                if !rec.missing && !mkr_shade::known_colorspace(&rec.colorspace) {
                    unsupported.push(format!("image {iname} in colour space {}", rec.colorspace));
                }
            }
            (c.wgsl, c.params, image_ids)
        } else {
            // without nodes EEVEE's background is the world's colour
            let c: Vec<f64> = w.get("color").and_then(Value::as_array)
                .map(|a| a.iter().take(3).map(|x| x.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_else(|| vec![0.0; 3]);
            let wgsl = format!("// mkr: the world's colour (no nodes)\nfn mat_world(s: Surface) -> Closures {{\n  \
                                return mks_closure_from_color(vec3f({:?}, {:?}, {:?}));\n}}\n", c[0], c[1], c[2]);
            let keyed: Vec<String> = w.get("keyed").and_then(|k| serde_json::from_value(k.clone()).ok()).unwrap_or_default();
            unsupported.extend(mkr_shade::keyed_reason(&keyed, None, groups));
            (wgsl, Vec::new(), Vec::new())
        };
        let world = World { name, wgsl, params, param_base, image_base, unsupported, props: props.unwrap_or_default(),
                            anim };
        Ok((world, image_ids))
    }

    /// A World property as a number (enums such as probe_resolution are their text's number).
    pub fn prop(&self, key: &str) -> Option<f64> {
        match self.props.get(key)? {
            Value::Number(n) => n.as_f64(),
            Value::Bool(b) => Some(*b as u32 as f64),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }
}

fn setting<'a>(s: &'a Value, key: &str) -> Option<&'a Value> {
    s.get(key)
}

/// A WGSL-safe id per material from its name (FNV-1a), unique within the set.
fn wgsl_id(name: &str, taken: &std::collections::BTreeSet<String>) -> String {
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    let mut id = format!("m{h:08x}");
    let mut k = 1;
    while taken.contains(&id) {
        id = format!("m{h:08x}_{k}");
        k += 1;
    }
    id
}

impl MaterialSet {
    /// Compile every material of the lit bake (and the default surface for empty slots).
    pub fn compile(bake: &Bake) -> Result<MaterialSet> {
        let lit = bake.lit()?;
        let raw_lit = bake.raw.get("lit").context("bake.json has no lit")?;
        let groups: BTreeMap<String, GroupDesc> = match raw_lit.get("groups") {
            Some(g) if !g.is_null() => serde_json::from_value(g.clone()).context("lit.groups")?,
            _ => BTreeMap::new(),
        };
        let image_descs = mkr_shade::images_from_lit(raw_lit);
        let mut images = Vec::new();
        for d in &image_descs {
            let rec = lit.images.get(&d.name).cloned().with_context(|| format!("image {} is not in lit.images", d.name))?;
            images.push((d.name.clone(), rec));
        }
        let raw_mats = raw_lit.get("materials").and_then(|m| m.as_object());
        let mut names: Vec<String> = lit.materials.keys().cloned().collect();
        names.push(DEFAULT.to_string());
        let mut list = Vec::new();
        let mut by_name = BTreeMap::new();
        let mut mat_anim = Vec::new();
        let mut taken = std::collections::BTreeSet::new();
        let mut param_base = 0u32;
        let mut image_map = Vec::new();
        for name in &names {
            let desc: MaterialDesc = if name == DEFAULT {
                MaterialDesc::default()
            } else {
                let v = raw_mats.and_then(|m| m.get(name)).with_context(|| format!("material {name}"))?;
                serde_json::from_value(v.clone()).with_context(|| format!("material {name}"))?
            };
            let id = wgsl_id(name, &taken);
            taken.insert(id.clone());
            // its images in the order it uses them, then compiled against that list alone: its code indexes its own
            // images and parameters, the same whatever else the scene holds
            let first = mkr_shade::compile(&id, &desc, &groups, &image_descs, 0)
                .map_err(|e| anyhow::anyhow!("material {name}: {e}"))?;
            let image_ids = first.images.clone();
            let c: CompiledMaterial = if image_ids.is_empty() {
                first
            } else {
                let own: Vec<ImageDesc> = image_ids.iter().map(|&i| image_descs[i].clone()).collect();
                mkr_shade::compile(&id, &desc, &groups, &own, 0).map_err(|e| anyhow::anyhow!("material {name}: {e}"))?
            };
            if c.images.iter().enumerate().any(|(k, &i)| k != i) {
                bail!("material {name}: its images do not keep their order when compiled alone");
            }
            let settings = Value::Object(desc.settings.clone());
            let flag = |k: &str, d: bool| setting(&settings, k).and_then(|v| v.as_bool()).unwrap_or(d);
            let blended = setting(&settings, "surface_render_method").and_then(|v| v.as_str()) == Some("BLENDED");
            let mut unsupported = c.unsupported.clone();
            for &ii in &image_ids {
                let (iname, rec) = &images[ii];
                if !rec.missing && !mkr_shade::known_colorspace(&rec.colorspace) {
                    unsupported.push(format!("image {iname} in colour space {}", rec.colorspace));
                }
            }
            if c.attributes.iter().any(|a| a.starts_with("tangent:")) {
                unsupported.push("tangents (the bake holds none)".into());
            }
            if c.attributes.len() > mkr_shade::MAX_ATTRS {
                unsupported.push(format!("{} attributes (mkr passes {})", c.attributes.len(), mkr_shade::MAX_ATTRS));
            }
            by_name.insert(name.clone(), list.len());
            mat_anim.push(lit.materials.get(name).map(|m| m.anim.clone()).unwrap_or_default());
            let image_base = image_map.len() as u32;
            image_map.extend(image_ids.iter().map(|&i| i as u32));
            list.push(Material {
                name: name.clone(),
                id,
                wgsl: c.wgsl,
                attributes: c.attributes,
                transparent: c.transparent,
                blended,
                cull: flag("use_backface_culling", false),
                show_back: flag("show_transparent_back", true),
                transparent_shadow: flag("use_transparent_shadow", true),
                uses_shader_to_rgb: c.uses_shader_to_rgb,
                params: c.params,
                param_base,
                image_ids,
                image_base,
                unsupported,
            });
            param_base += c.param_count;
        }
        let mut group_anim = BTreeMap::new();
        if let Some(g) = raw_lit.get("groups").and_then(|g| g.as_object()) {
            for (name, v) in g {
                let anim: Vec<crate::bake::AnimRec> = match v.get("anim") {
                    Some(a) if !a.is_null() => serde_json::from_value(a.clone()).with_context(|| format!("group {name} anim"))?,
                    _ => Vec::new(),
                };
                group_anim.insert(name.clone(), anim);
            }
        }
        let world = match lit.world.as_ref().filter(|w| world_shows(w)) {
            Some(w) => {
                let (world, image_ids) = World::compile(w, &groups, &image_descs, &images, param_base, image_map.len() as u32)?;
                param_base += world.params.iter().map(|p| p.slot + 1).max().unwrap_or(0);
                image_map.extend(image_ids.iter().map(|&i| i as u32));
                Some(world)
            }
            None => None,
        };
        Ok(MaterialSet {
            list,
            by_name,
            world,
            images,
            prelude: mkr_shade::PRELUDE.to_string(),
            sun_cascades: if lit.lights.iter().any(|l| l.kind == "SUN" && l.use_shadow) { crate::shadows::SUN_CASCADES } else { 1 },
            traced: crate::raytrace::screen_traced(&lit.eevee),
            param_count: param_base,
            image_map,
            mat_anim,
            group_anim,
        })
    }

    /// The attributes material `name` reads.
    pub fn attributes(&self, name: &str) -> Vec<String> {
        self.by_name.get(name).map(|&i| self.list[i].attributes.clone()).unwrap_or_default()
    }

    /// Index of a material by name.
    pub fn index(&self, name: &str) -> Result<usize> {
        self.by_name.get(name).copied().with_context(|| format!("material {name} is not in lit.materials"))
    }

    pub fn default_index(&self) -> usize {
        self.by_name[DEFAULT]
    }

    /// The animated sockets' values on bake frame `k`, one vec4 per slot (the materials', then the world's).
    pub fn params_at(&self, bake: &Bake, k: usize) -> Result<Vec<[f32; 4]>> {
        let mut out = vec![[0.0f32; 4]; self.param_count as usize];
        for (mi, m) in self.list.iter().enumerate() {
            self.fill(&mut out, bake, k, m.param_base, &m.params, &self.mat_anim[mi], &format!("material {}", m.name))?;
        }
        if let Some(w) = &self.world {
            self.fill(&mut out, bake, k, w.param_base, &w.params, &w.anim, &format!("world {}", w.name))?;
        }
        Ok(out)
    }

    /// The slots `params` (at `base`) of a material or the world on bake frame `k`, from its own anim records `own` or
    /// its groups'.
    #[allow(clippy::too_many_arguments)]
    fn fill(&self, out: &mut [[f32; 4]], bake: &Bake, k: usize, base: u32, params: &[mkr_shade::ParamSlot],
            own: &[crate::bake::AnimRec], what: &str) -> Result<()> {
        for p in params {
            let rec = match &p.source {
                AnimSource::Material(i) => own.get(*i),
                AnimSource::Group(g, i) => self.group_anim.get(g).and_then(|a| a.get(*i)),
                AnimSource::Object { .. } => continue,     // per draw: ParamLayout
            };
            let Some(rec) = rec else {
                bail!("{what}: animated socket slot {} has no record", p.slot);
            };
            let kk = if rec.frames == 1 { 0 } else { k };
            let v = bake.f(rec.offset + rec.size * kk, rec.size)?;
            let slot = &mut out[(base + p.slot) as usize];
            for (j, x) in v.iter().take(4).enumerate() {
                slot[j] = *x;
            }
            if rec.size == 1 {
                *slot = [v[0], v[0], v[0], v[0]];
            }
        }
        Ok(())
    }

    /// mkr-shade's LUT texture: (side, side, layers, rgba texels).
    pub fn shade_lut(&self) -> (u32, u32, u32, Vec<f32>) {
        let l = mkr_shade::lut::lut();
        (l.size, l.size, l.layers, l.data.clone())
    }
}

/// What mkr cannot draw (for `mkr info` and for frames handed to Blender): materials with node types mkr-shade lacks,
/// meshes mkr cannot hold, and what of the lit scene it does not draw at all.
pub struct Support {
    pub materials: BTreeMap<String, Vec<String>>,
    pub meshes: BTreeMap<String, String>,
    /// Why no lit frame of the scene is mkr's, when none is (`scene_problem`).
    pub scene: Option<String>,
}

impl Support {
    pub fn of(bake: &Bake) -> Result<Support> {
        let mut materials = BTreeMap::new();
        let mut meshes = BTreeMap::new();
        let Some(lit) = &bake.doc.lit else {
            return Ok(Support { materials, meshes, scene: None });
        };
        let set = MaterialSet::compile(bake)?;
        for m in &set.list {
            if !m.unsupported.is_empty() && m.name != DEFAULT {
                materials.insert(m.name.clone(), m.unsupported.clone());
            }
        }
        for m in &lit.meshes {
            if let Some(why) = mesh_problem(m) {
                meshes.insert(m.name.clone(), why);
            }
            // what a material reads off this mesh's object that the bake does not hold (a bake from before
            // object_attrs): the material is not drawn lit; the looks, which do not shade it, still are
            for mat in m.materials.iter().flatten().filter_map(|name| set.by_name.get(name).map(|&i| &set.list[i])) {
                for (_, key) in mat.object_slots().filter(|(_, key)| !m.object_attrs.contains_key(key)) {
                    let why = format!("object attribute {key}, which the bake does not hold (`mk build --bake`)");
                    let list = materials.entry(mat.name.clone()).or_insert_with(Vec::new);
                    if !list.contains(&why) {
                        list.push(why);
                    }
                }
            }
        }
        Ok(Support { materials, meshes, scene: scene_problem(lit, set.world.as_ref()) })
    }
}

/// Why mkr draws no lit frame of the scene, when it draws none: what it lacks of the world (`world`: the compiled one,
/// when it shows or lights: a volume, node types mkr-shade lacks, settings the bake does not hold, and EEVEE's sun
/// extraction: the light over a non-zero sun threshold taken out into a sun light, which mkr does not draw unless the
/// world is one colour under it) or of EEVEE's ray tracing (mkr traces the screen as its SCREEN method does:
/// raytrace.rs; not its light-probe method). The look passes are EEVEE's with a black world and no ray tracing: this
/// is not theirs.
pub fn scene_problem(lit: &crate::bake::Lit, world: Option<&World>) -> Option<String> {
    if let Some(w) = world {
        if !w.unsupported.is_empty() {
            return Some(format!("world {}: {}", w.name, w.unsupported.join(", ")));
        }
        let threshold = w.prop("sun_threshold").unwrap_or(0.0);
        if threshold > 0.0 && !world_under(lit.world.as_ref(), threshold) {
            return Some(format!("world {}: EEVEE takes its light over the sun threshold ({threshold}) out into a sun \
                                 light, which mkr does not draw", w.name));
        }
    }
    if lit.eevee.get("use_raytracing").and_then(Value::as_bool).unwrap_or(false) && !crate::raytrace::screen_traced(&lit.eevee) {
        return Some("EEVEE ray tracing by light probes (ray_tracing_method PROBE): mkr traces the screen only".into());
    }
    None
}

/// Whether a world record is one colour (no nodes, or a Background whose colour and strength are neither linked nor
/// keyed) whose radiance stays under `threshold` on every channel.
fn world_under(w: Option<&Value>, threshold: f64) -> bool {
    let Some(w) = w else { return true };
    let rgb = |v: &Value| v.as_array().map(|a| a.iter().take(3).filter_map(Value::as_f64).fold(0.0f64, f64::max));
    if !w.get("use_nodes").and_then(Value::as_bool).unwrap_or(false) {
        return w.get("color").and_then(rgb).is_some_and(|c| c < threshold);
    }
    let Some(tree) = w.get("tree").and_then(|t| serde_json::from_value::<mkr_shade::graph::Graph>(t.clone()).ok()) else {
        return false;
    };
    let keyed = w.get("anim").and_then(Value::as_array).is_some_and(|a| !a.is_empty());
    let mut backgrounds = tree.nodes.iter().filter(|n| n.kind == "BACKGROUND");
    let (Some(bg), None) = (backgrounds.next(), backgrounds.next()) else { return false };
    let value = |id: &str| bg.inputs.iter().find(|s| s.identifier == id && !s.linked).map(|s| s.value.clone());
    match (value("Color"), value("Strength")) {
        (Some(c), Some(s)) if !keyed && tree.nodes.len() <= 2 => {
            rgb(&c).zip(s.as_f64()).is_some_and(|(c, s)| c * s < threshold)
        }
        _ => false,
    }
}

/// Whether a world (bake.json `lit.world`) shows or lights anything as far as its record tells: without nodes, its
/// colour unless black; with them, any EEVEE output (target ALL or EEVEE) with a Volume, or a Surface other than a
/// Background left black (its colour or its strength 0, neither linked nor keyed).
fn world_shows(w: &Value) -> bool {
    let black = |v: &Value| match v {
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::Array(a) => a.iter().take(3).all(|x| x.as_f64() == Some(0.0)),
        _ => false,
    };
    if !w.get("use_nodes").and_then(Value::as_bool).unwrap_or(false) {
        return !w.get("color").is_some_and(black);
    }
    let Some(tree) = w.get("tree").and_then(|t| serde_json::from_value::<mkr_shade::graph::Graph>(t.clone()).ok()) else {
        return true;
    };
    let anim: Vec<mkr_shade::Anim> =
        w.get("anim").and_then(|a| serde_json::from_value(a.clone()).ok()).unwrap_or_default();
    let eevee = |n: &&mkr_shade::graph::NodeDesc| {
        n.kind == "OUTPUT_WORLD" && matches!(n.props.get("target").and_then(Value::as_str), None | Some("ALL" | "EEVEE"))
    };
    tree.nodes.iter().filter(eevee).any(|out| {
        tree.links.iter().filter(|l| l.2 == out.name).any(|(from, _, _, to)| {
            let Some(bg) = tree.nodes.iter().find(|n| &n.name == from && n.kind == "BACKGROUND" && !n.mute) else {
                return true;
            };
            let still = |id: &str| {
                bg.inputs.iter().find(|s| s.identifier == id && !s.linked)
                    .filter(|_| !anim.iter().any(|a| a.node == bg.name && a.socket == id))
            };
            let dark = still("Color").is_some_and(|s| black(&s.value)) || still("Strength").is_some_and(|s| black(&s.value));
            to != "Surface" || !dark
        })
    })
}

/// Why mkr cannot hold a lit mesh record, if it cannot.
fn mesh_problem(m: &crate::bake::MeshRec) -> Option<String> {
    if m.dqs {
        return Some("preserve-volume (dual quaternion) skinning".into());
    }
    match (&m.by_frame, m.kind.as_str()) {
        (Some(_), _) => None,
        (None, "by_frame") => Some("frame by frame without geometries".into()),
        (None, _) => {
            for need in ["position", "normal"] {
                if !m.attributes.contains_key(need) {
                    return Some(format!("no {need} attribute"));
                }
            }
            if m.triangles.is_none() {
                return Some("no triangles".into());
            }
            if m.kind == "skinned" && !m.attributes.contains_key("joints") {
                return Some("skinned without joints".into());
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sock(id: &str, ty: &str, value: Value) -> Value {
        json!({"identifier": id, "name": id, "type": ty, "enabled": true, "linked": false, "value": value})
    }

    /// A material showing what its Attribute node (type `kind`, name `name`) reads, its Emission strength keyed.
    fn shows_attribute(kind: &str, name: &str) -> Value {
        let out = |id: &str, ty: &str| sock(id, ty, Value::Null);
        json!({"settings": {}, "anim": [{"node": "em", "socket": "Strength", "io": "input", "size": 1, "offset": 12,
                                         "frames": 2}],
               "tree": {"nodes": [
                   {"name": "at", "type": "ATTRIBUTE", "props": {"attribute_type": kind, "attribute_name": name},
                    "inputs": [], "outputs": [out("Color", "RGBA"), out("Vector", "VECTOR"), out("Fac", "VALUE"),
                                              out("Alpha", "VALUE")]},
                   {"name": "em", "type": "EMISSION", "props": {}, "inputs": [sock("Color", "RGBA", json!([1, 1, 1, 1])),
                    sock("Strength", "VALUE", json!(1.0))], "outputs": [out("Emission", "SHADER")]},
                   {"name": "out", "type": "OUTPUT_MATERIAL", "props": {"is_active_output": true, "target": "ALL"},
                    "inputs": [sock("Surface", "SHADER", Value::Null)], "outputs": []}],
                        "links": [["at", "Color", "em", "Color"], ["em", "Emission", "out", "Surface"]]}})
    }

    /// Meshes `a` and `b` in material `ring` (which shows its object attribute mk_amp) with values of their own (with
    /// `lacking`, `c` in it too without them); `d` in `plain`, which reads none. bake.bin: a's mk_amp on two frames,
    /// b's on one, ring's strength.
    fn bake_dir(tag: &str, lacking: bool) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mkr_materials_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mesh = |name: &str, mat: &str, attrs: Value| json!({"name": name, "kind": "static", "materials": [mat],
            "object_attrs": attrs, "triangles": {"offset": 0, "count": 0},
            "attributes": {"position": {"offset": 0, "comp": 3, "type": "f32"},
                           "normal": {"offset": 0, "comp": 3, "type": "f32"}}});
        let mut meshes = vec![mesh("a", "ring", json!({"OBJECT:mk_amp": {"offset": 0, "frames": 2}})),
                              mesh("b", "ring", json!({"OBJECT:mk_amp": {"offset": 8, "frames": 1}})),
                              mesh("d", "plain", json!({}))];
        if lacking {
            meshes.push(mesh("c", "ring", json!({})));
        }
        let doc = json!({"version": 5, "fps": 30.0, "frame0": 1, "frames": 2, "outputs": [{"name": "o", "size": [4, 4]}],
            "lit": {"view": {"lut": {"file": "view_lut.bin", "n": 2, "lo": -1.0, "hi": 1.0}},
                    "materials": {"ring": shows_attribute("OBJECT", "mk_amp"),
                                  "plain": shows_attribute("GEOMETRY", "col")},
                    "meshes": meshes}});
        std::fs::write(dir.join("bake.json"), doc.to_string()).unwrap();
        let floats: [f32; 14] = [0.5, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 1.0, 0.75, 0.75, 0.75, 1.0, 2.0, 3.0];
        std::fs::write(dir.join("bake.bin"), bytemuck::cast_slice(&floats)).unwrap();
        std::fs::write(dir.join("lit.bin"), b"").unwrap();
        dir
    }

    #[test]
    fn object_attributes_fill_each_draws_own_parameter_block() {
        let dir = bake_dir("oattr", false);
        let bake = Bake::open(&dir).unwrap();
        let mats = MaterialSet::compile(&bake).unwrap();
        let (ring, plain) = (mats.index("ring").unwrap(), mats.index("plain").unwrap());
        assert!(mats.list[ring].unsupported.is_empty(), "{:?}", mats.list[ring].unsupported);
        let slots: Vec<(u32, String)> = mats.list[ring].object_slots().collect();
        assert_eq!(slots.iter().map(|s| s.1.as_str()).collect::<Vec<_>>(), ["OBJECT:mk_amp"]);
        let strength = mats.list[ring].params.iter().find(|p| matches!(p.source, AnimSource::Material(_))).unwrap().slot;

        let mut layout = ParamLayout::new(&mats);
        let (a, b) = (layout.draw_base(&mats, ring, 0), layout.draw_base(&mats, ring, 1));
        assert!(a >= mats.param_count && b >= mats.param_count && a != b, "{a} {b}");
        assert_eq!(layout.draw_base(&mats, plain, 3), mats.list[plain].param_base);     // reads nothing per object
        let own = mats.list[ring].param_base;
        for (k, a_amp, strength_k) in [(0usize, 0.5f32, 2.0f32), (1, 0.25, 3.0)] {
            let v = layout.values(&mats, &bake, k).unwrap();
            assert_eq!(v.len(), layout.count as usize);
            assert_eq!(v[(a + slots[0].0) as usize], [a_amp, a_amp, a_amp, 1.0], "frame {k}");
            assert_eq!(v[(b + slots[0].0) as usize], [0.75, 0.75, 0.75, 1.0], "frame {k}");
            for base in [a, b, own] {
                assert_eq!(v[(base + strength) as usize], [strength_k; 4], "the keyed strength, frame {k}");
            }
            assert_eq!(v[(own + slots[0].0) as usize], [0.0; 4], "the material's own block holds no object's value");
        }

        let support = Support::of(&bake).unwrap();
        assert!(support.materials.is_empty() && support.meshes.is_empty(), "{:?} {:?}", support.materials, support.meshes);
        std::fs::remove_dir_all(&dir).unwrap();

        // a bake that lacks the value for one of its meshes (made before object_attrs): the material is not drawn lit
        let dir = bake_dir("oattr_lacking", true);
        let bake = Bake::open(&dir).unwrap();
        let mats = MaterialSet::compile(&bake).unwrap();
        let mut layout = ParamLayout::new(&mats);
        let ring = mats.index("ring").unwrap();
        let c = layout.draw_base(&mats, ring, 3);
        let slot = mats.list[ring].object_slots().next().unwrap().0;
        assert_eq!(layout.values(&mats, &bake, 1).unwrap()[(c + slot) as usize], [0.0; 4]);
        let support = Support::of(&bake).unwrap();
        assert!(support.materials.get("ring").is_some_and(|why| why.iter().any(|w| w.contains("OBJECT:mk_amp"))),
                "{:?}", support.materials);
        assert!(support.meshes.is_empty(), "{:?}", support.meshes);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Keyed or driven values the bake holds one value of (bake.json `keyed`): a colour ramp's stops and a node's
    /// setting on a material with nodes, the colour of a material without nodes, and the colour of a world without nodes
    /// make those materials (and, for the world, the whole scene) Blender's, saying which paths; a driven socket value
    /// (its per-frame values in bake.bin: `anim`) and a plain material do not.
    #[test]
    fn keyed_values_the_bake_holds_once_are_left_to_blender() {
        let dir = std::env::temp_dir().join(format!("mkr_materials_keyed_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mesh = |name: &str, mat: &str| json!({"name": name, "kind": "static", "materials": [mat],
            "triangles": {"offset": 0, "count": 0},
            "attributes": {"position": {"offset": 0, "comp": 3, "type": "f32"},
                           "normal": {"offset": 0, "comp": 3, "type": "f32"}}});
        let mut ramp = shows_attribute("GEOMETRY", "col");
        ramp["keyed"] = json!([r#"nodes["ramp"].color_ramp.elements[0].color"#, r#"nodes["mix"].blend_type"#]);
        let doc = json!({"version": 5, "fps": 30.0, "frame0": 1, "frames": 2, "outputs": [{"name": "o", "size": [4, 4]}],
            "lit": {"view": {"lut": {"file": "view_lut.bin", "n": 2, "lo": -1.0, "hi": 1.0}},
                    "world": {"name": "w", "use_nodes": false, "color": [0.2, 0.2, 0.25], "props": {},
                              "keyed": ["color"]},
                    "materials": {"ramp": ramp, "driven": shows_attribute("GEOMETRY", "col"),
                                  "flat": {"settings": {"diffuse_color": [1.0, 0.2, 0.2, 1.0]}, "tree": null,
                                           "anim": [], "keyed": ["diffuse_color"]}},
                    "meshes": [mesh("a", "ramp"), mesh("b", "driven"), mesh("c", "flat")]}});
        std::fs::write(dir.join("bake.json"), doc.to_string()).unwrap();
        let floats: [f32; 14] = [0.0; 14];
        std::fs::write(dir.join("bake.bin"), bytemuck::cast_slice(&floats)).unwrap();
        std::fs::write(dir.join("lit.bin"), b"").unwrap();
        let bake = Bake::open(&dir).unwrap();
        let support = Support::of(&bake).unwrap();
        let why = |m: &str| support.materials.get(m).map(|w| w.join("; ")).unwrap_or_default();
        assert!(why("ramp").contains("values keyed or driven that the bake holds one of")
                && why("ramp").contains("color_ramp.elements[0].color") && why("ramp").contains("blend_type"),
                "{:?}", support.materials);
        assert!(why("flat").ends_with("diffuse_color"), "{:?}", support.materials);
        assert!(!support.materials.contains_key("driven"), "a driven socket is mkr's: {:?}", support.materials);
        let scene = support.scene.clone().unwrap_or_default();
        assert!(scene.starts_with("world w:") && scene.ends_with("color"), "{scene:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

