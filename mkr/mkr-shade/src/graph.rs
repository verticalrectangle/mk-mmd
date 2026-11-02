//! The bake's node graphs (bake v5 GRAPH, `local design: Bake v5 contract`) and their flattening: groups expanded,
//! reroutes, group sockets and muted nodes turned into pass-through nodes, links resolved to node indices.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use serde_json::Value;

/// One socket of a node as the bake records it.
#[derive(Clone, Debug, Deserialize)]
pub struct SocketDesc {
    pub identifier: String,
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type", default)]
    pub ty: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default)]
    pub linked: bool,
    #[serde(default)]
    pub value: Value,
}

fn yes() -> bool {
    true
}

/// One node of a GRAPH.
#[derive(Clone, Debug, Deserialize)]
pub struct NodeDesc {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub bl_idname: String,
    #[serde(default)]
    pub mute: bool,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub props: serde_json::Map<String, Value>,
    #[serde(default)]
    pub ramp: Option<Value>,
    #[serde(default)]
    pub curves: Option<Value>,
    #[serde(default)]
    pub inputs: Vec<SocketDesc>,
    #[serde(default)]
    pub outputs: Vec<SocketDesc>,
    #[serde(default)]
    pub internal_links: Vec<(String, String)>,
}

/// A node tree: nodes and valid, unmuted links `[from node, from socket identifier, to node, to socket identifier]`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Graph {
    #[serde(default)]
    pub nodes: Vec<NodeDesc>,
    #[serde(default)]
    pub links: Vec<(String, String, String, String)>,
}

/// An animated socket: its per-frame values live in bake.bin (`offset`, `size` floats per frame).
#[derive(Clone, Debug, Deserialize)]
pub struct Anim {
    pub node: String,
    pub socket: String,
    #[serde(default = "input_io")]
    pub io: String,
    #[serde(default = "one")]
    pub size: u32,
    #[serde(default)]
    pub offset: u64,
    #[serde(default = "one")]
    pub frames: u32,
}

fn input_io() -> String {
    "input".into()
}

fn one() -> u32 {
    1
}

/// A material of bake.json `lit.materials`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct MaterialDesc {
    #[serde(default)]
    pub settings: serde_json::Map<String, Value>,
    #[serde(default)]
    pub tree: Option<Graph>,
    #[serde(default)]
    pub anim: Vec<Anim>,
    /// What is keyed or driven on it that the bake holds one value of (bake_lit `_keyed`): mkr cannot follow it.
    #[serde(default)]
    pub keyed: Vec<String>,
}

/// A node group of bake.json `lit.groups`. Interface rows: [in_out, name, identifier, socket_type, default].
#[derive(Clone, Debug, Default, Deserialize)]
pub struct GroupDesc {
    #[serde(default)]
    pub interface: Vec<Vec<Value>>,
    #[serde(default)]
    pub tree: Graph,
    #[serde(default)]
    pub anim: Vec<Anim>,
    #[serde(default)]
    pub keyed: Vec<String>,
}

/// The world of bake.json `lit.world`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct WorldDesc {
    #[serde(default)]
    pub tree: Option<Graph>,
    #[serde(default)]
    pub anim: Vec<Anim>,
    #[serde(default)]
    pub settings: serde_json::Map<String, Value>,
    #[serde(default)]
    pub keyed: Vec<String>,
}

/// Value types the generated code carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Float,
    Vec3,
    Color,
    Shader,
}

impl Ty {
    pub fn of_socket(t: &str) -> Ty {
        match t {
            "VECTOR" | "ROTATION" => Ty::Vec3,
            "RGBA" => Ty::Color,
            "SHADER" => Ty::Shader,
            _ => Ty::Float,
        }
    }

    pub fn wgsl(self) -> &'static str {
        match self {
            Ty::Float => "f32",
            Ty::Vec3 => "vec3f",
            Ty::Color => "vec4f",
            Ty::Shader => "Closures",
        }
    }
}

/// A node-group interface socket type ("NodeSocketColor", ...) as the socket `type` a node reports ("RGBA", ...).
pub fn socket_type_of_interface(t: &str) -> &'static str {
    match t {
        "NodeSocketColor" => "RGBA",
        "NodeSocketShader" => "SHADER",
        "NodeSocketInt" | "NodeSocketIntFactor" | "NodeSocketIntPercentage" | "NodeSocketIntUnsigned" => "INT",
        "NodeSocketBool" => "BOOLEAN",
        "NodeSocketRotation" => "ROTATION",
        t if t.starts_with("NodeSocketVector") => "VECTOR",
        _ => "VALUE",
    }
}

/// Where an animated socket's value comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnimSource {
    /// Index into the material's (or world's) `anim`.
    Material(usize),
    /// (group name, index into that group's `anim`).
    Group(String, usize),
    /// An Attribute node of type `kind` (OBJECT, INSTANCER or VIEW_LAYER) reading attribute `name`: a value per
    /// object (the bake's `lit.meshes[].object_attrs`, RGBA as Blender converts the property), so a renderer fills
    /// the slot per draw, not per material. The slot is on every output of the node.
    Object { kind: String, name: String },
}

#[derive(Clone, Debug)]
pub struct FSocket {
    pub identifier: String,
    pub name: String,
    pub ty: Ty,
    pub raw_ty: String,
    pub enabled: bool,
    pub value: Value,
    /// Inputs: (source node, source output index).
    pub link: Option<(usize, usize)>,
    /// Animated value: index into `Flat::anims`.
    pub anim: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pass {
    /// A real node, evaluated by its type.
    No,
    /// Output i passes input `map[i]` through (None: the output's own default value).
    Map(Vec<Option<usize>>),
}

#[derive(Clone, Debug)]
pub struct FNode {
    /// "group node/inner node" path, for messages.
    pub path: String,
    pub kind: String,
    pub bl_idname: String,
    pub props: serde_json::Map<String, Value>,
    pub ramp: Option<Value>,
    pub curves: Option<Value>,
    pub image: Option<String>,
    pub inputs: Vec<FSocket>,
    pub outputs: Vec<FSocket>,
    pub pass: Pass,
}

impl FNode {
    pub fn input_index(&self, key: &str) -> Option<usize> {
        self.inputs
            .iter()
            .position(|s| s.identifier == key)
            .or_else(|| self.inputs.iter().position(|s| s.name == key && s.enabled))
            .or_else(|| self.inputs.iter().position(|s| s.name == key))
    }

    pub fn output_index(&self, key: &str) -> Option<usize> {
        self.outputs
            .iter()
            .position(|s| s.identifier == key)
            .or_else(|| self.outputs.iter().position(|s| s.name == key && s.enabled))
            .or_else(|| self.outputs.iter().position(|s| s.name == key))
    }

    pub fn prop_str(&self, key: &str) -> Option<&str> {
        self.props.get(key).and_then(|v| v.as_str())
    }

    pub fn prop_bool(&self, key: &str) -> Option<bool> {
        self.props.get(key).and_then(|v| v.as_bool().or_else(|| v.as_f64().map(|f| f != 0.0)))
    }

    pub fn prop_f64(&self, key: &str) -> Option<f64> {
        self.props.get(key).and_then(|v| v.as_f64().or_else(|| v.as_bool().map(|b| b as i32 as f64)))
    }
}

/// A flattened node tree.
#[derive(Clone, Debug, Default)]
pub struct Flat {
    pub nodes: Vec<FNode>,
    pub anims: Vec<AnimSource>,
    pub warnings: Vec<String>,
    pub unsupported: Vec<String>,
}

/// Group expansion depth limit (a group cannot contain itself; this only guards malformed input).
const MAX_DEPTH: usize = 32;

struct Flattener<'a> {
    groups: &'a BTreeMap<String, GroupDesc>,
    flat: Flat,
}


impl<'a> Flattener<'a> {
    /// Flatten `graph` (a material/world tree when `group` is None). `inputs_of_group`: for a group instance, the
    /// pass node standing for its Group Input nodes. Returns, per node name, the flat node index (group instances map
    /// to their Group Output pass node) and the output-socket redirects of group instances.
    fn tree(
        &mut self,
        graph: &Graph,
        prefix: &str,
        anims: &[Anim],
        anim_source: &dyn Fn(usize) -> AnimSource,
        group_in: Option<usize>,
        depth: usize,
    ) -> Option<usize> {
        // node name -> flat index of the node (group instances: their output pass node)
        let mut index: HashMap<&str, usize> = HashMap::new();
        // node name -> flat index of the input pass node (group instances only)
        let mut inst_in: HashMap<&str, usize> = HashMap::new();
        let mut group_out: Option<(usize, bool)> = None;

        for node in &graph.nodes {
            let path = format!("{prefix}{}", node.name);
            match node.kind.as_str() {
                "FRAME" => continue,
                "GROUP_INPUT" => {
                    if let Some(gi) = group_in {
                        index.insert(node.name.as_str(), gi);
                    } else {
                        // A Group Input at the top level has nothing to read: its outputs are zeros.
                        let id = self.push(node, &path, Pass::Map(vec![None; node.outputs.len()]));
                        index.insert(node.name.as_str(), id);
                    }
                    continue;
                }
                "GROUP_OUTPUT" => {
                    let active = node.props.get("is_active_output").and_then(|v| v.as_bool()).unwrap_or(false);
                    // Outputs mirror the inputs (minus the virtual extend socket).
                    let mut n = node.clone();
                    n.outputs = n
                        .inputs
                        .iter()
                        .filter(|s| s.identifier != "__extend__")
                        .cloned()
                        .collect();
                    n.inputs.retain(|s| s.identifier != "__extend__");
                    let map = (0..n.outputs.len()).map(Some).collect();
                    let id = self.push(&n, &path, Pass::Map(map));
                    index.insert(node.name.as_str(), id);
                    match group_out {
                        Some((_, true)) => {}
                        Some((_, false)) if !active => {}
                        _ => group_out = Some((id, active)),
                    }
                    continue;
                }
                "REROUTE" => {
                    let map = if node.inputs.is_empty() { vec![None] } else { vec![Some(0)] };
                    let id = self.push(node, &path, Pass::Map(map));
                    index.insert(node.name.as_str(), id);
                    continue;
                }
                _ => {}
            }
            if node.mute {
                let mut map: Vec<Option<usize>> = vec![None; node.outputs.len()];
                for (fi, fo) in &node.internal_links {
                    let i = node.inputs.iter().position(|s| &s.identifier == fi);
                    let o = node.outputs.iter().position(|s| &s.identifier == fo);
                    if let (Some(i), Some(o)) = (i, o) {
                        map[o] = Some(i);
                    }
                }
                let id = self.push(node, &path, Pass::Map(map));
                index.insert(node.name.as_str(), id);
                continue;
            }
            if node.kind == "GROUP" {
                let gname = node.group.clone().unwrap_or_default();
                let Some(group) = self.groups.get(&gname).filter(|_| depth < MAX_DEPTH) else {
                    self.flat.unsupported.push(format!("GROUP {gname:?} (missing node group)"));
                    let id = self.push(node, &path, Pass::Map(vec![None; node.outputs.len()]));
                    index.insert(node.name.as_str(), id);
                    continue;
                };
                // The instance's inputs, as a pass node whose outputs the group's Group Input nodes are.
                let mut gin = node.clone();
                gin.outputs = node.inputs.iter().filter(|s| s.identifier != "__extend__").cloned().collect();
                for o in gin.outputs.iter_mut() {
                    // The group's own interface type wins (it is the type inside the group).
                    if let Some(row) = group.interface.iter().find(|r| {
                        r.first().and_then(|v| v.as_str()) == Some("INPUT")
                            && r.get(2).and_then(|v| v.as_str()) == Some(o.identifier.as_str())
                    }) {
                        if let Some(t) = row.get(3).and_then(|v| v.as_str()) {
                            o.ty = socket_type_of_interface(t).to_string();
                        }
                    }
                }
                gin.inputs.retain(|s| s.identifier != "__extend__");
                let map = (0..gin.outputs.len()).map(Some).collect();
                let gin_id = self.push(&gin, &format!("{path}/[in]"), Pass::Map(map));
                inst_in.insert(node.name.as_str(), gin_id);
                let gname2 = gname.clone();
                let sub_prefix = format!("{path}/");
                let out_id = self.tree(
                    &group.tree,
                    &sub_prefix,
                    &group.anim,
                    &move |i| AnimSource::Group(gname2.clone(), i),
                    Some(gin_id),
                    depth + 1,
                );
                // The instance's outputs come from the group's Group Output (re-keyed by the instance's sockets).
                let out_id = match out_id {
                    Some(o) => {
                        let mut redirect = node.clone();
                        redirect.inputs = node.outputs.clone();
                        let map = (0..node.outputs.len()).map(Some).collect();
                        let rid = self.push(&redirect, &format!("{path}/[out]"), Pass::Map(map));
                        for (k, s) in node.outputs.iter().enumerate() {
                            let src = self.flat.nodes[o].outputs.iter().position(|t| t.identifier == s.identifier);
                            if let Some(si) = src {
                                self.flat.nodes[rid].inputs[k].link = Some((o, si));
                            }
                        }
                        rid
                    }
                    None => self.push(node, &path, Pass::Map(vec![None; node.outputs.len()])),
                };
                index.insert(node.name.as_str(), out_id);
                continue;
            }
            let id = self.push(node, &path, Pass::No);
            if let Some(source) = object_attribute(node) {
                let slot = self.flat.anims.len();
                self.flat.anims.push(source);
                for s in &mut self.flat.nodes[id].outputs {
                    s.anim = Some(slot);
                }
            }
            index.insert(node.name.as_str(), id);
        }

        // Links.
        for (fnode, fsock, tnode, tsock) in &graph.links {
            let (Some(&from), Some(&to_out)) = (index.get(fnode.as_str()), index.get(tnode.as_str())) else {
                self.flat.warnings.push(format!("{prefix}: link {fnode}.{fsock} -> {tnode}.{tsock}: unknown node"));
                continue;
            };
            // A link into a group instance lands on its input pass node.
            let to = inst_in.get(tnode.as_str()).copied().unwrap_or(to_out);
            let Some(fi) = self.flat.nodes[from].output_index(fsock) else {
                self.flat.warnings.push(format!("{prefix}: link from unknown socket {fnode}.{fsock}"));
                continue;
            };
            let Some(ti) = self.flat.nodes[to].inputs.iter().position(|s| &s.identifier == tsock) else {
                // Links into a Group Output's extend socket or a node's hidden socket carry nothing.
                if tsock != "__extend__" {
                    self.flat.warnings.push(format!("{prefix}: link into unknown socket {tnode}.{tsock}"));
                }
                continue;
            };
            self.flat.nodes[to].inputs[ti].link = Some((from, fi));
        }

        // Animated sockets.
        for (k, a) in anims.iter().enumerate() {
            let Some(&id) = index.get(a.node.as_str()) else {
                self.flat.warnings.push(format!("{prefix}: animated socket on unknown node {:?}", a.node));
                continue;
            };
            let target = inst_in.get(a.node.as_str()).copied().filter(|_| a.io == "input").unwrap_or(id);
            let slot = self.flat.anims.len();
            let node = &mut self.flat.nodes[target];
            let found = if a.io == "output" {
                node.outputs.iter_mut().find(|s| s.identifier == a.socket)
            } else {
                node.inputs.iter_mut().find(|s| s.identifier == a.socket)
            };
            match found {
                Some(s) => {
                    s.anim = Some(slot);
                    self.flat.anims.push(anim_source(k));
                }
                None => self.flat.warnings.push(format!(
                    "{prefix}: animated socket {}.{} ({}) not found",
                    a.node, a.socket, a.io
                )),
            }
        }
        group_out.map(|(id, _)| id)
    }

    fn push(&mut self, node: &NodeDesc, path: &str, pass: Pass) -> usize {
        let conv = |s: &SocketDesc| FSocket {
            identifier: s.identifier.clone(),
            name: s.name.clone(),
            ty: Ty::of_socket(&s.ty),
            raw_ty: s.ty.clone(),
            enabled: s.enabled,
            value: s.value.clone(),
            link: None,
            anim: None,
        };
        self.flat.nodes.push(FNode {
            path: path.to_string(),
            kind: node.kind.clone(),
            bl_idname: node.bl_idname.clone(),
            props: node.props.clone(),
            ramp: node.ramp.clone(),
            curves: node.curves.clone(),
            image: node.image.clone(),
            inputs: node.inputs.iter().map(conv).collect(),
            outputs: node.outputs.iter().map(conv).collect(),
            pass,
        });
        self.flat.nodes.len() - 1
    }
}

/// The per-object value an Attribute node of type Object, Instancer or View Layer reads, as its parameter source.
fn object_attribute(node: &NodeDesc) -> Option<AnimSource> {
    if node.kind != "ATTRIBUTE" {
        return None;
    }
    let prop = |k: &str| node.props.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let kind = prop("attribute_type");
    (!kind.is_empty() && kind != "GEOMETRY").then(|| AnimSource::Object { kind, name: prop("attribute_name") })
}

/// Flatten a material or world tree with its groups.
pub fn flatten(graph: &Graph, anims: &[Anim], groups: &BTreeMap<String, GroupDesc>) -> Flat {
    let mut f = Flattener { groups, flat: Flat::default() };
    f.tree(graph, "", anims, &AnimSource::Material, None, 0);
    f.flat
}
