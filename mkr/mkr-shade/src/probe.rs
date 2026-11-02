//! Reader for the scene probe's dump (`dump_scene.py`: materials with trees whose sockets are keyed by identifier,
//! groups with interfaces) into bake GRAPH records, so the compiler can be checked against a scene without a bake.
//! The probe keeps enabled sockets only and a subset of node properties; Blender's defaults fill the rest.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::graph::{Graph, GroupDesc, MaterialDesc, NodeDesc, SocketDesc};
use crate::ImageDesc;

/// A probe scene as compiler inputs.
pub struct ProbeScene {
    pub materials: BTreeMap<String, MaterialDesc>,
    pub groups: BTreeMap<String, GroupDesc>,
    pub images: Vec<ImageDesc>,
}

const META: &[&str] = &["name", "type", "bl_idname", "label", "group", "image", "ramp", "inputs", "outputs"];

fn socket_type_of_value(v: Option<&Value>) -> &'static str {
    match v {
        None | Some(Value::Null) => "SHADER",
        Some(Value::Array(a)) if a.len() >= 4 => "RGBA",
        Some(Value::Array(a)) if a.len() == 3 => "VECTOR",
        Some(Value::Bool(_)) => "BOOLEAN",
        _ => "VALUE",
    }
}

fn output_type(kind: &str, id: &str) -> &'static str {
    match id {
        "BSDF" | "Shader" | "Emission" | "Holdout" | "Background" | "Volume" => "SHADER",
        "Color" | "Result_Color" | "Image" => "RGBA",
        "Vector" | "Normal" | "UV" | "Generated" | "Object" | "Camera" | "Window" | "Reflection" | "Position"
        | "True Normal" | "Incoming" | "Tangent" | "Parametric" | "Result_Vector" | "Location" => "VECTOR",
        "Fac" | "Alpha" | "Value" | "Result_Float" | "Distance" | "W" | "Radius" | "Fresnel" | "Facing" => "VALUE",
        _ if kind == "SEPXYZ" || kind == "SEPARATE_COLOR" => "VALUE",
        _ => "VALUE",
    }
}

struct Ctx<'a> {
    groups_raw: &'a Map<String, Value>,
    images: BTreeMap<String, ImageDesc>,
}

impl Ctx<'_> {
    /// Output identifiers and types of a group, from its Group Output node (in order) and interface.
    fn group_outputs(&self, group: &str) -> Vec<SocketDesc> {
        let Some(g) = self.groups_raw.get(group) else { return Vec::new() };
        let mut out = Vec::new();
        let names: Vec<String> = g["interface"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter(|r| r.get(1).and_then(|v| v.as_str()) == Some("OUTPUT"))
                    .map(|r| r.get(2).and_then(|v| v.as_str()).unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();
        let nodes = g["tree"]["nodes"].as_array().cloned().unwrap_or_default();
        if let Some(go) = nodes.iter().find(|n| n["type"] == "GROUP_OUTPUT") {
            if let Some(ins) = go["inputs"].as_object() {
                for (k, (id, s)) in ins.iter().filter(|(id, _)| id.as_str() != "__extend__").enumerate() {
                    out.push(SocketDesc {
                        identifier: id.clone(),
                        name: names.get(k).cloned().unwrap_or_else(|| id.clone()),
                        ty: socket_type_of_value(s.get("value")).into(),
                        enabled: true,
                        linked: false,
                        value: s.get("value").cloned().unwrap_or(Value::Null),
                    });
                }
            }
        }
        out
    }

    fn tree(&mut self, t: &Value) -> Graph {
        let links: Vec<(String, String, String, String)> = t["links"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|l| {
                        let l = l.as_array()?;
                        Some((
                            l.first()?.as_str()?.to_string(),
                            l.get(1)?.as_str()?.to_string(),
                            l.get(2)?.as_str()?.to_string(),
                            l.get(3)?.as_str()?.to_string(),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut nodes = Vec::new();
        for n in t["nodes"].as_array().cloned().unwrap_or_default() {
            let name = n["name"].as_str().unwrap_or("").to_string();
            let kind = n["type"].as_str().unwrap_or("").to_string();
            let mut props = Map::new();
            for (k, v) in n.as_object().cloned().unwrap_or_default() {
                if !META.contains(&k.as_str()) {
                    props.insert(k, v);
                }
            }
            let mut image = None;
            if let Some(im) = n.get("image").filter(|v| v.is_object()) {
                let iname = im["name"].as_str().unwrap_or("").to_string();
                let alpha_mode = im["alpha_mode"].as_str().unwrap_or("STRAIGHT").to_string();
                self.images.entry(iname.clone()).or_insert_with(|| ImageDesc {
                    name: iname.clone(),
                    colorspace: im["colorspace"].as_str().unwrap_or("sRGB").to_string(),
                    premultiplied: alpha_mode == "PREMUL",
                    alpha_mode,
                    missing: false,
                });
                image = Some(iname);
            }
            let inputs: Vec<SocketDesc> = n["inputs"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .map(|(id, s)| SocketDesc {
                            identifier: id.clone(),
                            name: id.clone(),
                            ty: socket_type_of_value(s.get("value")).into(),
                            enabled: true,
                            linked: s["linked"].as_bool().unwrap_or(false),
                            value: s.get("value").cloned().unwrap_or(Value::Null),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let group = n["group"].as_str().map(|s| s.to_string());
            let mut outputs: Vec<SocketDesc> = if kind == "GROUP" {
                group.as_deref().map(|g| self.group_outputs(g)).unwrap_or_default()
            } else {
                n["outputs"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| SocketDesc {
                                identifier: s.to_string(),
                                name: s.to_string(),
                                ty: output_type(&kind, s).into(),
                                enabled: true,
                                linked: false,
                                value: Value::Null,
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };
            // Outputs whose identifier differs from their name show up only in links.
            for (f, fs, _, _) in &links {
                if f == &name && !outputs.iter().any(|o| &o.identifier == fs) {
                    if let Some(o) = outputs.iter_mut().find(|o| fs.starts_with(&o.name) && o.identifier == o.name) {
                        o.identifier = fs.clone();
                        o.ty = output_type(&kind, fs).into();
                    } else {
                        outputs.push(SocketDesc {
                            identifier: fs.clone(),
                            name: fs.clone(),
                            ty: output_type(&kind, fs).into(),
                            enabled: true,
                            linked: true,
                            value: Value::Null,
                        });
                    }
                }
            }
            nodes.push(NodeDesc {
                name,
                kind,
                bl_idname: n["bl_idname"].as_str().unwrap_or("").to_string(),
                mute: false,
                group,
                image,
                props,
                ramp: n.get("ramp").filter(|v| !v.is_null()).cloned(),
                curves: None,
                inputs,
                outputs,
                internal_links: Vec::new(),
            });
        }
        Graph { nodes, links }
    }
}

/// Read a probe dump (the JSON `dump_scene.py` writes).
pub fn load(doc: &Value) -> ProbeScene {
    let empty = Map::new();
    let groups_raw = doc["groups"].as_object().unwrap_or(&empty);
    let mut ctx = Ctx { groups_raw, images: BTreeMap::new() };
    let mut groups = BTreeMap::new();
    for (name, g) in groups_raw {
        let interface = g["interface"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .map(|r| {
                        let r = r.as_array().cloned().unwrap_or_default();
                        // probe rows: [item_type, in_out, name, socket_type, default]
                        vec![
                            r.get(1).cloned().unwrap_or(Value::Null),
                            r.get(2).cloned().unwrap_or(Value::Null),
                            json!(""),
                            r.get(3).cloned().unwrap_or(Value::Null),
                            r.get(4).cloned().unwrap_or(Value::Null),
                        ]
                    })
                    .collect()
            })
            .unwrap_or_default();
        let tree = ctx.tree(&g["tree"]);
        groups.insert(name.clone(), GroupDesc { interface, tree, anim: Vec::new(), keyed: Vec::new() });
    }
    let mut materials = BTreeMap::new();
    if let Some(mats) = doc["materials"].as_object() {
        for (name, m) in mats {
            let tree = if m["tree"].is_object() { Some(ctx.tree(&m["tree"])) } else { None };
            let settings = m["props"].as_object().cloned().unwrap_or_default();
            materials.insert(name.clone(), MaterialDesc { settings, tree, anim: Vec::new(), keyed: Vec::new() });
        }
    }
    ProbeScene { materials, groups, images: ctx.images.into_values().collect() }
}
