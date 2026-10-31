//! A silhouette's soft accents (a lightning bolt: Transparent mixed with Emission): their pass carries, beside their
//! coverage, the opaque share of their surface as styles.add_weight_aov wires it into an AOV in Blender: the factor of
//! every Mix Shader that mixes a Transparent BSDF in (the factor itself when Transparent is its first shader, one minus
//! it when the second). Each soft material is compiled once more by mkr-shade with that AOV Output node added (id
//! `<id>_soft`, against the material's own images and parameters, so its code indexes them as the material's does),
//! and the pass's entry point `fsoft_<id>` writes it: R the weight, A the coverage (looks_gpu.rs).

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use mkr_shade::{GroupDesc, ImageDesc, MaterialDesc};
use serde_json::{Value, json};

use crate::render::Renderer;

const AOV_NAME: &str = "mk_soft";

fn socket(identifier: &str, ty: &str, linked: bool, value: Value) -> Value {
    json!({"identifier": identifier, "name": identifier, "type": ty, "enabled": true, "linked": linked, "value": value})
}

/// A material graph (bake GRAPH) with an AOV Output node per Mix Shader that mixes a Transparent BSDF in, carrying the
/// mix's opaque share (styles.add_weight_aov); the graph as it is when there is none.
pub fn with_weight_aov(tree: &Value) -> Result<Value> {
    let mut tree = tree.clone();
    let nodes = tree.get("nodes").and_then(|n| n.as_array()).cloned().unwrap_or_default();
    let links: Vec<Value> = tree.get("links").and_then(|l| l.as_array()).cloned().unwrap_or_default();
    let kind_of: BTreeMap<String, String> = nodes.iter().filter_map(|n| {
        Some((n.get("name")?.as_str()?.to_string(), n.get("bl_idname").and_then(|b| b.as_str()).unwrap_or("").to_string()))
    }).collect();
    let link_into = |node: &str, sock: &str| -> Option<(String, String)> {
        links.iter().find_map(|l| {
            let s = |i: usize| l.get(i).and_then(|x| x.as_str());
            (s(2) == Some(node) && s(3) == Some(sock)).then(|| (s(0).unwrap_or("").to_string(), s(1).unwrap_or("").to_string()))
        })
    };
    let mut new_nodes = Vec::new();
    let mut new_links = Vec::new();
    for n in &nodes {
        if n.get("bl_idname").and_then(|b| b.as_str()) != Some("ShaderNodeMixShader") || n.get("mute").and_then(|m| m.as_bool()) == Some(true) {
            continue;
        }
        let name = n.get("name").and_then(|x| x.as_str()).context("a node without a name")?;
        let inputs = n.get("inputs").and_then(|i| i.as_array()).context("a Mix Shader without inputs")?;
        let ident = |i: usize| inputs.get(i).and_then(|s| s.get("identifier")).and_then(|x| x.as_str()).map(str::to_string);
        let (Some(fac_id), Some(s1), Some(s2)) = (ident(0), ident(1), ident(2)) else { continue };
        let fac_value = inputs[0].get("value").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let fac_src = link_into(name, &fac_id);
        for (slot, sid) in [(1, &s1), (2, &s2)] {
            let Some((from, _)) = link_into(name, sid) else { continue };
            if kind_of.get(&from).map(String::as_str) != Some("ShaderNodeBsdfTransparent") {
                continue;
            }
            let aov = format!("mk_soft_aov_{}_{slot}", new_nodes.len());
            let (value, linked) = if slot == 1 {
                match &fac_src {
                    Some((src, out)) => {
                        new_links.push(json!([src, out, aov, "Value"]));
                        (0.0, true)
                    }
                    None => (fac_value, false),
                }
            } else {
                let sub = format!("{aov}_sub");
                let (b, b_linked) = match &fac_src {
                    Some((src, out)) => {
                        new_links.push(json!([src, out, sub, "Value_001"]));
                        (0.0, true)
                    }
                    None => (fac_value, false),
                };
                new_nodes.push(json!({
                    "name": sub, "type": "MATH", "bl_idname": "ShaderNodeMath", "mute": false, "group": null, "image": null,
                    "props": {"operation": "SUBTRACT", "use_clamp": false}, "ramp": null, "curves": null,
                    "inputs": [socket("Value", "VALUE", false, json!(1.0)), socket("Value_001", "VALUE", b_linked, json!(b)),
                               socket("Value_002", "VALUE", false, json!(0.0))],
                    "outputs": [socket("Value", "VALUE", true, Value::Null)], "internal_links": []
                }));
                new_links.push(json!([sub, "Value", aov, "Value"]));
                (0.0, true)
            };
            new_nodes.push(json!({
                "name": aov, "type": "OUTPUT_AOV", "bl_idname": "ShaderNodeOutputAOV", "mute": false, "group": null,
                "image": null, "props": {"aov_name": AOV_NAME}, "ramp": null, "curves": null,
                "inputs": [socket("Color", "RGBA", false, json!([0.0, 0.0, 0.0, 1.0])), socket("Value", "VALUE", linked, json!(value))],
                "outputs": [], "internal_links": []
            }));
        }
    }
    if let Some(arr) = tree.get_mut("nodes").and_then(|n| n.as_array_mut()) {
        arr.extend(new_nodes);
    }
    if let Some(arr) = tree.get_mut("links").and_then(|l| l.as_array_mut()) {
        arr.extend(new_links);
    }
    Ok(tree)
}

/// The soft accents' code: per material flagged in `soft` (MaterialSet indices) its weighted variant and its pass's
/// entry point `fsoft_<id>`.
pub(crate) fn soft_entries(r: &Renderer<'_>, soft: &[bool]) -> Result<String> {
    if !soft.iter().any(|s| *s) {
        return Ok(String::new());
    }
    let raw_lit = r.bake.raw.get("lit").context("bake.json has no lit")?;
    let groups: BTreeMap<String, GroupDesc> = match raw_lit.get("groups") {
        Some(g) if !g.is_null() => serde_json::from_value(g.clone()).context("lit.groups")?,
        _ => BTreeMap::new(),
    };
    let all_images = mkr_shade::images_from_lit(raw_lit);
    let args = "in: VsOut, @builtin(front_facing) front: bool, @builtin(primitive_index) prim: u32";
    let mut src = String::new();
    for (i, m) in r.mats.list.iter().enumerate() {
        if !soft[i] || m.blended {
            continue;
        }
        let rec = raw_lit.get("materials").and_then(|ms| ms.get(&m.name)).with_context(|| format!("material {}", m.name))?;
        let mut rec = rec.clone();
        if let Some(tree) = rec.get("tree").filter(|t| !t.is_null()) {
            let weighted = with_weight_aov(tree)?;
            rec["tree"] = weighted;
        }
        let desc: MaterialDesc = serde_json::from_value(rec).with_context(|| format!("material {}", m.name))?;
        let own: Vec<ImageDesc> = m.image_ids.iter().map(|&k| all_images[k].clone()).collect();
        let id = format!("{}_soft", m.id);
        let c = mkr_shade::compile(&id, &desc, &groups, &own, 0).map_err(|e| anyhow::anyhow!("material {}: {e}", m.name))?;
        if c.images.iter().enumerate().any(|(k, &x)| k != x) || c.images.len() > own.len() {
            bail!("material {}: its weighted variant reads images the material does not", m.name);
        }
        if c.params.len() != m.params.len() {
            bail!("material {}: its weighted variant animates other sockets than the material", m.name);
        }
        src.push_str(&c.wgsl);
        src.push('\n');
        let weight = match c.aovs.iter().rev().find(|a| a.name == AOV_NAME) {
            Some(a) => format!("{}(s).a", a.function),
            None => "0.0".to_string(),
        };
        src.push_str(&format!(
            "\n@fragment\nfn fsoft_{}({args}) -> @location(0) vec4<f32> {{\n    let s = fragment_surface(in, front, prim);\n    return vec4<f32>(clamp({weight}, 0.0, 1.0), 0.0, 0.0, 1.0);\n}}\n",
            m.id
        ));
    }
    Ok(src)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix_tree(transparent_first: bool, fac_linked: bool) -> Value {
        let (a, b) = if transparent_first { ("Transparent BSDF", "Emission") } else { ("Emission", "Transparent BSDF") };
        let mut links = vec![json!([a, "BSDF", "Mix Shader", "Shader"]), json!([b, "BSDF", "Mix Shader", "Shader_001"]),
                             json!(["Mix Shader", "Shader", "Material Output", "Surface"])];
        if fac_linked {
            links.push(json!(["Value", "Value", "Mix Shader", "Fac"]));
        }
        json!({"nodes": [
            {"name": "Transparent BSDF", "type": "BSDF_TRANSPARENT", "bl_idname": "ShaderNodeBsdfTransparent", "inputs": [], "outputs": []},
            {"name": "Emission", "type": "EMISSION", "bl_idname": "ShaderNodeEmission", "inputs": [], "outputs": []},
            {"name": "Value", "type": "VALUE", "bl_idname": "ShaderNodeValue", "inputs": [], "outputs": []},
            {"name": "Mix Shader", "type": "MIX_SHADER", "bl_idname": "ShaderNodeMixShader",
             "inputs": [{"identifier": "Fac", "value": 0.3}, {"identifier": "Shader"}, {"identifier": "Shader_001"}], "outputs": []},
            {"name": "Material Output", "type": "OUTPUT_MATERIAL", "bl_idname": "ShaderNodeOutputMaterial", "inputs": [], "outputs": []}],
            "links": links})
    }

    #[test]
    fn transparent_first_weighs_by_the_factor_itself() {
        let t = with_weight_aov(&mix_tree(true, true)).unwrap();
        let aov = t["nodes"].as_array().unwrap().iter().find(|n| n["type"] == "OUTPUT_AOV").unwrap();
        assert_eq!(aov["props"]["aov_name"], AOV_NAME);
        let name = aov["name"].as_str().unwrap();
        assert!(t["links"].as_array().unwrap().iter().any(|l| l[0] == "Value" && l[2] == name && l[3] == "Value"));
    }

    #[test]
    fn transparent_second_weighs_by_one_minus_the_factor() {
        let t = with_weight_aov(&mix_tree(false, false)).unwrap();
        let nodes = t["nodes"].as_array().unwrap();
        let sub = nodes.iter().find(|n| n["type"] == "MATH").unwrap();
        assert_eq!(sub["props"]["operation"], "SUBTRACT");
        assert_eq!(sub["inputs"][0]["value"], 1.0);
        assert_eq!(sub["inputs"][1]["value"], 0.3);
        assert!(nodes.iter().any(|n| n["type"] == "OUTPUT_AOV"));
    }

    #[test]
    fn a_tree_without_a_transparent_mix_is_left_as_it_is() {
        let tree = json!({"nodes": [{"name": "Emission", "type": "EMISSION", "bl_idname": "ShaderNodeEmission"}], "links": []});
        assert_eq!(with_weight_aov(&tree).unwrap(), tree);
    }
}
