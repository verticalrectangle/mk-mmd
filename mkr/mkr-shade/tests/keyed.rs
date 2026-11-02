//! What mkr-shade reports it cannot follow over the frames: values keyed or driven that the bake holds one value of
//! (bake.json `keyed`, bake_lit `_keyed`): a colour ramp's stops, a node's setting, a material's own colour when it draws
//! without nodes, the same on a node group a material uses (followed down), and on the world.

use std::collections::BTreeMap;

use mkr_shade::{GroupDesc, MaterialDesc, WorldDesc};
use serde_json::{Value, json};

const REASON: &str = "values keyed or driven that the bake holds one of";

fn sock(id: &str, ty: &str, value: Value) -> Value {
    json!({"identifier": id, "name": id, "type": ty, "enabled": true, "linked": false, "value": value})
}

/// An Emission material with a group node of `group` beside it (when given), `keyed` its own keyed paths.
fn emitting(keyed: &[&str], group: Option<&str>) -> MaterialDesc {
    let mut nodes = vec![
        json!({"name": "em", "type": "EMISSION", "props": {}, "inputs": [sock("Color", "RGBA", json!([1, 0.5, 0.2, 1])),
               sock("Strength", "VALUE", json!(1.0))], "outputs": [sock("Emission", "SHADER", Value::Null)]}),
        json!({"name": "out", "type": "OUTPUT_MATERIAL", "props": {"is_active_output": true, "target": "ALL"},
               "inputs": [sock("Surface", "SHADER", Value::Null)], "outputs": []}),
    ];
    if let Some(g) = group {
        nodes.push(json!({"name": "grp", "type": "GROUP", "group": g, "props": {}, "inputs": [], "outputs": []}));
    }
    serde_json::from_value(json!({"settings": {}, "tree": {"nodes": nodes, "links": [["em", "Emission", "out", "Surface"]]},
                                  "anim": [], "keyed": keyed}))
        .unwrap()
}

fn group(keyed: &[&str], inner: Option<&str>) -> GroupDesc {
    let nodes: Vec<Value> = inner
        .map(|g| vec![json!({"name": "in", "type": "GROUP", "group": g, "props": {}, "inputs": [], "outputs": []})])
        .unwrap_or_default();
    serde_json::from_value(json!({"interface": [], "tree": {"nodes": nodes, "links": []}, "anim": [], "keyed": keyed}))
        .unwrap()
}

fn reasons(m: &mkr_shade::CompiledMaterial) -> Vec<&String> {
    m.unsupported.iter().filter(|u| u.contains(REASON)).collect()
}

#[test]
fn a_keyed_ramp_or_node_setting_is_reported_with_its_paths() {
    let ramp = r#"nodes["ramp"].color_ramp.elements[0].color"#;
    let setting = r#"nodes["mix"].blend_type"#;
    let c = mkr_shade::compile("k", &emitting(&[ramp, setting], None), &BTreeMap::new(), &[], 0).unwrap();
    let why = reasons(&c);
    assert_eq!(why.len(), 1, "{:?}", c.unsupported);
    assert!(why[0].contains(ramp) && why[0].contains(setting), "{}", why[0]);

    let plain = mkr_shade::compile("p", &emitting(&[], None), &BTreeMap::new(), &[], 0).unwrap();
    assert!(reasons(&plain).is_empty(), "{:?}", plain.unsupported);
}

#[test]
fn a_material_without_nodes_reports_its_keyed_colour() {
    let flat: MaterialDesc = serde_json::from_value(json!({"settings": {"diffuse_color": [1.0, 0.2, 0.2, 1.0]},
                                                         "tree": null, "anim": [], "keyed": ["diffuse_color"]}))
        .unwrap();
    let c = mkr_shade::compile("f", &flat, &BTreeMap::new(), &[], 0).unwrap();
    let why = reasons(&c);
    assert!(why.len() == 1 && why[0].ends_with("diffuse_color"), "{:?}", c.unsupported);
}

#[test]
fn a_keyed_node_group_is_reported_for_every_material_that_uses_it_however_deep() {
    let groups = BTreeMap::from([
        ("outer".to_string(), group(&[], Some("inner"))),
        ("inner".to_string(), group(&[r#"nodes["ramp"].color_ramp.elements[1].color"#], None)),
    ]);
    let c = mkr_shade::compile("g", &emitting(&[], Some("outer")), &groups, &[], 0).unwrap();
    let why = reasons(&c);
    assert!(why.len() == 1 && why[0].contains(r#"group inner: nodes["ramp"].color_ramp.elements[1].color"#),
            "{:?}", c.unsupported);
    assert_eq!(mkr_shade::keyed_reason(&[], None, &groups), None);
}

#[test]
fn a_keyed_world_is_reported() {
    let world: WorldDesc = serde_json::from_value(json!({"tree": {"nodes": [
        {"name": "bg", "type": "BACKGROUND", "props": {}, "inputs": [sock("Color", "RGBA", json!([0.1, 0.1, 0.2, 1])),
                                                                    sock("Strength", "VALUE", json!(1.0))],
         "outputs": [sock("Background", "SHADER", Value::Null)]},
        {"name": "out", "type": "OUTPUT_WORLD", "props": {"is_active_output": true, "target": "ALL"},
         "inputs": [sock("Surface", "SHADER", Value::Null)], "outputs": []}],
        "links": [["bg", "Background", "out", "Surface"]]}, "anim": [],
        "keyed": [r#"nodes["sky"].color_ramp.elements[0].color"#]}))
        .unwrap();
    let c = mkr_shade::compile_world("world", &world, &BTreeMap::new(), &[], 0).unwrap();
    assert_eq!(reasons(&c).len(), 1, "{:?}", c.unsupported);
}
