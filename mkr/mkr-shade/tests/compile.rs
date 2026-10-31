//! Consumer-visible behaviour of mkr-shade: graphs compile to WGSL that validates (in a fragment entry point and, for
//! the alpha functions, a compute one), unknown nodes are reported, values match Blender's.

mod common;

use std::collections::BTreeMap;

use common::gpu::{Gpu, Kind, film_error};
use mkr_shade::{GroupDesc, MaterialDesc};
use serde_json::{Value, json};

fn sock(id: &str, ty: &str, value: Value) -> Value {
    json!({"identifier": id, "name": id, "type": ty, "enabled": true, "linked": false, "value": value})
}

/// A material: `nodes` + a Material Output whose Surface comes from (node, socket).
fn material(nodes: Vec<Value>, mut links: Vec<Value>, surface: (&str, &str)) -> MaterialDesc {
    let mut nodes = nodes;
    nodes.push(json!({"name": "out", "type": "OUTPUT_MATERIAL", "props": {"is_active_output": true, "target": "ALL"},
                      "inputs": [sock("Surface", "SHADER", Value::Null), sock("Volume", "SHADER", Value::Null),
                                 sock("Displacement", "VECTOR", json!([0, 0, 0])), sock("Thickness", "VALUE", json!(0.0))],
                      "outputs": []}));
    links.push(json!([surface.0, surface.1, "out", "Surface"]));
    serde_json::from_value(json!({"settings": {}, "tree": {"nodes": nodes, "links": links}, "anim": []})).unwrap()
}

/// An Emission shader showing (node, socket), node `n` given.
fn emitting(n: Value, socket: &str) -> MaterialDesc {
    let name = n["name"].as_str().unwrap().to_string();
    let em = json!({"name": "em", "type": "EMISSION", "inputs": [sock("Color", "RGBA", json!([1, 1, 1, 1])),
                    sock("Strength", "VALUE", json!(1.0))], "outputs": [sock("Emission", "SHADER", Value::Null)]});
    material(vec![n, em], vec![json!([name, socket, "em", "Color"])], ("em", "Emission"))
}

fn node(name: &str, kind: &str, props: Value, inputs: Vec<Value>, outputs: Vec<Value>) -> Value {
    json!({"name": name, "type": kind, "props": props, "inputs": inputs, "outputs": outputs})
}

fn validate_both(m: &mkr_shade::CompiledMaterial) {
    let id = &m.id;
    let module = format!(
        "{}{}{}\n@fragment fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4f {{\n  var s: Surface;\n  \
         s.screen = p.xy;\n  let c = mat_{id}(s);\n  return vec4f(c.diffuse + c.emission, mat_{id}_alpha(s));\n}}\n\
         @compute @workgroup_size(1) fn cs_main() {{\n  var s: Surface;\n  let a = mat_{id}_alpha_lod(s);\n}}\n",
        mkr_shade::PRELUDE,
        mkr_shade::RENDERER_STUBS,
        m.wgsl
    );
    if let Err(e) = mkr_shade::validate(&module) {
        panic!("{}:\n{e}\n{}", m.id, m.wgsl);
    }
}

fn v3(id: &str) -> Value {
    sock(id, "VECTOR", json!([0.3, -0.2, 0.7]))
}
fn f(id: &str, v: f64) -> Value {
    sock(id, "VALUE", json!(v))
}
fn c4(id: &str) -> Value {
    sock(id, "RGBA", json!([0.6, 0.3, 0.2, 1.0]))
}
fn out(id: &str, ty: &str) -> Value {
    sock(id, ty, Value::Null)
}

#[test]
fn every_supported_node_type_compiles_and_validates() {
    let tex_in = || vec![v3("Vector"), f("W", 0.3), f("Scale", 2.0), f("Detail", 2.5), f("Roughness", 0.5),
                         f("Lacunarity", 2.0), f("Offset", 0.5), f("Gain", 1.0), f("Distortion", 0.4),
                         f("Smoothness", 0.5), f("Exponent", 1.5), f("Randomness", 1.0)];
    let ramp_table: Vec<Value> = (0..257).map(|i| json!([i as f64 / 256.0, 0.5, 0.2, 1.0])).collect();
    let samples: Vec<f64> = (0..257).map(|i| (i as f64 / 256.0).powf(0.8)).collect();
    let curve_list: Vec<Value> = ["C", "R", "G", "B"]
        .iter()
        .map(|n| json!({"name": n, "range": [0, 1], "slope_lo": 1.0, "slope_hi": 1.0, "samples": samples}))
        .collect();
    let cases: Vec<(Value, &str)> = vec![
        (node("n", "MATH", json!({"operation": "POWER", "use_clamp": true}), vec![f("Value", 0.5), f("Value_001", 2.0),
              f("Value_002", 0.0)], vec![out("Value", "VALUE")]), "Value"),
        (node("n", "VECT_MATH", json!({"operation": "REFRACT"}), vec![v3("Vector"), v3("Vector_001"), v3("Vector_002"),
              f("Scale", 1.3)], vec![out("Vector", "VECTOR"), out("Value", "VALUE")]), "Vector"),
        (node("n", "MIX_RGB", json!({"blend_type": "SOFT_LIGHT", "use_clamp": true}), vec![f("Fac", 0.4), c4("Color1"),
              c4("Color2")], vec![out("Color", "RGBA")]), "Color"),
        (node("n", "MIX", json!({"data_type": "RGBA", "blend_type": "HUE"}), vec![f("Factor_Float", 0.5), c4("A_Color"),
              c4("B_Color")], vec![out("Result_Color", "RGBA")]), "Result_Color"),
        (node("n", "MAP_RANGE", json!({"interpolation_type": "STEPPED"}), vec![f("Value", 0.4), f("From Min", 0.0),
              f("From Max", 1.0), f("To Min", 0.0), f("To Max", 1.0), f("Steps", 4.0)], vec![out("Result", "VALUE")]),
         "Result"),
        (node("n", "CLAMP", json!({"clamp_type": "RANGE"}), vec![f("Value", 0.4), f("Min", 1.0), f("Max", 0.0)],
              vec![out("Result", "VALUE")]), "Result"),
        (json!({"name": "n", "type": "VALTORGB", "props": {}, "inputs": [f("Fac", 0.4)],
                "outputs": [out("Color", "RGBA"), out("Alpha", "VALUE")],
                "ramp": {"interpolation": "B_SPLINE", "color_mode": "RGB", "stops": [[0.1, [1, 0, 0, 1]],
                         [0.5, [0, 1, 0, 1]], [0.9, [0, 0, 1, 1]]],
                         "table": ramp_table}}),
         "Color"),
        (json!({"name": "n", "type": "CURVE_RGB", "props": {}, "inputs": [f("Fac", 1.0), c4("Color")],
                "outputs": [out("Color", "RGBA")],
                "curves": {"black_level": [0, 0, 0, 0], "white_level": [1, 1, 1, 1], "curves": curve_list}}),
         "Color"),
        (node("n", "SEPARATE_COLOR", json!({"mode": "HSL"}), vec![c4("Color")], vec![out("Red", "VALUE"),
              out("Green", "VALUE"), out("Blue", "VALUE")]), "Green"),
        (node("n", "COMBINE_COLOR", json!({"mode": "HSV"}), vec![f("Red", 0.2), f("Green", 0.5), f("Blue", 0.7)],
              vec![out("Color", "RGBA")]), "Color"),
        (node("n", "MAPPING", json!({"vector_type": "TEXTURE"}), vec![v3("Vector"), v3("Location"), v3("Rotation"),
              v3("Scale")], vec![out("Vector", "VECTOR")]), "Vector"),
        (node("n", "VECT_TRANSFORM", json!({"vector_type": "NORMAL", "convert_from": "OBJECT", "convert_to": "CAMERA"}),
              vec![v3("Vector")], vec![out("Vector", "VECTOR")]), "Vector"),
        (node("n", "VECTOR_ROTATE", json!({"rotation_type": "EULER_XYZ", "invert": true}), vec![v3("Vector"),
              v3("Center"), v3("Axis"), f("Angle", 0.4), v3("Rotation")], vec![out("Vector", "VECTOR")]), "Vector"),
        (node("n", "HUE_SAT", json!({}), vec![f("Hue", 0.3), f("Saturation", 1.2), f("Value", 0.9), f("Fac", 1.0),
              c4("Color")], vec![out("Color", "RGBA")]), "Color"),
        (node("n", "INVERT", json!({}), vec![f("Fac", 1.0), c4("Color")], vec![out("Color", "RGBA")]), "Color"),
        (node("n", "GAMMA", json!({}), vec![c4("Color"), f("Gamma", 2.2)], vec![out("Color", "RGBA")]), "Color"),
        (node("n", "BRIGHTCONTRAST", json!({}), vec![c4("Color"), f("Bright", 0.1), f("Contrast", 0.2)],
              vec![out("Color", "RGBA")]), "Color"),
        (node("n", "RGBTOBW", json!({}), vec![c4("Color")], vec![out("Val", "VALUE")]), "Val"),
        (node("n", "TEX_COORD", json!({}), vec![], vec![out("Generated", "VECTOR"), out("Normal", "VECTOR"),
              out("UV", "VECTOR"), out("Object", "VECTOR"), out("Camera", "VECTOR"), out("Window", "VECTOR"),
              out("Reflection", "VECTOR")]), "Camera"),
        (node("n", "UVMAP", json!({"uv_map": "UVMap"}), vec![], vec![out("UV", "VECTOR")]), "UV"),
        (node("n", "ATTRIBUTE", json!({"attribute_type": "GEOMETRY", "attribute_name": "col"}), vec![],
              vec![out("Color", "RGBA"), out("Vector", "VECTOR"), out("Fac", "VALUE"), out("Alpha", "VALUE")]), "Fac"),
        (node("n", "VERTEX_COLOR", json!({"layer_name": ""}), vec![], vec![out("Color", "RGBA"), out("Alpha", "VALUE")]),
         "Color"),
        (node("n", "NEW_GEOMETRY", json!({}), vec![], vec![out("Position", "VECTOR"), out("Normal", "VECTOR"),
              out("Tangent", "VECTOR"), out("Incoming", "VECTOR"), out("Backfacing", "VALUE")]), "Tangent"),
        (node("n", "LAYER_WEIGHT", json!({}), vec![f("Blend", 0.3), v3("Normal")], vec![out("Fresnel", "VALUE"),
              out("Facing", "VALUE")]), "Facing"),
        (node("n", "FRESNEL", json!({}), vec![f("IOR", 1.45), v3("Normal")], vec![out("Fac", "VALUE")]), "Fac"),
        (node("n", "NORMAL_MAP", json!({"space": "TANGENT", "uv_map": ""}), vec![f("Strength", 1.0), c4("Color")],
              vec![out("Normal", "VECTOR")]), "Normal"),
        (node("n", "TANGENT", json!({"direction_type": "UV_MAP", "uv_map": "UVMap"}), vec![],
              vec![out("Tangent", "VECTOR")]), "Tangent"),
        (node("n", "TEX_NOISE", json!({"noise_dimensions": "4D", "noise_type": "HYBRID_MULTIFRACTAL"}), tex_in(),
              vec![out("Fac", "VALUE"), out("Color", "RGBA")]), "Color"),
        (node("n", "TEX_VORONOI", json!({"voronoi_dimensions": "2D", "feature": "SMOOTH_F1", "distance": "MINKOWSKI"}),
              tex_in(), vec![out("Distance", "VALUE"), out("Color", "RGBA"), out("Position", "VECTOR")]), "Position"),
        (node("n", "TEX_WAVE", json!({"wave_type": "RINGS", "rings_direction": "SPHERICAL", "wave_profile": "TRI"}),
              vec![v3("Vector"), f("Scale", 2.0), f("Distortion", 1.0), f("Detail", 2.0), f("Detail Scale", 1.0),
                   f("Detail Roughness", 0.5), f("Phase Offset", 0.2)], vec![out("Color", "RGBA"), out("Fac", "VALUE")]),
         "Fac"),
        (node("n", "TEX_GRADIENT", json!({"gradient_type": "RADIAL"}), vec![v3("Vector")], vec![out("Fac", "VALUE")]),
         "Fac"),
        (node("n", "TEX_CHECKER", json!({}), vec![v3("Vector"), c4("Color1"), c4("Color2"), f("Scale", 5.0)],
              vec![out("Color", "RGBA"), out("Fac", "VALUE")]), "Color"),
        (node("n", "TEX_BRICK", json!({"offset": 0.5, "offset_frequency": 2, "squash": 1.0, "squash_frequency": 2}),
              vec![v3("Vector"), c4("Color1"), c4("Color2"), c4("Mortar"), f("Scale", 5.0), f("Mortar Size", 0.02),
                   f("Mortar Smooth", 0.1), f("Bias", 0.0), f("Brick Width", 0.5), f("Row Height", 0.25)],
              vec![out("Color", "RGBA"), out("Fac", "VALUE")]), "Color"),
        (node("n", "TEX_MAGIC", json!({"turbulence_depth": 4}), vec![v3("Vector"), f("Scale", 5.0), f("Distortion", 1.0)],
              vec![out("Color", "RGBA"), out("Fac", "VALUE")]), "Color"),
        (node("n", "TEX_WHITE_NOISE", json!({"noise_dimensions": "3D"}), vec![v3("Vector"), f("W", 0.0)],
              vec![out("Value", "VALUE"), out("Color", "RGBA")]), "Color"),
        (node("n", "LIGHT_PATH", json!({}), vec![], vec![out("Is Camera Ray", "VALUE"), out("Ray Length", "VALUE")]),
         "Ray Length"),
        (node("n", "OBJECT_INFO", json!({}), vec![], vec![out("Location", "VECTOR"), out("Random", "VALUE")]), "Random"),
        (node("n", "CAMERA", json!({}), vec![], vec![out("View Vector", "VECTOR"), out("View Z Depth", "VALUE")]),
         "View Vector"),
    ];
    for (n, socket) in cases {
        let kind = n["type"].as_str().unwrap().to_string();
        let m = mkr_shade::compile("t", &emitting(n, socket), &BTreeMap::new(), &[], 0).unwrap();
        assert!(m.unsupported.is_empty(), "{kind}: {:?}", m.unsupported);
        validate_both(&m);
    }
}

#[test]
fn shader_nodes_bump_and_mixing_compile_and_alpha_needs_no_derivatives() {
    let bsdfs = [
        ("BSDF_PRINCIPLED", vec![c4("Base Color"), f("Roughness", 0.4), f("Alpha", 0.5), f("Sheen Weight", 0.3),
                                 f("Coat Weight", 0.2), f("Transmission Weight", 0.1)]),
        ("BSDF_GLOSSY", vec![c4("Color"), f("Roughness", 0.2)]),
        ("BSDF_GLASS", vec![c4("Color"), f("Roughness", 0.1), f("IOR", 1.45)]),
        ("BSDF_TRANSLUCENT", vec![c4("Color")]),
        ("BSDF_TRANSPARENT", vec![c4("Color")]),
        ("HOLDOUT", vec![]),
        ("EEVEE_SPECULAR", vec![c4("Base Color"), c4("Specular"), f("Roughness", 0.3), f("Transparency", 0.2)]),
    ];
    for (kind, inputs) in bsdfs {
        let wave = node("w", "TEX_WAVE", json!({}), vec![v3("Vector"), f("Scale", 3.0)], vec![out("Fac", "VALUE")]);
        let bump = node("b", "BUMP", json!({}), vec![f("Strength", 1.0), f("Distance", 0.1), f("Height", 0.0),
                        v3("Normal")], vec![out("Normal", "VECTOR")]);
        let mut ins = inputs.clone();
        ins.push(v3("Normal"));
        let bsdf = node("s", kind, json!({"distribution": "MULTI_GGX"}), ins, vec![out("BSDF", "SHADER")]);
        let diff = node("d", "BSDF_DIFFUSE", json!({}), vec![c4("Color"), v3("Normal")], vec![out("BSDF", "SHADER")]);
        let mix = node("m", "MIX_SHADER", json!({}), vec![f("Fac", 0.3), out("Shader", "SHADER"),
                       out("Shader_001", "SHADER")], vec![out("Shader", "SHADER")]);
        let links = vec![json!(["w", "Fac", "b", "Height"]), json!(["b", "Normal", "s", "Normal"]),
                         json!(["s", "BSDF", "m", "Shader"]), json!(["d", "BSDF", "m", "Shader_001"])];
        let mat = material(vec![wave, bump, bsdf, diff, mix], links, ("m", "Shader"));
        let m = mkr_shade::compile("s", &mat, &BTreeMap::new(), &[], 0).unwrap();
        assert!(m.unsupported.is_empty(), "{kind}: {:?}", m.unsupported);
        if !matches!(kind, "HOLDOUT" | "BSDF_TRANSPARENT") {
            assert!(m.uses_derivatives, "{kind}: the bump needs screen derivatives");
        }
        let alpha = m.wgsl.split("fn mat_s_alpha").nth(1).unwrap();
        assert!(!alpha.contains("dpdx") && !alpha.contains("dpdy"), "{kind}: derivatives in the alpha functions");
        let transparent = matches!(kind, "BSDF_PRINCIPLED" | "BSDF_TRANSPARENT" | "EEVEE_SPECULAR");
        assert_eq!(m.transparent, transparent, "{kind}: transparent flag");
        validate_both(&m);
    }
}

#[test]
fn unknown_nodes_and_missing_groups_are_reported() {
    let n = node("n", "SCRIPT", json!({}), vec![], vec![out("Value", "VALUE")]);
    let m = mkr_shade::compile("u", &emitting(n, "Value"), &BTreeMap::new(), &[], 0).unwrap();
    assert_eq!(m.unsupported, vec!["SCRIPT ()".to_string()]);
    validate_both(&m);
    let g = json!({"name": "g", "type": "GROUP", "group": "nowhere", "props": {}, "inputs": [],
                   "outputs": [out("Socket_0", "VALUE")]});
    let m = mkr_shade::compile("u", &emitting(g, "Socket_0"), &BTreeMap::new(), &[], 0).unwrap();
    assert!(m.unsupported.iter().any(|u| u.contains("nowhere")), "{:?}", m.unsupported);
    // a linked Material Output Thickness (EEVEE's thickness for translucency, refraction, subsurface) is not drawn
    let v = json!({"name": "v", "type": "VALUE", "props": {}, "inputs": [],
                   "outputs": [sock("Value", "VALUE", json!(0.2))]});
    let d = node("d", "BSDF_DIFFUSE", json!({}), vec![c4("Color"), v3("Normal")], vec![out("BSDF", "SHADER")]);
    let mat = material(vec![v, d], vec![json!(["v", "Value", "out", "Thickness"])], ("d", "BSDF"));
    let m = mkr_shade::compile("u", &mat, &BTreeMap::new(), &[], 0).unwrap();
    assert!(m.unsupported.iter().any(|u| u.contains("thickness")), "{:?}", m.unsupported);
}

/// An Attribute node of type Object, Instancer or View Layer reads a value per object: a parameter slot of its own (its
/// source says which attribute), filled per draw by the renderer, inside a node group too.
#[test]
fn object_attributes_are_parameter_slots_of_their_own() {
    let attr = |name: &str, kind: &str, attr: &str| {
        node(name, "ATTRIBUTE", json!({"attribute_type": kind, "attribute_name": attr}), vec![],
             vec![out("Color", "RGBA"), out("Vector", "VECTOR"), out("Fac", "VALUE"), out("Alpha", "VALUE")])
    };
    let group: GroupDesc = serde_json::from_value(json!({
        "interface": [["OUTPUT", "Out", "Socket_1", "NodeSocketFloat", 0.0]],
        "tree": {"nodes": [attr("ga", "INSTANCER", "tint"),
            {"name": "go", "type": "GROUP_OUTPUT", "props": {"is_active_output": true},
             "inputs": [f("Socket_1", 0.0)], "outputs": []}],
            "links": [["ga", "Alpha", "go", "Socket_1"]]},
        "anim": []})).unwrap();
    let grp = json!({"name": "grp", "type": "GROUP", "group": "g", "props": {}, "inputs": [],
                     "outputs": [out("Socket_1", "VALUE")]});
    let mix = node("mx", "MATH", json!({"operation": "ADD"}), vec![f("Value", 0.0), f("Value_001", 0.0),
                   f("Value_002", 0.0)], vec![out("Value", "VALUE")]);
    let em = json!({"name": "em", "type": "EMISSION", "inputs": [c4("Color"), f("Strength", 1.0)],
                    "outputs": [out("Emission", "SHADER")]});
    let links = vec![json!(["amp", "Fac", "mx", "Value"]), json!(["grp", "Socket_1", "mx", "Value_001"]),
                     json!(["mx", "Value", "em", "Strength"]), json!(["wc", "Color", "em", "Color"])];
    let mat = material(vec![attr("amp", "OBJECT", "mk_amp"), attr("wc", "VIEW_LAYER", "world.color"), grp, mix, em],
                       links, ("em", "Emission"));
    let mut groups = BTreeMap::new();
    groups.insert("g".to_string(), group);
    let m = mkr_shade::compile("oa", &mat, &groups, &[], 0).unwrap();
    assert!(m.unsupported.is_empty(), "{:?}", m.unsupported);
    let mut sources: Vec<(String, String)> = m.params.iter().filter_map(|p| match &p.source {
        mkr_shade::AnimSource::Object { kind, name } => Some((kind.clone(), name.clone())),
        _ => None,
    }).collect();
    sources.sort();
    assert_eq!(sources, [("INSTANCER".to_string(), "tint".to_string()), ("OBJECT".into(), "mk_amp".into()),
                         ("VIEW_LAYER".into(), "world.color".into())]);
    assert_eq!(m.param_count, 3);
    for p in &m.params {
        assert!(m.wgsl.contains(&format!("mkr_param({}u)", p.slot)), "slot {} unread:\n{}", p.slot, m.wgsl);
    }
    validate_both(&m);
}

/// EEVEE lights a Translucent BSDF by the material's thickness mode (SPHERE: from every side, SLAB: from behind): the
/// closures tell the renderer which.
#[test]
fn translucent_closures_carry_the_material_thickness_mode() {
    let gpu = Gpu::new().expect("these tests need the Vulkan GPU");
    for (mode, slab) in [(Some("SLAB"), 1.0f32), (Some("SPHERE"), 0.0), (None, 0.0)] {
        let tr = node("t", "BSDF_TRANSLUCENT", json!({}), vec![c4("Color"), v3("Normal")], vec![out("BSDF", "SHADER")]);
        let mut mat = material(vec![tr], vec![], ("t", "BSDF"));
        if let Some(mode) = mode {
            mat.settings.insert("thickness_mode".into(), json!(mode));
        }
        let m = mkr_shade::compile("lab", &mat, &BTreeMap::new(), &[], 0).unwrap();
        let o = gpu.eval(&m, 4, Kind::Flags, &[], &[], None).unwrap();
        assert_eq!(o[0], slab, "thickness_mode {mode:?}");
    }
}

/// A graph through a node group (with an interface type change), a reroute and a muted node.
fn plumbing_material() -> (MaterialDesc, BTreeMap<String, GroupDesc>) {
    // group: out = in * 2 (in is a float socket fed a colour: the colour is averaged at the boundary)
    let group: GroupDesc = serde_json::from_value(json!({
        "interface": [["INPUT", "In", "Socket_0", "NodeSocketFloat", 0.0], ["OUTPUT", "Out", "Socket_1",
                      "NodeSocketFloat", 0.0]],
        "tree": {"nodes": [
            {"name": "gi", "type": "GROUP_INPUT", "inputs": [], "outputs": [out("Socket_0", "VALUE")]},
            {"name": "x2", "type": "MATH", "props": {"operation": "MULTIPLY"}, "inputs": [f("Value", 0.0),
             f("Value_001", 2.0), f("Value_002", 0.0)], "outputs": [out("Value", "VALUE")]},
            {"name": "go", "type": "GROUP_OUTPUT", "props": {"is_active_output": true},
             "inputs": [f("Socket_1", 0.0)], "outputs": []}],
            "links": [["gi", "Socket_0", "x2", "Value"], ["x2", "Value", "go", "Socket_1"]]},
        "anim": []})).unwrap();
    let rgb = json!({"name": "rgb", "type": "RGB", "props": {}, "inputs": [],
                     "outputs": [sock("Color", "RGBA", json!([0.1, 0.2, 0.6, 1.0]))]});
    let grp = json!({"name": "grp", "type": "GROUP", "group": "dbl", "props": {}, "inputs": [f("Socket_0", 0.0)],
                     "outputs": [out("Socket_1", "VALUE")]});
    let rr = json!({"name": "rr", "type": "REROUTE", "props": {}, "inputs": [f("Input", 0.0)],
                    "outputs": [out("Output", "VALUE")]});
    // muted Invert passes Color through
    let inv = json!({"name": "inv", "type": "INVERT", "mute": true, "props": {}, "inputs": [f("Fac", 1.0),
                     sock("Color", "RGBA", json!([0, 0, 0, 1]))], "outputs": [out("Color", "RGBA")],
                     "internal_links": [["Color", "Color"]]});
    let links = vec![json!(["rgb", "Color", "grp", "Socket_0"]), json!(["grp", "Socket_1", "rr", "Input"]),
                     json!(["rr", "Output", "inv", "Color"])];
    let em = json!({"name": "em", "type": "EMISSION", "inputs": [c4("Color"), f("Strength", 1.0)],
                    "outputs": [out("Emission", "SHADER")]});
    let mut links = links;
    links.push(json!(["inv", "Color", "em", "Color"]));
    let mat = material(vec![rgb, grp, rr, inv, em], links, ("em", "Emission"));
    let mut groups = BTreeMap::new();
    groups.insert("dbl".to_string(), group);
    (mat, groups)
}

#[test]
fn groups_reroutes_and_muted_nodes_evaluate_as_blender() {
    let (mat, groups) = plumbing_material();
    let m = mkr_shade::compile("lab", &mat, &groups, &[], 0).unwrap();
    assert!(m.unsupported.is_empty(), "{:?}", m.unsupported);
    let gpu = Gpu::new().expect("these tests need the Vulkan GPU");
    let outv = gpu.eval(&m, 4, Kind::Field, &[], &[], None).unwrap();
    // colour (0.1, 0.2, 0.6) -> float socket: average 0.3 -> * 2 = 0.6 -> grey through the reroute and the mute
    for c in 0..3 {
        assert!((outv[c] - 0.6).abs() < 1e-5, "channel {c}: {}", outv[c]);
    }
}

/// Lab samples (tests/fixtures/*.json): a lab case's graph and a few quads with Blender's values.
#[test]
fn values_match_blender_lab_samples() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let gpu = Gpu::new().expect("these tests need the Vulkan GPU");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map(|e| e != "json").unwrap_or(true) {
            continue;
        }
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = doc["name"].as_str().unwrap();
        let mat: MaterialDesc = serde_json::from_value(doc["material"].clone()).unwrap();
        let m = mkr_shade::compile("lab", &mat, &BTreeMap::new(), &[], 0).unwrap();
        let kind = match doc["kind"].as_str() {
            Some("closure") => Kind::Closure,
            Some("shade") => Kind::Shade,
            _ => Kind::Field,
        };
        let n = doc["n"].as_u64().unwrap() as u32;
        let samples = doc["samples"].as_array().unwrap();
        // a quad per sample, at the sample's grid position, with its inputs
        let mut inputs = vec![0.0f32; (n * n * 9) as usize];
        for s in samples {
            let q = s["quad"].as_u64().unwrap() as usize;
            for (k, v) in s["in"].as_array().unwrap().iter().enumerate() {
                inputs[q * 9 + k] = v.as_f64().unwrap() as f32;
            }
        }
        let rows: Vec<f32> = doc["rows"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .unwrap_or_default();
        let outv = gpu.eval(&m, n, kind, &inputs, &rows, None).unwrap();
        let tol = doc["tolerance"].as_f64().unwrap_or(0.002);
        for s in samples {
            let q = s["quad"].as_u64().unwrap() as usize;
            for c in 0..3 {
                let r = s["ref"][c].as_f64().unwrap() as f32;
                let e = film_error(outv[q * 4 + c], r);
                assert!(e <= tol, "{name} quad {q} channel {c}: ours {} Blender {r} (error {e:.5})", outv[q * 4 + c]);
            }
        }
        checked += 1;
    }
    assert!(checked >= 10, "only {checked} fixtures");
}
