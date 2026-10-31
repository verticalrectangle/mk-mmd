//! Per-node WGSL emission: Blender 4.2 shader nodes as EEVEE evaluates them.

use serde_json::Value;

use crate::codegen::{Expr, Gen, lit};
use crate::graph::Ty;

type Outs = Vec<Option<Expr>>;

fn set(g: &Gen, n: usize, pairs: Vec<(&str, Expr)>) -> Outs {
    let node = g.node(n);
    let mut v: Outs = vec![None; node.outputs.len()];
    for (k, e) in pairs {
        if let Some(i) = node.output_index(k) {
            v[i] = Some(e);
        }
    }
    v
}

fn enum_index(g: &Gen, n: usize, prop: &str, names: &[&str], default: usize) -> usize {
    match g.node(n).prop_str(prop) {
        Some(s) => names.iter().position(|x| *x == s).unwrap_or(default),
        None => default,
    }
}

fn dims(g: &Gen, n: usize, prop: &str) -> u32 {
    match g.node(n).prop_str(prop) {
        Some("1D") => 1,
        Some("2D") => 2,
        Some("4D") => 4,
        _ => 3,
    }
}

fn pack(d: u32, v: &str, w: &str) -> String {
    match d {
        1 => format!("vec4f({w}, 0.0, 0.0, 0.0)"),
        2 => format!("vec4f(({v}).xy, 0.0, 0.0)"),
        3 => format!("vec4f({v}, 0.0)"),
        _ => format!("vec4f({v}, {w})"),
    }
}

fn b(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}

/// Emit node `n`; returns its outputs (aligned with the node's outputs).
pub fn emit(g: &mut Gen, n: usize) -> Outs {
    let node = g.node(n);
    let kind = node.kind.as_str();

    // Animated node outputs (Value / RGB nodes keyed or driven).
    if node.outputs.iter().any(|s| s.anim.is_some()) && matches!(kind, "VALUE" | "RGB") {
        let s = &node.outputs[0];
        let slot = g.sh.param_base + s.anim.unwrap() as u32;
        let e = if kind == "VALUE" {
            Expr::new(format!("mkr_param({slot}u).x"), Ty::Float)
        } else {
            Expr::new(format!("mkr_param({slot}u)"), Ty::Color)
        };
        return vec![Some(e)];
    }

    match kind {
        "VALUE" => {
            let s = &node.outputs[0];
            vec![Some(Expr::new(crate::codegen::const_of(&s.value, "VALUE", Ty::Float), Ty::Float))]
        }
        "RGB" => {
            let s = &node.outputs[0];
            vec![Some(Expr::new(crate::codegen::const_of(&s.value, "RGBA", Ty::Color), Ty::Color))]
        }
        "NORMAL" => normal_node(g, n),
        "MATH" => math(g, n),
        "VECT_MATH" => vect_math(g, n),
        "MIX_RGB" => mix_rgb(g, n),
        "MIX" => mix(g, n),
        "MAP_RANGE" => map_range(g, n),
        "CLAMP" => clamp_node(g, n),
        "VALTORGB" => color_ramp(g, n),
        "CURVE_RGB" | "CURVE_VEC" | "CURVE_FLOAT" => curves(g, n),
        "SEPARATE_COLOR" | "SEPRGB" | "SEPHSV" => separate_color(g, n),
        "COMBINE_COLOR" | "COMBRGB" | "COMBHSV" => combine_color(g, n),
        "SEPXYZ" => {
            let v = g.input_opt(n, "Vector", Ty::Vec3);
            let v = g.bind("v", v, Ty::Vec3);
            set(
                g,
                n,
                vec![
                    ("X", Expr::new(format!("{}.x", v.code), Ty::Float)),
                    ("Y", Expr::new(format!("{}.y", v.code), Ty::Float)),
                    ("Z", Expr::new(format!("{}.z", v.code), Ty::Float)),
                ],
            )
        }
        "COMBXYZ" => {
            let x = g.input_opt(n, "X", Ty::Float);
            let y = g.input_opt(n, "Y", Ty::Float);
            let z = g.input_opt(n, "Z", Ty::Float);
            let e = g.bind("v", format!("vec3f({x}, {y}, {z})"), Ty::Vec3);
            vec![Some(e)]
        }
        "MAPPING" => mapping(g, n),
        "VECT_TRANSFORM" => vect_transform(g, n),
        "VECTOR_ROTATE" => vector_rotate(g, n),
        "HUE_SAT" => {
            let h = g.input_f(n, "Hue");
            let s = g.input_f(n, "Saturation");
            let v = g.input_f(n, "Value");
            let f = g.input_f(n, "Fac");
            let c = g.input_c(n, "Color");
            vec![Some(g.bind("c", format!("mks_hue_sat({h}, {s}, {v}, {f}, {c})"), Ty::Color))]
        }
        "INVERT" => {
            let f = g.input_f(n, "Fac");
            let c = g.input_c(n, "Color");
            vec![Some(g.bind("c", format!("mks_invert({f}, {c})"), Ty::Color))]
        }
        "GAMMA" => {
            let c = g.input_c(n, "Color");
            let gm = g.input_f(n, "Gamma");
            vec![Some(g.bind("c", format!("mks_gamma({c}, {gm})"), Ty::Color))]
        }
        "BRIGHTCONTRAST" => {
            let c = g.input_c(n, "Color");
            let br = g.input_f(n, "Bright");
            let ct = g.input_f(n, "Contrast");
            vec![Some(g.bind("c", format!("mks_bright_contrast({c}, {br}, {ct})"), Ty::Color))]
        }
        "RGBTOBW" => {
            let c = g.input_c(n, "Color");
            vec![Some(g.bind("f", format!("mks_luma(({c}).rgb)"), Ty::Float))]
        }
        "TEX_COORD" => tex_coord(g, n),
        "UVMAP" => {
            if g.node(n).prop_bool("from_instancer").unwrap_or(false) {
                g.unsupported("UVMAP from_instancer");
            }
            let map = g.node(n).prop_str("uv_map").unwrap_or("").to_string();
            let uv = g.uv_attr(&map);
            vec![Some(g.bind("v", uv, Ty::Vec3))]
        }
        "ATTRIBUTE" => attribute(g, n),
        "VERTEX_COLOR" => {
            let layer = g.node(n).prop_str("layer_name").unwrap_or("").to_string();
            let a = g.src_attr(&format!("color:{layer}"));
            let a = g.bind("a", a, Ty::Color);
            set(
                g,
                n,
                vec![("Color", a.clone()), ("Alpha", Expr::new(format!("{}.w", a.code), Ty::Float))],
            )
        }
        "NEW_GEOMETRY" => geometry(g, n),
        "LAYER_WEIGHT" => {
            let blend = g.input_f(n, "Blend");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            let code = g.fresh("lw");
            g.stmt(format!("let {code}: vec2f = mks_layer_weight({blend}, {nrm}, s.I, s.backfacing);"));
            set(
                g,
                n,
                vec![
                    ("Fresnel", Expr::new(format!("{code}.x"), Ty::Float)),
                    ("Facing", Expr::new(format!("{code}.y"), Ty::Float)),
                ],
            )
        }
        "FRESNEL" => {
            let ior = g.input_f(n, "IOR");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            vec![Some(g.bind("f", format!("mks_fresnel({ior}, {nrm}, s.I, s.backfacing)"), Ty::Float))]
        }
        "NORMAL_MAP" => normal_map(g, n),
        "BUMP" => bump(g, n),
        "TANGENT" => tangent(g, n),
        "BEVEL" => {
            // EEVEE draws no bevel: the normal passes through.
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            vec![Some(g.bind("v", nrm, Ty::Vec3))]
        }
        "TEX_IMAGE" => tex_image(g, n),
        "TEX_ENVIRONMENT" => tex_environment(g, n),
        "TEX_NOISE" => tex_noise(g, n),
        "TEX_VORONOI" => tex_voronoi(g, n),
        "TEX_WAVE" => tex_wave(g, n),
        "TEX_GRADIENT" => {
            let co = tex_vector(g, n);
            let kind = enum_index(
                g,
                n,
                "gradient_type",
                &["LINEAR", "QUADRATIC", "EASING", "DIAGONAL", "RADIAL", "QUADRATIC_SPHERE", "SPHERICAL"],
                0,
            );
            let f = g.bind("f", format!("mks_tex_gradient({co}, {kind}u)"), Ty::Float);
            let c = Expr::new(format!("vec4f(vec3f({}), 1.0)", f.code), Ty::Color);
            set(g, n, vec![("Fac", f), ("Color", c)])
        }
        "TEX_CHECKER" => {
            let co = tex_vector(g, n);
            let c1 = g.input_c(n, "Color1");
            let c2 = g.input_c(n, "Color2");
            let sc = g.input_f(n, "Scale");
            let f = g.bind("f", format!("mks_tex_checker({co}, {sc})"), Ty::Float);
            let c = g.bind("c", format!("mix({c2}, {c1}, {})", f.code), Ty::Color);
            set(g, n, vec![("Fac", f), ("Color", c)])
        }
        "TEX_BRICK" => tex_brick(g, n),
        "TEX_MAGIC" => {
            let co = tex_vector(g, n);
            let sc = g.input_f(n, "Scale");
            let di = g.input_f(n, "Distortion");
            let depth = g.node(n).prop_f64("turbulence_depth").unwrap_or(2.0) as i32;
            let c = g.bind("v", format!("mks_tex_magic({co}, {sc}, {di}, {depth})"), Ty::Vec3);
            set(
                g,
                n,
                vec![
                    ("Color", Expr::new(format!("vec4f({}, 1.0)", c.code), Ty::Color)),
                    ("Fac", Expr::new(format!("mks_avg3({})", c.code), Ty::Float)),
                ],
            )
        }
        "TEX_WHITE_NOISE" => {
            let d = dims(g, n, "noise_dimensions");
            let v = g.linked_input(n, "Vector", Ty::Vec3).unwrap_or_else(|| "vec3f(0.0)".into());
            let w = g.input_f(n, "W");
            let p = pack(d, &v, &w);
            let r = g.bind("c", format!("mks_tex_white_noise({p}, {d}u)"), Ty::Color);
            set(
                g,
                n,
                vec![
                    ("Value", Expr::new(format!("{}.x", r.code), Ty::Float)),
                    ("Color", Expr::new(format!("vec4f({}.yzw, 1.0)", r.code), Ty::Color)),
                ],
            )
        }
        "LIGHT_PATH" => light_path(g, n),
        "OBJECT_INFO" => {
            let pi = g.sh.pass_index;
            set(
                g,
                n,
                vec![
                    ("Location", Expr::new("s.object[3].xyz", Ty::Vec3)),
                    ("Color", Expr::new("s.object_color", Ty::Color)),
                    ("Alpha", Expr::new("s.object_color.a", Ty::Float)),
                    ("Object Index", Expr::new("s.object_info.x", Ty::Float)),
                    ("Material Index", Expr::new(lit(pi), Ty::Float)),
                    ("Random", Expr::new("s.object_info.y", Ty::Float)),
                ],
            )
        }
        "CAMERA" => {
            let p = g.src_p();
            let v = g.bind("v", format!("(s.camera_inv * vec4f({p}, 1.0)).xyz * vec3f(1.0, 1.0, -1.0)"), Ty::Vec3);
            set(
                g,
                n,
                vec![
                    (
                        "View Vector",
                        Expr::new(
                            format!("normalize({})", v.code),
                            Ty::Vec3,
                        ),
                    ),
                    ("View Z Depth", Expr::new(format!("abs({}.z)", v.code), Ty::Float)),
                    ("View Distance", Expr::new(format!("length({})", v.code), Ty::Float)),
                ],
            )
        }
        "EMISSION" | "BACKGROUND" => {
            if g.is_alpha() {
                return vec![Some(Expr::new("mks_closures_none()", Ty::Shader))];
            }
            let c = g.input_c(n, "Color");
            let s = g.input_f(n, "Strength");
            vec![Some(g.bind("cl", format!("mks_bsdf_emission(({c}).rgb, {s})"), Ty::Shader))]
        }
        "BSDF_DIFFUSE" | "SUBSURFACE_SCATTERING" | "BSDF_SHEEN" | "BSDF_TOON" => {
            if kind == "BSDF_TOON" {
                g.warn(format!("{}: Toon BSDF drawn as diffuse (as EEVEE)", g.node(n).path));
            }
            if g.is_alpha() {
                return vec![Some(Expr::new("mks_closures_none()", Ty::Shader))];
            }
            let c = g.input_c(n, "Color");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            vec![Some(g.bind("cl", format!("mks_bsdf_diffuse(({c}).rgb, {nrm})"), Ty::Shader))]
        }
        "BSDF_TRANSLUCENT" => {
            if g.is_alpha() {
                return vec![Some(Expr::new("mks_closures_none()", Ty::Shader))];
            }
            let c = g.input_c(n, "Color");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            vec![Some(g.bind("cl", format!("mks_bsdf_translucent(({c}).rgb, {nrm})"), Ty::Shader))]
        }
        "BSDF_GLOSSY" | "BSDF_ANISOTROPIC" => {
            if g.is_alpha() {
                return vec![Some(Expr::new("mks_closures_none()", Ty::Shader))];
            }
            let c = g.input_c(n, "Color");
            let r = g.input_f(n, "Roughness");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            let multi = g.node(n).prop_str("distribution") == Some("MULTI_GGX");
            vec![Some(g.bind(
                "cl",
                format!("mks_bsdf_glossy(({c}).rgb, {r}, {nrm}, s.I, {})", b(multi)),
                Ty::Shader,
            ))]
        }
        "BSDF_REFRACTION" => {
            if g.is_alpha() {
                return vec![Some(Expr::new("mks_closures_none()", Ty::Shader))];
            }
            let c = g.input_c(n, "Color");
            let r = g.input_f(n, "Roughness");
            let ior = g.input_f(n, "IOR");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            vec![Some(g.bind(
                "cl",
                format!("mks_bsdf_refraction(({c}).rgb, {r}, select({ior}, 1.0 / {ior}, s.backfacing != 0.0), {nrm})"),
                Ty::Shader,
            ))]
        }
        "BSDF_GLASS" => {
            if g.is_alpha() {
                return vec![Some(Expr::new("mks_closures_none()", Ty::Shader))];
            }
            let c = g.input_c(n, "Color");
            let r = g.input_f(n, "Roughness");
            let ior = g.input_f(n, "IOR");
            let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
            let multi = g.node(n).prop_str("distribution") == Some("MULTI_GGX");
            vec![Some(g.bind(
                "cl",
                format!(
                    "mks_bsdf_glass(({c}).rgb, {r}, select({ior}, 1.0 / {ior}, s.backfacing != 0.0), {nrm}, s.I, {})",
                    b(multi)
                ),
                Ty::Shader,
            ))]
        }
        "BSDF_TRANSPARENT" => {
            let c = g.input_c(n, "Color");
            if g.is_alpha() {
                g.sh.transparent = true;
            }
            vec![Some(g.bind("cl", format!("mks_bsdf_transparent(({c}).rgb)"), Ty::Shader))]
        }
        "HOLDOUT" => vec![Some(g.bind("cl", "mks_bsdf_holdout()", Ty::Shader))],
        "BSDF_PRINCIPLED" => principled(g, n),
        "EEVEE_SPECULAR" => eevee_specular(g, n),
        "MIX_SHADER" => {
            let f = g.input_opt(n, "Fac", Ty::Float);
            let a = g.input_opt(n, "Shader", Ty::Shader);
            let c = g.input_opt(n, "Shader_001", Ty::Shader);
            vec![Some(g.bind("cl", format!("mks_closures_mix({a}, {c}, {f}, &rng)"), Ty::Shader))]
        }
        "ADD_SHADER" => {
            let a = g.input_opt(n, "Shader", Ty::Shader);
            let c = g.input_opt(n, "Shader_001", Ty::Shader);
            vec![Some(g.bind("cl", format!("mks_closures_add({a}, {c}, &rng)"), Ty::Shader))]
        }
        "SHADERTORGB" => {
            let c = g.input_opt(n, "Shader", Ty::Shader);
            let c = g.bind("cl", c, Ty::Shader);
            let rgb = if g.is_alpha() {
                g.warn(format!("{}: Shader to RGB feeding transparency uses emission only", g.node(n).path));
                format!("{}.emission", c.code)
            } else {
                g.sh.uses_shader_to_rgb = true;
                format!("mkr_shader_to_rgb({}, s).rgb", c.code)
            };
            let col = g.bind("c", format!("vec4f({rgb}, mks_opacity({}))", c.code), Ty::Color);
            set(
                g,
                n,
                vec![("Color", col.clone()), ("Alpha", Expr::new(format!("{}.a", col.code), Ty::Float))],
            )
        }
        _ => {
            let what = format!("{} ({})", kind, node.bl_idname);
            g.unsupported(what);
            vec![None; node.outputs.len()]
        }
    }
}

// --------------------------------------------------------------------------------------------------- Math
fn math(g: &mut Gen, n: usize) -> Outs {
    let op = g.node(n).prop_str("operation").unwrap_or("ADD").to_string();
    let clamp = g.node(n).prop_bool("use_clamp").unwrap_or(false);
    let a = g.input_opt(n, "Value", Ty::Float);
    let bb = g.input_opt(n, "Value_001", Ty::Float);
    let c = g.input_opt(n, "Value_002", Ty::Float);
    let e = match op.as_str() {
        "ADD" => format!("{a} + {bb}"),
        "SUBTRACT" => format!("{a} - {bb}"),
        "MULTIPLY" => format!("{a} * {bb}"),
        "DIVIDE" => format!("mks_safe_div({a}, {bb})"),
        "MULTIPLY_ADD" => format!("{a} * {bb} + {c}"),
        "POWER" => format!("mks_math_power({a}, {bb})"),
        "LOGARITHM" => format!("mks_math_log({a}, {bb})"),
        "SQRT" => format!("mks_math_sqrt({a})"),
        "INVERSE_SQRT" => format!("mks_math_inv_sqrt({a})"),
        "ABSOLUTE" => format!("abs({a})"),
        "EXPONENT" => format!("exp({a})"),
        "MINIMUM" => format!("min({a}, {bb})"),
        "MAXIMUM" => format!("max({a}, {bb})"),
        "LESS_THAN" => format!("select(0.0, 1.0, {a} < {bb})"),
        "GREATER_THAN" => format!("select(0.0, 1.0, {a} > {bb})"),
        "SIGN" => format!("sign({a})"),
        "COMPARE" => format!("mks_math_compare({a}, {bb}, {c})"),
        "SMOOTH_MIN" => format!("mks_smoothmin({a}, {bb}, {c})"),
        "SMOOTH_MAX" => format!("-mks_smoothmin(-({a}), -({bb}), {c})"),
        "ROUND" => format!("mks_round({a})"),
        "FLOOR" => format!("floor({a})"),
        "CEIL" => format!("ceil({a})"),
        "TRUNC" => format!("trunc({a})"),
        "FRACT" => format!("mks_frac({a})"),
        "MODULO" => format!("mks_fmod({a}, {bb})"),
        "FLOORED_MODULO" => format!("mks_floored_mod({a}, {bb})"),
        "WRAP" => format!("mks_wrap({a}, {bb}, {c})"),
        "SNAP" => format!("mks_snap({a}, {bb})"),
        "PINGPONG" => format!("mks_pingpong({a}, {bb})"),
        "SINE" => format!("sin({a})"),
        "COSINE" => format!("cos({a})"),
        "TANGENT" => format!("tan({a})"),
        "ARCSINE" => format!("mks_math_asin({a})"),
        "ARCCOSINE" => format!("mks_math_acos({a})"),
        "ARCTANGENT" => format!("atan({a})"),
        // A signed zero in y gives +pi on the GPU EEVEE ran on (measured): drop the sign.
        "ARCTAN2" => format!("atan2(select({a}, 0.0, {a} == 0.0), {bb})"),
        "SINH" => format!("sinh({a})"),
        "COSH" => format!("cosh({a})"),
        "TANH" => format!("tanh({a})"),
        "RADIANS" => format!("radians({a})"),
        "DEGREES" => format!("degrees({a})"),
        other => {
            g.unsupported(format!("MATH {other}"));
            "0.0".into()
        }
    };
    let e = if clamp { format!("clamp({e}, 0.0, 1.0)") } else { e };
    vec![Some(g.bind("f", e, Ty::Float))]
}

fn vect_math(g: &mut Gen, n: usize) -> Outs {
    let op = g.node(n).prop_str("operation").unwrap_or("ADD").to_string();
    let a = g.input_opt(n, "Vector", Ty::Vec3);
    let bb = g.input_opt(n, "Vector_001", Ty::Vec3);
    let c = g.input_opt(n, "Vector_002", Ty::Vec3);
    let sc = g.input_opt(n, "Scale", Ty::Float);
    let vec = |e: String| (Some(e), None::<String>);
    let val = |e: String| (None::<String>, Some(e));
    let (v, f) = match op.as_str() {
        "ADD" => vec(format!("{a} + {bb}")),
        "SUBTRACT" => vec(format!("{a} - {bb}")),
        "MULTIPLY" => vec(format!("{a} * {bb}")),
        "DIVIDE" => vec(format!("mks_safe_div3({a}, {bb})")),
        "MULTIPLY_ADD" => vec(format!("{a} * {bb} + {c}")),
        "CROSS_PRODUCT" => vec(format!("cross({a}, {bb})")),
        "PROJECT" => vec(format!("mks_project({a}, {bb})")),
        "REFLECT" => vec(format!("mks_reflect({a}, {bb})")),
        "REFRACT" => vec(format!("mks_refract({a}, {bb}, {sc})")),
        "FACEFORWARD" => vec(format!("mks_faceforward({a}, {bb}, {c})")),
        "DOT_PRODUCT" => val(format!("dot({a}, {bb})")),
        "DISTANCE" => val(format!("distance({a}, {bb})")),
        "LENGTH" => val(format!("length({a})")),
        "SCALE" => vec(format!("{a} * {sc}")),
        "NORMALIZE" => vec(format!("mks_safe_normalize({a})")),
        "SNAP" => vec(format!("mks_snap3({a}, {bb})")),
        "FLOOR" => vec(format!("floor({a})")),
        "CEIL" => vec(format!("ceil({a})")),
        "MODULO" => vec(format!("mks_fmod3({a}, {bb})")),
        "WRAP" => vec(format!("mks_wrap3({a}, {bb}, {c})")),
        "FRACTION" => vec(format!("({a}) - floor({a})")),
        "ABSOLUTE" => vec(format!("abs({a})")),
        "MINIMUM" => vec(format!("min({a}, {bb})")),
        "MAXIMUM" => vec(format!("max({a}, {bb})")),
        "SINE" => vec(format!("sin({a})")),
        "COSINE" => vec(format!("cos({a})")),
        "TANGENT" => vec(format!("tan({a})")),
        "POWER" => vec(format!("mks_pow3({a}, {bb})")),
        "SIGN" => vec(format!("sign({a})")),
        other => {
            g.unsupported(format!("VECT_MATH {other}"));
            vec("vec3f(0.0)".into())
        }
    };
    let ve = match v {
        Some(e) => g.bind("v", e, Ty::Vec3),
        None => Expr::new("vec3f(0.0)", Ty::Vec3),
    };
    let fe = match f {
        Some(e) => g.bind("f", e, Ty::Float),
        None => Expr::new("0.0", Ty::Float),
    };
    set(g, n, vec![("Vector", ve), ("Value", fe)])
}

fn blend_fn(mode: &str) -> Option<&'static str> {
    Some(match mode {
        "MIX" => "mks_blend_mix",
        "DARKEN" => "mks_blend_dark",
        "MULTIPLY" => "mks_blend_mult",
        "BURN" => "mks_blend_burn",
        "LIGHTEN" => "mks_blend_light",
        "SCREEN" => "mks_blend_screen",
        "DODGE" => "mks_blend_dodge",
        "ADD" => "mks_blend_add",
        "OVERLAY" => "mks_blend_overlay",
        "SOFT_LIGHT" => "mks_blend_soft",
        "LINEAR_LIGHT" => "mks_blend_linear",
        "DIFFERENCE" => "mks_blend_diff",
        "EXCLUSION" => "mks_blend_exclusion",
        "SUBTRACT" => "mks_blend_sub",
        "DIVIDE" => "mks_blend_div",
        "HUE" => "mks_blend_hue",
        "SATURATION" => "mks_blend_sat",
        "COLOR" => "mks_blend_color",
        "VALUE" => "mks_blend_val",
        _ => return None,
    })
}

fn mix_rgb(g: &mut Gen, n: usize) -> Outs {
    let mode = g.node(n).prop_str("blend_type").unwrap_or("MIX").to_string();
    let clamp = g.node(n).prop_bool("use_clamp").unwrap_or(false);
    let f = g.input(n, "Fac", Ty::Float);
    let c1 = g.input(n, "Color1", Ty::Color);
    let c2 = g.input(n, "Color2", Ty::Color);
    let func = blend_fn(&mode).unwrap_or_else(|| {
        g.unsupported(format!("MIX_RGB {mode}"));
        "mks_blend_mix"
    });
    let mut e = format!("{func}(clamp({f}, 0.0, 1.0), {c1}, {c2})");
    if clamp {
        e = format!("clamp({e}, vec4f(0.0), vec4f(1.0))");
    }
    vec![Some(g.bind("c", e, Ty::Color))]
}

fn mix(g: &mut Gen, n: usize) -> Outs {
    let dt = g.node(n).prop_str("data_type").unwrap_or("FLOAT").to_string();
    let clamp_f = g.node(n).prop_bool("clamp_factor").unwrap_or(true);
    let clamp_r = g.node(n).prop_bool("clamp_result").unwrap_or(false);
    match dt.as_str() {
        "VECTOR" => {
            let non_uniform = g.node(n).prop_str("factor_mode") == Some("NON_UNIFORM");
            let a = g.input(n, "A_Vector", Ty::Vec3);
            let bb = g.input(n, "B_Vector", Ty::Vec3);
            let e = if non_uniform {
                let f = g.input(n, "Factor_Vector", Ty::Vec3);
                let f = if clamp_f { format!("clamp({f}, vec3f(0.0), vec3f(1.0))") } else { f };
                format!("mix({a}, {bb}, {f})")
            } else {
                let f = g.input(n, "Factor_Float", Ty::Float);
                let f = if clamp_f { format!("clamp({f}, 0.0, 1.0)") } else { f };
                format!("mix({a}, {bb}, vec3f({f}))")
            };
            let v = g.bind("v", e, Ty::Vec3);
            set(g, n, vec![("Result_Vector", v)])
        }
        "RGBA" => {
            let mode = g.node(n).prop_str("blend_type").unwrap_or("MIX").to_string();
            let f = g.input(n, "Factor_Float", Ty::Float);
            let f = if clamp_f { format!("clamp({f}, 0.0, 1.0)") } else { f };
            let a = g.input(n, "A_Color", Ty::Color);
            let bb = g.input(n, "B_Color", Ty::Color);
            let func = blend_fn(&mode).unwrap_or_else(|| {
                g.unsupported(format!("MIX {mode}"));
                "mks_blend_mix"
            });
            let mut e = format!("{func}({f}, {a}, {bb})");
            if clamp_r {
                e = format!("clamp({e}, vec4f(0.0), vec4f(1.0))");
            }
            let c = g.bind("c", e, Ty::Color);
            set(g, n, vec![("Result_Color", c)])
        }
        "ROTATION" => {
            g.unsupported("MIX ROTATION");
            vec![None; g.node(n).outputs.len()]
        }
        _ => {
            let f = g.input(n, "Factor_Float", Ty::Float);
            let f = if clamp_f { format!("clamp({f}, 0.0, 1.0)") } else { f };
            let a = g.input(n, "A_Float", Ty::Float);
            let bb = g.input(n, "B_Float", Ty::Float);
            let v = g.bind("f", format!("mix({a}, {bb}, {f})"), Ty::Float);
            set(g, n, vec![("Result_Float", v)])
        }
    }
}

fn map_range(g: &mut Gen, n: usize) -> Outs {
    let dt = g.node(n).prop_str("data_type").unwrap_or("FLOAT").to_string();
    let interp = g.node(n).prop_str("interpolation_type").unwrap_or("LINEAR").to_string();
    let clamp = g.node(n).prop_bool("clamp").unwrap_or(true);
    if dt == "FLOAT_VECTOR" {
        let v = g.input(n, "Vector", Ty::Vec3);
        let v = g.bind("v", v, Ty::Vec3);
        let fmin = g.input(n, "From_Min_FLOAT3", Ty::Vec3);
        let fmax = g.input(n, "From_Max_FLOAT3", Ty::Vec3);
        let tmin = g.input(n, "To_Min_FLOAT3", Ty::Vec3);
        let tmax = g.input(n, "To_Max_FLOAT3", Ty::Vec3);
        let steps = g.input(n, "Steps_FLOAT3", Ty::Vec3);
        let (fmin, fmax, tmin, tmax, steps) = (
            g.bind("v", fmin, Ty::Vec3),
            g.bind("v", fmax, Ty::Vec3),
            g.bind("v", tmin, Ty::Vec3),
            g.bind("v", tmax, Ty::Vec3),
            g.bind("v", steps, Ty::Vec3),
        );
        let comp = |c: &str| -> String {
            let (v, a, bb, t0, t1, st) =
                (&v.code, &fmin.code, &fmax.code, &tmin.code, &tmax.code, &steps.code);
            let r = match interp.as_str() {
                "STEPPED" => format!("mks_map_range_stepped({v}.{c}, {a}.{c}, {bb}.{c}, {t0}.{c}, {t1}.{c}, {st}.{c})"),
                "SMOOTHSTEP" => format!("mks_map_range_smoothstep({v}.{c}, {a}.{c}, {bb}.{c}, {t0}.{c}, {t1}.{c})"),
                "SMOOTHERSTEP" => {
                    format!("mks_map_range_smootherstep({v}.{c}, {a}.{c}, {bb}.{c}, {t0}.{c}, {t1}.{c})")
                }
                _ => format!("mks_map_range_linear({v}.{c}, {a}.{c}, {bb}.{c}, {t0}.{c}, {t1}.{c})"),
            };
            if clamp && (interp == "LINEAR" || interp == "STEPPED") {
                format!("mks_clamp_range({r}, {t0}.{c}, {t1}.{c})")
            } else {
                r
            }
        };
        let e = format!("vec3f({}, {}, {})", comp("x"), comp("y"), comp("z"));
        let r = g.bind("v", e, Ty::Vec3);
        return set(g, n, vec![("Vector", r)]);
    }
    let v = g.input(n, "Value", Ty::Float);
    let fmin = g.input(n, "From Min", Ty::Float);
    let fmax = g.input(n, "From Max", Ty::Float);
    let tmin = g.input(n, "To Min", Ty::Float);
    let tmax = g.input(n, "To Max", Ty::Float);
    let r = match interp.as_str() {
        "STEPPED" => {
            let st = g.input(n, "Steps", Ty::Float);
            format!("mks_map_range_stepped({v}, {fmin}, {fmax}, {tmin}, {tmax}, {st})")
        }
        "SMOOTHSTEP" => format!("mks_map_range_smoothstep({v}, {fmin}, {fmax}, {tmin}, {tmax})"),
        "SMOOTHERSTEP" => format!("mks_map_range_smootherstep({v}, {fmin}, {fmax}, {tmin}, {tmax})"),
        _ => format!("mks_map_range_linear({v}, {fmin}, {fmax}, {tmin}, {tmax})"),
    };
    let r = if clamp && (interp == "LINEAR" || interp == "STEPPED") {
        let t0 = g.bind("f", tmin, Ty::Float);
        let t1 = g.bind("f", tmax, Ty::Float);
        format!("mks_clamp_range({r}, {}, {})", t0.code, t1.code)
    } else {
        r
    };
    let e = g.bind("f", r, Ty::Float);
    set(g, n, vec![("Result", e)])
}

fn clamp_node(g: &mut Gen, n: usize) -> Outs {
    let range = g.node(n).prop_str("clamp_type") == Some("RANGE");
    let v = g.input(n, "Value", Ty::Float);
    let lo = g.input(n, "Min", Ty::Float);
    let hi = g.input(n, "Max", Ty::Float);
    let e = if range { format!("mks_clamp_range({v}, {lo}, {hi})") } else { format!("mks_clamp_minmax({v}, {lo}, {hi})") };
    vec![Some(g.bind("f", e, Ty::Float))]
}

// --------------------------------------------------------------------------------------------------- Colour
fn rgba(v: &Value) -> [f64; 4] {
    let a: Vec<f64> = v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
    [
        a.first().copied().unwrap_or(0.0),
        a.get(1).copied().unwrap_or(0.0),
        a.get(2).copied().unwrap_or(0.0),
        a.get(3).copied().unwrap_or(1.0),
    ]
}

fn vec4_lit(c: [f64; 4]) -> String {
    format!("vec4f({}, {}, {}, {})", lit(c[0]), lit(c[1]), lit(c[2]), lit(c[3]))
}

fn color_ramp(g: &mut Gen, n: usize) -> Outs {
    let fac = g.input(n, "Fac", Ty::Float);
    let Some(ramp) = g.node(n).ramp.clone() else {
        g.unsupported("VALTORGB without ramp data");
        return vec![None; g.node(n).outputs.len()];
    };
    let interp = ramp.get("interpolation").and_then(|v| v.as_str()).unwrap_or("LINEAR").to_string();
    let mode = ramp.get("color_mode").and_then(|v| v.as_str()).unwrap_or("RGB").to_string();
    let stops: Vec<(f64, [f64; 4])> = ramp
        .get("stops")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| {
                    let s = s.as_array()?;
                    Some((s.first()?.as_f64()?, rgba(s.get(1)?)))
                })
                .collect()
        })
        .unwrap_or_default();
    let e = if stops.len() == 2 && mode == "RGB" && matches!(interp.as_str(), "LINEAR" | "CONSTANT" | "EASE") {
        // EEVEE's two-stop shortcut.
        let (p0, c0) = stops[0];
        let (p1, c1) = stops[1];
        let (c0, c1) = (vec4_lit(c0), vec4_lit(c1));
        match interp.as_str() {
            "CONSTANT" => format!("select({c0}, {c1}, {fac} > {})", lit(p0.max(p1))),
            _ => {
                let mul = 1.0 / (p1 - p0);
                let bias = -mul * p0;
                let f = format!("clamp({fac} * {} + {}, 0.0, 1.0)", lit(mul), lit(bias));
                let f = g.bind("f", f, Ty::Float);
                let f = if interp == "EASE" { format!("({0} * {0} * (3.0 - 2.0 * {0}))", f.code) } else { f.code };
                format!("mix({c0}, {c1}, {f})")
            }
        }
    } else {
        let table: Vec<[f64; 4]> = ramp
            .get("table")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(rgba).collect())
            .unwrap_or_default();
        if table.len() < 2 {
            g.unsupported("VALTORGB without a 257-sample table");
            return vec![None; g.node(n).outputs.len()];
        }
        let nearest = interp == "CONSTANT";
        let key = format!("ramp:{nearest}:{table:?}");
        let name = g.helper(key, |name| {
            let size = table.len();
            let mut s = format!("fn {name}(fac: f32) -> vec4f {{\n  var t = array<vec4f, {size}>(\n");
            for (i, c) in table.iter().enumerate() {
                let sep = if i + 1 == size { "" } else { "," };
                s.push_str(&format!("    {}{sep}\n", vec4_lit(*c)));
            }
            s.push_str("  );\n");
            if nearest {
                s.push_str(&format!("  return t[i32(clamp(fac, 0.0, 1.0) * {}.0)];\n}}\n", size - 1));
            } else {
                s.push_str(&format!(
                    "  let x = clamp(fac, 0.0, 1.0) * {m}.0;\n  let i0 = floor(x);\n  let f = x - i0;\n  \
                     let a = t[clamp(i32(i0), 0, {m})];\n  let b = t[clamp(i32(i0) + 1, 0, {m})];\n  \
                     return mix(a, b, f);\n}}\n",
                    m = size - 1
                ));
            }
            s
        });
        format!("{name}({fac})")
    };
    let c = g.bind("c", e, Ty::Color);
    set(g, n, vec![("Color", c.clone()), ("Alpha", Expr::new(format!("{}.a", c.code), Ty::Float))])
}

/// One sampled curve of a CurveMapping as a helper function `name(x) -> f32`.
fn curve_fn(g: &mut Gen, curve: &Value) -> Option<String> {
    let range = curve.get("range")?.as_array()?;
    let x0 = range.first()?.as_f64()?;
    let x1 = range.get(1)?.as_f64()?;
    let samples: Vec<f64> = curve.get("samples")?.as_array()?.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
    if samples.len() < 2 {
        return None;
    }
    let lo = curve.get("slope_lo").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let hi = curve.get("slope_hi").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let key = format!("curve:{x0}:{x1}:{lo}:{hi}:{samples:?}");
    Some(g.helper(key, |name| {
        let size = samples.len();
        let m = size - 1;
        let mut s = format!("fn {name}(x: f32) -> f32 {{\n  var t = array<f32, {size}>(\n");
        for (i, v) in samples.iter().enumerate() {
            let sep = if i + 1 == size { "" } else { "," };
            s.push_str(&format!("    {}{sep}\n", lit(*v)));
        }
        s.push_str("  );\n");
        s.push_str(&format!(
            "  if (x < {x0l}) {{ return t[0] + {lol} * (x - {x0l}); }}\n  \
             if (x > {x1l}) {{ return t[{m}] + {hil} * (x - {x1l}); }}\n  \
             let u = (x - {x0l}) / ({span}) * {m}.0;\n  let i0 = min(i32(floor(u)), {m1});\n  \
             return mix(t[i0], t[i0 + 1], u - f32(i0));\n}}\n",
            x0l = lit(x0),
            x1l = lit(x1),
            lol = lit(lo),
            hil = lit(hi),
            span = lit((x1 - x0).max(1e-12)),
            m1 = m - 1
        ));
        s
    }))
}

fn curves(g: &mut Gen, n: usize) -> Outs {
    let kind = g.node(n).kind.clone();
    let Some(data) = g.node(n).curves.clone() else {
        g.unsupported(format!("{kind} without curve data"));
        return vec![None; g.node(n).outputs.len()];
    };
    let list: Vec<Value> = data.get("curves").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let by_name = |name: &str, fallback: usize| -> Option<Value> {
        list.iter()
            .find(|c| c.get("name").and_then(|v| v.as_str()) == Some(name))
            .or_else(|| list.get(fallback))
            .cloned()
    };
    match kind.as_str() {
        "CURVE_FLOAT" => {
            let f = g.input(n, "Factor", Ty::Float);
            let v = g.input(n, "Value", Ty::Float);
            let v = g.bind("f", v, Ty::Float);
            let Some(c) = list.first().cloned().and_then(|c| curve_fn(g, &c)) else {
                g.unsupported("CURVE_FLOAT curve data");
                return vec![None];
            };
            vec![Some(g.bind("f", format!("mix({}, {c}({}), {f})", v.code, v.code), Ty::Float))]
        }
        "CURVE_VEC" => {
            let f = g.input(n, "Fac", Ty::Float);
            let v = g.input(n, "Vector", Ty::Vec3);
            let v = g.bind("v", v, Ty::Vec3);
            let mut comps = Vec::new();
            for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
                let c = by_name(name, i).and_then(|c| curve_fn(g, &c));
                comps.push(match c {
                    Some(c) => format!("{c}({}.{})", v.code, name.to_lowercase()),
                    None => format!("{}.{}", v.code, name.to_lowercase()),
                });
            }
            let e = format!("mix({}, vec3f({}, {}, {}), {f})", v.code, comps[0], comps[1], comps[2]);
            vec![Some(g.bind("v", e, Ty::Vec3))]
        }
        _ => {
            let f = g.input(n, "Fac", Ty::Float);
            let col = g.input(n, "Color", Ty::Color);
            let col = g.bind("c", col, Ty::Color);
            let black = rgba(data.get("black_level").unwrap_or(&Value::Null));
            let white = data.get("white_level").map(rgba).unwrap_or([1.0; 4]);
            let mul: Vec<f64> = (0..3).map(|i| 1.0 / (white[i] - black[i]).max(1e-12)).collect();
            let combined = by_name("C", 3).and_then(|c| curve_fn(g, &c));
            let mut comps = Vec::new();
            for (i, name) in ["R", "G", "B"].iter().enumerate() {
                let ch = ["r", "g", "b"][i];
                let x = format!("(({}.{ch} - {}) * {})", col.code, lit(black[i]), lit(mul[i]));
                let x = match &combined {
                    Some(c) => format!("{c}({x})"),
                    None => x,
                };
                let c = by_name(name, i).and_then(|c| curve_fn(g, &c));
                comps.push(match c {
                    Some(c) => format!("{c}({x})"),
                    None => x,
                });
            }
            let e = format!(
                "vec4f(mix({c}.rgb, vec3f({}, {}, {}), {f}), {c}.a)",
                comps[0],
                comps[1],
                comps[2],
                c = col.code
            );
            vec![Some(g.bind("c", e, Ty::Color))]
        }
    }
}

fn separate_color(g: &mut Gen, n: usize) -> Outs {
    let kind = g.node(n).kind.clone();
    let mode = match kind.as_str() {
        "SEPRGB" => "RGB".to_string(),
        "SEPHSV" => "HSV".to_string(),
        _ => g.node(n).prop_str("mode").unwrap_or("RGB").to_string(),
    };
    let key = match kind.as_str() {
        "SEPRGB" => "Image",
        _ => "Color",
    };
    let c = g.input_opt(n, key, Ty::Color);
    let e = match mode.as_str() {
        "HSV" => format!("mks_rgb_to_hsv(({c}).rgb)"),
        "HSL" => format!("mks_rgb_to_hsl(({c}).rgb)"),
        _ => format!("({c}).rgb"),
    };
    let v = g.bind("v", e, Ty::Vec3);
    let mut outs: Outs = vec![None; g.node(n).outputs.len()];
    for (i, o) in outs.iter_mut().enumerate().take(3) {
        *o = Some(Expr::new(format!("{}.{}", v.code, ["x", "y", "z"][i]), Ty::Float));
    }
    outs
}

fn combine_color(g: &mut Gen, n: usize) -> Outs {
    let kind = g.node(n).kind.clone();
    let mode = match kind.as_str() {
        "COMBRGB" => "RGB".to_string(),
        "COMBHSV" => "HSV".to_string(),
        _ => g.node(n).prop_str("mode").unwrap_or("RGB").to_string(),
    };
    let keys = match kind.as_str() {
        "COMBRGB" => ["R", "G", "B"],
        "COMBHSV" => ["H", "S", "V"],
        _ => ["Red", "Green", "Blue"],
    };
    let a = g.input_opt(n, keys[0], Ty::Float);
    let bb = g.input_opt(n, keys[1], Ty::Float);
    let c = g.input_opt(n, keys[2], Ty::Float);
    let e = match mode.as_str() {
        "HSV" => format!("vec4f(mks_hsv_to_rgb(vec3f({a}, {bb}, {c})), 1.0)"),
        "HSL" => format!("vec4f(mks_hsl_to_rgb(vec3f({a}, {bb}, {c})), 1.0)"),
        _ => format!("vec4f({a}, {bb}, {c}, 1.0)"),
    };
    vec![Some(g.bind("c", e, Ty::Color))]
}

// --------------------------------------------------------------------------------------------------- Vectors
fn mapping(g: &mut Gen, n: usize) -> Outs {
    let t = g.node(n).prop_str("vector_type").unwrap_or("POINT").to_string();
    let v = g.input(n, "Vector", Ty::Vec3);
    let loc = g.input(n, "Location", Ty::Vec3);
    let rot = g.input(n, "Rotation", Ty::Vec3);
    let sc = g.input(n, "Scale", Ty::Vec3);
    let e = match t.as_str() {
        "TEXTURE" => format!("mks_mapping_texture({v}, {loc}, {rot}, {sc})"),
        "VECTOR" => format!("mks_mapping_vector({v}, {rot}, {sc})"),
        "NORMAL" => format!("mks_mapping_normal({v}, {rot}, {sc})"),
        _ => format!("mks_mapping_point({v}, {loc}, {rot}, {sc})"),
    };
    vec![Some(g.bind("v", e, Ty::Vec3))]
}

fn vect_transform(g: &mut Gen, n: usize) -> Outs {
    let t = g.node(n).prop_str("vector_type").unwrap_or("VECTOR").to_string();
    let from = g.node(n).prop_str("convert_from").unwrap_or("WORLD").to_string();
    let to = g.node(n).prop_str("convert_to").unwrap_or("OBJECT").to_string();
    let v = g.input(n, "Vector", Ty::Vec3);
    let w = if t == "POINT" { "1.0" } else { "0.0" };
    let mut e = v;
    if from == "CAMERA" {
        e = format!("(({e}) * vec3f(1.0, 1.0, -1.0))");
    }
    if from != to {
        let to_world = match from.as_str() {
            "OBJECT" => format!("(s.object * vec4f({e}, {w})).xyz"),
            "CAMERA" => format!("(s.camera * vec4f({e}, {w})).xyz"),
            _ => e.clone(),
        };
        e = match to.as_str() {
            "OBJECT" => format!("(s.object_inv * vec4f({to_world}, {w})).xyz"),
            "CAMERA" => format!("(s.camera_inv * vec4f({to_world}, {w})).xyz"),
            _ => to_world,
        };
    }
    if t == "NORMAL" {
        e = format!("mks_safe_normalize({e})");
    }
    if to == "CAMERA" {
        e = format!("(({e}) * vec3f(1.0, 1.0, -1.0))");
    }
    vec![Some(g.bind("v", e, Ty::Vec3))]
}

fn vector_rotate(g: &mut Gen, n: usize) -> Outs {
    let t = g.node(n).prop_str("rotation_type").unwrap_or("AXIS_ANGLE").to_string();
    let inv = if g.node(n).prop_bool("invert").unwrap_or(false) { "1.0" } else { "0.0" };
    let v = g.input(n, "Vector", Ty::Vec3);
    let c = g.input(n, "Center", Ty::Vec3);
    let e = match t.as_str() {
        "EULER_XYZ" => {
            let r = g.input(n, "Rotation", Ty::Vec3);
            format!("mks_vector_rotate_euler({v}, {c}, {r}, {inv})")
        }
        "X_AXIS" | "Y_AXIS" | "Z_AXIS" => {
            let a = g.input(n, "Angle", Ty::Float);
            let axis = match t.as_str() {
                "X_AXIS" => "vec3f(1.0, 0.0, 0.0)",
                "Y_AXIS" => "vec3f(0.0, 1.0, 0.0)",
                _ => "vec3f(0.0, 0.0, 1.0)",
            };
            format!("mks_vector_rotate_axis_angle({v}, {c}, {axis}, {a}, {inv})")
        }
        _ => {
            let axis = g.input(n, "Axis", Ty::Vec3);
            let a = g.input(n, "Angle", Ty::Float);
            format!("mks_vector_rotate_axis_angle({v}, {c}, {axis}, {a}, {inv})")
        }
    };
    vec![Some(g.bind("v", e, Ty::Vec3))]
}

fn normal_node(g: &mut Gen, n: usize) -> Outs {
    let dir = crate::codegen::const_of(&g.node(n).outputs[0].value, "VECTOR", Ty::Vec3);
    let nin = g.input(n, "Normal", Ty::Vec3);
    let d = g.bind("v", format!("mks_safe_normalize({dir})"), Ty::Vec3);
    let dot = g.bind("f", format!("-dot({nin}, {})", d.code), Ty::Float);
    set(g, n, vec![("Normal", d), ("Dot", dot)])
}

// --------------------------------------------------------------------------------------------------- Inputs
/// Object-space normal (Texture Coordinate Normal, Box projection).
fn object_normal() -> String {
    "mks_safe_normalize((s.object_inv * vec4f(s.N, 0.0)).xyz)".into()
}

fn tex_coord(g: &mut Gen, n: usize) -> Outs {
    if g.node(n).prop_bool("from_instancer").unwrap_or(false) {
        g.unsupported("TEX_COORD from_instancer");
    }
    if g.node(n).props.get("object").map(|v| !v.is_null()).unwrap_or(false) {
        g.unsupported("TEX_COORD with an object");
    }
    let mut pairs = Vec::new();
    if g.output_used(n, "Generated") {
        let gc = g.src_generated();
        pairs.push(("Generated", Expr::new(gc, Ty::Vec3)));
    }
    if g.output_used(n, "Normal") {
        pairs.push(("Normal", g.bind("v", object_normal(), Ty::Vec3)));
    }
    if g.output_used(n, "UV") {
        let uv = g.uv_attr("");
        pairs.push(("UV", g.bind("v", uv, Ty::Vec3)));
    }
    if g.output_used(n, "Object") {
        let p = g.src_p();
        pairs.push(("Object", g.bind("v", format!("(s.object_inv * vec4f({p}, 1.0)).xyz"), Ty::Vec3)));
    }
    if g.output_used(n, "Camera") {
        let p = g.src_p();
        pairs.push((
            "Camera",
            g.bind("v", format!("(s.camera_inv * vec4f({p}, 1.0)).xyz * vec3f(1.0, 1.0, -1.0)"), Ty::Vec3),
        ));
    }
    if g.output_used(n, "Window") {
        let scr = g.src_screen();
        pairs.push(("Window", Expr::new(format!("vec3f({scr}, 0.0)"), Ty::Vec3)));
    }
    if g.output_used(n, "Reflection") {
        pairs.push(("Reflection", g.bind("v", "reflect(-s.I, s.N)", Ty::Vec3)));
    }
    set(g, n, pairs)
}

fn attribute(g: &mut Gen, n: usize) -> Outs {
    let t = g.node(n).prop_str("attribute_type").unwrap_or("GEOMETRY").to_string();
    let name = g.node(n).prop_str("attribute_name").unwrap_or("").to_string();
    let a = if t == "GEOMETRY" {
        g.src_attr(&format!("attr:{name}"))
    } else {
        // Object, Instancer, View Layer: the drawn object's value, in the parameter slot graph.rs gave the node
        let Some(slot) = g.node(n).outputs.first().and_then(|s| s.anim) else {
            g.unsupported(format!("ATTRIBUTE {t}"));
            return vec![None; g.node(n).outputs.len()];
        };
        format!("mkr_param({}u)", g.sh.param_base + slot as u32)
    };
    let a = g.bind("a", a, Ty::Color);
    let c = &a.code;
    set(
        g,
        n,
        vec![
            ("Color", Expr::new(format!("vec4f({c}.xyz, 1.0)"), Ty::Color)),
            ("Vector", Expr::new(format!("{c}.xyz"), Ty::Vec3)),
            ("Fac", Expr::new(format!("mks_avg3({c}.xyz)"), Ty::Float)),
            ("Alpha", Expr::new(format!("{c}.w"), Ty::Float)),
        ],
    )
}

/// Blender's default tangent: radial about the object's Z through the Generated coordinates.
fn radial_tangent(g: &mut Gen, axis: &str) -> String {
    let gc = g.src_generated();
    let t = match axis {
        "X" => format!("vec3f(0.0, -(({gc}).z - 0.5), ({gc}).y - 0.5)"),
        "Y" => format!("vec3f(-(({gc}).x - 0.5), 0.0, ({gc}).z - 0.5)"),
        _ => format!("vec3f(-(({gc}).y - 0.5), ({gc}).x - 0.5, 0.0)"),
    };
    format!("mks_safe_normalize(cross(s.N, mks_safe_normalize(cross((s.object * vec4f({t}, 0.0)).xyz, s.N))))")
}

fn geometry(g: &mut Gen, n: usize) -> Outs {
    let wants_tangent = g.output_used(n, "Tangent");
    let p = g.src_p();
    let mut pairs = vec![
        ("Position", Expr::new(p, Ty::Vec3)),
        ("Normal", Expr::new("s.N", Ty::Vec3)),
        ("True Normal", Expr::new("s.Ng", Ty::Vec3)),
        ("Incoming", Expr::new("s.I", Ty::Vec3)),
        ("Backfacing", Expr::new("s.backfacing", Ty::Float)),
        ("Pointiness", Expr::new("0.5", Ty::Float)),
        ("Random Per Island", Expr::new("0.0", Ty::Float)),
        ("Parametric", Expr::new("vec3f(0.0)", Ty::Vec3)),
    ];
    if wants_tangent {
        let t = radial_tangent(g, "Z");
        pairs.push(("Tangent", Expr::new(t, Ty::Vec3)));
    }
    set(g, n, pairs)
}

fn tangent(g: &mut Gen, n: usize) -> Outs {
    let dt = g.node(n).prop_str("direction_type").unwrap_or("RADIAL").to_string();
    let e = if dt == "UV_MAP" {
        let map = g.node(n).prop_str("uv_map").unwrap_or("").to_string();
        let t = g.src_attr(&format!("tangent:{map}"));
        format!("mks_safe_normalize(cross(s.N, mks_safe_normalize(cross(({t}).xyz, s.N))))")
    } else {
        let axis = g.node(n).prop_str("axis").unwrap_or("Z").to_string();
        radial_tangent(g, &axis)
    };
    vec![Some(g.bind("v", e, Ty::Vec3))]
}

fn light_path(g: &mut Gen, n: usize) -> Outs {
    let is = |t: &str| Expr::new(format!("select(0.0, 1.0, s.ray_type == {t})"), Ty::Float);
    let glossy = is("MKR_RAY_GLOSSY");
    let p = g.src_p();
    set(
        g,
        n,
        vec![
            ("Is Camera Ray", is("MKR_RAY_CAMERA")),
            ("Is Shadow Ray", is("MKR_RAY_SHADOW")),
            ("Is Diffuse Ray", is("MKR_RAY_DIFFUSE")),
            ("Is Glossy Ray", glossy.clone()),
            ("Is Singular Ray", glossy.clone()),
            ("Is Reflection Ray", glossy.clone()),
            ("Is Transmission Ray", glossy),
            ("Ray Length", Expr::new(format!("distance({p}, s.camera[3].xyz)"), Ty::Float)),
            ("Ray Depth", Expr::new("0.0", Ty::Float)),
            ("Diffuse Depth", Expr::new("0.0", Ty::Float)),
            ("Glossy Depth", Expr::new("0.0", Ty::Float)),
            ("Transparent Depth", Expr::new("0.0", Ty::Float)),
            ("Transmission Depth", Expr::new("0.0", Ty::Float)),
        ],
    )
}

// --------------------------------------------------------------------------------------------------- Normals
fn normal_map(g: &mut Gen, n: usize) -> Outs {
    let space = g.node(n).prop_str("space").unwrap_or("TANGENT").to_string();
    let strength = g.input(n, "Strength", Ty::Float);
    let c = g.input(n, "Color", Ty::Color);
    let c = g.bind("c", c, Ty::Color);
    // Object / world space: the decoded vector mixes in unnormalised (measured: lab normal_map_world); object space
    // turns it as a normal (inverse transpose of the object matrix).
    let normal_matrix = "transpose(mat3x3f(s.object_inv[0].xyz, s.object_inv[1].xyz, s.object_inv[2].xyz))";
    let mapped = match space.as_str() {
        "OBJECT" => format!("({normal_matrix} * (({0}).rgb * 2.0 - 1.0))", c.code),
        "WORLD" => format!("(({0}).rgb * 2.0 - 1.0)", c.code),
        "BLENDER_OBJECT" => {
            format!("({normal_matrix} * (vec3f(2.0, -2.0, -2.0) * ({0}).rgb - vec3f(1.0, -1.0, -1.0)))", c.code)
        }
        "BLENDER_WORLD" => format!("(vec3f(2.0, -2.0, -2.0) * ({0}).rgb - vec3f(1.0, -1.0, -1.0))", c.code),
        _ => {
            let map = g.node(n).prop_str("uv_map").unwrap_or("").to_string();
            let t = g.src_attr(&format!("tangent:{map}"));
            format!("mks_normal_map_tangent({t}, ({}).rgb * 2.0 - 1.0, s.N, s.backfacing, s.object)", c.code)
        }
    };
    let e = format!("normalize(mix(s.N, {mapped}, max({strength}, 0.0)))");
    vec![Some(g.bind("v", e, Ty::Vec3))]
}

fn bump(g: &mut Gen, n: usize) -> Outs {
    let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
    if g.is_alpha() || g.variant != 0 {
        return vec![Some(g.bind("v", nrm, Ty::Vec3))];
    }
    let strength = g.input(n, "Strength", Ty::Float);
    let dist = g.input(n, "Distance", Ty::Float);
    let invert = g.node(n).prop_bool("invert").unwrap_or(false);
    let h = g.input(n, "Height", Ty::Float);
    let h = g.bind("f", h, Ty::Float);
    let hx = g.input_in_variant(n, "Height", Ty::Float, 1);
    let hx = g.bind("f", hx, Ty::Float);
    let hy = g.input_in_variant(n, "Height", Ty::Float, 2);
    let hy = g.bind("f", hy, Ty::Float);
    g.sh.uses_derivatives = true;
    g.top_once("mks_dPdx", "let mks_dPdx: vec3f = dpdx(s.P) * MKS_BUMP_FILTER;".into());
    g.top_once("mks_dPdy", "let mks_dPdy: vec3f = dpdy(s.P) * MKS_BUMP_FILTER;".into());
    let e = format!(
        "mks_bump({strength}, {dist}, {}, {}, {}, {nrm}, {}, s.backfacing, mks_dPdx, mks_dPdy)",
        h.code,
        hx.code,
        hy.code,
        if invert { "1.0" } else { "0.0" }
    );
    vec![Some(g.bind("v", e, Ty::Vec3))]
}

// --------------------------------------------------------------------------------------------------- Textures
/// The vector of a procedural texture: linked, or Generated coordinates; with the node's texture mapping.
fn tex_vector(g: &mut Gen, n: usize) -> String {
    let v = match g.linked_input(n, "Vector", Ty::Vec3) {
        Some(v) => v,
        None => g.src_generated(),
    };
    texture_mapping(g, n, v)
}

/// Apply a texture node's (hidden) TexMapping when it is not the identity.
fn texture_mapping(g: &mut Gen, n: usize, v: String) -> String {
    let Some(tm) = g.node(n).props.get("texture_mapping").cloned() else {
        return v;
    };
    let get3 = |k: &str, d: [f64; 3]| -> [f64; 3] {
        tm.get(k)
            .and_then(|x| x.as_array())
            .map(|a| [0, 1, 2].map(|i| a.get(i).and_then(|x| x.as_f64()).unwrap_or(d[i])))
            .unwrap_or(d)
    };
    let loc = get3("translation", [0.0; 3]);
    let rot = get3("rotation", [0.0; 3]);
    let sc = get3("scale", [1.0; 3]);
    let use_min = tm.get("use_min").and_then(|x| x.as_bool()).unwrap_or(false);
    let use_max = tm.get("use_max").and_then(|x| x.as_bool()).unwrap_or(false);
    let identity = loc == [0.0; 3] && rot == [0.0; 3] && sc == [1.0; 3] && !use_min && !use_max;
    if identity {
        return v;
    }
    let vt = tm.get("vector_type").and_then(|x| x.as_str()).unwrap_or("POINT").to_string();
    let l = |a: [f64; 3]| format!("vec3f({}, {}, {})", lit(a[0]), lit(a[1]), lit(a[2]));
    let mut e = match vt.as_str() {
        "TEXTURE" => format!("mks_mapping_texture({v}, {}, {}, {})", l(loc), l(rot), l(sc)),
        "VECTOR" => format!("mks_mapping_vector({v}, {}, {})", l(rot), l(sc)),
        "NORMAL" => format!("mks_mapping_normal({v}, {}, {})", l(rot), l(sc)),
        _ => format!("mks_mapping_point({v}, {}, {}, {})", l(loc), l(rot), l(sc)),
    };
    if use_min {
        e = format!("max({e}, {})", l(get3("min", [0.0; 3])));
    }
    if use_max {
        e = format!("min({e}, {})", l(get3("max", [1.0; 3])));
    }
    g.bind("v", e, Ty::Vec3).code
}

fn interp_code(s: Option<&str>) -> u32 {
    match s {
        Some("Closest") => 1,
        Some("Cubic") => 2,
        Some("Smart") => 3,
        _ => 0,
    }
}

fn ext_code(s: Option<&str>) -> u32 {
    match s {
        Some("EXTEND") => 1,
        Some("CLIP") => 2,
        Some("MIRROR") => 3,
        _ => 0,
    }
}

/// (image index, stored premultiplied, alpha must not touch colour, alpha forced to 1)
fn image_info(g: &mut Gen, n: usize) -> Option<(usize, bool, bool, bool)> {
    let name = g.node(n).image.clone()?;
    let Some(&idx) = g.sh.image_index.get(name.as_str()) else {
        g.warn(format!("{}: image {name:?} not in the images list (drawn magenta)", g.node(n).path));
        return None;
    };
    let im = &g.sh.images[idx];
    let data = crate::is_data_colorspace(&im.colorspace);
    let clear = data || matches!(im.alpha_mode.as_str(), "CHANNEL_PACKED" | "NONE");
    if !crate::known_colorspace(&im.colorspace) {
        let cs = im.colorspace.clone();
        g.unsupported(format!("image colour space {cs:?}"));
    }
    Some((idx, im.premultiplied, clear, im.alpha_mode == "NONE"))
}

fn image_sample(g: &mut Gen, idx: usize, uv: &str, interp: u32, ext: u32) -> String {
    g.sh.used_images.insert(idx);
    let f = match g.mode {
        crate::codegen::Mode::Alpha { lod: true } => "mkr_image_lod",
        _ => "mkr_image",
    };
    format!("{f}({idx}u, {uv}, {interp}u, {ext}u)")
}

fn tex_image(g: &mut Gen, n: usize) -> Outs {
    let alpha_used = g.sh.alpha_used.contains(&n);
    let Some((idx, premul, clear, no_alpha)) = image_info(g, n) else {
        if g.node(n).image.is_none() {
            // An Image Texture without an image: transparent black.
            return set(g, n, vec![("Color", Expr::new("vec4f(0.0)", Ty::Color)), ("Alpha", Expr::new("0.0", Ty::Float))]);
        }
        return set(
            g,
            n,
            vec![("Color", Expr::new("vec4f(1.0, 0.0, 1.0, 1.0)", Ty::Color)), ("Alpha", Expr::new("1.0", Ty::Float))],
        );
    };
    let node = g.node(n);
    let interp = interp_code(node.prop_str("interpolation"));
    let ext = ext_code(node.prop_str("extension"));
    let projection = node.prop_str("projection").unwrap_or("FLAT").to_string();
    let v = match g.linked_input(n, "Vector", Ty::Vec3) {
        Some(v) => v,
        None => g.uv_attr(""),
    };
    let v = texture_mapping(g, n, v);
    let v = g.bind("v", v, Ty::Vec3);
    let t = match projection.as_str() {
        "BOX" => {
            let blend = g.node(n).prop_f64("projection_blend").unwrap_or(0.0);
            let w = g.bind("v", format!("mks_box_weights({}, {})", object_normal(), lit(blend)), Ty::Vec3);
            let on = g.bind("v", object_normal(), Ty::Vec3);
            let (c, onc, wc) = (v.code.clone(), on.code.clone(), w.code.clone());
            let sx = image_sample(g, idx, &format!("mks_box_uv_x({c}, {onc})"), interp, ext);
            let sy = image_sample(g, idx, &format!("mks_box_uv_y({c}, {onc})"), interp, ext);
            let sz = image_sample(g, idx, &format!("mks_box_uv_z({c}, {onc})"), interp, ext);
            format!("{wc}.x * {sx} + {wc}.y * {sy} + {wc}.z * {sz}")
        }
        "SPHERE" => image_sample(g, idx, &format!("mks_map_to_sphere(({} - 0.5) * 2.0)", v.code), interp, ext),
        "TUBE" => image_sample(g, idx, &format!("mks_map_to_tube(({} - 0.5) * 2.0)", v.code), interp, ext),
        _ => image_sample(g, idx, &format!("{}.xy", v.code), interp, ext),
    };
    let t = g.bind("t", t, Ty::Color);
    let color = g.bind(
        "c",
        format!("mks_image_color({}, {}, {}, {})", t.code, b(premul), b(clear), b(alpha_used)),
        Ty::Color,
    );
    let alpha = if no_alpha { Expr::new("1.0", Ty::Float) } else { Expr::new(format!("{}.a", t.code), Ty::Float) };
    set(g, n, vec![("Color", color), ("Alpha", alpha)])
}

fn tex_environment(g: &mut Gen, n: usize) -> Outs {
    let alpha_used = g.sh.alpha_used.contains(&n);
    let Some((idx, premul, clear, _)) = image_info(g, n) else {
        if g.node(n).image.is_none() {
            return set(g, n, vec![("Color", Expr::new("vec4f(0.0)", Ty::Color))]);
        }
        return set(g, n, vec![("Color", Expr::new("vec4f(1.0, 0.0, 1.0, 1.0)", Ty::Color))]);
    };
    let interp = interp_code(g.node(n).prop_str("interpolation"));
    let mirror = g.node(n).prop_str("projection") == Some("MIRROR_BALL");
    let v = g.linked_input(n, "Vector", Ty::Vec3).unwrap_or_else(|| "(-s.I)".into());
    let v = texture_mapping(g, n, v);
    let uv = if mirror { format!("mks_env_mirror_ball({v})") } else { format!("mks_env_equirect({v})") };
    let t = image_sample(g, idx, &uv, interp, 0);
    let t = g.bind("t", t, Ty::Color);
    let c = g.bind(
        "c",
        format!("mks_image_color({}, {}, {}, {})", t.code, b(premul), b(clear), b(alpha_used)),
        Ty::Color,
    );
    set(g, n, vec![("Color", c)])
}

fn tex_noise(g: &mut Gen, n: usize) -> Outs {
    let d = dims(g, n, "noise_dimensions");
    let kind = enum_index(
        g,
        n,
        "noise_type",
        &["MULTIFRACTAL", "RIDGED_MULTIFRACTAL", "HYBRID_MULTIFRACTAL", "FBM", "HETERO_TERRAIN"],
        3,
    );
    let normalize = g.node(n).prop_bool("normalize").unwrap_or(true);
    let v = tex_vector(g, n);
    let w = if d == 1 || d == 4 { g.input_f(n, "W") } else { "0.0".into() };
    let p = pack(d, &v, &w);
    let sc = g.input_f(n, "Scale");
    let det = g.input_f(n, "Detail");
    let r = g.input_f(n, "Roughness");
    let la = g.input_f(n, "Lacunarity");
    let off = g.input_f(n, "Offset");
    let gain = g.input_f(n, "Gain");
    let dis = g.input_f(n, "Distortion");
    let t = g.bind(
        "t",
        format!("mks_tex_noise({p}, {d}u, {kind}u, {sc}, {det}, {r}, {la}, {off}, {gain}, {dis}, {})", b(normalize)),
        Ty::Color,
    );
    set(
        g,
        n,
        vec![
            ("Fac", Expr::new(format!("{}.x", t.code), Ty::Float)),
            ("Color", Expr::new(format!("vec4f({}.yzw, 1.0)", t.code), Ty::Color)),
        ],
    )
}

fn tex_voronoi(g: &mut Gen, n: usize) -> Outs {
    let d = dims(g, n, "voronoi_dimensions");
    let feature =
        enum_index(g, n, "feature", &["F1", "F2", "SMOOTH_F1", "DISTANCE_TO_EDGE", "N_SPHERE_RADIUS"], 0);
    let metric = enum_index(g, n, "distance", &["EUCLIDEAN", "MANHATTAN", "CHEBYCHEV", "MINKOWSKI"], 0);
    let normalize = g.node(n).prop_bool("normalize").unwrap_or(false);
    let v = tex_vector(g, n);
    let w = if d == 1 || d == 4 { g.input_f(n, "W") } else { "0.0".into() };
    let p = pack(d, &v, &w);
    let sc = g.input_f(n, "Scale");
    let det = g.input_f(n, "Detail");
    let r = g.input_f(n, "Roughness");
    let la = g.input_f(n, "Lacunarity");
    let sm = g.input_f(n, "Smoothness");
    let ex = g.input_f(n, "Exponent");
    let ra = g.input_f(n, "Randomness");
    let name = g.fresh("vor");
    g.stmt(format!(
        "let {name}: MksVoronoi = mks_tex_voronoi({p}, {d}u, {feature}u, {metric}u, {sc}, {det}, {r}, {la}, {sm}, {ex}, {ra}, {});",
        b(normalize)
    ));
    let pos = if d == 2 {
        format!("vec3f({name}.position.xy, 0.0)")
    } else if d == 1 {
        "vec3f(0.0)".to_string()
    } else {
        format!("{name}.position.xyz")
    };
    set(
        g,
        n,
        vec![
            ("Distance", Expr::new(format!("{name}.distance"), Ty::Float)),
            ("Color", Expr::new(format!("vec4f({name}.color, 1.0)"), Ty::Color)),
            ("Position", Expr::new(pos, Ty::Vec3)),
            ("W", Expr::new(format!("{name}.position.w"), Ty::Float)),
            ("Radius", Expr::new(format!("{name}.color.x"), Ty::Float)),
        ],
    )
}

fn tex_wave(g: &mut Gen, n: usize) -> Outs {
    let wt = enum_index(g, n, "wave_type", &["BANDS", "RINGS"], 0);
    let dir = if wt == 0 {
        enum_index(g, n, "bands_direction", &["X", "Y", "Z", "DIAGONAL"], 0)
    } else {
        enum_index(g, n, "rings_direction", &["X", "Y", "Z", "SPHERICAL"], 0)
    };
    let profile = enum_index(g, n, "wave_profile", &["SIN", "SAW", "TRI"], 0);
    let v = tex_vector(g, n);
    let sc = g.input_f(n, "Scale");
    let dis = g.input_f(n, "Distortion");
    let det = g.input_f(n, "Detail");
    let ds = g.input_f(n, "Detail Scale");
    let dr = g.input_f(n, "Detail Roughness");
    let ph = g.input_f(n, "Phase Offset");
    let f = g.bind(
        "f",
        format!("mks_tex_wave({v}, {sc}, {dis}, {det}, {ds}, {dr}, {ph}, {wt}u, {dir}u, {profile}u)"),
        Ty::Float,
    );
    let c = Expr::new(format!("vec4f(vec3f({}), 1.0)", f.code), Ty::Color);
    set(g, n, vec![("Fac", f), ("Color", c)])
}

fn tex_brick(g: &mut Gen, n: usize) -> Outs {
    let co = tex_vector(g, n);
    let c1 = g.input_c(n, "Color1");
    let c2 = g.input_c(n, "Color2");
    let mortar = g.input_c(n, "Mortar");
    let sc = g.input_f(n, "Scale");
    let ms = g.input_f(n, "Mortar Size");
    let msm = g.input_f(n, "Mortar Smooth");
    let bias = g.input_f(n, "Bias");
    let bw = g.input_f(n, "Brick Width");
    let rh = g.input_f(n, "Row Height");
    let node = g.node(n);
    let off = node.prop_f64("offset").unwrap_or(0.5);
    let off_f = node.prop_f64("offset_frequency").unwrap_or(2.0) as i32;
    let sq = node.prop_f64("squash").unwrap_or(1.0);
    let sq_f = node.prop_f64("squash_frequency").unwrap_or(2.0) as i32;
    let bt = g.fresh("brick");
    g.stmt(format!(
        "let {bt}: vec2f = mks_brick({co} * {sc}, {ms}, {msm}, {bias}, {bw}, {rh}, {}, {off_f}, {}, {sq_f});",
        lit(off),
        lit(sq)
    ));
    let c1b = g.bind("c", c1, Ty::Color);
    let c2b = g.bind("c", c2, Ty::Color);
    let tinted = format!(
        "select((1.0 - {bt}.x) * {} + {bt}.x * {}, {}, {bt}.y == 1.0)",
        c1b.code, c2b.code, c1b.code
    );
    let col = g.bind("c", format!("mix({tinted}, {mortar}, {bt}.y)"), Ty::Color);
    set(g, n, vec![("Color", col), ("Fac", Expr::new(format!("{bt}.y"), Ty::Float))])
}

// --------------------------------------------------------------------------------------------------- Shaders
fn principled(g: &mut Gen, n: usize) -> Outs {
    if g.is_alpha() {
        if g.input_const(n, "Alpha").map(|a| a < 1.0).unwrap_or(true) {
            g.sh.transparent = true;
        }
        let a = g.input_f(n, "Alpha");
        return vec![Some(g.bind("cl", format!("mks_bsdf_transparent(vec3f(1.0 - clamp({a}, 0.0, 1.0)))"), Ty::Shader))];
    }
    let multi = g.node(n).prop_str("distribution") != Some("GGX");
    let base = g.input_c(n, "Base Color");
    let metallic = g.input_f(n, "Metallic");
    let rough = g.input_f(n, "Roughness");
    let ior = g.input_f(n, "IOR");
    let alpha = g.input_f(n, "Alpha");
    let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
    let ssw = g.input_f(n, "Subsurface Weight");
    let spec = g.input_f(n, "Specular IOR Level");
    let tint = g.input_c(n, "Specular Tint");
    let trans = g.input_f(n, "Transmission Weight");
    let cw = g.input_f(n, "Coat Weight");
    let cr = g.input_f(n, "Coat Roughness");
    let cior = g.input_f(n, "Coat IOR");
    let ctint = g.input_c(n, "Coat Tint");
    let cn = g.linked_input(n, "Coat Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
    let shw = g.input_f(n, "Sheen Weight");
    let shr = g.input_f(n, "Sheen Roughness");
    let sht = g.input_c(n, "Sheen Tint");
    let em = g.input_c(n, "Emission Color");
    let ems = g.input_f(n, "Emission Strength");
    if g.input_const(n, "Anisotropic").map(|v| v != 0.0).unwrap_or(true) {
        g.warn(format!("{}: anisotropy is not drawn (as EEVEE)", g.node(n).path));
    }
    let p = g.fresh("pr");
    g.stmt(format!(
        "let {p} = MksPrincipled(({base}).rgb, {metallic}, {rough}, {ior}, {alpha}, {nrm}, {ssw}, {spec}, ({tint}).rgb, {trans}, {cw}, {cr}, {cior}, ({ctint}).rgb, {cn}, {shw}, {shr}, ({sht}).rgb, ({em}).rgb, {ems});"
    ));
    vec![Some(g.bind("cl", format!("mks_bsdf_principled({p}, s.I, {})", b(multi)), Ty::Shader))]
}

fn eevee_specular(g: &mut Gen, n: usize) -> Outs {
    let transp = g.input_f(n, "Transparency");
    if g.is_alpha() {
        if g.input_const(n, "Transparency").map(|t| t > 0.0).unwrap_or(true) {
            g.sh.transparent = true;
        }
        return vec![Some(g.bind("cl", format!("mks_bsdf_transparent(vec3f({transp}))"), Ty::Shader))];
    }
    let base = g.input_c(n, "Base Color");
    let spec = g.input_c(n, "Specular");
    let rough = g.input_f(n, "Roughness");
    let emis = g.input_c(n, "Emissive Color");
    let nrm = g.linked_input(n, "Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
    let cc = g.input_f(n, "Clear Coat");
    let ccr = g.input_f(n, "Clear Coat Roughness");
    let ccn = g.linked_input(n, "Clear Coat Normal", Ty::Vec3).unwrap_or_else(|| "s.N".into());
    let e = format!(
        "mks_eevee_specular(({base}).rgb, ({spec}).rgb, {rough}, ({emis}).rgb, {transp}, {nrm}, {cc}, {ccr}, {ccn}, s.I)"
    );
    vec![Some(g.bind("cl", e, Ty::Shader))]
}
