//! WGSL generation for a flattened node tree: each node needed by the output becomes a few `let` statements;
//! links convert types as Blender's GPU code does.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use serde_json::Value;

use crate::ImageDesc;
use crate::graph::{FNode, Flat, Pass, Ty};

/// What function is being generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// mat_<id>: every closure.
    Surface,
    /// mat_<id>_alpha / _alpha_lod: only what decides transmittance and holdout; no derivatives.
    Alpha { lod: bool },
}

/// Bump re-evaluation variants: 0 at the point, 1 one pixel along screen x, 2 along screen y.
pub type Variant = u8;

#[derive(Clone, Debug)]
pub struct Expr {
    pub code: String,
    pub ty: Ty,
}

impl Expr {
    pub fn new(code: impl Into<String>, ty: Ty) -> Expr {
        Expr { code: code.into(), ty }
    }
}

/// State shared by the functions of one material.
pub struct Shared<'a> {
    pub flat: &'a Flat,
    pub images: &'a [ImageDesc],
    pub image_index: HashMap<&'a str, usize>,
    pub param_base: u32,
    pub pass_index: f64,
    pub attrs: Vec<String>,
    pub unsupported: BTreeSet<String>,
    pub warnings: BTreeSet<String>,
    pub uses_derivatives: bool,
    pub uses_shader_to_rgb: bool,
    /// Image Texture nodes whose Alpha output feeds the tree (Blender then outputs straight colour).
    pub alpha_used: HashSet<usize>,
    /// Module-scope helpers (colour ramp and curve tables), each emitted once.
    pub helpers: Vec<String>,
    pub helper_names: HashMap<String, String>,
    pub id: String,
    /// A transmittance source was emitted in the alpha function.
    pub transparent: bool,
    /// Images the code samples.
    pub used_images: BTreeSet<usize>,
    /// (node, output index) pairs linked into a node the output reaches.
    pub used_outputs: HashSet<(usize, usize)>,
}

/// One generated function.
pub struct Gen<'s, 'a> {
    pub sh: &'s mut Shared<'a>,
    pub mode: Mode,
    pub top: Vec<String>,
    pub top_keys: HashSet<String>,
    pub body: Vec<String>,
    memo: HashMap<(usize, Variant), Vec<Option<Expr>>>,
    visiting: HashSet<(usize, Variant)>,
    pub variant: Variant,
    tmp: usize,
}

pub fn lit(v: f64) -> String {
    let f = v as f32;
    if !f.is_finite() {
        return if f.is_nan() { "0.0".into() } else if f > 0.0 { "3.4e38".into() } else { "(-3.4e38)".into() };
    }
    let s = format!("{f:?}");
    if f < 0.0 || s.starts_with('-') { format!("({s})") } else { s }
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_bool().map(|b| b as i32 as f64))
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| num(x).unwrap_or(0.0)).collect(),
        other => num(other).map(|x| vec![x]).unwrap_or_default(),
    }
}

/// A socket's stored value as a WGSL constant of `ty`.
pub fn const_of(v: &Value, raw_ty: &str, ty: Ty) -> String {
    let n = nums(v);
    let src_ty = match raw_ty {
        "VECTOR" | "ROTATION" => Ty::Vec3,
        "RGBA" => Ty::Color,
        "SHADER" => Ty::Shader,
        _ if n.len() >= 4 => Ty::Color,
        _ if n.len() == 3 => Ty::Vec3,
        _ => Ty::Float,
    };
    let g = |i: usize| n.get(i).copied().unwrap_or(0.0);
    let code = match src_ty {
        Ty::Float => lit(g(0)),
        Ty::Vec3 => format!("vec3f({}, {}, {})", lit(g(0)), lit(g(1)), lit(g(2))),
        Ty::Color => format!(
            "vec4f({}, {}, {}, {})",
            lit(g(0)),
            lit(g(1)),
            lit(g(2)),
            lit(if n.len() >= 4 { g(3) } else { 1.0 })
        ),
        Ty::Shader => "mks_closures_none()".into(),
    };
    convert(&Expr::new(code, src_ty), ty)
}

/// Blender's implicit socket conversions.
pub fn convert(e: &Expr, to: Ty) -> String {
    let c = &e.code;
    match (e.ty, to) {
        (a, b) if a == b => c.clone(),
        (Ty::Float, Ty::Vec3) => format!("vec3f({c})"),
        (Ty::Float, Ty::Color) => format!("vec4f(vec3f({c}), 1.0)"),
        (Ty::Float, Ty::Shader) => format!("mks_closure_from_color(vec3f({c}))"),
        (Ty::Vec3, Ty::Float) => format!("mks_avg3({c})"),
        (Ty::Vec3, Ty::Color) => format!("vec4f({c}, 1.0)"),
        (Ty::Vec3, Ty::Shader) => format!("mks_closure_from_color({c})"),
        // EEVEE averages the channels (measured: lab color_to_float); the RGB to BW node uses luminance.
        (Ty::Color, Ty::Float) => format!("mks_avg3(({c}).rgb)"),
        (Ty::Color, Ty::Vec3) => format!("({c}).rgb"),
        (Ty::Color, Ty::Shader) => format!("mks_closure_from_color(({c}).rgb)"),
        (Ty::Shader, Ty::Float) => "0.0".into(),
        (Ty::Shader, Ty::Vec3) => "vec3f(0.0)".into(),
        (Ty::Shader, Ty::Color) => "vec4f(0.0)".into(),
        _ => c.clone(),
    }
}

pub fn zero(ty: Ty) -> String {
    match ty {
        Ty::Float => "0.0".into(),
        Ty::Vec3 => "vec3f(0.0)".into(),
        Ty::Color => "vec4f(0.0)".into(),
        Ty::Shader => "mks_closures_none()".into(),
    }
}

impl<'s, 'a> Gen<'s, 'a> {
    pub fn new(sh: &'s mut Shared<'a>, mode: Mode) -> Self {
        Gen {
            sh,
            mode,
            top: Vec::new(),
            top_keys: HashSet::new(),
            body: Vec::new(),
            memo: HashMap::new(),
            visiting: HashSet::new(),
            variant: 0,
            tmp: 0,
        }
    }

    pub fn node(&self, n: usize) -> &'a FNode {
        &self.sh.flat.nodes[n]
    }

    /// A fresh local name.
    pub fn fresh(&mut self, stem: &str) -> String {
        self.tmp += 1;
        let v = if self.variant == 0 { String::new() } else { format!("d{}", self.variant) };
        format!("{stem}{v}_{}", self.tmp)
    }

    /// `let <name> = <code>;` -> name.
    pub fn bind(&mut self, stem: &str, code: impl AsRef<str>, ty: Ty) -> Expr {
        let name = self.fresh(stem);
        self.body.push(format!("  let {name}: {} = {};", ty.wgsl(), code.as_ref()));
        Expr::new(name, ty)
    }

    pub fn stmt(&mut self, s: impl Into<String>) {
        self.body.push(format!("  {}", s.into()));
    }

    /// Function-top statement (uniform control flow, before everything), emitted once per key.
    pub fn top_once(&mut self, key: &str, line: String) {
        if self.top_keys.insert(key.to_string()) {
            self.top.push(format!("  {line}"));
        }
    }

    pub fn unsupported(&mut self, what: impl Into<String>) {
        self.sh.unsupported.insert(what.into());
    }

    pub fn warn(&mut self, what: impl Into<String>) {
        self.sh.warnings.insert(what.into());
    }

    pub fn is_alpha(&self) -> bool {
        matches!(self.mode, Mode::Alpha { .. })
    }

    // ------------------------------------------------------------------------------------------- surface sources
    /// Screen derivative of a surface quantity for the current bump variant (`None` at the point itself).
    fn offset(&mut self, key: &str, expr: &str, ty: Ty) -> String {
        if self.variant == 0 || self.is_alpha() {
            return expr.to_string();
        }
        self.sh.uses_derivatives = true;
        let f = if self.variant == 1 { "dpdx" } else { "dpdy" };
        let name = format!("mks_b{}_{key}", self.variant);
        self.top_once(&name, format!("let {name}: {} = {expr} + {f}({expr}) * MKS_BUMP_FILTER;", ty.wgsl()));
        name
    }

    pub fn src_p(&mut self) -> String {
        self.offset("P", "s.P", Ty::Vec3)
    }

    pub fn src_generated(&mut self) -> String {
        self.offset("gen", "s.generated", Ty::Vec3)
    }

    pub fn src_screen(&mut self) -> String {
        if self.variant == 0 || self.is_alpha() {
            return "s.screen".into();
        }
        self.sh.uses_derivatives = true;
        let f = if self.variant == 1 { "dpdx" } else { "dpdy" };
        let name = format!("mks_b{}_screen", self.variant);
        self.top_once(&name, format!("let {name}: vec2f = s.screen + {f}(s.screen) * MKS_BUMP_FILTER;"));
        name
    }

    /// Attribute `key` ("uv:", "uv:<name>", "color:...", "attr:...", "tangent:...") as the vec4 the renderer fills.
    pub fn src_attr(&mut self, key: &str) -> String {
        let idx = match self.sh.attrs.iter().position(|a| a == key) {
            Some(i) => i,
            None => {
                self.sh.attrs.push(key.to_string());
                self.sh.attrs.len() - 1
            }
        };
        if idx >= crate::MAX_ATTRS {
            self.unsupported(format!("more than {} vertex attributes", crate::MAX_ATTRS));
            return "vec4f(0.0)".into();
        }
        let e = format!("s.attrs[{idx}]");
        if key.starts_with("tangent:") {
            return e;
        }
        self.offset(&format!("a{idx}"), &e, Ty::Color)
    }

    pub fn uv_attr(&mut self, map: &str) -> String {
        let a = self.src_attr(&format!("uv:{map}"));
        format!("({a}).xyz")
    }

    // ------------------------------------------------------------------------------------------- inputs / outputs
    /// The value of input `idx` of node `n` as type `want`.
    pub fn input_idx(&mut self, n: usize, idx: usize, want: Ty) -> String {
        let sock = &self.node(n).inputs[idx];
        if let Some((src, out)) = sock.link {
            let e = self.output(src, out);
            return convert(&e, want);
        }
        if let Some(a) = sock.anim {
            let slot = self.sh.param_base + a as u32;
            let e = match sock.ty {
                Ty::Float => Expr::new(format!("mkr_param({slot}u).x"), Ty::Float),
                Ty::Vec3 => Expr::new(format!("mkr_param({slot}u).xyz"), Ty::Vec3),
                Ty::Color => Expr::new(format!("mkr_param({slot}u)"), Ty::Color),
                Ty::Shader => Expr::new("mks_closures_none()", Ty::Shader),
            };
            return convert(&e, want);
        }
        const_of(&sock.value, &sock.raw_ty, want)
    }

    /// Whether output `key` of node `n` feeds anything the material output reaches.
    pub fn output_used(&self, n: usize, key: &str) -> bool {
        self.node(n).output_index(key).map(|i| self.sh.used_outputs.contains(&(n, i))).unwrap_or(false)
    }

    /// Input by identifier (or name); zero, silently, when the node has no such socket (sockets a probe omits).
    pub fn input_opt(&mut self, n: usize, key: &str, want: Ty) -> String {
        match self.node(n).input_index(key) {
            Some(i) => self.input_idx(n, i, want),
            None => zero(want),
        }
    }

    /// Input by identifier (or name), `want` typed; a missing socket gives zero.
    pub fn input(&mut self, n: usize, key: &str, want: Ty) -> String {
        match self.node(n).input_index(key) {
            Some(i) => self.input_idx(n, i, want),
            None => {
                let path = self.node(n).path.clone();
                self.warn(format!("{path}: no input {key:?}"));
                zero(want)
            }
        }
    }

    /// The input if it is linked or animated (nodes whose unlinked inputs mean an implicit value: normals, texture
    /// coordinates).
    pub fn linked_input(&mut self, n: usize, key: &str, want: Ty) -> Option<String> {
        let i = self.node(n).input_index(key)?;
        let s = &self.node(n).inputs[i];
        if s.link.is_none() && s.anim.is_none() {
            return None;
        }
        Some(self.input_idx(n, i, want))
    }

    pub fn input_f(&mut self, n: usize, key: &str) -> String {
        self.input(n, key, Ty::Float)
    }

    pub fn input_c(&mut self, n: usize, key: &str) -> String {
        self.input(n, key, Ty::Color)
    }

    /// Stored constant value of an input (unlinked), as f64.
    pub fn input_const(&self, n: usize, key: &str) -> Option<f64> {
        let node = self.node(n);
        let s = &node.inputs[node.input_index(key)?];
        if s.link.is_some() || s.anim.is_some() {
            return None;
        }
        nums(&s.value).first().copied()
    }

    /// Output `out` of node `n` (pass nodes resolve through their inputs).
    pub fn output(&mut self, n: usize, out: usize) -> Expr {
        let node = self.node(n);
        let oty = node.outputs.get(out).map(|s| s.ty).unwrap_or(Ty::Float);
        if let Pass::Map(map) = &node.pass {
            if let Some(a) = node.outputs.get(out).and_then(|s| s.anim) {
                let slot = self.sh.param_base + a as u32;
                let code = match oty {
                    Ty::Float => format!("mkr_param({slot}u).x"),
                    Ty::Vec3 => format!("mkr_param({slot}u).xyz"),
                    Ty::Color => format!("mkr_param({slot}u)"),
                    Ty::Shader => "mks_closures_none()".into(),
                };
                return Expr::new(code, oty);
            }
            return match map.get(out).copied().flatten() {
                Some(i) if i < node.inputs.len() => {
                    let key = (n, self.variant);
                    if !self.visiting.insert(key) {
                        self.warn(format!("{}: cycle", node.path));
                        return Expr::new(zero(oty), oty);
                    }
                    let code = self.input_idx(n, i, oty);
                    self.visiting.remove(&key);
                    Expr::new(code, oty)
                }
                _ => {
                    let s = &node.outputs[out];
                    Expr::new(const_of(&s.value, &s.raw_ty, oty), oty)
                }
            };
        }
        let key = (n, self.variant);
        if let Some(outs) = self.memo.get(&key) {
            return outs.get(out).cloned().flatten().unwrap_or_else(|| Expr::new(zero(oty), oty));
        }
        if !self.visiting.insert(key) {
            let path = node.path.clone();
            self.warn(format!("{path}: cycle"));
            return Expr::new(zero(oty), oty);
        }
        let outs = crate::nodes::emit(self, n);
        self.visiting.remove(&key);
        let r = outs.get(out).cloned().flatten().unwrap_or_else(|| Expr::new(zero(oty), oty));
        self.memo.insert(key, outs);
        r
    }

    /// Evaluate input `key` of node `n` in bump variant `v` (the height subgraph again, one pixel away).
    pub fn input_in_variant(&mut self, n: usize, key: &str, want: Ty, v: Variant) -> String {
        let saved = self.variant;
        self.variant = v;
        let r = self.input(n, key, want);
        self.variant = saved;
        r
    }

    /// Module helper `name` (emitted once per material).
    pub fn helper(&mut self, key: String, make: impl FnOnce(&str) -> String) -> String {
        if let Some(n) = self.sh.helper_names.get(&key) {
            return n.clone();
        }
        let name = format!("mat_{}_h{}", self.sh.id, self.sh.helper_names.len());
        let code = make(&name);
        self.sh.helpers.push(code);
        self.sh.helper_names.insert(key, name.clone());
        name
    }

    /// Assemble `fn <name>(s: Surface) -> <ret> { ... return <ret_expr>; }`.
    pub fn finish(self, name: &str, ret_ty: &str, ret: &str) -> String {
        let mut f = String::new();
        let _ = writeln!(f, "fn {name}(s: Surface) -> {ret_ty} {{");
        let _ = writeln!(f, "  var rng: f32 = s.rand.x;");
        for l in &self.top {
            let _ = writeln!(f, "{l}");
        }
        for l in &self.body {
            let _ = writeln!(f, "{l}");
        }
        let _ = writeln!(f, "  return {ret};");
        let _ = writeln!(f, "}}");
        f
    }
}
