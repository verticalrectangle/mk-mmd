// inks: flat colours laid by coverages over the picture, in order (mkmmd.labels._ink: a label's box, its text, its
// leader; a stamp's letters): out = out * (1 - a) + colour * a, a = alpha x coverage, the coverage's top left at
// (P.x0, P.y0) (it may stick out of the picture), or a solid box where there is none.

struct Params {
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    cw: u32,
    ch: u32,
    solid: u32,
    pad0: u32,
    alpha: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cov: array<f32>;

@compute @workgroup_size(16, 16)
fn ink(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.cw || id.y >= P.ch) {
        return;
    }
    let x = P.x0 + i32(id.x);
    let y = P.y0 + i32(id.y);
    if (x < 0 || y < 0 || x >= i32(P.w) || y >= i32(P.h)) {
        return;
    }
    var c = 1.0;
    if (P.solid == 0u) {
        c = cov[id.y * P.cw + id.x];
    }
    let a = P.alpha * c;
    let o = u32(y) * P.w + u32(x);
    let d = dst[o];
    dst[o] = vec4<f32>(d.xyz * (1.0 - a) + P.colour.xyz * a, d.w);
}
