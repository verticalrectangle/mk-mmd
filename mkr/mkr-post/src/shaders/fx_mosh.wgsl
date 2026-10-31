// needs sample.wgsl
// mkmmd.framefx.mosh_step, mosh_carry and FrameFx._mosh: the held picture carried along block moves, and mixed over
// the frame.
//   step   the held picture one frame on along the cut's own motion: each pixel takes what stood its block's move
//          behind it (cv2.remap at (x - mx, y - my), INTER_LINEAR, BORDER_REPLICATE), the move its block's ((cols[x],
//          rows[y]) of the gw x gh blocks, as mosh_field's nearest resizes read them) / MOSH_SCALE
//   carry  the held picture `carries` steps along a made-up flow's whole-pixel block moves (mosh_carry: cv2.remap
//          INTER_NEAREST, BORDER_REPLICATE, again and again): each step a pixel takes the one its block's move behind
//          it, so the steps chain into one read through where each lands
//   mix    img * (1 - a) + held * a, a the strength (times the mask's region: the figure's coverage or 1 - it)
// Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    gw: u32,
    masked: u32,        // 0 none, 1 the figure, 2 the ground
    s: f32,
    one_minus: f32,     // 1 - s as numpy rounds it
    carries: u32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> held: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> blocks: array<f32>;      // (gh, gw, 2): x, y moves at MOSH_SCALE
@group(0) @binding(4) var<storage, read> cols: array<u32>;
@group(0) @binding(5) var<storage, read> rows: array<u32>;
@group(0) @binding(6) var<storage, read> img: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> fig: array<u32>;
@group(0) @binding(8) var<storage, read> lut: array<f32>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    return held[i];
}

@compute @workgroup_size(16, 16)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let k = 2u * (rows[id.y] * P.gw + cols[id.x]);
    let mx = blocks[k] / 0.25;
    let my = blocks[k + 1u] / 0.25;
    dst[id.y * P.w + id.x] = sample_bilinear(0u, f32(id.x) - mx, f32(id.y) - my, P.w, P.h, BORDER_REPLICATE,
                                             vec4<f32>(0.0));
}

@compute @workgroup_size(16, 16)
fn carry(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    var x = i32(id.x);
    var y = i32(id.y);
    for (var n = 0u; n < P.carries; n++) {
        let k = 2u * (rows[y] * P.gw + cols[x]);
        x = clamp(x - i32(blocks[k]), 0, i32(P.w) - 1);
        y = clamp(y - i32(blocks[k + 1u]), 0, i32(P.h) - 1);
    }
    dst[id.y * P.w + id.x] = held[u32(y) * P.w + u32(x)];
}

@compute @workgroup_size(16, 16)
fn mix(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let p = img[i];
    let q = held[i];
    if (P.masked == 0u) {
        dst[i] = vec4<f32>(p.xyz * P.one_minus + q.xyz * P.s, p.w);
        return;
    }
    let m = lut[fig[i] & 255u];
    let a = P.s * select(1.0 - m, m, P.masked == 1u);
    dst[i] = vec4<f32>(p.xyz * (1.0 - a) + q.xyz * a, p.w);
}
