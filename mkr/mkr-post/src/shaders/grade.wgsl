// The grade after the halation (mkmmd.post.Grade.__call__): saturation (the keep mask's share at its own), contrast
// about the pivot, split toning, clip and vignette, the floor's toe, grain, the final clip. `vig` and `grain` are
// the arrays Python made (mkmmd.post.Grade.vig, .grain[k % n]); every constant is float32 as numpy uses it.

struct Params {
    w: u32,
    h: u32,
    keep: u32,          // 1: the keep mask applies
    split: u32,         // 1: split toning
    floor: u32,         // 1: a floor
    grain_offset: u32,  // first float of this frame's grain in `grain`
    pad0: u32,
    pad1: u32,
    sat: f32,
    keep_delta: f32,    // keep saturation - saturation, as Python computes it (float64, then float32)
    pivot: f32,
    contrast: f32,
    amount: f32,        // split amount
    yf: f32,            // the floor's luma
    lim: f32,           // yf + toe
    lim_yf: f32,        // lim - yf, as Python computes it (float64, then float32)
    grain_amount: f32,
    lo: f32,            // the final clip's floor: yf * 0.98, or 0
    pad2: f32,
    pad3: f32,
    shadows: vec4<f32>,     // split: shadows - 0.5 (float32)
    highlights: vec4<f32>,  // split: highlights - 0.5
    floor_rgb: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vig: array<f32>;
@group(0) @binding(4) var<storage, read> grain: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;

@compute @workgroup_size(16, 16)
fn grade(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    var c = src[i].xyz;
    let y0 = luma(c);
    var sat = P.sat;
    if (P.keep != 0u) {
        sat = P.sat + P.keep_delta * clamp(mask[i], 0.0, 1.0);
    }
    c = y0 + (c - y0) * sat;
    c = P.pivot + (c - P.pivot) * P.contrast;
    if (P.split != 0u) {
        let y = clamp(luma(c), 0.0, 1.0);
        c = c + P.amount * ((1.0 - y) * P.shadows.xyz + y * P.highlights.xyz);
    }
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * vig[i];
    if (P.floor != 0u) {
        let y = luma(c);
        let u = clamp(y / P.lim, 0.0, 1.0);
        let f = u * u * (2.0 - u);
        let ny = select(y, P.yf + P.lim_yf * f, y < P.lim);
        let scale = ny / max(y, 1e-4);
        let w = (1.0 - u) * (1.0 - u);
        c = c * scale * (1.0 - w) + (P.floor_rgb.xyz + (c - y) * 0.3) * w;
    }
    c = c + grain[P.grain_offset + i] * P.grain_amount;
    dst[i] = vec4<f32>(clamp(c, vec3<f32>(P.lo), vec3<f32>(1.0)), 1.0);
}
