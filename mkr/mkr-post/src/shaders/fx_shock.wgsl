// needs sample.wgsl
// mkmmd.framefx.shock: rings running out from (cx, cy). Per wave (R, A, s): u = (r - R) / width pushes the picture
// out ahead of the front and pulls it in behind, disp += A * u * exp(-u * u), and lights the front, rim += s *
// exp(-(u / rim_k)^2); the pixel shows the picture at p - disp * d / r (cv2.remap, INTER_LINEAR, BORDER_REFLECT), then
// with a glow clip(out + glow * rim * colour, 0, 1). Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    n: u32,
    lit: u32,
    cx: f32,
    cy: f32,
    width: f32,
    rim_k: f32,
    glow: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> waves: array<vec4<f32>>;   // (R, A, s, 0)

fn fetch(which: u32, i: u32) -> vec4<f32> {
    return src[i];
}

@compute @workgroup_size(16, 16)
fn shock(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let xs = f32(id.x);
    let ys = f32(id.y);
    let dx = xs - P.cx;
    let dy = ys - P.cy;
    let r = sqrt(dx * dx + dy * dy) + 1e-3;
    var disp = 0.0;
    var rim = 0.0;
    for (var j = 0u; j < P.n; j = j + 1u) {
        let wv = waves[j];
        let u = (r - wv.x) / P.width;
        disp = disp + wv.y * u * exp(-u * u);
        let q = u / P.rim_k;
        rim = rim + wv.z * exp(-(q * q));
    }
    var o = sample_bilinear(0u, xs - disp * dx / r, ys - disp * dy / r, P.w, P.h, BORDER_REFLECT, vec4<f32>(0.0));
    if (P.lit != 0u) {
        o = vec4<f32>(clamp(o.xyz + P.glow * rim * P.colour.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), o.w);
    }
    dst[id.y * P.w + id.x] = o;
}
