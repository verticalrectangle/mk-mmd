// mkmmd.framefx.echo: one copy (a frame of the cut, packed 8-bit) laid on the picture in place at opacity `alpha`:
// `lighten` keeps the brighter of the two, max(out, alpha * copy * tint); else the darker,
// min(out, 1 - alpha * (1 - copy) * (1 - tint)); without a tint its factor is left out (numpy multiplies by 1.0).

struct Params {
    w: u32,
    h: u32,
    lighten: u32,
    tinted: u32,
    alpha: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    tint: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pic: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> frame: array<u32>;
@group(0) @binding(3) var<storage, read> lut: array<f32>;

@compute @workgroup_size(16, 16)
fn echo(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let q = frame[i];
    let c = vec3<f32>(lut[q & 255u], lut[(q >> 8u) & 255u], lut[(q >> 16u) & 255u]);
    let o = pic[i];
    var rgb: vec3<f32>;
    if (P.lighten != 0u) {
        var v = P.alpha * c;
        if (P.tinted != 0u) {
            v = v * P.tint.xyz;
        }
        rgb = max(o.xyz, v);
    } else {
        var v = P.alpha * (1.0 - c);
        if (P.tinted != 0u) {
            v = v * (1.0 - P.tint.xyz);
        }
        rgb = min(o.xyz, 1.0 - v);
    }
    pic[i] = vec4<f32>(rgb, o.w);
}
