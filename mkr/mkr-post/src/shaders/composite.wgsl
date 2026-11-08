// Screen type laid over a picture (mkmmd.core.screentype.composite): the straight RGBA layer over the RGB picture,
// base * (1 - a) + layer * a with a = clip(alpha, 0, 1); with `keep` the keep mask is multiplied by 1 - alpha too
// (the type over a kept eye is graded as type: mkmmd.postframes.Output._before_grade).
//
// `negate` is the negative type (`negative = true`) the same function lays after it: `neg` the coverage a of the type
// (the alpha of its layer, unpack_alpha), the picture under it inverted, base * (1 - a) + (1 - base) * a per channel,
// the keep mask multiplied by 1 - a too.

struct Params {
    w: u32,
    h: u32,
    keep: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> base: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> layer: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> mask: array<f32>;
@group(0) @binding(4) var<storage, read> neg: array<f32>;

@compute @workgroup_size(16, 16)
fn composite(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let l = layer[i];
    let a = clamp(l.w, 0.0, 1.0);
    let b = base[i];
    base[i] = vec4<f32>(b.xyz * (1.0 - a) + l.xyz * a, b.w);
    if (P.keep != 0u) {
        mask[i] = mask[i] * (1.0 - l.w);
    }
}

@compute @workgroup_size(16, 16)
fn negate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let n = neg[i];
    let a = clamp(n, 0.0, 1.0);
    let b = base[i];
    base[i] = vec4<f32>(b.xyz * (1.0 - a) + (vec3<f32>(1.0) - b.xyz) * a, b.w);
    if (P.keep != 0u) {
        mask[i] = mask[i] * (1.0 - n);
    }
}
