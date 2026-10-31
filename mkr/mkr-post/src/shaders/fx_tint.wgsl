// tint (mkmmd.framefx.gradient_map; FrameFx._tint): each pixel by its brightness through dark .. colour .. light,
// mixed in by the strength.

struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
    s: f32,             // the strength and 1 - it, float32
    s1: f32,
    pad2: f32,
    pad3: f32,
    dark: vec4<f32>,
    colour: vec4<f32>,
    light: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn tint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let img = src[i];
    let l = clamp(luma(img.xyz), 0.0, 1.0);
    let lo = P.dark.xyz + (P.colour.xyz - P.dark.xyz) * (2.0 * l);
    let hi = P.colour.xyz + (P.light.xyz - P.colour.xyz) * (2.0 * l - 1.0);
    let g = select(hi, lo, l < 0.5);
    dst[i] = vec4<f32>(img.xyz * P.s1 + g * P.s, img.w);
}
