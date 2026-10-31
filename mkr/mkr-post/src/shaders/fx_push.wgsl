// needs sample.wgsl
// mkmmd.framefx.push: the picture zooming into a disc that shows another frame (`inner`, packed 8-bit). With d the
// pixel's offset from the disc's centre p: outer = the picture at c + d / z, inside = the inner frame at m + d / s
// (cv2.remap, INTER_LINEAR, BORDER_REFLECT), cov = clip(rr - |d|, 0, 1) (rr the disc's radius + 0.5), the pixel
// outer * (1 - cov) + inside * cov; each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
    c: vec2<f32>,
    p: vec2<f32>,
    m: vec2<f32>,
    z: f32,
    s: f32,
    rr: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> inner: array<u32>;
@group(0) @binding(4) var<storage, read> lut: array<f32>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    if (which == 0u) {
        return src[i];
    }
    let q = inner[i];
    return vec4<f32>(lut[q & 255u], lut[(q >> 8u) & 255u], lut[(q >> 16u) & 255u], lut[q >> 24u]);
}

@compute @workgroup_size(16, 16)
fn push(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let dx = f32(id.x) - P.p.x;
    let dy = f32(id.y) - P.p.y;
    let outer = sample_bilinear(0u, P.c.x + dx / P.z, P.c.y + dy / P.z, P.w, P.h, BORDER_REFLECT, vec4<f32>(0.0));
    let inside = sample_bilinear(1u, P.m.x + dx / P.s, P.m.y + dy / P.s, P.w, P.h, BORDER_REFLECT, vec4<f32>(0.0));
    let cov = clamp(P.rr - sqrt(dx * dx + dy * dy), 0.0, 1.0);
    dst[id.y * P.w + id.x] = vec4<f32>(outer.xyz * (1.0 - cov) + inside.xyz * cov, outer.w);
}
