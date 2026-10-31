// A decoded PNG (RGBA8 per pixel, r | g << 8 | b << 16 | a << 24) to float as mkmmd.post.read makes it: v / 255 in
// float32 (`lut`, from numpy). `unpack_picture` keeps all four channels, `unpack_mask` the red one (a keep mask read
// as RGB, its first channel used); `expand` widens a worker's float RGB picture (w * h * 3 float32); `widen_luma` a
// figure layer's luma bytes (four pixels a word, D.w pixels) to such words (the byte in r, g and b, alpha 255).

struct Dims {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> D: Dims;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<storage, read_write> picture: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> mask: array<f32>;
@group(0) @binding(5) var<storage, read> raw: array<f32>;
@group(0) @binding(6) var<storage, read_write> wide: array<u32>;

@compute @workgroup_size(16, 16)
fn unpack_picture(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= D.w || id.y >= D.h) {
        return;
    }
    let i = id.y * D.w + id.x;
    let p = packed[i];
    picture[i] = vec4<f32>(lut[p & 255u], lut[(p >> 8u) & 255u], lut[(p >> 16u) & 255u], lut[p >> 24u]);
}

@compute @workgroup_size(16, 16)
fn unpack_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= D.w || id.y >= D.h) {
        return;
    }
    let i = id.y * D.w + id.x;
    mask[i] = lut[packed[i] & 255u];
}

@compute @workgroup_size(16, 16)
fn expand(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= D.w || id.y >= D.h) {
        return;
    }
    let i = id.y * D.w + id.x;
    picture[i] = vec4<f32>(raw[3u * i], raw[3u * i + 1u], raw[3u * i + 2u], 1.0);
}

@compute @workgroup_size(256)
fn widen_luma(@builtin(global_invocation_id) id: vec3<u32>) {
    if (4u * id.x >= D.w) {
        return;
    }
    let p = packed[id.x];
    for (var j = 0u; j < 4u; j++) {
        let i = 4u * id.x + j;
        if (i < D.w) {
            let v = (p >> (8u * j)) & 255u;
            wide[i] = v | (v << 8u) | (v << 16u) | 0xff000000u;
        }
    }
}
