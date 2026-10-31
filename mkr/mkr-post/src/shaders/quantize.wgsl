// The graded picture to the encoder's bytes (mkmmd.post.to_u8: clip(x, 0, 1) * 255 + 0.5, truncated) as rgb24, four
// pixels to three words, and its darkest luma (mkmmd.post.luma_min; the graded picture is never below 0, so the
// float's bits order as the numbers do).

struct Dims {
    w: u32,
    h: u32,
    quads: u32,     // w * h / 4
    pad: u32,
}

@group(0) @binding(0) var<uniform> D: Dims;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rgb: array<u32>;
@group(0) @binding(3) var<storage, read_write> darkest: atomic<u32>;

fn level(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 1.0) * 255.0 + 0.5);
}

@compute @workgroup_size(256)
fn quantize(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let q = id.x + id.y * groups.x * 256u;
    if (q >= D.quads) {
        return;
    }
    var b: array<u32, 12>;
    var lowest = 3.0e38;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let c = src[4u * q + j].xyz;
        lowest = min(lowest, luma(c));
        b[3u * j] = level(c.x);
        b[3u * j + 1u] = level(c.y);
        b[3u * j + 2u] = level(c.z);
    }
    rgb[3u * q] = b[0] | (b[1] << 8u) | (b[2] << 16u) | (b[3] << 24u);
    rgb[3u * q + 1u] = b[4] | (b[5] << 8u) | (b[6] << 16u) | (b[7] << 24u);
    rgb[3u * q + 2u] = b[8] | (b[9] << 8u) | (b[10] << 16u) | (b[11] << 24u);
    atomicMin(&darkest, bitcast<u32>(max(lowest, 0.0)));
}
