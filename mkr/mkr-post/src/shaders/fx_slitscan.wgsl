// mkmmd.framefx.slitscan: one band of columns (`across`) or rows of the picture, [i0, i1), replaced in place by the
// same pixels of an older frame of the cut (packed 8-bit, v / 255 as numpy reads it). The dispatch covers the band.

struct Params {
    w: u32,
    h: u32,
    across: u32,
    i0: u32,
    i1: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pic: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> frame: array<u32>;
@group(0) @binding(3) var<storage, read> lut: array<f32>;

@compute @workgroup_size(16, 16)
fn slitscan(@builtin(global_invocation_id) id: vec3<u32>) {
    var x = id.x;
    var y = id.y;
    if (P.across != 0u) {
        x = x + P.i0;
        if (x >= P.i1 || y >= P.h) {
            return;
        }
    } else {
        y = y + P.i0;
        if (y >= P.i1 || x >= P.w) {
            return;
        }
    }
    let i = y * P.w + x;
    let q = frame[i];
    pic[i] = vec4<f32>(lut[q & 255u], lut[(q >> 8u) & 255u], lut[(q >> 16u) & 255u], pic[i].w);
}
