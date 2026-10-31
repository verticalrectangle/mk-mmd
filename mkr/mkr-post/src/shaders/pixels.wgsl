// Per-pixel passes of mk post (mkmmd/post.py, mkmmd/core/screentype.py, mkmmd/cli/post.py's quantisation), on pictures
// held as one vec4<f32> per pixel (rgb, the fourth component unused), row by row. Every operation follows the numpy
// expression it stands for, in the same order, in float32; constants come from the Python side as numpy rounds them.

struct Dims {
    w: u32,
    h: u32,
    flag: u32,       // per pass: unpack: 0 picture, 1 mask (red channel only); composite: 1 = the keep mask too
    pad: u32,
}

@group(0) @binding(0) var<uniform> D: Dims;

const Y = vec3<f32>(0.2126, 0.7152, 0.0722);

fn luma(c: vec3<f32>) -> f32 {
    return c.x * Y.x + c.y * Y.y + c.z * Y.z;
}

fn index(id: vec3<u32>) -> i32 {
    if (id.x >= D.w || id.y >= D.h) {
        return -1;
    }
    return i32(id.y * D.w + id.x);
}

// ---------------------------------------------------------------- unpack: RGBA8 (r | g << 8 | b << 16 | a << 24) -> float
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;          // 256 entries: v / 255 as numpy makes it
@group(0) @binding(3) var<storage, read_write> picture: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = index(id);
    if (i < 0) {
        return;
    }
    let p = packed[i];
    picture[i] = vec4<f32>(lut[p & 255u], lut[(p >> 8u) & 255u], lut[(p >> 16u) & 255u], lut[p >> 24u]);
}
