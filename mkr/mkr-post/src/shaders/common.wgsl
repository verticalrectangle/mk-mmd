// Shared by every kernel of mkr-post (prepended to each). Pictures are one vec4<f32> per pixel (rgb; the fourth
// component is alpha where a layer has one), row by row, top row first, as numpy holds them. Each kernel follows the
// numpy / OpenCV expression it stands for, in float32 and in the same order; constants come from the Python side as
// numpy rounds them. OpenCV 5's warps and remaps sample at float coordinates, interpolating with fused multiply-adds:
// `bilinear` does the same, bit for bit.

const Y = vec3<f32>(0.2126, 0.7152, 0.0722);       // mkmmd.post.Y_W

fn luma(c: vec3<f32>) -> f32 {
    return c.x * Y.x + c.y * Y.y + c.z * Y.z;
}

// OpenCV's border modes (cv2.BORDER_*): the pixel `p` of a line of `n` reads, or -1 for the constant colour.
const BORDER_CONSTANT: u32 = 0u;
const BORDER_REPLICATE: u32 = 1u;
const BORDER_REFLECT: u32 = 2u;
const BORDER_REFLECT_101: u32 = 4u;

fn border(p: i32, n: i32, mode: u32) -> i32 {
    if (p >= 0 && p < n) {
        return p;
    }
    if (mode == BORDER_CONSTANT) {
        return -1;
    }
    if (mode == BORDER_REPLICATE || n == 1) {
        return clamp(p, 0, n - 1);
    }
    let delta = select(0, 1, mode == BORDER_REFLECT_101);
    var q = p;
    loop {
        if (q < 0) {
            q = -q - 1 + delta;
        } else if (q >= n) {
            q = n - 1 - (q - n) - delta;
        } else {
            break;
        }
    }
    return q;
}
