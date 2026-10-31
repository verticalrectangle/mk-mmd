// cv2.warpAffine(src, M, size, INTER_LINEAR | WARP_INVERSE_MAP, border) as OpenCV 5 computes it on a float32 picture:
// each pixel (x, y) reads src at sx = fma(M00, x, M01 * y + M02), sy = fma(M10, x, M11 * y + M12) (float32 M), the
// row terms worked out on the CPU (`rows`: two floats per row, so nothing here can fuse them), bilinear between the four
// pixels about it with fused multiply-adds, OpenCV's border rule for those outside (bit for bit with cv2). `dst` may be
// a box of the warped picture: P.w x P.h with its left column at P.x0 (its rows' terms are those of its own rows).

struct Params {
    w: u32,
    h: u32,
    sw: u32,            // the source's size
    sh: u32,
    border: u32,        // common.wgsl BORDER_*
    x0: u32,            // the box's left column in the warped picture
    m00: f32,
    m10: f32,
    cval: vec4<f32>,    // the colour outside, BORDER_CONSTANT
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rows: array<f32>;

fn tap(x: i32, y: i32) -> vec4<f32> {
    let bx = border(x, i32(P.sw), P.border);
    let by = border(y, i32(P.sh), P.border);
    if (bx < 0 || by < 0) {
        return P.cval;
    }
    return src[u32(by) * P.sw + u32(bx)];
}

@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let fx = f32(P.x0 + id.x);
    let sx = fma(P.m00, fx, rows[2u * id.y]);
    let sy = fma(P.m10, fx, rows[2u * id.y + 1u]);
    let x0 = floor(sx);
    let y0 = floor(sy);
    let tx = sx - x0;
    let ty = sy - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    let p00 = tap(ix, iy);
    let p01 = tap(ix + 1, iy);
    let p10 = tap(ix, iy + 1);
    let p11 = tap(ix + 1, iy + 1);
    let a = fma(p01 - p00, vec4<f32>(tx), p00);
    let b = fma(p11 - p10, vec4<f32>(tx), p10);
    dst[id.y * P.w + id.x] = fma(b - a, vec4<f32>(ty), a);
}
