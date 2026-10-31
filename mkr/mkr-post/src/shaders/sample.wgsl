// Sampling a picture as cv2.remap and cv2.warpAffine do in OpenCV 5 (INTER_LINEAR on float pictures): at the float
// coordinate (sx, sy), bilinear between the four pixels about it with fused multiply-adds, OpenCV's border rule for
// those outside (bit for bit with cv2.remap, all four border modes). A kernel that samples starts with
// `// needs sample.wgsl` and defines `fn fetch(which: u32, i: u32) -> vec4<f32>`: pixel i (row by row) of its source
// `which` (its picture, a packed frame, ...).

// cv2.borderInterpolate for p on a line of n pixels: the pixel read, or -1 for the constant colour (BORDER_* of
// common.wgsl). Reflections are worked out modulo their period, so a coordinate far outside costs nothing more.
fn sample_border(p: i32, n: i32, mode: u32) -> i32 {
    if (p >= 0 && p < n) {
        return p;
    }
    if (mode == BORDER_CONSTANT) {
        return -1;
    }
    if (mode == BORDER_REPLICATE || n == 1) {
        return clamp(p, 0, n - 1);
    }
    var period = 2 * n;
    var last = 2 * n - 1;
    if (mode == BORDER_REFLECT_101) {
        period = 2 * n - 2;
        last = 2 * n - 2;
    }
    var q = p % period;
    if (q < 0) {
        q = q + period;
    }
    if (q >= n) {
        q = last - q;
    }
    return q;
}

fn sample_tap(which: u32, x: i32, y: i32, sw: u32, sh: u32, mode: u32, cval: vec4<f32>) -> vec4<f32> {
    let bx = sample_border(x, i32(sw), mode);
    let by = sample_border(y, i32(sh), mode);
    if (bx < 0 || by < 0) {
        return cval;
    }
    return fetch(which, u32(by) * sw + u32(bx));
}

// Source `which` (sw x sh) at (sx, sy), INTER_LINEAR with border `mode` (cval outside with BORDER_CONSTANT).
fn sample_bilinear(which: u32, sx: f32, sy: f32, sw: u32, sh: u32, mode: u32, cval: vec4<f32>) -> vec4<f32> {
    let x0 = floor(clamp(sx, -1.0e9, 1.0e9));
    let y0 = floor(clamp(sy, -1.0e9, 1.0e9));
    let tx = clamp(sx, -1.0e9, 1.0e9) - x0;
    let ty = clamp(sy, -1.0e9, 1.0e9) - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    let p00 = sample_tap(which, ix, iy, sw, sh, mode, cval);
    let p01 = sample_tap(which, ix + 1, iy, sw, sh, mode, cval);
    let p10 = sample_tap(which, ix, iy + 1, sw, sh, mode, cval);
    let p11 = sample_tap(which, ix + 1, iy + 1, sw, sh, mode, cval);
    let a = fma(p01 - p00, vec4<f32>(tx), p00);
    let b = fma(p11 - p10, vec4<f32>(tx), p10);
    return fma(b - a, vec4<f32>(ty), a);
}
