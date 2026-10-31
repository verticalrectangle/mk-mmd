// The sticker work the collages and the masked effects share (mkmmd.framefx: _grow, _cast, _paste, _warp, plate_shift;
// mkmmd.mandala: soft, lay) on planes of one vec4<f32> per pixel, row by row (a coverage in .x, a premultiplied
// sticker in rgb and its coverage in .w): OpenCV's Gaussian blur (the kernel from the CPU, REFLECT_101, rows then
// columns), the exact Euclidean distance transform out to a reach (DIST_L2, DIST_MASK_PRECISE: a pixel's distance is
// the square root of a whole number of px squared), OpenCV's INTER_AREA and INTER_LINEAR resizes driven by the tables
// the CPU works out as OpenCV does, and the composites, each the numpy expression it stands for in float32.

struct Params {
    w: u32,             // the plane written
    h: u32,
    sw: u32,            // the plane read, where it is another size
    sh: u32,
    n: u32,             // blur: taps; edt: reach in px
    flag: u32,          // per kernel (below)
    x0: u32,            // blit: where the picture's top left lands
    y0: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> taps: array<f32>;      // a blur's weights
@group(0) @binding(4) var<storage, read> aux: array<vec4<f32>>; // a second plane read
@group(0) @binding(5) var<storage, read> tab: array<u32>;       // a resize's tables (floats as their bits)
@group(0) @binding(6) var<storage, read> packed: array<u32>;    // a picture as read: r | g << 8 | b << 16 | a << 24
@group(0) @binding(7) var<storage, read> lut: array<f32>;       // v / 255 as numpy makes it

const FAR: f32 = 1.0e9;                                         // no pixel of the figure within the reach

fn here(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

// ---------------------------------------------------------------- Gaussian blur (cv2.GaussianBlur, sigmaY = sigmaX)
// `taps` holds getGaussianKernel(n, sigma, CV_32F); blur_h runs along the rows, then blur_v down the columns. blur_h
// takes a run of 256 pixels of a row per workgroup: the run and its reach are read into the workgroup's memory once
// (BLUR_TILE pixels: a reach up to 256 px), then each pixel's taps summed in order from there; blur_h_wide, for a reach
// past that, reads each tap from the plane.
const BLUR_RUN: u32 = 256u;
const BLUR_TILE: u32 = 768u;
var<workgroup> blur_tile: array<vec4<f32>, 768>;

@compute @workgroup_size(256)
fn blur_h(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let y = wg.y;
    let first = wg.x * BLUR_RUN;
    let r = i32(P.n / 2u);
    let len = BLUR_RUN + P.n - 1u;
    let row = y * P.w;
    for (var k = t; k < len; k = k + BLUR_RUN) {
        let x = border(i32(first + k) - r, i32(P.w), BORDER_REFLECT_101);
        blur_tile[k] = src[row + u32(x)];
    }
    workgroupBarrier();
    let x = first + t;
    if (x >= P.w || y >= P.h) {
        return;
    }
    var s = vec4<f32>(0.0);
    for (var j = 0u; j < P.n; j = j + 1u) {
        s = s + taps[j] * blur_tile[t + j];
    }
    dst[row + x] = s;
}

@compute @workgroup_size(16, 16)
fn blur_h_wide(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let r = i32(P.n / 2u);
    let row = id.y * P.w;
    var s = vec4<f32>(0.0);
    for (var j = 0; j < i32(P.n); j = j + 1) {
        let x = border(i32(id.x) + j - r, i32(P.w), BORDER_REFLECT_101);
        s = s + taps[j] * src[row + u32(x)];
    }
    dst[i] = s;
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let r = i32(P.n / 2u);
    var s = vec4<f32>(0.0);
    for (var j = 0; j < i32(P.n); j = j + 1) {
        let y = border(i32(id.y) + j - r, i32(P.h), BORDER_REFLECT_101);
        s = s + taps[j] * src[u32(y) * P.w + id.x];
    }
    dst[i] = s;
}

// ---------------------------------------------------------------- _grow: the die-cut ring round a coverage
// edt_cols: .x the rows to the nearest pixel of the figure (coverage .x >= 0.5) in the same column, FAR past the reach.
@compute @workgroup_size(16, 16)
fn edt_cols(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let y = i32(id.y);
    var v = FAR;
    for (var k = 0; k <= i32(P.n); k = k + 1) {
        let up = y - k;
        let down = y + k;
        if ((up >= 0 && src[u32(up) * P.w + id.x].x >= 0.5) || (down < i32(P.h) && src[u32(down) * P.w + id.x].x >= 0.5)) {
            v = f32(k);
            break;
        }
    }
    dst[i] = vec4<f32>(v, 0.0, 0.0, 0.0);
}

// edt_rows: the distance d to the nearest pixel of the figure (from edt_cols in `src`), then the ring before its
// blur: clip(P.a - d, 0, 1), P.a = float32(r + 0.5).
@compute @workgroup_size(16, 16)
fn edt_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let x = i32(id.x);
    let row = id.y * P.w;
    var best = FAR;
    let reach = i32(P.n);
    for (var k = -reach; k <= reach; k = k + 1) {
        let q = x + k;
        if (q < 0 || q >= i32(P.w)) {
            continue;
        }
        let c = src[row + u32(q)].x;
        if (c < FAR) {
            best = min(best, f32(k * k) + c * c);
        }
    }
    let d = select(FAR, sqrt(best), best < FAR);
    dst[i] = vec4<f32>(clamp(P.a - d, 0.0, 1.0), 0.0, 0.0, 0.0);
}

// grow_finish: the ring (`src`, blurred when flag = 1) sharpened back, clip((g - 0.5) * P.b + 0.5, 0, 1), and never
// under the coverage itself (`aux`): np.maximum(g, a).
@compute @workgroup_size(16, 16)
fn grow_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    var g = src[i].x;
    if (P.flag == 1u) {
        g = clamp((g - 0.5) * P.b + 0.5, 0.0, 1.0);
    }
    dst[i] = vec4<f32>(max(g, aux[i].x), 0.0, 0.0, 0.0);
}

// ---------------------------------------------------------------- resizes (cv2.resize of float32 planes)
// INTER_AREA as OpenCV's general path: `tab` starts with the written columns' offsets (dw + 1 words, from 0) and the
// written rows' (dh + 1 words, from P.flag), each the word where that column's (row's) (source px, weight as bits)
// pairs start, the next one's where they end; each written pixel is the sum over its rows of the row weight times the
// sum over its columns of the column weight times the pixel read.
@compute @workgroup_size(16, 16)
fn resize_area(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let ty = P.flag;
    var s = vec4<f32>(0.0);
    for (var e = tab[ty + id.y]; e < tab[ty + id.y + 1u]; e = e + 2u) {
        let sy = tab[e];
        let beta = bitcast<f32>(tab[e + 1u]);
        var row = vec4<f32>(0.0);
        for (var q = tab[id.x]; q < tab[id.x + 1u]; q = q + 2u) {
            row = row + src[sy * P.sw + tab[q]] * bitcast<f32>(tab[q + 1u]);
        }
        s = s + row * beta;
    }
    dst[i] = s;
}

// INTER_LINEAR: `tab` holds, per written column, (source column, its weight, the next one's weight) and from tab[P.flag]
// the same per written row; the next column (row) is the last one at the edge, where its weight is 0.
@compute @workgroup_size(16, 16)
fn resize_linear(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let cx = 3u * id.x;
    let cy = P.flag + 3u * id.y;
    let sx = tab[cx];
    let sx1 = min(sx + 1u, P.sw - 1u);
    let sy = tab[cy];
    let sy1 = min(sy + 1u, P.sh - 1u);
    let a0 = bitcast<f32>(tab[cx + 1u]);
    let a1 = bitcast<f32>(tab[cx + 2u]);
    let b0 = bitcast<f32>(tab[cy + 1u]);
    let b1 = bitcast<f32>(tab[cy + 2u]);
    let r0 = src[sy * P.sw + sx] * a0 + src[sy * P.sw + sx1] * a1;
    let r1 = src[sy1 * P.sw + sx] * a0 + src[sy1 * P.sw + sx1] * a1;
    dst[i] = r0 * b0 + r1 * b1;
}

// INTER_CUBIC as OpenCV resizes a float plane: `tab` holds, per written column, (first source column as i32 bits, its
// four weights as bits) and from tab[P.flag] the same per written row; taps outside the plane read its edge. Along the
// rows first, each sum in order.
fn cubic_row(y: u32, sx: i32, a: vec4<f32>) -> vec4<f32> {
    let last = i32(P.sw) - 1;
    let row = y * P.sw;
    let p0 = src[row + u32(clamp(sx - 1, 0, last))];
    let p1 = src[row + u32(clamp(sx, 0, last))];
    let p2 = src[row + u32(clamp(sx + 1, 0, last))];
    let p3 = src[row + u32(clamp(sx + 2, 0, last))];
    return ((p0 * a.x + p1 * a.y) + p2 * a.z) + p3 * a.w;
}

@compute @workgroup_size(16, 16)
fn resize_cubic(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let cx = 5u * id.x;
    let cy = P.flag + 5u * id.y;
    let sx = bitcast<i32>(tab[cx]);
    let sy = bitcast<i32>(tab[cy]);
    let a = vec4<f32>(bitcast<f32>(tab[cx + 1u]), bitcast<f32>(tab[cx + 2u]), bitcast<f32>(tab[cx + 3u]),
                      bitcast<f32>(tab[cx + 4u]));
    let b = vec4<f32>(bitcast<f32>(tab[cy + 1u]), bitcast<f32>(tab[cy + 2u]), bitcast<f32>(tab[cy + 3u]),
                      bitcast<f32>(tab[cy + 4u]));
    let last = i32(P.sh) - 1;
    let h0 = cubic_row(u32(clamp(sy - 1, 0, last)), sx, a);
    let h1 = cubic_row(u32(clamp(sy, 0, last)), sx, a);
    let h2 = cubic_row(u32(clamp(sy + 1, 0, last)), sx, a);
    let h3 = cubic_row(u32(clamp(sy + 2, 0, last)), sx, a);
    dst[i] = ((h0 * b.x + h1 * b.y) + h2 * b.z) + h3 * b.w;
}

// ---------------------------------------------------------------- composites (a pasted figure's: fx_plane.wgsl)
// premultiply: rgb of `src` times coverage aux.x into `dst` (pic * m[..., None]), its coverage in .w.
@compute @workgroup_size(16, 16)
fn premultiply(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let m = aux[i].x;
    dst[i] = vec4<f32>(src[i].xyz * m, m);
}

// mix: the effect's picture `dst` mixed with the picture as it reached it (`src`) by the strength:
// img * (1 - s) + out * s, P.a = s and P.b = 1 - s as numpy rounds them.
@compute @workgroup_size(16, 16)
fn mix(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let c = dst[i];
    dst[i] = vec4<f32>(src[i].xyz * P.b + c.xyz * P.a, c.w);
}

// fill: every pixel P.colour (a flat back).
@compute @workgroup_size(16, 16)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    dst[i] = P.colour;
}

// blit: a packed 8-bit picture of P.sw x P.sh (a contained clip) widened by `lut` into `dst` (P.w x P.h) with its top
// left at (P.x0, P.y0): out[y0:y0 + sh, x0:x0 + sw] = small / 255.
@compute @workgroup_size(16, 16)
fn blit(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.sw || id.y >= P.sh) {
        return;
    }
    let p = packed[id.y * P.sw + id.x];
    dst[(P.y0 + id.y) * P.w + P.x0 + id.x] = vec4<f32>(lut[p & 255u], lut[(p >> 8u) & 255u], lut[(p >> 16u) & 255u], 1.0);
}

// flip: `src` reflected left for right about x = P.n / 2 (P.n as i32 bits): pixel x shows pixel n - x, nothing where
// that is outside (mkmmd.stack.lay's reflection).
@compute @workgroup_size(16, 16)
fn flip(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let x = bitcast<i32>(P.n) - i32(id.x);
    var c = vec4<f32>(0.0);
    if (x >= 0 && x < i32(P.w)) {
        c = src[id.y * P.w + u32(x)];
    }
    dst[i] = c;
}
