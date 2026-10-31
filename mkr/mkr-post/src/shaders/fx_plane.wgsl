// Single-float planes (a coverage, a ring, a shadow: one f32 per pixel, row by row) and the sticker work on them
// (fx/plane.rs): each kernel the one of fx_figure.wgsl or fx_masked.wgsl it stands for (named alike), the same
// arithmetic in the same order on the same values, a quarter of the memory traffic. A plane may be a box of a larger
// frame: P.w x P.h the box, (P.x0, P.y0) its top left in the frame, P.fw x P.fh the frame; a kernel then works each
// pixel out as it would over the whole frame (its coordinates and borders the frame's), a pixel of the frame outside
// the box reading 0 (the box holds everything over 0 of what is read). A plane of its own is the whole of a frame its
// size. Pictures (`pic`) are the frame's vec4 planes.

struct Params {
    w: u32,             // the box written
    h: u32,
    x0: u32,            // its top left in the frame
    y0: u32,
    fw: u32,            // the frame
    fh: u32,
    n: u32,             // blur: taps; edt: reach; warp: border mode
    flag: u32,          // per kernel (below)
    sw: u32,            // warp, resizes: the plane read
    sh: u32,
    sx0: u32,           // warp: its top left in the frame it is a box of, sfw x sfh
    sy0: u32,
    sfw: u32,
    sfh: u32,
    pad0: u32,
    pad1: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fa: array<f32>;             // a plane read
@group(0) @binding(2) var<storage, read_write> fb: array<f32>;       // a plane written
@group(0) @binding(3) var<storage, read> fc: array<f32>;             // a second plane read
@group(0) @binding(4) var<storage, read> taps: array<f32>;           // a blur's weights
@group(0) @binding(5) var<storage, read> tab: array<u32>;            // a resize's tables, a warp's row terms (bits)
@group(0) @binding(6) var<storage, read> packed: array<u32>;         // a packed 8-bit input (r | g << 8 | ...)
@group(0) @binding(7) var<storage, read> lut: array<f32>;            // v / 255 as numpy makes it
@group(0) @binding(8) var<storage, read> src: array<vec4<f32>>;      // a picture read
@group(0) @binding(9) var<storage, read_write> pic: array<vec4<f32>>; // a picture drawn on (or a box plane written)

const FAR: f32 = 1.0e9;                 // no pixel of the figure within the reach

fn here(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

// The frame's pixel of box pixel (x, y).
fn framed(x: u32, y: u32) -> u32 {
    return (P.y0 + y) * P.fw + P.x0 + x;
}

// unpack: the box of a packed 8-bit input of the frame's size widened by `lut`, its first channel (a figure layer's
// coverage, as unpack_picture's .x).
@compute @workgroup_size(16, 16)
fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    fb[i] = lut[packed[framed(id.x, id.y)] & 255u];
}

// crop: the box of the frame's picture `src`, its .x.
@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    fb[i] = src[framed(id.x, id.y)].x;
}

// ---------------------------------------------------------------- Gaussian blur (cv2.GaussianBlur, REFLECT_101)
// blur_h: the row pass (P.n taps, `taps` getGaussianKernel's) over runs of 256 pixels of a row, one workgroup a run:
// the run and its reach read into the workgroup's memory once (BLUR_TILE floats: a reach up to 1024 px), then each
// pixel's taps summed in order from there. blur_v: the column pass, each pixel's taps read from the plane.
const BLUR_RUN: u32 = 256u;
const BLUR_TILE: u32 = 2304u;
var<workgroup> blur_tile: array<f32, 2304>;

@compute @workgroup_size(256)
fn blur_h(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let y = wg.y;
    let first = wg.x * BLUR_RUN;
    let r = i32(P.n / 2u);
    let len = BLUR_RUN + P.n - 1u;
    let row = y * P.w;
    for (var k = t; k < len; k = k + BLUR_RUN) {
        let x = border(i32(P.x0 + first + k) - r, i32(P.fw), BORDER_REFLECT_101) - i32(P.x0);
        var v = 0.0;
        if (x >= 0 && x < i32(P.w)) {
            v = fa[row + u32(x)];
        }
        blur_tile[k] = v;
    }
    workgroupBarrier();
    let x = first + t;
    if (x >= P.w || y >= P.h) {
        return;
    }
    var s = 0.0;
    for (var j = 0u; j < P.n; j = j + 1u) {
        s = s + taps[j] * blur_tile[t + j];
    }
    fb[row + x] = s;
}

// blur_h_wide: the row pass for a reach past the workgroup's tile, each pixel's taps read from the plane.
@compute @workgroup_size(16, 16)
fn blur_h_wide(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let r = i32(P.n / 2u);
    let row = id.y * P.w;
    var s = 0.0;
    for (var j = 0; j < i32(P.n); j = j + 1) {
        let x = border(i32(P.x0 + id.x) + j - r, i32(P.fw), BORDER_REFLECT_101) - i32(P.x0);
        var v = 0.0;
        if (x >= 0 && x < i32(P.w)) {
            v = fa[row + u32(x)];
        }
        s = s + taps[j] * v;
    }
    fb[i] = s;
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let r = i32(P.n / 2u);
    var s = 0.0;
    for (var j = 0; j < i32(P.n); j = j + 1) {
        let y = border(i32(P.y0 + id.y) + j - r, i32(P.fh), BORDER_REFLECT_101) - i32(P.y0);
        var v = 0.0;
        if (y >= 0 && y < i32(P.h)) {
            v = fa[u32(y) * P.w + id.x];
        }
        s = s + taps[j] * v;
    }
    fb[i] = s;
}

// ---------------------------------------------------------------- _grow: the die-cut ring round a coverage
// The box holds the figure and the ring's reach: the distances found in it are the frame's.
// edt_cols: the rows to the nearest pixel of the figure (coverage `fa` >= 0.5) in the same column, FAR past P.n.
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
        if ((up >= 0 && fa[u32(up) * P.w + id.x] >= 0.5) || (down < i32(P.h) && fa[u32(down) * P.w + id.x] >= 0.5)) {
            v = f32(k);
            break;
        }
    }
    fb[i] = v;
}

// edt_rows: the distance d to the nearest pixel of the figure (edt_cols's in `fa`), then the ring before its blur:
// clip(P.a - d, 0, 1), P.a = float32(r + 0.5).
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
        let c = fa[row + u32(q)];
        if (c < FAR) {
            best = min(best, f32(k * k) + c * c);
        }
    }
    let d = select(FAR, sqrt(best), best < FAR);
    fb[i] = clamp(P.a - d, 0.0, 1.0);
}

// grow_finish: the ring (`fa`, blurred when P.flag = 1) sharpened back, clip((g - 0.5) * P.b + 0.5, 0, 1), never under
// the coverage itself (`fc`): np.maximum(g, a).
@compute @workgroup_size(16, 16)
fn grow_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    var g = fa[i];
    if (P.flag == 1u) {
        g = clamp((g - 0.5) * P.b + 0.5, 0.0, 1.0);
    }
    fb[i] = max(g, fc[i]);
}

// ---------------------------------------------------------------- warp (cv2.warpAffine, INTER_LINEAR, WARP_INVERSE_MAP)
// As warp.wgsl: the box's pixel (x, y) reads the plane at sx = fma(P.a, x0 + x, row term), sy = fma(P.b, x0 + x, row
// term) (frame coordinates; `tab` the float32 row terms of the box's rows, as bits), bilinear with fused multiply-adds.
// The plane read is a box (P.sw x P.sh at P.sx0, P.sy0) of a frame P.sfw x P.sfh whose border rule (P.n) applies; a
// pixel of it outside the box reads 0, one outside it the rule's (BORDER_CONSTANT: 0).
fn tap(x: i32, y: i32) -> f32 {
    let bx = border(x, i32(P.sfw), P.n);
    let by = border(y, i32(P.sfh), P.n);
    if (bx < 0 || by < 0) {
        return 0.0;
    }
    let lx = bx - i32(P.sx0);
    let ly = by - i32(P.sy0);
    if (lx < 0 || ly < 0 || lx >= i32(P.sw) || ly >= i32(P.sh)) {
        return 0.0;
    }
    return fa[u32(ly) * P.sw + u32(lx)];
}

@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let fx = f32(P.x0 + id.x);
    let sx = fma(P.a, fx, bitcast<f32>(tab[2u * id.y]));
    let sy = fma(P.b, fx, bitcast<f32>(tab[2u * id.y + 1u]));
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
    let a = fma(p01 - p00, tx, p00);
    let b = fma(p11 - p10, tx, p10);
    fb[i] = fma(b - a, ty, a);
}

// ---------------------------------------------------------------- resizes (cv2.resize of a plane of its own, P.sw x P.sh)
// resize_area: as fx_figure.wgsl's (`tab` the columns' offsets from 0, the rows' from P.flag).
@compute @workgroup_size(16, 16)
fn resize_area(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let ty = P.flag;
    var s = 0.0;
    for (var e = tab[ty + id.y]; e < tab[ty + id.y + 1u]; e = e + 2u) {
        let sy = tab[e];
        let beta = bitcast<f32>(tab[e + 1u]);
        var row = 0.0;
        for (var q = tab[id.x]; q < tab[id.x + 1u]; q = q + 2u) {
            row = row + fa[sy * P.sw + tab[q]] * bitcast<f32>(tab[q + 1u]);
        }
        s = s + row * beta;
    }
    fb[i] = s;
}

// resize_linear: as fx_figure.wgsl's (`tab` per written column (source column, weights), from P.flag per row).
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
    let r0 = fa[sy * P.sw + sx] * a0 + fa[sy * P.sw + sx1] * a1;
    let r1 = fa[sy1 * P.sw + sx] * a0 + fa[sy1 * P.sw + sx1] * a1;
    fb[i] = r0 * b0 + r1 * b1;
}

// ---------------------------------------------------------------- composites over the box of the picture `pic`
// colour_by: under coverage `fa` in P.colour: flag 0 a shadow, a = P.a * fa (_cast); flag 1 an edge's ring, a = fa.
// pic * (1 - a) + colour * a.
@compute @workgroup_size(16, 16)
fn colour_by(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    var a = fa[i];
    if (P.flag == 0u) {
        a = P.a * a;
    }
    let f = framed(id.x, id.y);
    let c = pic[f];
    pic[f] = vec4<f32>(c.xyz * (1.0 - a) + P.colour.xyz * a, c.w);
}

// premultiply: the picture `src` (P.sw wide, its top left at (P.sx0, P.sy0) of the frame: the frame's own picture, or
// a box of it) times coverage `fa` over the box into the box plane `pic` (pic * m[..., None]), the coverage in .w.
@compute @workgroup_size(16, 16)
fn premultiply(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let m = fa[i];
    let s = (P.y0 + id.y - P.sy0) * P.sw + P.x0 + id.x - P.sx0;
    pic[i] = vec4<f32>(src[s].xyz * m, m);
}

// paste: a figure over the box of the picture `pic` (_paste): flag 0 its premultiplied picture (`src`, a box plane)
// and coverage `fa`, rgb + pic * (1 - cov); flag 1 the flat P.colour, colour * cov + pic * (1 - cov).
@compute @workgroup_size(16, 16)
fn paste(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let cov = fa[i];
    var rgb = P.colour.xyz * cov;
    if (P.flag == 0u) {
        rgb = src[i].xyz;
    }
    let f = framed(id.x, id.y);
    let c = pic[f];
    pic[f] = vec4<f32>(rgb + c.xyz * (1.0 - cov), c.w);
}
