// needs sample.wgsl
// camcorder (mkmmd.picturefx.tape; the stamp is laid after by fx_inks.wgsl): a picture as a home video tape plays it,
// in float32 as OpenCV and numpy work it out. `split` takes it to YCrCb (cv2.COLOR_RGB2YCrCb, the colour less its
// half); the luma's softness and the colour's Gaussian are fx_figure.wgsl's blurs, `bleed` the colour's box along the
// rows (cv2.blur, BORDER_REFLECT_101), `shift` moves it right (cv2.warpAffine of a two-channel picture, which OpenCV 5
// still draws the old way: coordinates in 1/32 px); `combine` scales the colour by the saturation, adds the noise
// stretched over the picture (cv2.resize INTER_LINEAR of Python's normals) and goes back to RGB
// (cv2.COLOR_YCrCb2RGB); `lines` tears each line sideways (cv2.remap INTER_LINEAR, BORDER_REFLECT_101), washes the
// tracking band out under its snow (stretched INTER_NEAREST), darkens the scanlines and clips.
//
// `tab` holds per written column (source column, its weight, the next one's: OpenCV's INTER_LINEAR) the luma noise's
// columns from 0 and its rows from 3w, the colour noise's columns from 3w + 3h and its rows from 6w + 3h; from
// 6w + 6h the snow's source column per written one (INTER_NEAREST).

struct Params {
    w: u32,
    h: u32,
    n: u32,             // the colour's box along the rows (px)
    noise: u32,         // 1: noise added
    tear: u32,          // 1: the lines torn
    snow: u32,          // 1: a tracking band with snow
    nyw: u32,           // the luma noise's size
    ncw: u32,           // the colour noise's size (two values per pixel)
    nch: u32,
    sw: u32,            // the snow's width (its height is the picture's)
    nyh: u32,
    qx: i32,            // `shift`: cvRound(-dx * 1024) + 16, the move in 1/1024 px and half a step of 1/32
    sat: f32,           // float32 saturation
    ny_amt: f32,        // float32 noise, and half of it
    nc_amt: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ya: array<vec4<f32>>;     // the luma (.x)
@group(0) @binding(4) var<storage, read_write> ca: array<vec4<f32>>;     // the colour less its half (Cr, Cb in .xy)
@group(0) @binding(5) var<storage, read> ny: array<f32>;
@group(0) @binding(6) var<storage, read> nc: array<f32>;
@group(0) @binding(7) var<storage, read> tab: array<u32>;
@group(0) @binding(8) var<storage, read> rows: array<vec4<f32>>;   // per line: tear (px), wash, snow, scanline factor
@group(0) @binding(9) var<storage, read> snow: array<f32>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    return src[i];
}

fn at(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

@compute @workgroup_size(16, 16)
fn split(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let s = src[i];
    let p = clamp(s.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let y = p.x * 0.299 + p.y * 0.587 + p.z * 0.114;
    let cr = (p.x - y) * 0.713 + 0.5;
    let cb = (p.z - y) * 0.564 + 0.5;
    ya[i] = vec4<f32>(y, 0.0, 0.0, s.w);
    ca[i] = vec4<f32>(cr - 0.5, cb - 0.5, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn bleed(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let r = i32(P.n / 2u);
    let row = id.y * P.w;
    var s = vec4<f32>(0.0);
    for (var j = 0; j < i32(P.n); j = j + 1) {
        s = s + src[row + u32(border(i32(id.x) + j - r, i32(P.w), BORDER_REFLECT_101))];
    }
    dst[i] = s * (1.0 / f32(P.n));
}

// cv2.warpAffine(c, [[1, 0, dx], [0, 1, 0]], INTER_LINEAR, BORDER_REPLICATE) of the colour (two channels) as OpenCV's
// old warp draws it: the source column in 1/32 px, (x * 1024 + qx) >> 5, weighted by its 32nds (the rows stay put).
@compute @workgroup_size(16, 16)
fn shift(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let q = (i32(id.x) * 1024 + P.qx) >> 5u;
    let sx = q >> 5u;
    let b = f32(q & 31) * (1.0 / 32.0);
    let last = i32(P.w) - 1;
    let row = id.y * P.w;
    let p0 = src[row + u32(clamp(sx, 0, last))];
    let p1 = src[row + u32(clamp(sx + 1, 0, last))];
    dst[i] = p0 * (1.0 - b) + p1 * b;
}

// OpenCV's INTER_LINEAR weights at `c` in `tab`: (source px, its weight, the next one's).
fn lin(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(tab[c]), bitcast<f32>(tab[c + 1u]), bitcast<f32>(tab[c + 2u]));
}

@compute @workgroup_size(16, 16)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let l = ya[i];
    var y = l.x;
    var c = ca[i].xy * P.sat;
    if (P.noise == 1u) {
        let w3 = 3u * P.w;
        let h3 = 3u * P.h;
        // the luma's noise
        let ax = lin(3u * id.x);
        let ay = lin(w3 + 3u * id.y);
        let sx = u32(ax.x);
        let sx1 = min(sx + 1u, P.nyw - 1u);
        let sy = u32(ay.x);
        let sy1 = min(sy + 1u, P.nyh - 1u);
        let r0 = ny[sy * P.nyw + sx] * ax.y + ny[sy * P.nyw + sx1] * ax.z;
        let r1 = ny[sy1 * P.nyw + sx] * ax.y + ny[sy1 * P.nyw + sx1] * ax.z;
        y = y + P.ny_amt * (r0 * ay.y + r1 * ay.z);
        // the colour's, two values a pixel
        let bx = lin(w3 + h3 + 3u * id.x);
        let by = lin(2u * w3 + h3 + 3u * id.y);
        let tx = u32(bx.x);
        let tx1 = min(tx + 1u, P.ncw - 1u);
        let ty = u32(by.x);
        let ty1 = min(ty + 1u, P.nch - 1u);
        let a00 = 2u * (ty * P.ncw + tx);
        let a01 = 2u * (ty * P.ncw + tx1);
        let a10 = 2u * (ty1 * P.ncw + tx);
        let a11 = 2u * (ty1 * P.ncw + tx1);
        let q0 = vec2<f32>(nc[a00], nc[a00 + 1u]) * bx.y + vec2<f32>(nc[a01], nc[a01 + 1u]) * bx.z;
        let q1 = vec2<f32>(nc[a10], nc[a10 + 1u]) * bx.y + vec2<f32>(nc[a11], nc[a11 + 1u]) * bx.z;
        c = c + P.nc_amt * (q0 * by.y + q1 * by.z);
    }
    let held = c + 0.5;                       // as np.dstack holds it, then cvtColor takes the half off again
    let cr = held.x - 0.5;
    let cb = held.y - 0.5;
    let b = y + cb * 1.773;
    let g = y + cb * -0.344 + cr * -0.714;
    let r = y + cr * 1.403;
    dst[i] = vec4<f32>(r, g, b, l.w);
}

@compute @workgroup_size(16, 16)
fn lines(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let line = rows[id.y];
    let here = src[i];
    var out = here.xyz;
    if (P.tear == 1u) {
        out = sample_bilinear(0u, f32(id.x) + line.x, f32(id.y), P.w, P.h, BORDER_REFLECT_101, vec4<f32>(0.0)).xyz;
    }
    if (P.snow == 1u) {
        let s = snow[id.y * P.sw + tab[6u * P.w + 6u * P.h + id.x]] * line.z;
        let grey = (out.x + out.y + out.z) / 3.0;
        out = out * (1.0 - line.y) + vec3<f32>(grey * 0.8 + 0.2) * line.y;
        out = out + (1.0 - out) * (0.85 * s);
    }
    out = out * line.w;
    dst[i] = vec4<f32>(clamp(out, vec3<f32>(0.0), vec3<f32>(1.0)), here.w);
}
