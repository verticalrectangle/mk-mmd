// shapes (mkmmd.shapes.picture; FrameFx._shapes): the picture as flat shapes, a cell each. Python (mkmmd.gpufx.shapes)
// writes the frame's tables (`tab`, float32) for each layout (the new one, and the one before while the new one pops
// in): per cell 8 numbers (the pixel its colour is read at, its shape's code, its ink, the cosine and sine of its turn,
// its radius, 1 once its shape has begun to pop in), per picture row 4 (the row's first cell, the pixel's offset down
// from the cell's middle, 1 in an offset row), per picture column 2 for each of the two kinds of row (the cell's column,
// the pixel's offset along from its middle); then the inks (4 each).
//   cells_one / cells_group   each cell's colour: the picture's box blur (cv2.blur, a box of the layer's k, BORDER_REFLECT)
//                             at the cell's pixel, then the ink it takes and whether it draws at all (a flat back's
//                             skip). A thread a cell (a small box) or a workgroup of 256 a cell.
//   draw                      each pixel's cell and its shape's signed distance there, the alpha (where the new
//                             layout's shape has not begun, the layout before's), the colour over the picture or the
//                             flat back, kept to the mask's region, mixed in by the strength.
// Each step in numpy's order and float32.

// One layout of cells: `cells` of them in rows `columns` wide (box blur side `k`), its tables starting at `at` in
// `tab` (the cells, then the picture's rows, then its columns), its cells' colours at `paint_at` in `paint`.
struct Layer {
    cells: u32,
    columns: u32,
    k: u32,
    at: u32,
    rows_at: u32,
    cols_at: u32,
    paint_at: u32,
    pad: u32,
}

struct Params {
    w: u32,
    h: u32,
    inks: u32,
    inks_at: u32,       // where the inks start in `tab` (4 each)
    masked: u32,        // 0 none, 1 the figure, 2 the ground
    flat: u32,          // 1: on a flat back (a cell near it draws nothing)
    mixed: u32,         // 1: s < 1
    layers: u32,        // 2: the layout before shows where the new one's shapes have not begun
    which: u32,         // the layer a cells pass works on
    pad0: u32,
    pad1: u32,
    pad2: u32,
    skip: f32,
    s: f32,             // the strength and 1 - it
    s1: f32,
    pad3: f32,
    back: vec4<f32>,
    a: Layer,
    b: Layer,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tab: array<f32>;
@group(0) @binding(4) var<storage, read_write> paint: array<vec4<f32>>;   // each cell's colour, and 1 in w where it draws
@group(0) @binding(5) var<storage, read> fig: array<u32>;
@group(0) @binding(6) var<storage, read> lut: array<f32>;

fn layer_of(which: u32) -> Layer {
    if (which == 1u) {
        return P.b;
    }
    return P.a;
}

const SQ3: f32 = 1.7320508075688772;
const HALF_SQ3: f32 = 0.8660254037844386;
const INV_SQ3: f32 = 0.5773502691896258;
const INV_SQ2: f32 = 0.7071067811865476;

// Cell `i`'s colour from the sum `sum` of the picture under it.
fn finish(L: Layer, i: u32, sum: vec3<f32>) {
    var col = sum;
    if (L.k > 1u) {
        col = sum / f32(L.k * L.k);
    }
    var on = 1.0;
    if (P.flat == 1u) {
        let d = col - P.back.xyz;
        let e = d.x * d.x + d.y * d.y + d.z * d.z;
        on = select(0.0, 1.0, sqrt(e / 3.0) > P.skip);
    }
    var out = col;
    if (P.inks > 0u) {
        var j = i32(tab[L.at + i * 8u + 3u]);
        if (j < 0) {
            var best = 3.4e38;
            for (var n = 0u; n < P.inks; n++) {
                let at = P.inks_at + 4u * n;
                let d = col - vec3<f32>(tab[at], tab[at + 1u], tab[at + 2u]);
                let e = d.x * d.x + d.y * d.y + d.z * d.z;
                if (e < best) {
                    best = e;
                    j = i32(n);
                }
            }
        }
        let at = P.inks_at + 4u * u32(j);
        out = vec3<f32>(tab[at], tab[at + 1u], tab[at + 2u]);
    }
    paint[L.paint_at + i] = vec4<f32>(out, on);
}

// The sum over the cell's box of the pixels t, t + stride, ... of it (row by row), the box's reach reflected at the
// picture's edge.
fn box_at(L: Layer, i: u32, t: u32, stride: u32) -> vec3<f32> {
    let cx = i32(tab[L.at + i * 8u]);
    let cy = i32(tab[L.at + i * 8u + 1u]);
    let side = i32(L.k);
    let x0 = cx - side / 2;
    let y0 = cy - side / 2;
    let w = i32(P.w);
    let h = i32(P.h);
    var sum = vec3<f32>(0.0);
    for (var n = t; n < L.k * L.k; n += stride) {
        let y = border(y0 + i32(n / L.k), h, BORDER_REFLECT);
        let x = border(x0 + i32(n % L.k), w, BORDER_REFLECT);
        sum += src[u32(y * w + x)].xyz;
    }
    return sum;
}

@compute @workgroup_size(64)
fn cells_one(@builtin(global_invocation_id) id: vec3<u32>) {
    let L = layer_of(P.which);
    if (id.x >= L.cells) {
        return;
    }
    if (L.k <= 1u) {
        let cx = u32(tab[L.at + id.x * 8u]);
        let cy = u32(tab[L.at + id.x * 8u + 1u]);
        finish(L, id.x, src[cy * P.w + cx].xyz);
        return;
    }
    finish(L, id.x, box_at(L, id.x, 0u, 1u));
}

var<workgroup> part: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn cells_group(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let L = layer_of(P.which);
    let i = wg.y * 65535u + wg.x;
    let live = i < L.cells;
    var sum = vec3<f32>(0.0);
    if (live) {
        sum = box_at(L, i, t, 256u);
    }
    part[t] = sum;
    for (var s = 128u; s > 0u; s = s >> 1u) {
        workgroupBarrier();
        if (t < s) {
            part[t] = part[t] + part[t + s];
        }
    }
    if (t == 0u && live) {
        finish(L, i, part[0]);
    }
}

// The signed distance (px, negative inside) of a shape of circumradius `rad` about its cell's middle, at offsets
// (up, vp) from it in the shape's own turned frame (mkmmd.shapes._sdf): 0 circle, 1 square, 2 diamond, 3 triangle.
fn sdf(kind: u32, up: f32, vp: f32, rad: f32) -> f32 {
    if (kind == 0u) {
        return sqrt(up * up + vp * vp) - rad;
    }
    if (kind == 1u) {
        return max(abs(up), abs(vp)) - rad * INV_SQ2;
    }
    if (kind == 2u) {
        return (abs(up) + abs(vp) - rad) * INV_SQ2;
    }
    let rh = rad * HALF_SQ3;
    var px = abs(up) - rh;
    var py = -vp + rh * INV_SQ3;
    if (px + SQ3 * py > 0.0) {
        let qx = (px - SQ3 * py) * 0.5;
        let qy = (-SQ3 * px - py) * 0.5;
        px = qx;
        py = qy;
    }
    px = px - clamp(px, -2.0 * rh, 0.0);
    return -sqrt(px * px + py * py) * sign(py);
}

// What layer `L` puts at pixel (x, y): its cell's colour in xyz, the shape's coverage there (0 where the cell draws
// nothing) in w; `begun` whether its cell's shape has begun to pop in.
struct Shaded {
    paint: vec4<f32>,
    begun: bool,
}

fn shade(L: Layer, x: u32, y: u32) -> Shaded {
    let r = L.rows_at + 4u * y;
    let odd = u32(tab[r + 2u]);
    let c = L.cols_at + 2u * (odd * P.w + x);
    let cell = u32(tab[r]) + u32(tab[c]);
    let u = tab[c + 1u];
    let v = tab[r + 1u];
    let at = L.at + cell * 8u;
    let ca = tab[at + 4u];
    let sa = tab[at + 5u];
    let up = ca * u + sa * v;
    let vp = ca * v - sa * u;
    let alpha = clamp(0.5 - sdf(u32(tab[at + 2u]), up, vp, tab[at + 6u]), 0.0, 1.0);
    let p = paint[L.paint_at + cell];
    return Shaded(vec4<f32>(p.xyz, alpha * p.w), tab[at + 7u] != 0.0);
}

@compute @workgroup_size(16, 16)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let img = src[i];
    var sh = shade(P.a, id.x, id.y);
    if (P.layers == 2u && !sh.begun) {
        sh = shade(P.b, id.x, id.y);
    }
    let base = select(img.xyz, P.back.xyz, P.flat == 1u);
    var out = base + (sh.paint.xyz - base) * sh.paint.w;
    if (P.masked != 0u) {
        let m = lut[fig[i] & 255u];
        let work = select(1.0 - m, m, P.masked == 1u);
        out = img.xyz + (out - img.xyz) * work;
    }
    if (P.mixed == 1u) {
        out = img.xyz * P.s1 + out * P.s;
    }
    dst[i] = vec4<f32>(out, img.w);
}
