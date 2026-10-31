// The colour fields' shared math (mkmmd.fields): the value noise of mkmmd.fields.noise, the op-art patterns of
// mkmmd.fields.pattern, `bump` and the blends a field is laid on the picture by. A kernel that uses them starts with
// `// needs fields.wgsl` and binds `field_cubic` (binding 10: OpenCV's bicubic weights of the 32 phases of a lattice
// resized 32 times, float32 as interpolateCubic makes them) and `field_calls` (binding 11: its noise calls) when it
// draws noise.
//
// Noise: mkmmd.fields.noise resizes the hashed lattice NOISE_UP = 32 times by bicubic (cv2.resize) and reads it
// linearly (cv2.remap) at (X - (x0 - 0.5)) * 32 - 0.5 for a lattice box starting at x0. The fine samples sit at the
// same lattice positions whatever the box (x0 is whole), so here they are indexed from lattice 0: fine sample J lies
// at lattice position J / 32 + 1 / 64 - 1 / 2, and the noise at X reads between fine samples about 32 X + 15.5.

struct NoiseCall {
    k: i32,             // the time slice: slices k - 1 .. k + 2
    seed: i32,
    pad0: u32,
    pad1: u32,
    w: vec4<f32>,       // their Catmull-Rom weights (mkmmd.fields._catmull, float32)
}

@group(0) @binding(10) var<storage, read> field_cubic: array<vec4<f32>, 32>;
@group(0) @binding(11) var<storage, read> field_calls: array<NoiseCall>;

const NOISE_UP: f32 = 32.0;

// mkmmd.fields._hash: uniform 0..1 of lattice point (ix, iy) in slice k (the uint64 products keep only their low 32
// bits there, as these do).
fn field_hash(ix: i32, iy: i32, k: i32, seed: i32) -> f32 {
    let salt = u32(k) * 0xCB1AB31Fu + u32(seed) * 0x9E3779B1u + 0x632BE5ABu;
    var h = (u32(ix) * 0x8DA6B343u) ^ (u32(iy) * 0xD8163841u) ^ salt;
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) / 16777216.0;
}

// The lattice of a call at (ix, iy): its four slices weighted, summed in order from 0.
fn field_lattice(ix: i32, iy: i32, c: NoiseCall) -> f32 {
    var lat = 0.0;
    lat = lat + c.w.x * field_hash(ix, iy, c.k - 1, c.seed);
    lat = lat + c.w.y * field_hash(ix, iy, c.k, c.seed);
    lat = lat + c.w.z * field_hash(ix, iy, c.k + 1, c.seed);
    lat = lat + c.w.w * field_hash(ix, iy, c.k + 2, c.seed);
    return lat;
}

// The lattice column (row) of the first of the four taps of fine sample J: floor of its position, minus one.
fn field_first(j: i32) -> i32 {
    let n = j >> 5u;
    return select(n, n - 1, (j & 31) < 16) - 1;
}

var<private> field_win: array<f32, 25>;     // the 5 x 5 lattice window the four fine samples about a point read

// Fine sample (J, I) from the window whose first column and row are (cx0, cy0): the bicubic of the lattice, along the
// rows first, each sum in order.
fn field_fine(j: i32, i: i32, cx0: i32, cy0: i32) -> f32 {
    let ox = field_first(j) - cx0;
    let oy = field_first(i) - cy0;
    let a = field_cubic[j & 31];
    let b = field_cubic[i & 31];
    var h: array<f32, 4>;
    for (var r = 0; r < 4; r = r + 1) {
        let row = (oy + r) * 5 + ox;
        h[r] = ((field_win[row] * a.x + field_win[row + 1] * a.y) + field_win[row + 2] * a.z) + field_win[row + 3] * a.w;
    }
    return ((h[0] * b.x + h[1] * b.y) + h[2] * b.z) + h[3] * b.w;
}

// mkmmd.fields.noise of call `ci` at lattice coordinates (X, Y).
fn field_noise(x: f32, y: f32, ci: u32) -> f32 {
    let c = field_calls[ci];
    let u = x * NOISE_UP + 15.5;
    let v = y * NOISE_UP + 15.5;
    let fu = floor(u);
    let fv = floor(v);
    let tx = u - fu;
    let ty = v - fv;
    let j = i32(fu);
    let i = i32(fv);
    let cx0 = field_first(j);
    let cy0 = field_first(i);
    for (var r = 0; r < 5; r = r + 1) {
        for (var q = 0; q < 5; q = q + 1) {
            field_win[r * 5 + q] = field_lattice(cx0 + q, cy0 + r, c);
        }
    }
    let p00 = field_fine(j, i, cx0, cy0);
    let p01 = field_fine(j + 1, i, cx0, cy0);
    let p10 = field_fine(j, i + 1, cx0, cy0);
    let p11 = field_fine(j + 1, i + 1, cx0, cy0);
    let a = fma(p01 - p00, tx, p00);
    let b = fma(p11 - p10, tx, p10);
    return fma(b - a, ty, a);
}

// mkmmd.fields.bump: (1 - x^2)^2 inside +-1, 0 beyond.
fn field_bump(x: f32) -> f32 {
    let b = clamp(1.0 - x * x, 0.0, 1.0);
    return b * b;
}

// mkmmd.fields.blend: `pic` laid on `img` by `mode` (0 over, 1 screen, 2 multiply, 3 soft light), mixed in by `a`.
fn field_blend(img: vec3<f32>, pic: vec3<f32>, a: f32, mode: u32) -> vec3<f32> {
    var top = pic;
    if (mode == 1u) {
        top = 1.0 - (1.0 - img) * (1.0 - pic);
    } else if (mode == 2u) {
        top = img * pic;
    } else if (mode == 3u) {
        top = (1.0 - 2.0 * pic) * img * img + 2.0 * pic * img;
    }
    return img + (top - img) * a;
}

// ---------------------------------------------------------------- op-art patterns (mkmmd.fields.pattern)
struct Pattern {
    kind: u32,          // 0 spiral, 1 rings, 2 sunburst, 3 moire, 4 checker
    m: u32,             // colours
    moire: u32,         // a moire's sets: 0 rings, 1 lines
    pad0: u32,
    n: f32,             // spiral, sunburst: colours x arms
    tw: f32,            // spiral: twist / 0.5
    size: f32,          // rings, moire, checker: a stripe's size in units
    q0: f32,            // the pattern moved on by q0 stripes
    ca: f32,            // moire lines, checker: cos and sin of the angle
    sa: f32,
    o: f32,             // moire rings: the centres' offset (offset x size / 2)
    warp: f32,          // checker: BULGE[0] x warp
    g: f32,             // rings, moire: stripes per px, 1 / (size x unit) as float64 makes it
    unit: f32,
    c2k: f32,           // checker: float32(2 / BULGE[1]^2)
    pad2: f32,
    colors: array<vec4<f32>, 4>,
    mean: vec4<f32>,    // the colours' mean as numpy makes it
}

const TWO_PI: f32 = 6.2831855;      // np.float32(2 * pi)
const BULGE1: f32 = 0.42;

// _axis's antialiased stripe: (k, lo, hi) of stripe coordinate q at g stripes per px.
fn pattern_axis(q: f32, g: f32) -> vec3<f32> {
    let k = floor(q);
    let f = q - k;
    let gg = max(g, 1e-6);
    return vec3<f32>(k, clamp(0.5 - f / gg, 0.0, 1.0), clamp(0.5 - (1.0 - f) / gg, 0.0, 1.0));
}

fn pattern_fine(g: f32) -> f32 {
    return clamp(2.0 * g - 1.0, 0.0, 1.0);
}

fn pattern_mod(k: i32, m: u32) -> u32 {
    let mm = i32(m);
    return u32(((k % mm) + mm) % mm);
}

fn pattern_parity(q: f32, g: f32) -> f32 {
    let a = pattern_axis(q, g);
    let odd = f32(i32(a.x) & 1);
    let mid = 1.0 - a.y - a.z;
    let v = mid * odd + (a.y + a.z) * (1.0 - odd);
    return v + (0.5 - v) * pattern_fine(g);
}

// The pattern's colour at (dx, dy) units from its centre (r, th: their distance and angle, as mkmmd.fields.polar).
fn pattern_colour(p: Pattern, dx: f32, dy: f32, r: f32, th: f32) -> vec3<f32> {
    let m = p.m;
    if (p.kind <= 2u) {
        var q: f32;
        var g: f32;
        if (p.kind == 1u) {
            q = r / p.size - p.q0;
            g = p.g;
        } else {
            q = p.n * th / TWO_PI - p.q0;
            g = p.n / (TWO_PI * max(r, 1e-6) * p.unit);
            if (p.kind == 0u) {
                q = q + p.n * p.tw * r;
                let gt = p.n * p.tw / p.unit;
                g = sqrt(g * g + gt * gt);
            }
        }
        let a = pattern_axis(q, g);
        let k = i32(a.x);
        let j = pattern_mod(k, m);
        let c = p.colors[j].xyz;
        var out = c;
        if (a.y > 0.0) {
            out = out + a.y * (p.colors[pattern_mod(k - 1, m)].xyz - c);
        }
        if (a.z > 0.0) {
            out = out + a.z * (p.colors[pattern_mod(k + 1, m)].xyz - c);
        }
        if (p.kind != 1u || g > 0.5) {
            let fine = pattern_fine(g);
            if (fine > 0.0) {
                out = out + (p.mean.xyz - out) * fine;
            }
        }
        return out;
    }
    if (p.kind == 3u) {
        var q1: f32;
        var q2: f32;
        if (p.moire == 1u) {
            q1 = dx / p.size;
            q2 = (dx * p.ca - dy * p.sa) / p.size - p.q0;
        } else {
            q1 = sqrt((dx + p.o) * (dx + p.o) + dy * dy) / p.size;
            q2 = sqrt((dx - p.o) * (dx - p.o) + dy * dy) / p.size - p.q0;
        }
        let v1 = pattern_parity(q1, p.g);
        let v2 = pattern_parity(q2, p.g);
        if (m == 2u) {
            return p.colors[0].xyz + (v1 + v2 - 2.0 * v1 * v2) * (p.colors[1].xyz - p.colors[0].xyz);
        }
        let n1 = 1.0 - v1;
        let n2 = 1.0 - v2;
        var out = vec3<f32>(0.0);
        if (m == 3u) {
            out = out + (n1 * n2) * p.colors[0].xyz;
            out = out + (v1 * n2 + n1 * v2) * p.colors[1].xyz;
            out = out + (v1 * v2) * p.colors[2].xyz;
        } else {
            out = out + (n1 * n2) * p.colors[0].xyz;
            out = out + (v1 * n2) * p.colors[1].xyz;
            out = out + (n1 * v2) * p.colors[2].xyz;
            out = out + (v1 * v2) * p.colors[3].xyz;
        }
        return out;
    }
    // the checkerboard, turned and bulged
    let x = dx * p.ca - dy * p.sa;
    let y = dx * p.sa + dy * p.ca;
    let rb = r / BULGE1;
    let e = p.warp * exp(-(rb * rb));
    let kf = 1.0 - e;
    let q1 = x * kf / p.size - p.q0;
    let q2 = y * kf / p.size - p.q0;
    let c2 = p.c2k * e;
    let j12 = c2 * x * y;
    let su = p.size * p.unit;
    let ga = kf + c2 * x * x;
    let gb = kf + c2 * y * y;
    let g1 = sqrt(ga * ga + j12 * j12) / su;
    let g2 = sqrt(j12 * j12 + gb * gb) / su;
    let a1 = pattern_axis(q1, g1);
    let a2 = pattern_axis(q2, g2);
    let k = i32(a1.x) + i32(a2.x);
    let j = pattern_mod(k, m);
    let c = p.colors[j].xyz;
    var out = c;
    if (a1.y > 0.0 || a1.z > 0.0 || a2.y > 0.0 || a2.z > 0.0) {
        let m1 = 1.0 - a1.y - a1.z;
        let m2 = 1.0 - a2.y - a2.z;
        var add = vec3<f32>(0.0);
        add = add + (a1.y * a2.y) * (p.colors[pattern_mod(k - 2, m)].xyz - c);
        add = add + (a1.y * m2 + m1 * a2.y) * (p.colors[pattern_mod(k - 1, m)].xyz - c);
        add = add + (a1.z * m2 + m1 * a2.z) * (p.colors[pattern_mod(k + 1, m)].xyz - c);
        add = add + (a1.z * a2.z) * (p.colors[pattern_mod(k + 2, m)].xyz - c);
        out = out + add;
    }
    let fine = pattern_fine(max(g1, g2));
    if (fine > 0.0) {
        out = out + (p.mean.xyz - out) * fine;
    }
    return out;
}

// mkmmd.fields.polar at pixel (x, y) about pixel `c` in units of `unit` px: (dx, dy, r, theta).
fn field_polar(x: u32, y: u32, c: vec2<f32>, unit: f32) -> vec4<f32> {
    let dx = (f32(x) - c.x) / unit;
    let dy = (f32(y) - c.y) / unit;
    return vec4<f32>(dx, dy, sqrt(dx * dx + dy * dy), atan2(-dy, dx));
}
