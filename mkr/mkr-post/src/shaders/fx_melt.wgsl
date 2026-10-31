// needs sample.wgsl
// mkmmd.core.remap.melt_field and RemapFx._melt: the picture displaced by a melt's field, the pixel at p showing the
// picture at p + (fx, fy) (cv2.remap, INTER_LINEAR, BORDER_REFLECT), or left as it is when the field moves nothing by
// 0.01 px or more (`melt_reach` first: the largest |fx|, |fy| into `reach`). With d the offset from the centre, r = |d|:
//   0 breathe  k = k0 * exp(-(r / su)^2); (fx, fy) = -k * d
//   1 swirl    th = k0 * exp(-(r / su)^2); the pixel turned th about the centre: (c dx - s dy - dx, s dx + c dy - dy)
//   2 ripple   q = r / su; k = a * sin(tau * (q - phase)) * (1 - exp(-q^2)) / max(r, 1e-3), a = k0 or, with a front,
//              the amplitude table at r (np.interp on n points `step` px apart); (fx, fy) = -k * d
//   3 drip     depth = y / h; sag = k0 * depth * sqrt(depth) * col * (0.9 + 0.1 sin(tau (depth - phase) + 3 drip));
//              (0, -sag), drip and col per column
//   4 noise    two smooth lattices carried onto the pixels by cubic B-splines (4 taps per row and per column), divided
//              by `spread`, squeezed by k = k0 / sqrt(1 + fx^2 + fy^2)
// Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    mode: u32,
    n: u32,           // ripple: amplitude table points; noise: lattice columns
    cx: f32,
    cy: f32,
    k0: f32,
    su: f32,
    tau: f32,
    phase: f32,
    step: f32,
    spread: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> table: array<f32>;       // ripple amplitudes; drip (drip, col) per column;
                                                                   // noise row taps, column taps, then the lattices
@group(0) @binding(4) var<storage, read_write> reach: atomic<u32>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    return src[i];
}

// A ripple's amplitude at distance r: np.interp over the table's points 0, step, 2 step, ...
fn amplitude(r: f32) -> f32 {
    let last = P.n - 1u;
    let u = r / P.step;
    if (u <= 0.0) {
        return table[0];
    }
    if (u >= f32(last)) {
        return table[last];
    }
    let j = min(u32(floor(u)), last - 1u);
    let f = u - f32(j);
    return table[j] + (table[j + 1u] - table[j]) * f;
}

// The noise field (fx, fy) before the squeeze: sum over 4 x 4 lattice points of row weight x lattice x column weight,
// the rows' sums first (numpy's (Wy @ L) @ Wx.T).
fn lattice(x: u32, y: u32) -> vec2<f32> {
    let rows = 5u * y;                        // per row: first lattice row, 4 weights
    let cols = 5u * P.h + 5u * x;             // per column: first lattice column, 4 weights
    let lat = 5u * P.h + 5u * P.w;            // the lattices: (rows x n x 2)
    let i0 = u32(table[rows]);
    let j0 = u32(table[cols]);
    var out = vec2<f32>(0.0);
    for (var b = 0u; b < 4u; b = b + 1u) {
        var t = vec2<f32>(0.0);
        for (var a = 0u; a < 4u; a = a + 1u) {
            let at = lat + 2u * ((i0 + a) * P.n + j0 + b);
            t = t + table[rows + 1u + a] * vec2<f32>(table[at], table[at + 1u]);
        }
        out = out + t * table[cols + 1u + b];
    }
    return out / P.spread;
}

fn field(x: u32, y: u32) -> vec2<f32> {
    let dx = f32(x) - P.cx;
    let dy = f32(y) - P.cy;
    let r = sqrt(dx * dx + dy * dy);
    if (P.mode == 0u) {
        let q = r / P.su;
        let k = P.k0 * exp(-(q * q));
        return vec2<f32>(-k * dx, -k * dy);
    }
    if (P.mode == 1u) {
        let q = r / P.su;
        let th = P.k0 * exp(-(q * q));
        let c = cos(th);
        let s = sin(th);
        return vec2<f32>(c * dx - s * dy - dx, s * dx + c * dy - dy);
    }
    if (P.mode == 2u) {
        var a = P.k0;
        if (P.n > 0u) {
            a = amplitude(r);
        }
        let q = r / P.su;
        let ring = sin(P.tau * (q - P.phase));
        let k = a * ring * (1.0 - exp(-(q * q))) / max(r, 1e-3);
        return vec2<f32>(-k * dx, -k * dy);
    }
    if (P.mode == 3u) {
        let depth = f32(y) / f32(P.h);
        let drip = table[2u * x];
        let col = table[2u * x + 1u];
        let run = 0.9 + 0.1 * sin(P.tau * (depth - P.phase) + 3.0 * drip);
        let sag = P.k0 * depth * sqrt(depth) * col * run;
        return vec2<f32>(0.0, -sag);
    }
    let f = lattice(x, y);
    let k = P.k0 / sqrt(1.0 + f.x * f.x + f.y * f.y);
    return vec2<f32>(f.x * k, f.y * k);
}

@compute @workgroup_size(16, 16)
fn melt_reach(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let f = abs(field(id.x, id.y));
    atomicMax(&reach, bitcast<u32>(max(f.x, f.y)));
}

@compute @workgroup_size(16, 16)
fn melt(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    if (bitcast<f32>(atomicLoad(&reach)) <= 0.01) {
        dst[i] = src[i];
        return;
    }
    let f = field(id.x, id.y);
    dst[i] = sample_bilinear(0u, f32(id.x) + f.x, f32(id.y) + f.y, P.w, P.h, BORDER_REFLECT, vec4<f32>(0.0));
}

// melt_keep: under a mask, the picture `src` as it is over the masked work's `dst` where the field moves nothing by
// 0.01 px (as RemapFx._melt leaves it then, unfilled).
@compute @workgroup_size(16, 16)
fn melt_keep(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    if (bitcast<f32>(atomicLoad(&reach)) <= 0.01) {
        let i = id.y * P.w + id.x;
        dst[i] = src[i];
    }
}
