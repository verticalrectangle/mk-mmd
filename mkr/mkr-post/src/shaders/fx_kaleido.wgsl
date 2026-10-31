// needs sample.wgsl
// mkmmd.remap.kaleido and mkmmd.core.remap.kaleido_coords: the frame folded into n wedges about the centre c, each
// showing the source wedge about s (mirrored in turn), turned, spun and zoomed. With d = (x - cx, cy - y) (y up):
//   r = |d|, rel = atan2(d) - off1, folded into the first wedge (mirror: mod(rel, 2a), reflected past a; plain copies:
//   mod(rel, a)), psi = fold + off2, rho = r * inv_zoom (with a radius R folded back, R - |mod(rho, 2R) - R|), and the
//   pixel reads (sx + rho cos psi, sy - rho sin psi) (cv2.remap, INTER_LINEAR, BORDER_REFLECT), from the pyramid's
//   levels l0 (weight w0) and l1 (w1) at (sx, sy) * k (remap.sample).
//   coords_half   the map at half resolution (remap.half_map: each coarse point where cv2.resize reads it), for
//                 mirrored wedges (its creases are smooth); resized to the frame on the CPU's tables (figure.rs)
//   sample_map    the picture at the map's coordinates
//   sample_plain  plain copies: the map worked out at every pixel and the pixels within one of a seam blended with
//                 the copy across it by clip(0.5 + d, 0, 1) (remap.seam_blend)
// Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    hw: u32,            // the half map's size
    hh: u32,
    lw0: u32,           // the pyramid levels read
    lh0: u32,
    lw1: u32,
    lh1: u32,
    two: u32,           // 1: a second level is blended in
    has_radius: u32,
    pad0: u32,
    pad1: u32,
    c: vec2<f32>,
    s: vec2<f32>,
    off1: f32,
    off2: f32,
    a: f32,             // the wedge, 2 pi / n
    a2: f32,            // 2 a
    inv_zoom: f32,
    radius: f32,
    kx: f32,            // half map: w / hw, h / hh
    ky: f32,
    w0: f32,
    w1: f32,
    k0: f32,
    k1: f32,
    mirror: u32,
    pad2: u32,
    half_a: f32,        // seams: a / 2
    pi2: f32,           // pi / 2
    off2p: f32,         // the copies across a seam: off2 for turn + one wedge and turn - one
    off2m: f32,
    pad3: f32,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> lev0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lev1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> map: array<vec4<f32>>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    if (which == 0u) {
        return lev0[i];
    }
    return lev1[i];
}

// np.mod for float32: the remainder with the divisor's sign.
fn modf32(x: f32, y: f32) -> f32 {
    let r = x - y * floor(x / y);
    return select(r, 0.0, r >= y);
}

fn coords(xs: f32, ys: f32, off2: f32, mirror: bool) -> vec2<f32> {
    let dx = xs - P.c.x;
    let dy = P.c.y - ys;
    let r = sqrt(dx * dx + dy * dy);
    let rel = atan2(dy, dx) - P.off1;
    var f: f32;
    if (mirror) {
        let u = modf32(rel, P.a2);
        f = select(u, P.a2 - u, u > P.a);
    } else {
        f = modf32(rel, P.a);
    }
    let psi = f + off2;
    var rho = r * P.inv_zoom;
    if (P.has_radius != 0u) {
        let R = P.radius;
        rho = R - abs(modf32(rho, 2.0 * R) - R);
    }
    return vec2<f32>(P.s.x + rho * cos(psi), P.s.y - rho * sin(psi));
}

// remap.sample: the levels' remaps, each times its weight, summed.
fn lods(sx: f32, sy: f32) -> vec4<f32> {
    var out: vec4<f32>;
    if (P.k0 == 1.0) {
        out = sample_bilinear(0u, sx, sy, P.lw0, P.lh0, BORDER_REFLECT, vec4<f32>(0.0)) * P.w0;
    } else {
        out = sample_bilinear(0u, sx * P.k0, sy * P.k0, P.lw0, P.lh0, BORDER_REFLECT, vec4<f32>(0.0)) * P.w0;
    }
    if (P.two != 0u) {
        out = out + sample_bilinear(1u, sx * P.k1, sy * P.k1, P.lw1, P.lh1, BORDER_REFLECT, vec4<f32>(0.0)) * P.w1;
    }
    return out;
}

@compute @workgroup_size(16, 16)
fn coords_half(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.hw || id.y >= P.hh) {
        return;
    }
    let xs = (f32(id.x) + 0.5) * P.kx - 0.5;
    let ys = (f32(id.y) + 0.5) * P.ky - 0.5;
    dst[id.y * P.hw + id.x] = vec4<f32>(coords(xs, ys, P.off2, true), 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn sample_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let q = map[i];
    dst[i] = lods(q.x, q.y);
}

@compute @workgroup_size(16, 16)
fn sample_plain(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let xs = f32(id.x);
    let ys = f32(id.y);
    let q = coords(xs, ys, P.off2, false);
    var out = lods(q.x, q.y);
    // the seams between plain copies (remap.seam_blend, core.remap.kaleido_seams)
    let dx = xs - P.c.x;
    let dy = P.c.y - ys;
    let r = sqrt(dx * dx + dy * dy);
    let u = modf32(atan2(dy, dx) - P.off1, P.a);
    let near_start = u < P.half_a;
    let d = r * sin(min(select(P.a - u, u, near_start), P.pi2));
    if (d < 1.0) {
        let o = coords(xs, ys, select(P.off2m, P.off2p, near_start), false);
        let wt = clamp(0.5 + d, 0.0, 1.0);
        out = out * wt + lods(o.x, o.y) * (1.0 - wt);
    }
    dst[id.y * P.w + id.x] = out;
}
