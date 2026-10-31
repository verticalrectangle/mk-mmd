// needs fields.wgsl
// liquid (mkmmd.fields.liquid; FieldFx._liquid): the light show, step by step as the CPU draws it. On the work grid
// (`f` render px per cell): the water's swirl (liquid_psi, then a Gaussian blur, then liquid_warp: the warped
// coordinates and the dye clouds), each dish's oil field (liquid_field) and its slope (liquid_slope), the oil cells
// (liquid_cells). On the render picture (those resized up by bicubic): the dyed water (liquid_dye), each dish's oil laid
// on it (liquid_oil), the cell walls (liquid_walls). On the frame (the render picture resized up): the bubbles and the
// lay (liquid_lay).

struct Params {
    ww: u32,            // the work grid
    hh: u32,
    rw: u32,            // the render picture
    rh: u32,
    w: u32,             // the frame
    h: u32,
    f: u32,             // render px per work cell
    nink: u32,
    dish: u32,          // this oil dish's noise call
    clear: u32,         // 1: the clear oil's dish
    mask: u32,          // 0 none, 1 the figure, 2 the ground
    blend: u32,
    nballs: u32,
    nbubbles: u32,
    cells_call: u32,    // the oil cells' patch noise call
    cells_seed: i32,    // and their Worley hash's seed
    rcx: f32,           // the centre on the render picture (px)
    rcy: f32,
    runit: f32,         // the render picture's short side (px)
    z: f32,             // float32(1 + pulse x swell)
    lw: f32,            // float32 of 0.6 x scale, 0.5 x scale (the swirl's and the dye's sizes)
    lc: f32,
    amp: f32,           // float32(SWIRL x scale x lw)
    gdx: f32,           // float32(f / runit): the work grid's spacing in units
    level: f32,
    flare: f32,
    aa: f32,
    rpx17: f32,         // float32 of 1.7, 4 and 2.8 render px, and 0.06 render units
    rpx4: f32,
    rpx28: f32,
    runit06: f32,
    fi: f32,            // float32(f)
    rough: f32,         // float32(0.25 x scale): the oil's roughness noise size
    marble: f32,
    sizeu: f32,         // the oil cells: float32 of 0.06 x scale, half of it, 0.45 x scale (the patches)
    sizeu05: f32,
    patchk: f32,
    cut: f32,           // float32(CELL_CUT[0] - CELL_CUT[1] x cells)
    px15: f32,          // float32 of 1.5 render px, 0.06 x scale x runit x 0.22, 2.4 and 1.4 render px
    wallk: f32,
    px24: f32,
    px14: f32,
    strength: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    ca: f32,            // this oil dish's turn (cos, sin) and pivot (units)
    sa: f32,
    pvx: f32,
    pvy: f32,
    pad3: f32,
    pad4: f32,
    light: vec4<f32>,
    unink: vec4<f32>,   // float32(1 - ink) of this oil dish's ink
    bub: vec4<f32>,     // float32 of 1.4, 1.8, 1.5 and 1.6 px of the frame (the bubbles)
}

// An ink's dish of dye: its turn (cos, sin) and pivot (units), its colour.
struct Dye {
    ca: f32,
    sa: f32,
    pvx: f32,
    pvy: f32,
    ink: vec4<f32>,
}

// A bubble: its centre and radius (px), its glint's centre and radius (float32 as mkmmd.fields._bubbles makes them),
// and the box it is drawn in.
struct Bubble {
    c: vec4<f32>,       // cx, cy, rad, glint radius
    g: vec4<f32>,       // glint x, glint y
    box: vec4<u32>,     // x0, y0, x1, y1
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> aux: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dst2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> balls: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> dyes: array<Dye>;
@group(0) @binding(7) var<storage, read> bubbles: array<Bubble>;
@group(0) @binding(8) var<storage, read> region: array<vec4<f32>>;

// The work cell's centre in units about the render centre (mkmmd.fields._work).
fn work_xy(x: u32, y: u32) -> vec2<f32> {
    let fx = f32(P.f);
    return vec2<f32>(((f32(x) + 0.5) * fx - 0.5 - P.rcx) / P.runit, ((f32(y) + 0.5) * fx - 0.5 - P.rcy) / P.runit);
}

// np.gradient of plane `src` (.x) at (x, y) of a w x h grid, `dx` apart: (d/dx, d/dy).
fn gradient(x: u32, y: u32, w: u32, h: u32, dx: f32) -> vec2<f32> {
    let i = y * w + x;
    var gx: f32;
    var gy: f32;
    if (x == 0u) {
        gx = (src[i + 1u].x - src[i].x) / dx;
    } else if (x == w - 1u) {
        gx = (src[i].x - src[i - 1u].x) / dx;
    } else {
        gx = (src[i + 1u].x - src[i - 1u].x) / (2.0 * dx);
    }
    if (y == 0u) {
        gy = (src[i + w].x - src[i].x) / dx;
    } else if (y == h - 1u) {
        gy = (src[i].x - src[i - w].x) / dx;
    } else {
        gy = (src[i + w].x - src[i - w].x) / (2.0 * dx);
    }
    return vec2<f32>(gx, gy);
}

// (x, y) turned into a dish turned (ca, sa) about `pivot` (mkmmd.fields._turned).
fn turned(x: f32, y: f32, ca: f32, sa: f32, pvx: f32, pvy: f32) -> vec2<f32> {
    let dx = x - pvx;
    let dy = y - pvy;
    return vec2<f32>(dx * ca + dy * sa, dy * ca - dx * sa);
}

// ---------------------------------------------------------------- the work grid
@compute @workgroup_size(16, 16)
fn liquid_psi(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.ww || id.y >= P.hh) {
        return;
    }
    let xy = work_xy(id.x, id.y);
    let xs = xy.x / P.z;
    let ys = xy.y / P.z;
    dst[id.y * P.ww + id.x] = vec4<f32>(field_noise(xs / P.lw, ys / P.lw, 0u), 0.0, 0.0, 0.0);
}

// `src` the blurred swirl: dst the warped coordinates (Xw, Yw), dst2 the dye (density, tint rgb).
@compute @workgroup_size(16, 16)
fn liquid_warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.ww || id.y >= P.hh) {
        return;
    }
    let i = id.y * P.ww + id.x;
    let xy = work_xy(id.x, id.y);
    let g = gradient(id.x, id.y, P.ww, P.hh, P.gdx);
    let xw = xy.x / P.z + P.amp * g.y;
    let yw = xy.y / P.z - P.amp * g.x;
    dst[i] = vec4<f32>(xw, yw, 0.0, 0.0);
    var pres: array<f32, 6>;
    var most = 0.0;
    for (var k = 0u; k < P.nink; k = k + 1u) {
        let d = dyes[k];
        let t = turned(xw, yw, d.ca, d.sa, d.pvx, d.pvy);
        let xi = t.x / P.lc;
        let yi = t.y / P.lc;
        let c = 1u + 5u * k;
        let qx = field_noise(xi, yi, c) + 0.5 * field_noise(2.0 * xi, 2.0 * yi, c + 1u);
        let qy = field_noise(xi, yi, c + 2u) + 0.5 * field_noise(2.0 * xi, 2.0 * yi, c + 3u);
        let cl = field_noise(xi + P.marble * (qx - 0.75), yi + P.marble * (qy - 0.75), c + 4u);
        pres[k] = clamp((cl - 0.38) / 0.3, 0.0, 1.0);
        most = max(most, pres[k]);
    }
    var ks: array<f32, 6>;
    var sum = 0.0;
    for (var k = 0u; k < P.nink; k = k + 1u) {
        let p2 = pres[k] * pres[k];
        ks[k] = p2 * p2 * p2;
        sum = sum + ks[k];
    }
    var tint = vec3<f32>(0.0);
    for (var k = 0u; k < P.nink; k = k + 1u) {
        tint = tint + (ks[k] / max(sum, 1e-6)) * dyes[k].ink.xyz;
    }
    dst2[i] = vec4<f32>(0.7 * (most * sqrt(most)), tint);
}

// `aux` the warped coordinates: dst this dish's oil field F (balls, roughened).
@compute @workgroup_size(16, 16)
fn liquid_field(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.ww || id.y >= P.hh) {
        return;
    }
    let i = id.y * P.ww + id.x;
    let c = aux[i];
    var g = 0.0;
    for (var b = 0u; b < P.nballs; b = b + 1u) {
        let ball = balls[b];
        let dx = c.x - ball.x;
        let dy = c.y - ball.y;
        g = g + exp(-(dx * dx + dy * dy) / ball.z);
    }
    let f = g + 0.3 * (field_noise(c.x / P.rough, c.y / P.rough, P.dish) - 0.5);
    dst[i] = vec4<f32>(f, 0.0, 0.0, 0.0);
}

// `src` the field F: dst (F, its slope's length / f, the slope along the light, 0).
@compute @workgroup_size(16, 16)
fn liquid_slope(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.ww || id.y >= P.hh) {
        return;
    }
    let i = id.y * P.ww + id.x;
    let g = gradient(id.x, id.y, P.ww, P.hh, 1.0);
    let gn = max(sqrt(g.x * g.x + g.y * g.y), 1e-6);
    dst[i] = vec4<f32>(src[i].x, gn / P.fi, (0.6 * g.x + 0.8 * g.y) / gn, 0.0);
}

// `aux` the warped coordinates: dst the oil cells (wall x runit, the lens, the patch), riding the first ink's dish.
@compute @workgroup_size(16, 16)
fn liquid_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.ww || id.y >= P.hh) {
        return;
    }
    let i = id.y * P.ww + id.x;
    let c = aux[i];
    let d0 = dyes[0];
    let t = turned(c.x, c.y, d0.ca, d0.sa, d0.pvx, d0.pvy);
    let cx = t.x / P.sizeu;
    let cy = t.y / P.sizeu;
    let ix = i32(floor(cx));
    let iy = i32(floor(cy));
    var d1 = 9.0;
    var d2 = 9.0;
    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let jx = ix + ox;
            let jy = iy + oy;
            let px = cx - (f32(jx) + 0.1 + 0.8 * field_hash(jx, jy, 0, P.cells_seed));
            let py = cy - (f32(jy) + 0.1 + 0.8 * field_hash(jx, jy, 1, P.cells_seed));
            let d = sqrt(px * px + py * py);
            d2 = select(min(d2, d), d1, d < d1);
            d1 = min(d1, d);
        }
    }
    let wall = (d2 - d1) * P.sizeu05;
    let mid = d1 * P.sizeu;
    let l = clamp(1.0 - mid / P.sizeu05, 0.0, 1.0);
    let pn = field_noise(t.x / P.patchk, t.y / P.patchk, P.cells_call);
    dst[i] = vec4<f32>(wall * P.runit, l * l, clamp((pn - P.cut) / 0.05, 0.0, 1.0), 0.0);
}

// ---------------------------------------------------------------- the render picture (P: rgb, the oil in .w)
// `aux` the dye resized up: dst the dyed water.
@compute @workgroup_size(16, 16)
fn liquid_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.rw || id.y >= P.rh) {
        return;
    }
    let i = id.y * P.rw + id.x;
    let d = aux[i];
    dst[i] = vec4<f32>(P.light.xyz * (1.0 - d.x * (1.0 - d.yzw)), 0.0);
}

// `aux` the dish's (F, slope / f, light slope) resized up: its oil laid on the water in dst.
@compute @workgroup_size(16, 16)
fn liquid_oil(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.rw || id.y >= P.rh) {
        return;
    }
    let i = id.y * P.rw + id.x;
    let s = aux[i];
    var p = dst[i];
    let d = (s.x - P.level) / max(s.y, 1e-5);
    let c = clamp(d / P.aa + 0.5, 0.0, 1.0);
    let oil = p.w + c * (1.0 - p.w);
    var dark = 1.0 - 0.3 * field_bump(d / P.rpx17);
    var lit = P.flare * field_bump((d - P.rpx4) / P.rpx28);
    let r = clamp(1.0 - d / P.runit06, 0.0, 1.0);
    let lens = s.z * c * (r * r);
    dark = dark * (1.0 - 0.22 * max(-lens, 0.0));
    lit = lit + 0.3 * max(lens, 0.0);
    var rgb = p.xyz;
    if (P.clear == 1u) {
        let a = 0.85 * c;
        rgb = rgb + (P.light.xyz - rgb) * a;
    } else {
        let a = c * (0.55 + 0.45 * clamp((s.x - P.level) / 0.6, 0.0, 1.0));
        let push = 0.7 * c;
        rgb = rgb + (P.light.xyz - rgb) * push;
        rgb = rgb * (1.0 - a * P.unink.xyz);
    }
    rgb = rgb * dark;
    rgb = rgb + (1.0 - rgb) * lit;
    dst[i] = vec4<f32>(rgb, oil);
}

// `aux` the oil cells resized up: their walls on the oil in dst.
@compute @workgroup_size(16, 16)
fn liquid_walls(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.rw || id.y >= P.rh) {
        return;
    }
    let i = id.y * P.rw + id.x;
    let s = aux[i];
    let p = dst[i];
    let pm = s.z * clamp(p.w * 1.4, 0.0, 1.0);
    let k = clamp(1.0 - s.x / P.wallk, 0.0, 1.0);
    let dark = 1.0 - pm * (0.18 * field_bump(s.x / P.px15) + 0.2 * (k * k));
    let lit = pm * (0.28 * s.y + 0.3 * field_bump((s.x - P.px24) / P.px14));
    var rgb = p.xyz * dark;
    rgb = rgb + (1.0 - rgb) * lit;
    dst[i] = vec4<f32>(rgb, p.w);
}

// ---------------------------------------------------------------- the frame
// The bubbles over the light show in dst, in order.
@compute @workgroup_size(16, 16)
fn liquid_bubbles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    var p = dst[i].xyz;
    let x = f32(id.x);
    let y = f32(id.y);
    for (var b = 0u; b < P.nbubbles; b = b + 1u) {
        let bb = bubbles[b];
        if (id.x < bb.box.x || id.y < bb.box.y || id.x >= bb.box.z || id.y >= bb.box.w) {
            continue;
        }
        let dx = x - bb.c.x;
        let dy = y - bb.c.y;
        let d = sqrt(dx * dx + dy * dy) - bb.c.z;
        let inside = clamp(0.5 - d, 0.0, 1.0);
        let lit = 0.75 * field_bump((d + P.bub.x) / P.bub.y);
        let dark = 0.3 * field_bump((d - P.bub.z) / P.bub.w);
        let gx = x - bb.g.x;
        let gy = y - bb.g.y;
        let gl = 0.9 * field_bump(sqrt(gx * gx + gy * gy) / bb.c.w);
        p = p * (1.0 - dark);
        p = p + (1.0 - p) * (0.18 * inside + lit + gl);
        p = clamp(p, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    dst[i] = vec4<f32>(p, 0.0);
}

// `aux` the light show: laid on the picture (src) into dst by the blend, the strength and the region.
@compute @workgroup_size(16, 16)
fn liquid_lay(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    var a = P.strength;
    if (P.mask == 1u) {
        a = region[i].x * a;
    } else if (P.mask == 2u) {
        a = (1.0 - region[i].x) * a;
    }
    let img = src[i];
    dst[i] = vec4<f32>(field_blend(img.xyz, aux[i].xyz, a, P.blend), img.w);
}
