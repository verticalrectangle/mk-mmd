// key (mkmmd.colorkey; FrameFx._key): a colour of the picture keyed out and what lies behind laid in plane by plane.
//   matte_now    each pixel's distance from the key colour (`distance`: RGB, chroma or luma) and its three planes'
//                mattes (`matte`: each at its own tolerance, a hard step or soft, crushed into bands), into `acc`
//                (the mattes in xyz, the distance in w)
//   matte_trail  a cut frame before (`pic`, packed) keyed at its own tolerance (P.tol, P.cx, P.cy), the mattes kept as bytes as
//                the CPU path keeps them (round(a * 255)) and weighed by fade ** j (P.factor): the strongest of the
//                trail and the matte so far
//   lay          the picture with what shows through (`lay`): the back (a colour, packed pixels or the picture) times
//                the gain laid in by each plane's matte, the spill's stain, kept to the mask's region, clipped, mixed
//                in by the strength
// Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    space: u32,         // 0 rgb, 1 chroma, 2 luma
    bands: u32,         // 0: smooth
    hard: u32,          // 1: no soft edge (a plane is keyed or not)
    masked: u32,        // 0 none, 1 the figure, 2 the ground
    back: u32,          // 0 a colour, 1 the pixels of `pic`, 2 the picture
    spill: u32,
    mixed: u32,         // 1: s < 1
    pad0: u32,
    pad1: u32,
    pad2: u32,
    inv: f32,           // 1 / soft
    bscale: f32,        // 1 / bands
    factor: f32,        // a trail frame's fade ** j / 255
    amount: f32,        // the spill's
    edge: f32,          // the spill's soft + reach
    sinv: f32,
    s: f32,             // the strength and 1 - it
    s1: f32,
    icr: f32,
    tol: f32,           // the tolerance before the spread, + soft
    soft: f32,
    spread: f32,        // the spread / the frame's height: tolerance lost per px from the centre (0: none)
    cx: f32,            // the spread's centre (px)
    cy: f32,
    pad3: f32,
    pad4: f32,
    split: vec4<f32>,   // each plane's tolerance offset
    color: vec4<f32>,   // the key colour
    refs: vec4<f32>,    // its luma, its Cb and Cr (the scales applied), 1 / the Cb scale
    gain: vec4<f32>,
    flat: vec4<f32>,    // the flat back
    stain: vec4<f32>,   // the spill's colour
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acc: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pic: array<u32>;
@group(0) @binding(5) var<storage, read> fig: array<u32>;
@group(0) @binding(6) var<storage, read> lut: array<f32>;

fn unpacked(p: u32) -> vec3<f32> {
    return vec3<f32>(lut[p & 255u], lut[(p >> 8u) & 255u], lut[(p >> 16u) & 255u]);
}

// mkmmd.colorkey.distance
fn distance(c: vec3<f32>) -> f32 {
    if (P.space == 0u) {
        let d = c - P.color.xyz;
        return sqrt((d.x * d.x + d.y * d.y + d.z * d.z) * 0.3333333333333333);
    }
    let y = luma(c);
    if (P.space == 2u) {
        return abs(y - P.refs.x);
    }
    let cb = (c.z - y) * P.refs.w - P.refs.y;
    let cr = (c.x - y) * P.icr - P.refs.z;
    return sqrt(cb * cb + cr * cr);
}

// The tolerance at pixel (x, y) (KeyFx.key_spread: the spread's share of its distance from the centre taken off).
fn tolerance(x: u32, y: u32) -> f32 {
    if (P.spread == 0.0) {
        return P.tol;
    }
    let xs = f32(x) - P.cx;
    let ys = f32(y) - P.cy;
    return P.tol - sqrt(xs * xs + ys * ys) * P.spread;
}

// mkmmd.colorkey.matte, at tolerance `tol`
fn matte(d: f32, tol: f32) -> vec3<f32> {
    let t = vec3<f32>(tol) + P.split.xyz;
    var a: vec3<f32>;
    if (P.hard == 1u) {
        a = select(vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(d) <= t);
    } else {
        a = clamp(((t + P.soft) - d) * P.inv, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (P.bands > 0u) {
        a = round(a * f32(P.bands)) * P.bscale;
    }
    return a;
}

@compute @workgroup_size(16, 16)
fn matte_now(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let d = distance(src[i].xyz);
    acc[i] = vec4<f32>(matte(d, tolerance(id.x, id.y)), d);
}

@compute @workgroup_size(16, 16)
fn matte_trail(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let a = round(matte(distance(unpacked(pic[i])), tolerance(id.x, id.y)) * 255.0);
    let before = acc[i];
    acc[i] = vec4<f32>(max(before.xyz, a * P.factor), before.w);
}

@compute @workgroup_size(16, 16)
fn lay(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let img = src[i];
    let m = acc[i];
    let a = m.xyz;
    var back = P.flat.xyz;
    if (P.back == 1u) {
        back = unpacked(pic[i]);
    } else if (P.back == 2u) {
        back = img.xyz;
    }
    let b = back * P.gain.xyz;
    var out = img.xyz + (b - img.xyz) * a;
    if (P.spill == 1u) {
        var s = clamp(((tolerance(id.x, id.y) + P.edge) - m.w) * P.sinv, 0.0, 1.0);
        s = s * (P.amount * (1.0 - max(a.x, max(a.y, a.z))));
        out = out + (P.stain.xyz - out) * s;
    }
    if (P.masked != 0u) {
        let k = lut[fig[i] & 255u];
        let work = select(1.0 - k, k, P.masked == 1u);
        out = img.xyz + (out - img.xyz) * work;
    }
    out = clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
    if (P.mixed == 1u) {
        out = img.xyz * P.s1 + out * P.s;
    }
    dst[i] = vec4<f32>(out, img.w);
}
