// film (mkmmd.picturefx.film; PictureFx._film): a film print of the picture as it reached the film on its step's first
// frame, kept for the step's frames (fx/film.rs). `film_tone` puts the picture through the print (black and white, or
// its colours through a dye print) and its S curve; then (fx/film.rs) the gate's weave; `film_damage` lays the dust,
// scratches and hairs (masks Python drew, put in planes by `film_put` and blurred) and the flicker's gain;
// `film_lights` and `film_halo` the halation; `film_grain` the grain (strongest in the mid tones), the vignette and the
// final clip. A film in a `rect` prints the box `film_crop` cuts out, and `film_region` lays it inside the rect.

struct Params {
    w: u32,
    h: u32,
    mono: u32,
    dye_on: u32,        // 1: a dye print (dye > 0 or saturation != 1)
    curve: u32,         // 1: the S curve (4 x contrast >= 1e-3)
    light: u32,         // 1: light damage
    dark: u32,          // 1: dark damage
    grain: u32,         // 1: grain
    vig: u32,           // 1: a vignette
    x0: u32,            // film_put: where the mask's part goes, and its size; film_crop, film_region: the box
    y0: u32,
    cw: u32,
    ch: u32,
    mixed: u32,         // film_region: 1 the print mixed in by the strength
    pad1: u32,
    pad2: u32,
    sat: f32,           // float32 saturation
    dye45: f32,         // float32 of 0.45 x dye, 0.3 x dye, dye
    dye3: f32,
    dye: f32,
    c2: f32,            // float32(c / 2) and float32(tanh(c / 4)) of the curve
    tc4: f32,
    gain: f32,
    halo_amt: f32,      // float32(halation)
    grain_amt: f32,     // float32(grain amount)
    pad4: f32,
    vig_amt: f32,
    pad3: f32,
    halo_rgb: vec4<f32>,
    mix: vec4<f32>,     // film_region: float32(1 - strength), float32(strength)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> light: array<vec4<f32>>;   // the light damage, blurred (.x)
@group(0) @binding(4) var<storage, read> halo: array<vec4<f32>>;    // the lights blurred near (.x) and far (.y)
@group(0) @binding(5) var<storage, read> noise: array<vec4<f32>>;   // the grain field, resized (.x)
@group(0) @binding(6) var<storage, read> dark: array<vec4<f32>>;    // the dark damage, blurred (.x)
@group(0) @binding(7) var<storage, read> part: array<f32>;          // film_put: a mask's part, v / 255
@group(0) @binding(9) var<storage, read> print: array<vec4<f32>>;   // film_region: the box's print
@group(0) @binding(10) var<storage, read> ramps: array<f32>;        // its rows' share of the rect, then its columns'
@group(0) @binding(11) var<storage, read> spread: array<f32>;       // film_grain: float32 of the grain's spread

// The box (P.x0, P.y0, P.cw x P.ch) of the picture `src` (P.w wide) as a picture of its own.
@compute @workgroup_size(16, 16)
fn film_crop(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.cw || id.y >= P.ch) {
        return;
    }
    dst[id.y * P.cw + id.x] = src[(P.y0 + id.y) * P.w + P.x0 + id.x];
}

// The picture `src` with the box's print kept inside the rect (FrameFx._apply): out = img * (1 - s) + print * s
// (PictureFx._mix; the print alone at full strength), then img + (out - img) * rows[y] * columns[x]; the picture as it
// is outside the box.
@compute @workgroup_size(16, 16)
fn film_region(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    let img = src[i];
    if (id.x < P.x0 || id.y < P.y0 || id.x >= P.x0 + P.cw || id.y >= P.y0 + P.ch) {
        dst[i] = img;
        return;
    }
    let u = id.x - P.x0;
    let v = id.y - P.y0;
    var out = print[v * P.cw + u].xyz;
    if (P.mixed == 1u) {
        out = img.xyz * P.mix.x + out * P.mix.y;
    }
    let m = ramps[v] * ramps[P.ch + u];
    dst[i] = vec4<f32>(img.xyz + (out - img.xyz) * m, img.w);
}

fn film_here(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

fn film_curve(v: vec3<f32>) -> vec3<f32> {
    if (P.curve == 0u) {
        return v;
    }
    return 0.5 + 0.5 * tanh(P.c2 * (v - 0.5)) / P.tc4;
}

@compute @workgroup_size(16, 16)
fn film_tone(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    var x = clamp(src[i].xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    if (P.mono == 1u) {
        let l = film_curve(vec3<f32>(luma(x))).x;
        dst[i] = vec4<f32>(l, l, l, 1.0);
        return;
    }
    if (P.dye_on == 1u) {
        let l0 = luma(x);
        var out = l0 + (x - l0) * P.sat;
        if (P.dye > 0.0) {
            let mean = (out.x + out.y + out.z) / 3.0;
            out = out + P.dye45 * (out - mean);
            let red = clamp(2.0 * (out.x - max(out.y, out.z)), 0.0, 1.0);
            let l = luma(out);
            out = out + P.dye3 * red * (out - l);
            out = out + P.dye * (vec3<f32>(-0.02, 0.0, 0.035) * ((1.0 - l) * (1.0 - l)) +
                                 vec3<f32>(0.03, 0.01, -0.03) * (l * l));
        }
        x = clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    dst[i] = vec4<f32>(film_curve(x), 1.0);
}

// A mask's part (P.cw x P.ch at (P.x0, P.y0)) into a plane of nothing (`dst`, .x).
@compute @workgroup_size(16, 16)
fn film_put(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    var v = 0.0;
    if (id.x >= P.x0 && id.y >= P.y0 && id.x < P.x0 + P.cw && id.y < P.y0 + P.ch) {
        v = part[(id.y - P.y0) * P.cw + id.x - P.x0];
    }
    dst[i] = vec4<f32>(v, 0.0, 0.0, 0.0);
}

// The dust, scratches and hairs over `dst`, then the flicker's gain.
@compute @workgroup_size(16, 16)
fn film_damage(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    var out = dst[i].xyz;
    if (P.dark == 1u) {
        out = out * (1.0 - 0.92 * dark[i].x);
    }
    if (P.light == 1u) {
        out = out + (1.0 - out) * light[i].x;
    }
    dst[i] = vec4<f32>(out * P.gain, 1.0);
}

// The lights the halation is made of, clip((luma(clip(src)) - 0.72) / 0.28, 0, 1), into `dst` (.x).
@compute @workgroup_size(16, 16)
fn film_lights(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    let x = clamp(src[i].xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    dst[i] = vec4<f32>(clamp((luma(x) - 0.72) / 0.28, 0.0, 1.0), 0.0, 0.0, 0.0);
}

// The halation over `dst`: `halo` .x the near blur, .y the far one.
@compute @workgroup_size(16, 16)
fn film_halo(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    let x = clamp(dst[i].xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let g = halo[i].x + 0.5 * halo[i].y;
    let add = clamp(P.halo_amt * g, 0.0, 1.0) * P.halo_rgb.xyz;
    dst[i] = vec4<f32>(1.0 - (1.0 - x) * (1.0 - add), 1.0);
}

// The near (.x of `src`) and far (.x of `light`, as bound here) blurs of the lights side by side into `dst`.
@compute @workgroup_size(16, 16)
fn film_pair(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    dst[i] = vec4<f32>(src[i].x, light[i].x, 0.0, 0.0);
}

// The grain, the vignette and the final clip over `dst`.
@compute @workgroup_size(16, 16)
fn film_grain(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = film_here(id);
    if (i < 0) {
        return;
    }
    var out = dst[i].xyz;
    if (P.grain == 1u) {
        let n = noise[i].x / spread[0];
        let lum = clamp(luma(out), 0.0, 1.0);
        out = out + P.grain_amt * n * (0.35 + 2.6 * lum * (1.0 - lum));
    }
    if (P.vig == 1u) {
        let xx = (f32(id.x) + 0.5) / (f32(P.w) / 2.0) - 1.0;
        let yy = (f32(id.y) + 0.5) / (f32(P.h) / 2.0) - 1.0;
        let r = sqrt(xx * xx + yy * yy) / 1.4142135;
        let u = clamp((r - 0.3) / 0.7, 0.0, 1.0);
        out = out * (1.0 - P.vig_amt * u * u * (3.0 - 2.0 * u));
    }
    dst[i] = vec4<f32>(clamp(out, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// The grain field's sums per row (`noise` .x: Σ v and Σ v², in order along the row), its spread worked out from them.
@group(0) @binding(8) var<storage, read_write> sums: array<vec2<f32>>;

@compute @workgroup_size(64)
fn film_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.h) {
        return;
    }
    var s = 0.0;
    var q = 0.0;
    for (var x = 0u; x < P.w; x = x + 1u) {
        let v = noise[id.x * P.w + x].x;
        s = s + v;
        q = q + v * v;
    }
    sums[id.x] = vec2<f32>(s, q);
}
