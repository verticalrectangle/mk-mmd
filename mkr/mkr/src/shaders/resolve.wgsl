// The film, part 2: the weighted sum divided by the weights, through the view LUT, quantised to 8 bits after Blender's
// dither (its fixed per-pixel pattern, computed on the CPU: `dither.rs`).

struct Resolve {
    width: u32,
    height: u32,
    lut_n: u32,
    per_channel: u32,               // 1: the view transforms each channel alone (Standard)
    lo: f32,                        // log2 of the LUT's first input
    hi: f32,
    dither: f32,                    // multiplier on the dither pattern: dither_intensity x the `dither` tunable
    bias: f32,                      // 8-bit levels added before rounding
    zero: vec4<f32>,                // the view's output for black
    exposure: f32,                  // scale on the scene-linear values
    frame: u32,
    pad2: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> film_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(3) var<storage, read> dither: array<f32>;

var<immediate> res: Resolve;

fn lut_at(r: u32, g: u32, b: u32) -> vec3<f32> {
    let n = res.lut_n;
    let i = ((b * n + g) * n + r) * 3u;
    return vec3<f32>(lut[i], lut[i + 1u], lut[i + 2u]);
}

fn lut_index(v: f32) -> f32 {
    var l = res.lo;
    if (v > 0.0) {
        l = log2(v);
    }
    return clamp((l - res.lo) / (res.hi - res.lo), 0.0, 1.0) * f32(res.lut_n - 1u);
}

fn view(c: vec3<f32>) -> vec3<f32> {
    if (c.r <= 0.0 && c.g <= 0.0 && c.b <= 0.0) {
        return res.zero.rgb;
    }
    let x = vec3<f32>(lut_index(c.r), lut_index(c.g), lut_index(c.b));
    let i0 = min(vec3<u32>(floor(x)), vec3<u32>(res.lut_n - 2u));
    let f = x - vec3<f32>(i0);
    var out = vec3<f32>(0.0);
    for (var k = 0u; k < 8u; k = k + 1u) {
        let d = vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        let w = select(1.0 - f.x, f.x, d.x == 1u) * select(1.0 - f.y, f.y, d.y == 1u) * select(1.0 - f.z, f.z, d.z == 1u);
        out = out + w * lut_at(i0.x + d.x, i0.y + d.y, i0.z + d.z);
    }
    // under the LUT's first input the picture heads linearly for the view's output for black: per channel where the
    // view transforms channels alone (Standard: linear there), else only when every channel is that dark (AgX mixes
    // channels: a dark channel of a bright colour still shows; its first entry equals black's)
    let first = exp2(res.lo);
    if (res.per_channel != 0u) {
        let toward = res.zero.rgb + (out - res.zero.rgb) * (max(c, vec3<f32>(0.0)) / first);
        return select(out, toward, c < vec3<f32>(first));
    }
    let m = max(max(c.r, c.g), c.b);
    if (m < first) {
        return res.zero.rgb + (out - res.zero.rgb) * (m / first);
    }
    return out;
}

// Blender's 8 bits: the dither (display units) added to the display value, then 255 x rounded half up, clamped.
fn to8(v: f32, noise: f32) -> u32 {
    let level = 255.0 * (v + noise) + 0.5 + res.bias;
    return u32(clamp(floor(level), 0.0, 255.0));
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= res.width || gid.y >= res.height) {
        return;
    }
    let i = gid.y * res.width + gid.x;
    let f = film_in[i];
    var c = vec3<f32>(0.0);
    if (f.a > 0.0) {
        c = f.rgb / f.a * res.exposure;
    }
    let d = view(c);
    let n = dither[i] * res.dither;
    pixels[i] = to8(d.r, n) | (to8(d.g, n) << 8u) | (to8(d.b, n) << 16u) | (255u << 24u);
}
