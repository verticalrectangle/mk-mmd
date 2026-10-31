// The layers' film (layers_gpu.rs), in one pass over a pass's pictures (one layer of `samples` per sample): each
// sample's premultiplied RGBA spread over the pixels around it with the pass's filter and summed with the weights,
// sample after sample, then resolved to the 8 bits Blender writes for that layer. A pass of more samples than the
// layers hold runs in batches, the sums kept between them in `film`.

struct Film {
    width: u32,
    height: u32,
    mode: u32,                      // 0 figure, 1 keep, 2 screen
    count: u32,                     // this batch's samples, in layers 0..count
    taps: u32,                      // the batch's first sample's taps in `consts` (sample index)
    flags: u32,                     // 1: the batch starts the film (else it goes on from `film`); 2: it ends it (resolve)
    stride: u32,                    // figure and keep: u32 words per row of grey bytes
    lut_n: u32,
    lo: f32,                        // log2 of the LUT's first input (screen)
    hi: f32,
    black: f32,                     // figure: Workbench's film alpha over nothing
    white: f32,                     // and over a whole pixel
    zero: vec4<f32>,                // the LUT's output for black
}

@group(0) @binding(0) var samples: texture_2d_array<f32>;
// the figure's transfer (5 vec4: Workbench's alpha for the weighted coverage a = i / 16, i = 0..16), then per sample 25
// weights (7 vec4), entry (dy + 2) * 5 + dx + 2 for the sample drawn on p + (dx, dy)
@group(0) @binding(1) var<storage, read> consts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
// figure and keep: a grey byte per pixel, rows `stride` words apart; screen: RGBA8 per pixel
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
// per pixel: RGBA sum, (weight sum, 0, 0, 0)
@group(0) @binding(4) var<storage, read_write> film: array<vec4<f32>>;

var<immediate> par: Film;
// the workgroup's grey bytes, packed four to a word
var<workgroup> grey: array<u32, 64>;

fn tap(s: u32, i: u32) -> f32 {
    return consts[5u + (par.taps + s) * 7u + i / 4u][i % 4u];
}

fn lut_at(r: u32, g: u32, b: u32) -> vec3<f32> {
    let n = par.lut_n;
    let i = ((b * n + g) * n + r) * 3u;
    return vec3<f32>(lut[i], lut[i + 1u], lut[i + 2u]);
}

fn lut_index(v: f32) -> f32 {
    var l = par.lo;
    if (v > 0.0) {
        l = log2(v);
    }
    return clamp((l - par.lo) / (par.hi - par.lo), 0.0, 1.0) * f32(par.lut_n - 1u);
}

// The view transform of the captured LUT (trilinear in log2 space), black as its `zero`.
fn view(c: vec3<f32>) -> vec3<f32> {
    if (c.r <= 0.0 && c.g <= 0.0 && c.b <= 0.0) {
        return par.zero.rgb;
    }
    let x = vec3<f32>(lut_index(c.r), lut_index(c.g), lut_index(c.b));
    let i0 = min(vec3<u32>(floor(x)), vec3<u32>(par.lut_n - 2u));
    let f = x - vec3<f32>(i0);
    var out = vec3<f32>(0.0);
    for (var k = 0u; k < 8u; k = k + 1u) {
        let d = vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        let w = select(1.0 - f.x, f.x, d.x == 1u) * select(1.0 - f.y, f.y, d.y == 1u) * select(1.0 - f.z, f.z, d.z == 1u);
        out = out + w * lut_at(i0.x + d.x, i0.y + d.y, i0.z + d.z);
    }
    return out;
}

// Blender's float to byte: clamped, rounded half up.
fn byte(v: f32) -> u32 {
    return u32(clamp(floor(v * 255.0 + 0.5), 0.0, 255.0));
}

fn rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(8, 8)
fn film_pass(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let inside = gid.x < par.width && gid.y < par.height;
    let starts = (par.flags & 1u) != 0u;
    let ends = (par.flags & 2u) != 0u;
    var v = 0u;
    if (inside) {
        let p = vec2<i32>(gid.xy);
        let i = gid.y * par.width + gid.x;
        var total = vec4<f32>(0.0);
        var weight = 0.0;
        if (!starts) {
            total = film[2u * i];
            weight = film[2u * i + 1u].x;
        }
        for (var s = 0u; s < par.count; s = s + 1u) {
            var sum = vec4<f32>(0.0);
            var wsum = 0.0;
            for (var dy = -2; dy <= 2; dy = dy + 1) {
                for (var dx = -2; dx <= 2; dx = dx + 1) {
                    let q = p + vec2<i32>(dx, dy);
                    let w = tap(s, u32((dy + 2) * 5 + dx + 2));
                    if (w > 0.0 && q.x >= 0 && q.y >= 0 && q.x < i32(par.width) && q.y < i32(par.height)) {
                        let c = textureLoad(samples, q, s, 0);
                        sum = sum + select(vec4<f32>(0.0), max(c, vec4<f32>(0.0)), c == c) * w;
                        wsum = wsum + w;
                    }
                }
            }
            // the film starts from the pass's first sample and adds each one after
            if (starts && s == 0u) {
                total = sum;
                weight = wsum;
            } else {
                total = total + sum;
                weight = weight + wsum;
            }
        }
        if (!ends) {
            film[2u * i] = total;
            film[2u * i + 1u] = vec4<f32>(weight, 0.0, 0.0, 0.0);
        } else {
            var c = vec4<f32>(0.0);
            if (weight > 0.0) {
                c = total / weight;
            }
            let a = clamp(c.a, 0.0, 1.0);
            if (par.mode == 0u) {
                // Workbench's alpha: the weighted coverage through its transfer, between its film's levels, in 8 bits;
                // mk divides it back by a whole pixel's 254 (shotstyle.ALPHA_FULL)
                let x = a * 16.0;
                let k = min(u32(floor(x)), 15u);
                let t = x - f32(k);
                let g = mix(consts[k / 4u][k % 4u], consts[(k + 1u) / 4u][(k + 1u) % 4u], t);
                let a8 = f32(byte(par.black + (par.white - par.black) * g));
                v = byte(min(a8 / 254.0, 1.0));
            } else {
                // the straight colour Blender's 8-bit RGBA keeps: un-premultiplied, through the view
                var straight = vec3<f32>(0.0);
                if (a > 0.0) {
                    straight = c.rgb / a;
                }
                if (par.mode == 1u) {
                    // the keep mask: R x A of the Raw (linear) RGBA, both 8-bit, back to 8 bits
                    let r8 = f32(byte(straight.r));
                    let a8 = f32(byte(a));
                    v = u32(clamp(floor(r8 * a8 / 255.0 + 0.5), 0.0, 255.0));
                } else {
                    var d = vec3<f32>(0.0);
                    if (a > 0.0) {
                        d = view(straight);
                    }
                    pixels[i] = rgba(byte(d.r), byte(d.g), byte(d.b), byte(a));
                }
            }
        }
    }
    // figure and keep: the grey bytes four to a word (rows padded to whole words)
    if (ends && par.mode != 2u) {
        grey[lid.y * 8u + lid.x] = v;
        workgroupBarrier();
        if (inside && lid.x % 4u == 0u) {
            let b = lid.y * 8u + lid.x;
            pixels[gid.y * par.stride + gid.x / 4u] = grey[b] | (grey[b + 1u] << 8u) | (grey[b + 2u] << 16u)
                | (grey[b + 3u] << 24u);
        }
    }
}
