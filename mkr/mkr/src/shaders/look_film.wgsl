// The looks' films (looks_gpu.rs). Workbench's: each of its 16 samples spread over the sample's 3 x 3 neighbours with
// their Blackman-Harris weights and summed in log2, the sum kept in half floats rounded toward zero (what Blender's
// half-float accumulation gives on this GPU: a whole pixel's alpha lands on 254 of 255, a dark flat colour a level
// up), resolved as exp2(sum / weights) - 0.5. EEVEE's: each sample's premultiplied RGBA spread with the film filter's
// taps and summed, resolved as the weighted mean. The resolves write the 8 bits Blender's files hold.

struct WbAccumulate {
    w: array<vec4<f32>, 3>,         // the sample's nine weights, entry (dx + 1) * 3 + (dy + 1) for the neighbour (dx, dy)
    width: u32,
    height: u32,
    first: u32,                     // 1 on the first sample: the sum starts from it
    pad: u32,
}

struct EeAccumulate {
    taps: array<vec4<f32>, 7>,      // 25 weights, entry (dy + 2) * 5 + dx + 2 for the sample drawn on p + (dx, dy)
    width: u32,
    height: u32,
    first: u32,
    pad: u32,
}

@group(0) @binding(0) var sample_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> film: array<vec4<f32>>;     // per pixel two vec4: the sum, its weight

var<immediate> wb: WbAccumulate;

// A float rounded toward zero to the half floats' 11 significant bits (normal range; smaller values keep theirs).
fn rtz16(x: f32) -> f32 {
    if (abs(x) < 6.103515625e-05) {
        return x;
    }
    return bitcast<f32>(bitcast<u32>(x) & 0xffffe000u);
}

fn rtz16v(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(rtz16(v.x), rtz16(v.y), rtz16(v.z), rtz16(v.w));
}

@compute @workgroup_size(8, 8)
fn wb_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= wb.width || gid.y >= wb.height) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let top = vec2<i32>(i32(wb.width) - 1, i32(wb.height) - 1);
    // copied: RADV gives a whole wave one element when an immediate array is indexed per invocation
    let wts = wb.w;
    var sum = vec4<f32>(0.0);
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let i = u32((dx + 1) * 3 + (dy + 1));
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), top);
            let c = clamp(textureLoad(sample_tex, q, 0), vec4<f32>(0.0), vec4<f32>(1e10));
            sum = sum + log2(c + 0.5) * wts[i / 4u][i % 4u];
        }
    }
    let i = 2u * (gid.y * wb.width + gid.x);
    let s = rtz16v(sum);
    if (wb.first == 1u) {
        film[i] = s;
    } else {
        film[i] = rtz16v(film[i] + s);
    }
}

var<immediate> ee: EeAccumulate;

@compute @workgroup_size(8, 8)
fn ee_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ee.width || gid.y >= ee.height) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let taps = ee.taps;
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let q = p + vec2<i32>(dx, dy);
            let k = u32((dy + 2) * 5 + dx + 2);
            let w = taps[k / 4u][k % 4u];
            if (w > 0.0 && q.x >= 0 && q.y >= 0 && q.x < i32(ee.width) && q.y < i32(ee.height)) {
                let c = textureLoad(sample_tex, q, 0);
                sum = sum + select(vec4<f32>(0.0), max(c, vec4<f32>(0.0)), c == c) * w;
                wsum = wsum + w;
            }
        }
    }
    let i = 2u * (gid.y * ee.width + gid.x);
    if (ee.first == 1u) {
        film[i] = sum;
        film[i + 1u] = vec4<f32>(wsum, 0.0, 0.0, 0.0);
    } else {
        film[i] = film[i] + sum;
        film[i + 1u] = film[i + 1u] + vec4<f32>(wsum, 0.0, 0.0, 0.0);
    }
}

// ------------------------------------------------------------------------------------------------------- resolve
struct Resolve {
    width: u32,
    height: u32,
    mode: u32,                      // 0 flat, 1 coverage, 2 shade, 3 texture (Workbench); 4 RGBA, 5 soft (EEVEE)
    lut_n: u32,
    lo: f32,
    hi: f32,
    total: f32,                     // Workbench: the samples' weights, all of them
    pad: f32,
    zero: vec4<f32>,                // the LUT's output for black
}

@group(0) @binding(0) var<storage, read> film_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<u32>;

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

// The Standard view of the captured LUT (trilinear in log2 space), black as its `zero`.
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
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= res.width || gid.y >= res.height) {
        return;
    }
    let i = gid.y * res.width + gid.x;
    var c = vec4<f32>(0.0);
    if (res.mode < 4u) {
        // Workbench: out of log2, into the half floats of its colour buffer
        c = rtz16v(exp2(film_in[2u * i] / res.total) - 0.5);
    } else {
        let w = film_in[2u * i + 1u].x;
        if (w > 0.0) {
            c = film_in[2u * i] / w;
        }
    }
    let a = clamp(c.a, 0.0, 1.0);
    var straight = vec3<f32>(0.0);
    if (a > 0.0) {
        straight = c.rgb / a;
    }
    switch res.mode {
        case 1u: {
            pixels[i] = byte(a);
        }
        case 2u: {
            pixels[i] = byte(straight.r);
        }
        case 5u: {
            // the soft pass as the compositor writes it: R alpha, B the accent weight (Raw, not divided by alpha)
            pixels[i] = rgba(byte(a), byte(c.r), 0u, 255u);
        }
        default: {
            var d = vec3<f32>(0.0);
            if (a > 0.0) {
                d = view(straight);
            }
            pixels[i] = rgba(byte(d.r), byte(d.g), byte(d.b), byte(a));
        }
    }
}
