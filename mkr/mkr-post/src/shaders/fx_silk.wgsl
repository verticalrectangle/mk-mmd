// A silkscreen print's plates (mkmmd.framefx.silk_bands), for the collages that fill a figure with one: `silk_luma`
// gives the figure's luma times its coverage and the coverage (_within blurs both at once), `silk_mean` divides them
// back (the luma smoothed inside the figure; `silk_plain` takes it as it is), `silk_vals` writes the luma where the
// figure covers more than half (NaN elsewhere) for the quantiles worked out on the CPU, and `silk_plates` cuts the
// bands: the key plate (the darkest band and the lines, times the coverage) and the colour steps, each a hard edge
// antialiased over a pixel (np.clip((L - t) / g + 0.5, 0, 1), g half the gradient from cv2.Sobel(ksize=1)).

struct Params {
    w: u32,
    h: u32,
    steps: u32,         // the colour steps (n inks - 2)
    line: u32,          // 1: the lines into the key plate
    t0: f32,            // float32 thresholds: the key's, then the steps'
    t1: f32,
    t2: f32,
    depth: f32,         // SILK_DEPTH
    nan: f32,           // a NaN (silk_vals' mark for a pixel left out)
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cov: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mean: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> vals: array<f32>;

fn at(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

// (L x m, m, L) of the picture `src` and the coverage `cov` (.x): what _within blurs, and the luma itself.
@compute @workgroup_size(16, 16)
fn silk_luma(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let p = src[i];
    let m = cov[i].x;
    let l = p.x * 0.2126 + p.y * 0.7152 + p.z * 0.0722;
    dst[i] = vec4<f32>(l * m, m, l, 0.0);
}

// The blurred (L x m, m) in `src` divided back into dst.x (the coverage in .y): num / max(den, 1e-4).
@compute @workgroup_size(16, 16)
fn silk_mean(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let b = src[i];
    dst[i] = vec4<f32>(b.x / max(b.y, 1e-4), cov[i].x, 0.0, 0.0);
}

// The luma (`src` .z of silk_luma's) as it is into dst.x, the coverage in .y.
@compute @workgroup_size(16, 16)
fn silk_plain(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    dst[i] = vec4<f32>(src[i].z, cov[i].x, 0.0, 0.0);
}

// (L x m, m) of the smoothed luma (`src`: L in .x, m in .y), for the lines' mean (_within over their reach).
@compute @workgroup_size(16, 16)
fn silk_masked(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let b = src[i];
    dst[i] = vec4<f32>(b.x * b.y, b.y, 0.0, 0.0);
}

// The luma (`src` .x) where the coverage (`cov` .x) is over half, NaN elsewhere: what silk_levels takes quantiles of.
@compute @workgroup_size(16, 16)
fn silk_vals(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    vals[i] = select(P.nan, src[i].x, cov[i].x > 0.5);
}

fn luma_at(x: i32, y: i32) -> f32 {
    let bx = border(x, i32(P.w), BORDER_REFLECT_101);
    let by = border(y, i32(P.h), BORDER_REFLECT_101);
    return src[u32(by) * P.w + u32(bx)].x;
}

fn past(l: f32, t: f32, g: f32) -> f32 {
    return clamp((l - t) / g + 0.5, 0.0, 1.0);
}

// The plates from the smoothed luma (`src` .x, the coverage in .y) and, with lines, its mean over the lines' reach
// (`mean`: the blurred (L x m, m)): dst = (key x m, step 1, step 2, 0).
@compute @workgroup_size(16, 16)
fn silk_plates(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = at(id);
    if (i < 0) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let here = src[i];
    let l = here.x;
    let gx = luma_at(x + 1, y) - luma_at(x - 1, y);
    let gy = luma_at(x, y + 1) - luma_at(x, y - 1);
    let g = 0.5 * sqrt(gx * gx + gy * gy) + 1e-4;
    var key = 1.0 - past(l, P.t0, g);
    if (P.line == 1u) {
        let b = mean[i];
        key = max(key, 1.0 - past(l, b.x / max(b.y, 1e-4) - P.depth, g));
    }
    var s1 = 0.0;
    var s2 = 0.0;
    if (P.steps >= 1u) {
        s1 = past(l, P.t1, g);
    }
    if (P.steps >= 2u) {
        s2 = past(l, P.t2, g);
    }
    dst[i] = vec4<f32>(key * here.y, s1, s2, 0.0);
}
