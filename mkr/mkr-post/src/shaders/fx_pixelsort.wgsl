// mkmmd.picturefx.sort_key and sort_runs: every run of eligible pixels along a row (axis x) or a column (axis y)
// sorted by its key. A pixel is eligible when its key is within [lo, hi] (and, with a mask, the figure's coverage or
// what it leaves is over 0.5); a run is unbroken along its line. The CPU path packs (run, q, index) into one integer
// and sorts them all: q the key quantised to `top` (rint(key * top), top - q with `reverse`), ties kept in line order.
//   keys        each pixel's q, or NONE when it is not eligible; its key exactly as numpy and OpenCV work it out: luma
//               ((r * 0.2126 + g * 0.7152) + b * 0.0722, each step rounded), hue / 360 and saturation (cv2.cvtColor
//               RGB2HSV on float: OpenCV's vector code for columns under `tail`, its scalar code past it), the
//               divisions correctly rounded
//   sort_lines  one workgroup a line: run ids by a scan, a bitonic sort of (run, q << 12 | position), then each
//               eligible position takes the pixel the sorted order puts there

const LINE: u32 = 4096u;            // the longest line a workgroup sorts
const THREADS: u32 = 256u;
const PER: u32 = 16u;               // LINE / THREADS
const NONE: u32 = 0xffffffffu;
const EPS: f32 = 1.1920928955078125e-07;     // FLT_EPSILON

struct Params {
    w: u32,
    h: u32,
    axis: u32,          // 0 x (rows), 1 y (columns)
    by: u32,            // 0 luma, 1 hue, 2 saturation
    reverse: u32,
    masked: u32,        // 0 none, 1 the figure, 2 the ground
    top: u32,
    tail: u32,          // columns from here: OpenCV's RGB2HSV works them out one by one
    lo: f32,
    hi: f32,
    zero: u32,          // 0, unknown to the compiler: keeps a product from fusing into a sum
    pad: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> fig: array<u32>;
@group(0) @binding(5) var<storage, read> lut: array<f32>;

// x rounded to float32 on its own: what is computed from it is not fused with what made it.
fn exact(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ P.zero);
}

// a / b correctly rounded (a >= 0, b > 0): of the floats round the GPU's quotient, the one whose exact residual
// a - c b is smallest (the even one on a tie).
fn div_rn(a: f32, b: f32) -> f32 {
    let q = bitcast<u32>(a / b);
    var best = bitcast<f32>(q);
    var err = abs(fma(-best, b, a));
    for (var d = 0u; d < 5u; d = d + 1u) {
        let bits = q + d - 2u;
        if (d == 2u || (d < 2u && q < 2u - d)) {
            continue;
        }
        let c = bitcast<f32>(bits);
        let e = abs(fma(-c, b, a));
        if (e < err || (e == err && (bits & 1u) == 0u)) {
            best = c;
            err = e;
        }
    }
    return best;
}

fn luma_key(c: vec3<f32>) -> f32 {
    let s = exact(exact(c.x * 0.2126) + exact(c.y * 0.7152));
    return clamp(s + exact(c.z * 0.0722), 0.0, 1.0);
}

fn hsv_key(c0: vec3<f32>, x: u32) -> f32 {
    let r = clamp(c0.x, 0.0, 1.0);
    let g = clamp(c0.y, 0.0, 1.0);
    let b = clamp(c0.z, 0.0, 1.0);
    let v = max(max(r, g), b);
    let diff = v - min(min(r, g), b);
    if (P.by == 2u) {
        return div_rn(diff, abs(v) + EPS);
    }
    var h: f32;
    if (x < P.tail) {
        let hd = select(select(r - g, b - r, g == v), g - b, r == v);
        let res = select(select(240.0, 120.0, g == v), select(0.0, 360.0, g < b), r == v);
        h = fma(hd, div_rn(60.0, diff + EPS), res);
    } else {
        let d = div_rn(60.0, diff + EPS);
        if (r == v) {
            h = exact((g - b) * d);
        } else if (g == v) {
            h = exact(exact((b - r) * d) + 120.0);
        } else {
            h = exact(exact((r - g) * d) + 240.0);
        }
        if (h < 0.0) {
            h = h + 360.0;
        }
    }
    return div_rn(h, 360.0);
}

@compute @workgroup_size(16, 16)
fn key(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let c = src[i].xyz;
    var k: f32;
    if (P.by == 0u) {
        k = luma_key(c);
    } else {
        k = hsv_key(c, id.x);
    }
    var ok = k >= P.lo && k <= P.hi;
    if (P.masked != 0u) {
        let m = lut[fig[i] & 255u];
        let region = select(1.0 - m, m, P.masked == 1u);
        ok = ok && region > 0.5;
    }
    var q = u32(round(exact(k * f32(P.top))));
    if (P.reverse != 0u) {
        q = P.top - q;
    }
    keys[i] = select(NONE, q, ok);
}

var<workgroup> khi: array<u32, LINE>;
var<workgroup> klo: array<u32, LINE>;
var<workgroup> part: array<u32, THREADS>;

fn at(line: u32, j: u32) -> u32 {
    if (P.axis == 0u) {
        return line * P.w + j;
    }
    return j * P.w + line;
}

fn eligible(line: u32, j: u32, len: u32) -> bool {
    return j < len && keys[at(line, j)] != NONE;
}

// part[t] (each thread's count) scanned to the counts before each thread's run of the line (exclusive).
fn scan_parts(t: u32) {
    workgroupBarrier();
    for (var s = 1u; s < THREADS; s = s * 2u) {
        var v = part[t];
        if (t >= s) {
            v = v + part[t - s];
        }
        workgroupBarrier();
        part[t] = v;
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn sort_lines(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let line = wg.x + wg.y * 65535u;
    let len = select(P.h, P.w, P.axis == 0u);
    let lines = select(P.w, P.h, P.axis == 0u);
    let live = line < lines;
    let a = t * PER;
    // run ids: the runs started at or before each position
    var starts = 0u;
    if (live) {
        for (var k = 0u; k < PER; k = k + 1u) {
            let j = a + k;
            if (eligible(line, j, len) && (j == 0u || !eligible(line, j - 1u, len))) {
                starts = starts + 1u;
            }
        }
    }
    part[t] = starts;
    scan_parts(t);
    var run = 0u;
    if (t > 0u) {
        run = part[t - 1u];
    }
    var n = 2u;
    while (n < len) {
        n = n * 2u;
    }
    for (var k = 0u; k < PER; k = k + 1u) {
        let j = a + k;
        var hi = NONE;
        var lo = NONE;
        if (live && eligible(line, j, len)) {
            if (j == 0u || !eligible(line, j - 1u, len)) {
                run = run + 1u;
            }
            hi = run;
            lo = (keys[at(line, j)] << 12u) | j;
        }
        khi[j] = hi;
        klo[j] = lo;
    }
    workgroupBarrier();
    // bitonic sort of the first n entries by (hi, lo)
    for (var size = 2u; size <= n; size = size * 2u) {
        for (var stride = size / 2u; stride > 0u; stride = stride / 2u) {
            for (var p = t; p < n / 2u; p = p + THREADS) {
                let i = 2u * p - (p & (stride - 1u));
                let l = i + stride;
                let up = (i & size) == 0u;
                let bigger = khi[i] > khi[l] || (khi[i] == khi[l] && klo[i] > klo[l]);
                if (bigger == up) {
                    let th = khi[i];
                    let tl = klo[i];
                    khi[i] = khi[l];
                    klo[i] = klo[l];
                    khi[l] = th;
                    klo[l] = tl;
                }
            }
            workgroupBarrier();
        }
    }
    // each eligible position, in line order, takes the next pixel of the sorted order
    var count = 0u;
    if (live) {
        for (var k = 0u; k < PER; k = k + 1u) {
            if (eligible(line, a + k, len)) {
                count = count + 1u;
            }
        }
    }
    part[t] = count;
    scan_parts(t);
    if (!live) {
        return;
    }
    var next = 0u;
    if (t > 0u) {
        next = part[t - 1u];
    }
    for (var k = 0u; k < PER; k = k + 1u) {
        let j = a + k;
        if (j >= len) {
            break;
        }
        let i = at(line, j);
        if (keys[i] != NONE) {
            dst[i] = src[at(line, klo[next] & 0xfffu)];
            next = next + 1u;
        } else {
            dst[i] = src[i];
        }
    }
}
