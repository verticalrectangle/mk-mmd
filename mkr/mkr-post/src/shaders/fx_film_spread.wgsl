// film's grain spread (fx/film.rs grain_spread) on the GPU in 64-bit floats, for a device that has them: the grain
// field's rows' sums (film_rows: Σ v and Σ v² along each row, float32) folded in order in float64, n = w h, mean = s / n,
// std = sqrt(max(q / n - mean², 0)), float32(max(std, 1e-6)) into spread[0], the float film_grain divides by: what the
// CPU worked out from the sums read back. One invocation. mean² goes to spread[1] as well: used twice, it is rounded
// on its own as the CPU rounds it, never fused into the subtraction.

struct Dims {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> D: Dims;
@group(0) @binding(1) var<storage, read> sums: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> spread: array<f32>;

@compute @workgroup_size(1)
fn film_spread() {
    var s = f64(0.0);
    var q = f64(0.0);
    for (var y = 0u; y < D.h; y = y + 1u) {
        let r = sums[y];
        s = s + f64(r.x);
        q = q + f64(r.y);
    }
    let n = f64(D.w) * f64(D.h);
    let mean = s / n;
    let m2 = mean * mean;
    let dev = sqrt(max(q / n - m2, f64(0.0)));
    spread[0] = f32(max(dev, f64(1e-6)));
    spread[1] = f32(m2);
}
