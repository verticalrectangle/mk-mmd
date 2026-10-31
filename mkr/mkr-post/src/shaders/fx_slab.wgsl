// mkmmd.framefx.slab_cover and FrameFx._slab: a slanted bar's coverage averaged over the n places it stands at over
// its shutter, each counted only inside its box (x0..x1, y0..y1, as slab_cover's sums run), then the bar's colour mixed
// in place by alpha * coverage. Per place: X = x - cx, Y = cy - y (y up), u = X ca + Y sa, v = -X sa + Y ca,
// d = max(|v| - hh, (|u - v t| - hw) cs), clip(0.5 - d, 0, 1); each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    n: u32,
    pad0: u32,
    ca: f32,
    sa: f32,
    cs: f32,
    t: f32,
    hh: f32,
    hw: f32,
    alpha: f32,
    pad1: f32,
    colour: vec4<f32>,
}

struct Place {
    cx: f32,
    cy: f32,
    pad0: f32,
    pad1: f32,
    box: vec4<i32>,     // x0, x1, y0, y1
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pic: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> places: array<Place>;

@compute @workgroup_size(16, 16)
fn slab(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    var acc = 0.0;
    for (var j = 0u; j < P.n; j = j + 1u) {
        let c = places[j];
        if (x < c.box.x || x >= c.box.y || y < c.box.z || y >= c.box.w) {
            continue;
        }
        let X = f32(id.x) - c.cx;
        let Y = c.cy - f32(id.y);
        let u = X * P.ca + Y * P.sa;
        let v = -X * P.sa + Y * P.ca;
        let d = max(abs(v) - P.hh, (abs(u - v * P.t) - P.hw) * P.cs);
        acc = acc + clamp(0.5 - d, 0.0, 1.0);
    }
    let a = P.alpha * (acc / f32(max(P.n, 1u)));
    let i = id.y * P.w + id.x;
    let o = pic[i];
    pic[i] = vec4<f32>(o.xyz * (1.0 - a) + P.colour.xyz * a, o.w);
}
