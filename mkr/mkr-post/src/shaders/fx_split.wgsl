// mkmmd.framefx.split: a broken digital signal. Each row's red, green and blue planes are rolled sideways by its own
// whole-pixel shifts (np.roll along the row: the pixel at x shows the one at (x - shift) mod w); mkmmd.gpufx.split
// hands over each row's three shifts, worked out by the CPU path's own code.

struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rows: array<vec4<i32>>;    // (red, green, blue, 0) shifts per row

fn rolled(y: u32, x: u32, s: i32) -> vec4<f32> {
    let w = i32(P.w);
    var q = (i32(x) - s) % w;
    if (q < 0) {
        q = q + w;
    }
    return src[y * P.w + u32(q)];
}

@compute @workgroup_size(16, 16)
fn split(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let s = rows[id.y];
    let i = id.y * P.w + id.x;
    dst[i] = vec4<f32>(rolled(id.y, id.x, s.x).x, rolled(id.y, id.x, s.y).y, rolled(id.y, id.x, s.z).z, src[i].w);
}
