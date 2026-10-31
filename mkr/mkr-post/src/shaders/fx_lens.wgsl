// needs sample.wgsl
// mkmmd.framefx.lens: a point mass at (cx, cy) with Einstein radius sqrt(theta2): the pixel at d = (dx, dy) from it,
// r2 = max(|d|^2, 1e-6), shows the picture at c + d * (1 - theta2 / r2) (cv2.remap, INTER_LINEAR, black outside);
// with `shadow` the pixels with r2 < lim are black (lim: the float32 the float64 bound compares as).

struct Params {
    w: u32,
    h: u32,
    shadow: u32,
    pad: u32,
    cx: f32,
    cy: f32,
    theta2: f32,
    lim: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    return src[i];
}

@compute @workgroup_size(16, 16)
fn lens(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let dx = f32(id.x) - P.cx;
    let dy = f32(id.y) - P.cy;
    let r2 = max(dx * dx + dy * dy, 1e-6);
    let k = 1.0 - P.theta2 / r2;
    var o = sample_bilinear(0u, P.cx + dx * k, P.cy + dy * k, P.w, P.h, BORDER_CONSTANT, vec4<f32>(0.0));
    if (P.shadow != 0u && r2 < P.lim) {
        o = vec4<f32>(0.0, 0.0, 0.0, o.w);
    }
    dst[id.y * P.w + id.x] = o;
}
