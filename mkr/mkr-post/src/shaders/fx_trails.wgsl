// mkmmd.mandala.MandalaFx._trails: a trail's copies laid on the picture `out` in place, oldest first, each over the box
// of its figure's coverage (_box: past 1/512, a pixel more round it; `box` as fx_masked.wgsl's bbox leaves it), then
// the figure of this frame laid back over them as it is.
//   copy         its colour (with its own picture `pic`: pic + (colour - pic) * tint) laid on at a = alpha * coverage:
//                over dst + (c - dst) a, screen 1 - (1 - dst)(1 - c a), add min(dst + c a, 1)
//   figure_back  out * (1 - k) + img * k, k the figure's coverage, img the picture as it reached the trail (`pic`)
// Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    mode: u32,          // 0 over, 1 screen, 2 add
    tinted: u32,        // 1: the copy's colour mixed with its own picture
    alpha: f32,
    tint: f32,
    pad0: f32,
    pad1: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cov: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pic: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> box: array<u32, 4>;

// Whether pixel (x, y) is in the box of the coverage, a pixel more round it (_box's pad), as numpy slices it.
fn in_box(x: u32, y: u32) -> bool {
    if (box[2] == 0u) {
        return false;
    }
    let x0 = P.w - box[0];
    let y0 = P.h - box[1];
    return x + 1u >= x0 && x < box[2] + 1u && y + 1u >= y0 && y < box[3] + 1u;
}

@compute @workgroup_size(16, 16)
fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h || !in_box(id.x, id.y)) {
        return;
    }
    let i = id.y * P.w + id.x;
    var c = P.colour.xyz;
    if (P.tinted != 0u) {
        let p = pic[i].xyz;
        c = p + (c - p) * P.tint;
    }
    let a = P.alpha * cov[i].x;
    let d = out[i];
    var rgb: vec3<f32>;
    if (P.mode == 0u) {
        rgb = d.xyz + (c - d.xyz) * a;
    } else if (P.mode == 1u) {
        rgb = 1.0 - (1.0 - d.xyz) * (1.0 - c * a);
    } else {
        rgb = min(d.xyz + c * a, vec3<f32>(1.0));
    }
    out[i] = vec4<f32>(rgb, d.w);
}

@compute @workgroup_size(16, 16)
fn figure_back(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h || !in_box(id.x, id.y)) {
        return;
    }
    let i = id.y * P.w + id.x;
    let k = cov[i].x;
    let d = out[i];
    out[i] = vec4<f32>(d.xyz * (1.0 - k) + pic[i].xyz * k, d.w);
}
