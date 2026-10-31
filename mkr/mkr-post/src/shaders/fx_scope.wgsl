// mkmmd.framefx.draw_trace: a scope's trace added to the picture in place. The CPU draws the line (cv2.polylines,
// LINE_AA) and hands over its coverage a = mask / 255 over the box (x0, y0, cw, ch) it touches; outside it 0.
//   place  the coverage as a plane (.x), 0 outside the box
//   add    a (with a glow, a + glow * blurred a * 3) times the opacity and the colour added, clip(img + .., 0, 1)
// Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    glowed: u32,
    pad0: u32,
    opacity: f32,
    glow: f32,
    pad1: f32,
    pad2: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> crop: array<f32>;
@group(0) @binding(2) var<storage, read_write> plane: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> pic: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> cover: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn place(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    var a = 0.0;
    if (id.x >= P.x0 && id.x < P.x0 + P.cw && id.y >= P.y0 && id.y < P.y0 + P.ch) {
        a = crop[(id.y - P.y0) * P.cw + id.x - P.x0];
    }
    plane[id.y * P.w + id.x] = vec4<f32>(a, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    var a = cover[i].x;
    if (P.glowed != 0u) {
        a = a + P.glow * blurred[i].x * 3.0;
    }
    let p = pic[i];
    pic[i] = vec4<f32>(clamp(p.xyz + P.opacity * a * P.colour.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), p.w);
}
