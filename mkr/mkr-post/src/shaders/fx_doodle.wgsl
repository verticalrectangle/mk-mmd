// doodle (mkmmd.picturefx.draw_strokes; PictureFx._doodle): pen strokes over the picture, their masks drawn by Python
// with OpenCV's antialiased lines in the box they cover (`masks`: rim, ink as v / 255): the white rim, then the ink.

struct Params {
    w: u32,
    h: u32,
    x0: u32,            // the box
    y0: u32,
    bw: u32,
    bh: u32,
    pad0: u32,
    pad1: u32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> masks: array<vec2<f32>>;

@compute @workgroup_size(16, 16)
fn doodle(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.bw || id.y >= P.bh) {
        return;
    }
    let m = masks[id.y * P.bw + id.x];
    let o = (P.y0 + id.y) * P.w + P.x0 + id.x;
    let c = dst[o];
    var reg = c.xyz * (1.0 - m.x) + m.x;
    reg = reg * (1.0 - m.y) + P.colour.xyz * m.y;
    dst[o] = vec4<f32>(reg, c.w);
}
