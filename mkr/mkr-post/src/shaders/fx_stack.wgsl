// stack (mkmmd.stack.lay; StackFx._stack): each layer laid within its box of the picture as mkmmd.stack.lay lays it, so
// its warp samples where the CPU's does. Each figure is cut over the box of the picture it covers at all (its support,
// fx/stack.rs: from its layer, or found here by `support`), its coverage a single-float plane over that box and its
// picture times it a vec4 one; `cut` and `cut1` copy a layer's part of them (reflected about x = P.flip / 2 for a
// mirrored layer) into planes of its own, and after the warp into the layer's box (fx/stack.rs), its ring and its
// shadow (fx_plane.wgsl, single floats), `layer_box` lays the shadow, the edge and the figure over the picture at the
// box's place.

struct Params {
    w: u32,             // the picture
    h: u32,
    x0: u32,            // cut: the part's first column and row (in the picture, as laid); layer_box: the box's place
    y0: u32,
    cw: u32,            // cut: the part's size; layer_box: the box's size
    ch: u32,
    flip: i32,          // cut: x of the picture shows x' = flip - x of the figure; < 0: not reflected
    slot: u32,          // support: where its box goes in `boxes`
    fill: u32,          // layer_box: 1 a flat colour
    edged: u32,
    shadow: u32,
    ax0: u32,           // cut, support: the box of the picture the cut planes cover (left, top, width, height)
    ay0: u32,
    aw: u32,
    ah: u32,
    alpha: f32,
    colour: vec4<f32>,
    edge: vec4<f32>,
    shade: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cov: array<f32>;
@group(0) @binding(4) var<storage, read> ring: array<f32>;
@group(0) @binding(5) var<storage, read> dark: array<f32>;
@group(0) @binding(6) var<storage, read> pic: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> boxes: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> dst1: array<f32>;

// support: the box (x0, y0, x1, y1; x1, y1 past it) of the picture that coverage `cov` (over the cut planes' box)
// covers at all, by atomics on boxes[4 slot ..] (set to (u32 max, u32 max, 0, 0) first).
@compute @workgroup_size(16, 16)
fn support(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.aw || id.y >= P.ah) {
        return;
    }
    if (cov[id.y * P.aw + id.x] > 0.0) {
        let b = 4u * P.slot;
        atomicMin(&boxes[b], P.ax0 + id.x);
        atomicMin(&boxes[b + 1u], P.ay0 + id.y);
        atomicMax(&boxes[b + 2u], P.ax0 + id.x + 1u);
        atomicMax(&boxes[b + 3u], P.ay0 + id.y + 1u);
    }
}

// The cut planes' pixel showing the picture's (x, y) (an index into them), or -1: outside the picture, or outside
// their box (where the figure covers nothing).
fn cut_at(x: i32, y: u32) -> i32 {
    let lx = x - i32(P.ax0);
    let ly = i32(y) - i32(P.ay0);
    if (x < 0 || x >= i32(P.w) || lx < 0 || ly < 0 || lx >= i32(P.aw) || ly >= i32(P.ah)) {
        return -1;
    }
    return ly * i32(P.aw) + lx;
}

// The part's pixel (x, y): the picture's (P.x0 + x, P.y0 + y), or with P.flip >= 0 the figure's
// (P.flip - (P.x0 + x), P.y0 + y).
fn part_x(id: vec3<u32>) -> i32 {
    var x = i32(P.x0 + id.x);
    if (P.flip >= 0) {
        x = P.flip - x;
    }
    return x;
}

// cut: the part of the vec4 cut plane `src` a layer is laid from into `dst` (P.cw x P.ch).
@compute @workgroup_size(16, 16)
fn cut(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.cw || id.y >= P.ch) {
        return;
    }
    let s = cut_at(part_x(id), P.y0 + id.y);
    var c = vec4<f32>(0.0);
    if (s >= 0) {
        c = src[u32(s)];
    }
    dst[id.y * P.cw + id.x] = c;
}

// cut1: the same of the single-float coverage `cov` into `dst1`.
@compute @workgroup_size(16, 16)
fn cut1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.cw || id.y >= P.ch) {
        return;
    }
    let s = cut_at(part_x(id), P.y0 + id.y);
    var c = 0.0;
    if (s >= 0) {
        c = cov[u32(s)];
    }
    dst1[id.y * P.cw + id.x] = c;
}

// layer_box: one layer over the picture `dst` at its box (P.x0, P.y0, P.cw x P.ch): its shadow (`dark`, alpha times
// the shifted, blurred coverage), its edge's ring at its opacity, then its picture (premultiplied `pic`) or flat fill
// by its coverage, at its opacity.
@compute @workgroup_size(16, 16)
fn layer_box(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.cw || id.y >= P.ch) {
        return;
    }
    let i = id.y * P.cw + id.x;
    let o = (P.y0 + id.y) * P.w + P.x0 + id.x;
    var sub = dst[o].xyz;
    if (P.shadow == 1u) {
        let a = P.shade.w * dark[i];
        sub = sub * (1.0 - a) + P.shade.xyz * a;
    }
    if (P.edged == 1u) {
        var g = ring[i];
        if (P.alpha < 1.0) {
            g = g * P.alpha;
        }
        sub = sub * (1.0 - g) + P.edge.xyz * g;
    }
    let c = cov[i];
    var col = pic[i].xyz;
    if (P.fill == 1u) {
        col = P.colour.xyz * c;
    }
    var a = c;
    if (P.alpha < 1.0) {
        col = col * P.alpha;
        a = a * P.alpha;
    }
    dst[o] = vec4<f32>(col + sub * (1.0 - a), dst[o].w);
}
