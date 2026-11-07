// Stickers laid as mkmmd.mandala.lay lays them (a mandala's copies, its figure in the middle): each within its box of
// the picture (P.bw x P.bh at (P.x0, P.y0)), the tile warped into it (fx/figure.rs), then `lay_merge` puts the edge
// under the picture (`a` in the box plane `aux2`'s .x for the shadow), `lay_put` the shadow under both and the sticker
// over the picture. `figure_cut` and `figure_tile` make a tile of the figure itself (mkmmd.mandala.figure_tile).

struct Params {
    bw: u32,            // the box (or the tile)
    bh: u32,
    x0: u32,            // the box's place in the picture (or the part of the frame a figure tile is cut from)
    y0: u32,
    w: u32,             // the picture (the frame)
    h: u32,
    edged: u32,         // 1: the tile has a ring and the sticker an edge
    shadow: u32,        // 1: a shadow
    pad: u32,           // figure_tile: the padding round the tile
    fill: u32,          // figure_tile: 1 a flat fill, 0 the picture
    pad0: u32,
    pad1: u32,
    alpha: f32,         // the shadow's opacity
    opacity: f32,       // lay_merge: the sticker's (the tile's picture and ring times it)
    pad3: f32,
    pad4: f32,
    edge: vec4<f32>,    // the edge's colour (or a figure tile's flat fill)
    shade: vec4<f32>,   // the shadow's colour
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> aux: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> aux2: array<vec4<f32>>;

// `dst` the tile warped into the box (rgb premultiplied, a), `aux` its ring warped: the edge under the picture.
@compute @workgroup_size(16, 16)
fn lay_merge(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.bw || id.y >= P.bh) {
        return;
    }
    let i = id.y * P.bw + id.x;
    let k = P.opacity;
    let g = dst[i] * k;
    var a = clamp(g.w, 0.0, 1.0);
    var rgb = g.xyz;
    if (P.edged == 1u) {
        let ring = clamp(aux[i].x * k, 0.0, 1.0);
        rgb = rgb + P.edge.xyz * (ring * (1.0 - a));
        a = a + ring - a * ring;
    }
    dst[i] = vec4<f32>(rgb, a);
    aux2[i] = vec4<f32>(a, 0.0, 0.0, 0.0);
}

// `src` the sticker in its box (rgb, a), `aux` its shadow (shifted and soft): the shadow under it, the two over the
// picture `dst` at the box's place.
@compute @workgroup_size(16, 16)
fn lay_put(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.bw || id.y >= P.bh) {
        return;
    }
    let i = id.y * P.bw + id.x;
    let g = src[i];
    var rgb = g.xyz;
    var a = g.w;
    if (P.shadow == 1u) {
        let k = P.alpha * aux[i].x;
        rgb = rgb + P.shade.xyz * (k * (1.0 - a));
        a = a + k - a * k;
    }
    let o = (P.y0 + id.y) * P.w + P.x0 + id.x;
    let c = dst[o];
    dst[o] = vec4<f32>(c.xyz * (1.0 - a) + rgb, c.w);
}

// figure_cut: the part of the frame (P.x0, P.y0, P.bw x P.bh) a figure tile is cut from: `dst` its coverage (.x, from
// the figure plane `aux`), `aux2` its picture times its coverage (from `src`).
@compute @workgroup_size(16, 16)
fn figure_cut(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.bw || id.y >= P.bh) {
        return;
    }
    let i = id.y * P.bw + id.x;
    let o = (P.y0 + id.y) * P.w + P.x0 + id.x;
    let m = aux[o].x;
    dst[i] = vec4<f32>(m, 0.0, 0.0, 0.0);
    aux2[i] = vec4<f32>(src[o].xyz * m, 0.0);
}

// figure_tile: the tile (P.bw x P.bh, P.pad round the cut part) from the cut part resized (`src` its coverage, `aux`
// its picture times coverage; (P.bw - 2 pad) x (P.bh - 2 pad)): rgb = min(clip(rgb), cov), or the fill times cov; its
// coverage alone in `aux2` (.x: the ring is grown from it).
@compute @workgroup_size(16, 16)
fn figure_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.bw || id.y >= P.bh) {
        return;
    }
    let i = id.y * P.bw + id.x;
    let nw = P.bw - 2u * P.pad;
    let nh = P.bh - 2u * P.pad;
    if (id.x < P.pad || id.y < P.pad || id.x >= P.pad + nw || id.y >= P.pad + nh) {
        dst[i] = vec4<f32>(0.0);
        aux2[i] = vec4<f32>(0.0);
        return;
    }
    let j = (id.y - P.pad) * nw + id.x - P.pad;
    let cov = clamp(src[j].x, 0.0, 1.0);
    var rgb = P.edge.xyz * cov;
    if (P.fill == 0u) {
        rgb = min(clamp(aux[j].xyz, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(cov));
    }
    dst[i] = vec4<f32>(rgb, cov);
    aux2[i] = vec4<f32>(cov, 0.0, 0.0, 0.0);
}
