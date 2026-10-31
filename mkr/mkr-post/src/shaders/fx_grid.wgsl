// grid (mkmmd.framefx.cells, FrameFx._grid): the frame shrunk into boxes, each the whole frame contain-fitted and
// centred in it, the figure on the box's back. The figure's tile (its coverage and picture shrunk by area into a
// padded tile, its ring, its shadow, its silkscreen plates) is made once (fx/grid.rs); `cell_back` lays a box's back
// from the shrunk frame reflected past its edges (mkmmd.framefx._framed), `cell_put` lays the tile into a box, cut at
// its edges.

struct Params {
    w: u32,             // the picture
    h: u32,
    x0: u32,            // the box (x0, y0, x1, y1)
    y0: u32,
    x1: u32,
    y1: u32,
    tw: u32,            // the tile, or the shrunk frame's part a back shows
    th: u32,
    tx: i32,            // where the tile's (or the part's) top left lands in the picture
    ty: i32,
    px: u32,            // cell_back: the part's first column and row in the shrunk frame
    py: u32,
    sw: u32,            // cell_back: the shrunk frame's width
    fill: u32,          // cell_put: 0 the picture, 1 a flat colour
    shadow: u32,        // cell_put: 1 a shadow
    edged: u32,         // cell_put: 1 an edge
    silk: u32,          // cell_put: 1 a silkscreen print (mkmmd.framefx.silk_print)
    steps: u32,         // its colour steps (inks - 2)
    pad0: u32,
    pad1: u32,
    colour: vec4<f32>,  // the fill
    edge: vec4<f32>,
    shade: vec4<f32>,
    inks: array<vec4<f32>, 4>,    // the silkscreen's inks, the key's first
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cov: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> ring: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> dark: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> plates: array<vec4<f32>>;    // the silkscreen's colour steps (.y .z)
@group(0) @binding(7) var<storage, read> key: array<vec4<f32>>;       // its key plate off register (.x)

// The box's back from the shrunk frame `src`: its part (P.tw x P.th from (P.px, P.py)) put at (P.tx, P.ty) and
// reflected past its edges (cv2.BORDER_REFLECT) to fill the box.
@compute @workgroup_size(16, 16)
fn cell_back(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = P.x0 + id.x;
    let y = P.y0 + id.y;
    if (x >= P.x1 || y >= P.y1) {
        return;
    }
    let u = border(i32(x) - P.tx, i32(P.tw), BORDER_REFLECT);
    let v = border(i32(y) - P.ty, i32(P.th), BORDER_REFLECT);
    let c = src[(P.py + u32(v)) * P.sw + P.px + u32(u)];
    dst[y * P.w + x] = vec4<f32>(c.xyz, 1.0);
}

// The tile laid into the box: its shadow, its edge, the figure (`src` its picture times its coverage, `cov` .x; a
// flat colour; or its silkscreen print: the colour plates, then the key plate over them).
@compute @workgroup_size(16, 16)
fn cell_put(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = P.x0 + id.x;
    let y = P.y0 + id.y;
    if (x >= P.x1 || y >= P.y1) {
        return;
    }
    let u = i32(x) - P.tx;
    let v = i32(y) - P.ty;
    if (u < 0 || v < 0 || u >= i32(P.tw) || v >= i32(P.th)) {
        return;
    }
    let t = u32(v) * P.tw + u32(u);
    let o = y * P.w + x;
    var sub = dst[o].xyz;
    if (P.shadow == 1u) {
        let a = dark[t].x;
        sub = sub * (1.0 - a) + P.shade.xyz * a;
    }
    if (P.edged == 1u) {
        let g = ring[t].x;
        sub = sub * (1.0 - g) + P.edge.xyz * g;
    }
    let c = cov[t].x;
    if (P.silk == 1u) {
        let st = plates[t];
        var colour = P.inks[1].xyz;
        if (P.steps >= 1u) {
            colour = colour + (P.inks[2].xyz - P.inks[1].xyz) * st.y;
        }
        if (P.steps >= 2u) {
            colour = colour + (P.inks[3].xyz - P.inks[2].xyz) * st.z;
        }
        sub = sub * (1.0 - c) + colour * c;
        let k = key[t].x;
        sub = sub * (1.0 - k) + P.inks[0].xyz * k;
    } else if (P.fill == 1u) {
        sub = P.colour.xyz * c + sub * (1.0 - c);
    } else {
        sub = src[t].xyz + sub * (1.0 - c);
    }
    dst[o] = vec4<f32>(sub, 1.0);
}

// A shrunk plane (`src`, (P.tw - 2 pad) x (P.th - 2 pad), the pad in P.px) put in a tile with `pad` px of nothing
// round it (`dst`, P.tw x P.th).
@compute @workgroup_size(16, 16)
fn tile_pad(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.tw || id.y >= P.th) {
        return;
    }
    let pad = P.px;
    let i = id.y * P.tw + id.x;
    let sw = P.tw - 2u * pad;
    let sh = P.th - 2u * pad;
    if (id.x < pad || id.y < pad || id.x >= pad + sw || id.y >= pad + sh) {
        dst[i] = vec4<f32>(0.0);
        return;
    }
    let j = (id.y - pad) * sw + id.x - pad;
    dst[i] = src[j];
}

// dark = alpha x the shifted, blurred coverage (in place in `dst`, .x).
@compute @workgroup_size(16, 16)
fn tile_dark(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.tw || id.y >= P.th) {
        return;
    }
    let i = id.y * P.tw + id.x;
    dst[i] = vec4<f32>(P.shade.w * dst[i].x, 0.0, 0.0, 0.0);
}

// A box filled with a flat back.
@compute @workgroup_size(16, 16)
fn cell_fill(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = P.x0 + id.x;
    let y = P.y0 + id.y;
    if (x >= P.x1 || y >= P.y1) {
        return;
    }
    dst[y * P.w + x] = vec4<f32>(P.colour.xyz, 1.0);
}
