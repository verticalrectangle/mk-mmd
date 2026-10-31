// needs sample.wgsl
// mkmmd.remap.tunnel and level_cover: one level of a Droste tunnel drawn over its box (x0, y0, bw, bh) of `out`. The
// level's picture: the pyramid's levels l0 (weight w0) and l1 (w1) warped by their float32 maps from the box's pixels
// (cv2.warpAffine, INTER_LINEAR | WARP_INVERSE_MAP: x term fused, the rows' terms from the CPU as warp.wgsl takes
// them), BORDER_REFLECT nested, BORDER_CONSTANT for a figure's queue; with `lined` its frame line in `col` laid in by
// line * cov. The level's coverage of each pixel and its line (level_cover, the screen -> frame map a):
// d = min(min(qx + 0.5, wr - qx), min(qy + 0.5, hr - qy)), cov = clip(d m + 0.5, 0, 1), line = clip((width - d) m +
// 0.5, 0, 1). Nested: the outermost level is the picture (`first`), each next laid over by cov * alpha; a figure's
// queue: each copy times alpha laid over the farther ones by its own alpha. Each step in numpy's order and float32.

struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    bw: u32,
    bh: u32,
    lw0: u32,
    lh0: u32,
    lw1: u32,
    lh1: u32,
    two: u32,
    first: u32,
    nest: u32,
    lined: u32,
    border: u32,
    pad0: u32,
    m00_0: f32,
    m10_0: f32,
    m00_1: f32,
    m10_1: f32,
    w0: f32,
    w1: f32,
    alpha: f32,
    m: f32,
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    width: f32,
    wr: f32,
    hr: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
    col: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> lev0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lev1: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> rows0: array<f32>;
@group(0) @binding(5) var<storage, read> rows1: array<f32>;

fn fetch(which: u32, i: u32) -> vec4<f32> {
    if (which == 0u) {
        return lev0[i];
    }
    return lev1[i];
}

@compute @workgroup_size(16, 16)
fn level(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.bw || id.y >= P.bh) {
        return;
    }
    let lx = f32(id.x);
    var got = sample_bilinear(0u, fma(P.m00_0, lx, rows0[2u * id.y]), fma(P.m10_0, lx, rows0[2u * id.y + 1u]),
                              P.lw0, P.lh0, P.border, vec4<f32>(0.0)) * P.w0;
    if (P.two != 0u) {
        got = got + sample_bilinear(1u, fma(P.m00_1, lx, rows1[2u * id.y]), fma(P.m10_1, lx, rows1[2u * id.y + 1u]),
                                    P.lw1, P.lh1, P.border, vec4<f32>(0.0)) * P.w1;
    }
    let x = P.x0 + id.x;
    let y = P.y0 + id.y;
    var cov = 0.0;
    if (P.first == 0u || P.lined != 0u) {
        let X = f32(x);
        let Y = f32(y);
        let qx = P.a00 * X + P.a01 * Y + P.a02;
        let qy = P.a10 * X + P.a11 * Y + P.a12;
        let d = min(min(qx + 0.5, P.wr - qx), min(qy + 0.5, P.hr - qy));
        cov = clamp(d * P.m + 0.5, 0.0, 1.0);
        if (P.lined != 0u) {
            let line = clamp((P.width - d) * P.m + 0.5, 0.0, 1.0);
            got = got + (P.col - got) * (line * cov);
        }
    }
    let i = y * P.w + x;
    if (P.first != 0u) {
        out[i] = got;
    } else if (P.nest != 0u) {
        let o = out[i];
        out[i] = o + (got - o) * (cov * P.alpha);
    } else {
        let g = got * P.alpha;
        let o = out[i];
        out[i] = g + o * (1.0 - g.w);
    }
}
