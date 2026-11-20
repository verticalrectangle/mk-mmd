// The vector look's construction-paper cut-out (cutout_gpu.rs; mkmmd.core.cutout is the reference): the look's flat
// fills from the passes, the scissors' cut, the rim, the drop shadow and the pieces laid far to near, each with its
// shadow, then boxed down to the frame and written as 8-bit RGBA. The float steps are numpy's (float32, in its order).
//
// Coordinates: `pass` pixels are the passes' (twice the frame's size); `box` pixels are the figure's box (Frame.bbox),
// where every buffer but the passes, the lit mask and the frame lives; an Op's window is in box pixels (or pass pixels
// for the lit mask, frame pixels for the finish).

struct Frame {
    bg: vec4<f32>,
    rim: vec4<f32>,
    shc: vec4<f32>,
    size: vec4<u32>,   // pass w, pass h, frame w, frame h
    bbox: vec4<u32>,   // box x0, y0, w, h in pass pixels
    misc: vec4<u32>,   // the pieces' sheet's width, has tex, has type, the least lit shade byte
    misc2: vec4<u32>,  // material ids, box capacity (the second half of the mask and blur buffers), pass pixels
    fl: vec4<f32>,     // rough px (float32), grain
}

struct Tone {
    kind: u32,
    cel: u32,
    steps: u32,
    first: u32,
    table: u32,
    grey: u32,
    last: u32,
    pad: u32,
}

struct Op {
    win: vec4<u32>,    // x0, y0, x1, y1 (end excluded)
    a: vec4<i32>,
    b: vec4<i32>,
    f: vec4<f32>,
}

const NONE: u32 = 0xffffffffu;
const PAD: u32 = 64u;
const FLAT: u32 = 0u;
const LIT: u32 = 1u;

@group(0) @binding(0) var<uniform> fr: Frame;
@group(0) @binding(1) var<storage, read> idb: array<u32>;
@group(0) @binding(2) var<storage, read> shadeb: array<u32>;
@group(0) @binding(3) var<storage, read> texb: array<u32>;
@group(0) @binding(4) var<storage, read> typeb: array<u32>;
@group(0) @binding(5) var<storage, read> idinfo: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> tones: array<Tone>;
@group(0) @binding(7) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> steps: array<u32>;
@group(0) @binding(9) var<storage, read> units: array<f32>;
@group(0) @binding(10) var<storage, read_write> lit: array<u32>;
@group(0) @binding(11) var<storage, read> sgrid: array<f32>;
@group(0) @binding(12) var<storage, read> saxes: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read> fgrid: array<f32>;
@group(0) @binding(14) var<storage, read> faxes: array<vec2<u32>>;
@group(0) @binding(15) var<storage, read> noises: array<vec4<u32>>;
@group(0) @binding(16) var<storage, read_write> wsrc: array<u32>;
@group(0) @binding(17) var<storage, read_write> pw: array<u32>;
@group(0) @binding(18) var<storage, read_write> texf: array<f32>;
@group(0) @binding(19) var<storage, read_write> pic: array<vec4<f32>>;
@group(0) @binding(20) var<storage, read_write> mask: array<u32>;
@group(0) @binding(21) var<storage, read_write> srcb: array<u32>;
@group(0) @binding(22) var<storage, read_write> stamp: array<u32>;
@group(0) @binding(23) var<storage, read_write> sbuf: array<f32>;
@group(0) @binding(24) var<storage, read_write> bboxes: array<atomic<u32>>;
@group(0) @binding(25) var<storage, read_write> out8: array<u32>;

var<immediate> op: Op;

// ------------------------------------------------------------------------------------------------ the passes
fn byte_at(i: u32, which: u32) -> u32 {
    switch which {
        case 0u: { return (shadeb[i >> 2u] >> (8u * (i & 3u))) & 255u; }
        case 1u: { return (texb[i >> 2u] >> (8u * (i & 3u))) & 255u; }
        default: { return (typeb[i >> 2u] >> (8u * (i & 3u))) & 255u; }
    }
}

// every id is one the tables hold: cutout.rs (Plan::new) stops on one they do not, as the CPU look does
fn id_at(i: u32) -> u32 {
    return (idb[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
}

fn group_at(i: u32) -> u32 {
    return idinfo[id_at(i)].x;
}

fn is_lit(i: u32) -> bool {
    return lit[i] != 0u;
}

// _vector_slots and _flat: a pass pixel's fill colour
fn flat(i: u32) -> vec3<f32> {
    let id = id_at(i);
    if id == 0u {
        return fr.bg.rgb;
    }
    let t = tones[idinfo[id].y];
    var band = 0u;
    var tc = vec3<u32>(0u);
    if fr.misc.y != 0u {
        tc = vec3<u32>(byte_at(3u * i, 1u), byte_at(3u * i + 1u, 1u), byte_at(3u * i + 2u, 1u));
    }
    let c = vec3<f32>(units[tc.x], units[tc.y], units[tc.z]);
    if t.kind == FLAT {
        band = 0u;
    } else if t.kind == LIT {
        band = select(1u, 0u, is_lit(i));
    } else {
        for (var k = 0u; k < t.steps; k++) {
            if tc.z >= steps[t.table + k * 65536u + tc.x * 256u + tc.y] {
                band += 1u;
            }
        }
        let hi = max(max(c.x, c.y), c.z);
        let lo = min(min(c.x, c.y), c.z);
        if band == t.last && (hi - lo) >= bitcast<f32>(t.grey) * max(hi, 1e-6) {
            band = t.last - 1u;
        }
    }
    var col = colors[t.first + band].rgb;
    if t.cel != 0u {
        col = col * c;
    }
    return col;
}

// ------------------------------------------------------------------------------------------------ rounding
// A product rounded to float32 before what it is added to, as numpy rounds it: the driver may fuse a multiply and an
// add into one fma, rounded once, and a scissors' cut that falls exactly between two pixels then goes the other way.
// A select on a uniform it cannot know (fl.w is always 0) keeps the product apart.
fn keep(x: f32) -> f32 {
    return select(x, 0.0, fr.fl.w != 0.0);
}

fn keep3(x: vec3<f32>) -> vec3<f32> {
    return select(x, vec3<f32>(0.0), fr.fl.w != 0.0);
}

// ------------------------------------------------------------------------------------------------ the noise
// crayon.noise: the grid blended along y, then along x (rows[:, ix] * (1 - fx) + rows[:, ix + 1] * fx)
fn blend(a: f32, b: f32, f: f32) -> f32 {
    return keep(a * (1.0 - f)) + keep(b * f);
}

fn snoise(n: u32, y: u32, x: u32) -> f32 {
    let d = noises[n];
    let ty = saxes[d.z + y];
    let tx = saxes[d.w + x];
    let fy = bitcast<f32>(ty.y);
    let fx = bitcast<f32>(tx.y);
    let g0 = d.x + ty.x * d.y + tx.x;
    let g1 = g0 + d.y;
    return blend(blend(sgrid[g0], sgrid[g1], fy), blend(sgrid[g0 + 1u], sgrid[g1 + 1u], fy), fx);
}

fn fnoise(n: u32, y: u32, x: u32) -> f32 {
    let d = noises[n];
    let ty = faxes[d.z + y];
    let tx = faxes[d.w + x];
    let fy = bitcast<f32>(ty.y);
    let fx = bitcast<f32>(tx.y);
    let g0 = d.x + ty.x * d.y + tx.x;
    let g1 = g0 + d.y;
    return blend(blend(fgrid[g0], fgrid[g1], fy), blend(fgrid[g0 + 1u], fgrid[g1 + 1u], fy), fx);
}

// crayon.fibres' sum (`amount` of the mottle, the fibres and the fine grain)
fn fibre_sum(amount: f32, mottle: f32, fib: f32, fine: f32) -> f32 {
    return 1.0 + keep(amount * (keep(0.035 * mottle) + keep(0.06 * fib) + keep(0.03 * fine)));
}

// crayon.fibres: the background's sheet (noises 0..3, `grain` of it), at a pass pixel
fn sheet(y: u32, x: u32) -> f32 {
    return fibre_sum(fr.fl.y, snoise(0u, y, x), snoise(1u, y, x) * snoise(2u, y, x), snoise(3u, y, x));
}

fn background(y: u32, x: u32) -> vec3<f32> {
    return fr.bg.rgb * sheet(y, x);
}

// ------------------------------------------------------------------------------------------------ helpers
fn in_win(x: i32, y: i32) -> bool {
    return x >= i32(op.win.x) && y >= i32(op.win.y) && x < i32(op.win.z) && y < i32(op.win.w);
}

fn at(id: vec3<u32>) -> vec2<u32> {
    return id.xy + op.win.xy;
}

fn inside(p: vec2<u32>) -> bool {
    return p.x < op.win.z && p.y < op.win.w;
}

fn bi(p: vec2<u32>) -> u32 {
    return p.y * fr.bbox.z + p.x;
}

// ------------------------------------------------------------------------------------------------ the lit mask
@compute @workgroup_size(16, 16)
fn lit_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let i = p.y * fr.size.x + p.x;
    lit[i] = select(0u, 1u, byte_at(i, 0u) >= fr.misc.w);
}

// dilate_round(m, 2) (a.x 0) or its erosion ~dilate_round(~m, 2) (a.x 1): the octagon |dx|, |dy| <= 2, |dx| + |dy| <= 3,
// what lies outside the pass counting for nothing; from the first half into the second (a.y 0) or back
@compute @workgroup_size(16, 16)
fn lit_morph(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let n = fr.misc2.z;
    let src0 = select(0u, n, op.a.y != 0);
    let dst0 = select(n, 0u, op.a.y != 0);
    let erode = op.a.x != 0;
    var hit = erode;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            if abs(dx) + abs(dy) > 3 { continue; }
            let x = i32(p.x) + dx;
            let y = i32(p.y) + dy;
            if x < 0 || y < 0 || x >= i32(fr.size.x) || y >= i32(fr.size.y) { continue; }
            let v = lit[src0 + u32(y) * fr.size.x + u32(x)] != 0u;
            if erode { hit = hit && v; } else { hit = hit || v; }
        }
    }
    lit[dst0 + p.y * fr.size.x + p.x] = select(0u, 1u, hit);
}

// ------------------------------------------------------------------------------------------------ the figure's box
// the pieces' sheet (crayon.fibres, noises 8..11, the whole of it) over the box and its pad
@compute @workgroup_size(16, 16)
fn paper_sheet(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let fib = fnoise(9u, p.y, p.x) * fnoise(10u, p.y, p.x);
    texf[p.y * fr.misc.x + p.x] = fibre_sum(1.0, fnoise(8u, p.y, p.x), fib, fnoise(11u, p.y, p.x));
}

// _warped: each box pixel takes the pass pixel the scissors' wander (noises 4, 5 for dy, 6, 7 for dx) puts there; the
// box of each piece once cut
@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let rough = fr.fl.x;
    let dy = keep((keep(0.7 * fnoise(4u, p.y, p.x)) + keep(0.3 * fnoise(5u, p.y, p.x))) * rough);
    let dx = keep((keep(0.7 * fnoise(6u, p.y, p.x)) + keep(0.3 * fnoise(7u, p.y, p.x))) * rough);
    let yy = u32(clamp(round(f32(p.y) + dy), 0.0, f32(fr.bbox.w - 1u)));
    let xx = u32(clamp(round(f32(p.x) + dx), 0.0, f32(fr.bbox.z - 1u)));
    let s = (fr.bbox.y + yy) * fr.size.x + fr.bbox.x + xx;
    let i = bi(p);
    wsrc[i] = s;
    let g = group_at(s);
    pw[i] = g;
    if g > 0u {
        atomicMin(&bboxes[4u * g], p.x);
        atomicMin(&bboxes[4u * g + 1u], p.y);
        atomicMax(&bboxes[4u * g + 2u], p.x);
        atomicMax(&bboxes[4u * g + 3u], p.y);
    }
}

// the background's paper under the figure
@compute @workgroup_size(16, 16)
fn pic_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    pic[bi(p)] = vec4<f32>(background(fr.bbox.y + p.y, fr.bbox.x + p.x), 0.0);
}

// the figure's mask (a.x 0) or piece a.x's (with the colours it starts from: its own pixels), into the first half
@compute @workgroup_size(16, 16)
fn mask_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let i = bi(p);
    let g = u32(op.a.x);
    let m = select(pw[i] > 0u, pw[i] == g, g > 0u);
    mask[i] = select(0u, 1u, m);
    if g > 0u {
        srcb[i] = select(NONE, i, m);
        stamp[i] = select(NONE, 0u, m);
    }
}

fn half(which: i32) -> u32 {
    return select(0u, fr.misc2.y, which != 0);
}

// shotstyle.dilate's square: a.x px along x (a.y 0) or y (a.y 1), from half a.z into the other, inside the window
@compute @workgroup_size(16, 16)
fn dil_square(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let src0 = half(op.a.z);
    let dst0 = half(1 - op.a.z);
    var hit = false;
    for (var k = -op.a.x; k <= op.a.x; k++) {
        let x = i32(p.x) + select(0, k, op.a.y == 0);
        let y = i32(p.y) + select(k, 0, op.a.y == 0);
        if in_win(x, y) && mask[src0 + u32(y) * fr.bbox.z + u32(x)] != 0u {
            hit = true;
            break;
        }
    }
    mask[dst0 + bi(p)] = select(0u, 1u, hit);
}

// dilate_round's step of a cross, from half a.z into the other, inside the window
@compute @workgroup_size(16, 16)
fn dil_cross(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let src0 = half(op.a.z);
    let dst0 = half(1 - op.a.z);
    let x = i32(p.x);
    let y = i32(p.y);
    var hit = mask[src0 + bi(p)] != 0u;
    var d = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
    for (var k = 0; k < 4; k++) {
        let q = vec2<i32>(x, y) + d[k];
        if in_win(q.x, q.y) && mask[src0 + u32(q.y) * fr.bbox.z + u32(q.x)] != 0u {
            hit = true;
        }
    }
    mask[dst0 + bi(p)] = select(0u, 1u, hit);
}

// _spread's step in direction (a.x, a.y), sub-step a.z: a pixel of the grown piece (half b.x) without a colour takes
// its neighbour's, if that one had it before this sub-step
@compute @workgroup_size(16, 16)
fn spread(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let i = bi(p);
    if mask[half(op.b.x) + i] == 0u || stamp[i] != NONE { return; }
    let sx = i32(p.x) - op.a.x;
    let sy = i32(p.y) - op.a.y;
    if !in_win(sx, sy) { return; }
    let j = u32(sy) * fr.bbox.z + u32(sx);
    if stamp[j] < u32(op.a.z) {
        srcb[i] = srcb[j];
        stamp[i] = u32(op.a.z);
    }
}

// a mask (half b.x) shifted by (a.z, a.w) and that by the shadow's (a.x, a.y), in the window: what is boxed into a
// shadow, into the first half of the blur buffer
@compute @workgroup_size(16, 16)
fn cast_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let q = vec2<i32>(p) - op.a.xy;
    let r = q - op.a.zw;
    var v = 0.0;
    if in_win(q.x, q.y) && in_win(r.x, r.y) && mask[half(op.b.x) + u32(r.y) * fr.bbox.z + u32(r.x)] != 0u {
        v = 1.0;
    }
    sbuf[bi(p)] = v;
}

// crayon.blur's box of b.x px each way along x (b.y 0) or y (b.y 1), the window's edges held, from half b.z to the other
@compute @workgroup_size(16, 16)
fn blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let src0 = half(op.b.z);
    let dst0 = half(1 - op.b.z);
    let r = op.b.x;
    var sum = 0.0;
    for (var k = -r; k <= r; k++) {
        var x = i32(p.x);
        var y = i32(p.y);
        if op.b.y == 0 {
            x = clamp(x + k, i32(op.win.x), i32(op.win.z) - 1);
        } else {
            y = clamp(y + k, i32(op.win.y), i32(op.win.w) - 1);
        }
        sum += sbuf[src0 + u32(y) * fr.bbox.z + u32(x)];
    }
    sbuf[dst0 + bi(p)] = sum / f32(2 * r + 1);
}

fn shadowed(c: vec3<f32>, i: u32) -> vec3<f32> {
    let a = keep(op.f.x * sbuf[half(op.b.z) + i]);
    return keep3(c * (1.0 - a)) + keep3(fr.shc.rgb * a);
}

fn paper(p: vec2<u32>, oy: i32, ox: i32) -> f32 {
    let grain = fr.fl.y;
    return 1.0 + keep(grain * (texf[(p.y + u32(oy)) * fr.misc.x + p.x + u32(ox)] - 1.0));
}

// the figure's drop shadow (b.y: one, its blur in half b.z, alpha f.x) and its rim (b.w: one, the figure in half b.x,
// its paper at (a.x, a.y))
@compute @workgroup_size(16, 16)
fn figure(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let i = bi(p);
    var c = pic[i].rgb;
    if op.b.y != 0 {
        c = shadowed(c, i);
    }
    if op.b.w != 0 && mask[half(op.b.x) + i] != 0u {
        c = fr.rim.rgb * paper(p, op.a.x, op.a.y);
    }
    pic[i] = vec4<f32>(c, 0.0);
}

// a piece: its shadow on what is under it (b.y: one, blurred in half b.z, alpha f.x), then the piece itself where its
// grown mask (half b.x) is once lifted by (a.x, a.y), in its colours carried out to its margin, on its paper at (a.z, a.w)
@compute @workgroup_size(16, 16)
fn piece(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let i = bi(p);
    var c = pic[i].rgb;
    if op.b.y != 0 {
        c = shadowed(c, i);
    }
    let q = vec2<i32>(p) - op.a.xy;
    if in_win(q.x, q.y) {
        let j = u32(q.y) * fr.bbox.z + u32(q.x);
        if mask[half(op.b.x) + j] != 0u {
            var s = srcb[j];
            if s == NONE { s = j; }
            c = flat(wsrc[s]) * paper(p, op.a.z, op.a.w);
        }
    }
    pic[i] = vec4<f32>(c, 0.0);
}

// ------------------------------------------------------------------------------------------------ the frame
// compose_vector's box down (numpy's order: ((p00 + p01) + p10) + p11, over 4), the type laid over it, 8 bits
fn composed(y: u32, x: u32) -> vec3<f32> {
    let b = fr.bbox;
    if x >= b.x && y >= b.y && x < b.x + b.z && y < b.y + b.w {
        return pic[(y - b.y) * b.z + (x - b.x)].rgb;
    }
    return background(y, x);
}

@compute @workgroup_size(16, 16)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = at(id);
    if !inside(p) { return; }
    let y = 2u * p.y;
    let x = 2u * p.x;
    var c = ((composed(y, x) + composed(y, x + 1u)) + composed(y + 1u, x)) + composed(y + 1u, x + 1u);
    c = c / 4.0;
    let o = p.y * fr.size.z + p.x;
    if fr.misc.z != 0u {
        let t = vec4<f32>(units[byte_at(4u * o, 2u)], units[byte_at(4u * o + 1u, 2u)], units[byte_at(4u * o + 2u, 2u)],
                          units[byte_at(4u * o + 3u, 2u)]);
        let a = clamp(t.w, 0.0, 1.0);
        c = keep3(c * (1.0 - a)) + keep3(t.rgb * a);
    }
    let v = vec3<u32>(keep3(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0) + 0.5);
    out8[o] = v.x | (v.y << 8u) | (v.z << 16u) | (255u << 24u);
}
