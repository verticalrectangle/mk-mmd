// EEVEE Next's motion blur as a post process on each sample's picture (motion_blur.rs; records.wgsl comes first): the
// picture drawn at the shutter's middle is smeared along each pixel's velocity to where the shutter's open and its close
// show it (the G-buffer's motion target, pixels). A gather after McGuire et al. (2012) and Guertin et al. (2014) as
// Jimenez (2014) describes it, its constants measured on EEVEE's frames:
// - tiles: per TILE x TILE pixels the longest velocity toward the open and toward the close (and each pixel's velocity
//   kept for the gather, where nothing is drawn the sky's: the camera's turn alone);
// - dilate: each tile's longest velocities spread over the tiles their blur reaches (the tiles within sqrt 2 tiles of
//   the line from the tile along either of them), each tile keeping the longest of each half that reaches it (a tile
//   reached along the open's way takes the close's too, and the other way: measured, EEVEE's weights need it);
// - gather: per pixel and per half of the shutter, TAPS samples along the longest velocity of its tile (the tile picked
//   a random quarter tile off) and TAPS along its own, a sample counting as background (behind the pixel, within the
//   pixel's own blur) or foreground (in front, its blur reaching the pixel) by a soft depth test and a soft test of the
//   blur's reach, samples moving away from the pixel dropped; the background's mean stands for the pixel's own colour
//   where the samples leave it uncovered.

const TILE: i32 = 32;
// TAPS, injected by motion_blur.rs (the `mb_taps` param): EEVEE takes 8 per sample; over a frame's 64 samples 6 leave
// mkr's frames as close to its (measured: the lab's mb cases and two MVs' moving pixels within 0.02 levels either way)
// for 1-2 ms less per 1080p frame; fewer samples take more (final frames: 8 over 36 samples)

struct Mb {
    size: vec2<u32>,
    tiles: vec2<u32>,
    sample: u32,
    ortho: u32,
    near: f32,
    far: f32,
    depth_scale: f32,       // the soft depth test's slope (per metre of view depth)
    noise: f32,             // the sample's random offset of the per-pixel noise
    pad: vec2<u32>,
}

var<immediate> mb: Mb;

@group(0) @binding(0) var<storage, read> samples: array<SampleData>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var color_in: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> tiles: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> dilated: array<atomic<u32>>;
@group(0) @binding(7) var color_out: texture_storage_2d<rgba16float, write>;
// per pixel: its velocity toward the open and toward the close (half floats)
@group(0) @binding(8) var<storage, read_write> pixels: array<vec2<u32>>;

// The view depth (metres along the axis) of a reversed depth value.
fn view_depth(d: f32) -> f32 {
    if (mb.ortho != 0u) {
        return mb.far - d * (mb.far - mb.near);
    }
    return mb.near * mb.far / ((mb.far - mb.near) * d + mb.near);
}

// A pixel's velocity (pixels: toward the open, toward the close) at reversed depth `depth`. Background pixels (nothing
// drawn) move as the sky does: the camera's turn alone.
fn velocity(px: vec2<i32>, depth: f32) -> vec4<f32> {
    if (depth > 0.0) {
        return textureLoad(motion_tex, px, 0);
    }
    let sd = samples[mb.sample];
    let size = vec2<f32>(mb.size);
    let ndc = vec2<f32>((f32(px.x) + 0.5) / size.x * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) / size.y * 2.0);
    let far = sd.vp_inv * vec4<f32>(ndc, 0.0, 1.0);
    let near = sd.vp_inv * vec4<f32>(ndc, 1.0, 1.0);
    let dir = vec4<f32>(far.xyz / far.w - near.xyz / near.w, 0.0);
    let c = sd.vp_cur * dir;
    let a = sd.vp_prev * dir;
    let b = sd.vp_next * dir;
    if (mb.ortho != 0u || c.w <= 0.0 || a.w <= 0.0 || b.w <= 0.0) {
        return vec4<f32>(0.0);
    }
    let cur = c.xy / c.w;
    let to = vec2<f32>(0.5, -0.5) * size;
    return vec4<f32>((a.xy / a.w - cur) * to, (b.xy / b.w - cur) * to);
}

// ------------------------------------------------------------------------------------------------------------ tiles
var<workgroup> wg_prev: array<vec2<f32>, 256>;
var<workgroup> wg_next: array<vec2<f32>, 256>;

fn longer(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return select(a, b, dot(b, b) > dot(a, a));
}

// One workgroup per tile, each invocation a 2 x 2 block of it, the longest found by halving.
@compute @workgroup_size(16, 16)
fn tile_max(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    var p = vec2<f32>(0.0);
    var n = vec2<f32>(0.0);
    let origin = vec2<i32>(wg.xy) * TILE + vec2<i32>(lid.xy) * 2;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let px = origin + vec2<i32>(x, y);
            if (px.x < i32(mb.size.x) && px.y < i32(mb.size.y)) {
                let v = velocity(px, textureLoad(depth_tex, px, 0));
                p = longer(p, v.xy);
                n = longer(n, v.zw);
                pixels[u32(px.y) * mb.size.x + u32(px.x)] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
            }
        }
    }
    wg_prev[li] = p;
    wg_next[li] = n;
    for (var span = 128u; span > 0u; span = span >> 1u) {
        workgroupBarrier();
        if (li < span) {
            wg_prev[li] = longer(wg_prev[li], wg_prev[li + span]);
            wg_next[li] = longer(wg_next[li], wg_next[li + span]);
        }
    }
    if (li == 0u) {
        tiles[wg.y * mb.tiles.x + wg.x] = vec4<f32>(wg_prev[0], wg_next[0]);
    }
}

// --------------------------------------------------------------------------------------------------------- dilation
// A tile's claim: its velocity's length (whole pixels) over its index, so that the longest wins an atomic maximum.
fn claim(v: vec2<f32>, tile: u32) -> u32 {
    return (min(u32(ceil(length(v))), 16383u) << 18u) | (tile & 0x3ffffu);
}

fn spread(src: vec2<i32>, v: vec2<f32>, slot: u32, claim_a: u32, claim_b: u32) {
    let reach = vec2<i32>(sign(v) * ceil(abs(v) / f32(TILE)));
    let last = vec2<i32>(mb.tiles) - 1;
    let lo = clamp(min(src, src + reach), vec2<i32>(0), last);
    let hi = clamp(max(src, src + reach), vec2<i32>(0), last);
    let l = length(v);
    let dir = select(vec2<f32>(1.0, 0.0), v / l, l > 0.0);
    let nrm = vec2<f32>(-dir.y, dir.x);
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let d = dot(vec2<f32>(vec2<i32>(x, y) - src), nrm);
            if (abs(d) < 1.41421356) {
                let t = u32(y) * mb.tiles.x + u32(x);
                atomicMax(&dilated[t * 2u + slot], claim_a);
                atomicMax(&dilated[t * 2u + 1u - slot], claim_b);
            }
        }
    }
}

@compute @workgroup_size(8, 8)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mb.tiles.x || gid.y >= mb.tiles.y) {
        return;
    }
    let tile = gid.y * mb.tiles.x + gid.x;
    let v = tiles[tile];
    let src = vec2<i32>(gid.xy);
    let cp = claim(v.xy, tile);
    let cn = claim(v.zw, tile);
    spread(src, v.xy, 0u, cp, cn);
    spread(src, v.zw, 1u, cn, cp);
}

// ----------------------------------------------------------------------------------------------------------- gather
// Interleaved gradient noise (Jimenez 2014), a second domain shifted across the picture, offset per sample.
fn noise(px: vec2<f32>, domain: f32) -> f32 {
    let q = px + domain * vec2<f32>(47.0, 17.0);
    return fract(mb.noise + 52.9829189 * fract(0.06711056 * q.x + 0.00583715 * q.y));
}

struct Acc {
    fg: vec4<f32>,
    bg: vec4<f32>,
    w: vec3<f32>,           // background, foreground, the samples that count (moving the pixel's way or still)
}

fn tap(acc: ptr<function, Acc>, px: vec2<i32>, center_depth: f32, center_len: f32, offset: vec2<f32>, len: f32,
       next: bool) {
    let last = vec2<i32>(mb.size) - 1;
    let q = clamp(vec2<i32>(floor(vec2<f32>(px) + 0.5 - offset)), vec2<i32>(0), last);
    let data = pixels[u32(q.y) * mb.size.x + u32(q.x)];
    let m = unpack2x16float(select(data.x, data.y, next));
    let m_len = length(m);
    let d = view_depth(textureLoad(depth_tex, q, 0)) - center_depth;
    // x: behind the pixel and inside its own blur; y: in front and its blur reaching the pixel
    let depth = saturate(vec2<f32>(0.5) + vec2<f32>(mb.depth_scale, -mb.depth_scale) * d);
    let reach = saturate(vec2<f32>(center_len, m_len) - len + 1.0);
    let toward = select(select(0.0, 1.0, dot(offset, m) > 0.0), 1.0, m_len < 0.5);
    let w = depth * reach * toward;
    let c = textureLoad(color_in, q, 0);
    (*acc).fg += c * w.y;
    (*acc).bg += c * w.x;
    (*acc).w += vec3<f32>(w, toward);
}

fn half_shutter(acc: ptr<function, Acc>, px: vec2<i32>, center: vec2<f32>, center_depth: f32, tile_max: vec2<f32>,
                r: f32, next: bool) {
    let center_len = length(center);
    var most = tile_max;
    var most_len = length(most);
    // the tile picked off its own can move less than the pixel
    if (most_len < center_len) {
        most = center;
        most_len = center_len;
    }
    if (most_len < 0.5) {
        return;
    }
    for (var i = 0; i < TAPS; i++) {
        let t = (f32(i) + r) / f32(TAPS);
        tap(acc, px, center_depth, center_len, most * t, most_len * t, next);
    }
    if (center_len < 0.5) {
        return;
    }
    // along the pixel's own way too: what moves against the tile's longest
    for (var i = 0; i < TAPS; i++) {
        let t = (f32(i) + r) / f32(TAPS);
        tap(acc, px, center_depth, center_len, center * t, center_len * t, next);
    }
}

@compute @workgroup_size(8, 8)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mb.size.x || gid.y >= mb.size.y) {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let fpx = vec2<f32>(gid.xy);
    let center_color = textureLoad(color_in, px, 0);
    // the tile a random quarter tile off (the same way along both axes), its dilated longest velocities
    let jitter = i32((noise(fpx, 0.0) * 2.0 - 1.0) * f32(TILE) * 0.25);
    let last = vec2<i32>(mb.tiles) - 1;
    let tile = clamp((px + vec2<i32>(jitter)) / TILE, vec2<i32>(0), last);
    let t = u32(tile.y) * mb.tiles.x + u32(tile.x);
    let claim_prev = atomicLoad(&dilated[t * 2u]);
    let claim_next = atomicLoad(&dilated[t * 2u + 1u]);
    // nothing within reach of that tile nor of the pixel's own moves at all: the picture stays
    let own_tile = min(px / TILE, last);
    let o = u32(own_tile.y) * mb.tiles.x + u32(own_tile.x);
    if (((claim_prev | claim_next | atomicLoad(&dilated[o * 2u]) | atomicLoad(&dilated[o * 2u + 1u])) >> 18u) == 0u) {
        textureStore(color_out, px, center_color);
        return;
    }
    let data = pixels[gid.y * mb.size.x + gid.x];
    let center = vec4<f32>(unpack2x16float(data.x), unpack2x16float(data.y));
    let most_prev = tiles[claim_prev & 0x3ffffu].xy;
    let most_next = tiles[claim_next & 0x3ffffu].zw;
    let center_depth = view_depth(textureLoad(depth_tex, px, 0));
    var acc: Acc;
    acc.fg = vec4<f32>(0.0);
    acc.bg = vec4<f32>(0.0);
    acc.w = vec3<f32>(0.0, 0.0, 1.0);
    let r = noise(fpx, 1.0);
    half_shutter(&acc, px, center.xy, center_depth, most_prev, r, false);
    half_shutter(&acc, px, center.zw, center_depth, most_next, r, true);
    // the pixel's own colour weighs little: the background's mean stands for it where the samples leave it uncovered
    let own = 1.0 / (50.0 * f32(TAPS) * 4.0);
    acc.bg += center_color * own;
    acc.w.x += own;
    let behind = acc.bg / acc.w.x;
    let fg = acc.fg + acc.bg;
    let covered = acc.w.y + acc.w.x;
    let rest = saturate(1.0 - covered / acc.w.z);
    textureStore(color_out, px, fg / acc.w.z + behind * rest);
}
