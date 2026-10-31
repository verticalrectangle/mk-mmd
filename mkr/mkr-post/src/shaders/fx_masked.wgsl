// The masked remaps' shared work (mkmmd.remap: RemapFx._remapped, fill, pushpull, sticker, shade; the kinds kaleido,
// tunnel and melt), row by row on pictures of one vec4<f32> per pixel and coverages of one f32 (the figure, the hole,
// a box's coverage: `fa`, `fb`, `fc`, `fd`), each the numpy / OpenCV expression it stands for in float32 and in its
// order:
//   bbox, bbox1    the box of the pixels whose coverage (.x of a picture plane; a coverage) is over P.a
//                  (cv2.boundingRect of cov > HOLE_CUT)
//   crop, uncrop   a box of a coverage copied out, as numpy slices it, and laid back
//   cover_of       clip(alpha, 0, 1) of a remapped premultiplied figure, as its coverage
//   gate           pushpull's passes dispatched (all their workgroups) only when fill_lost found them needed
//   known          pushpull's first level: the picture times known = (hole <= cut), known in .w
//   pyr_down       cv2.pyrDown (5 x 5 [1 4 6 4 1] / 256, BORDER_REFLECT_101, rows then columns)
//   pull_top       pushpull's coarsest colour: where(a > 1e-4, P / max(a, 1e-4), 0.5)
//   pull_up        pushpull's way back up: P + (1 - A) * up
//   fill_scan      per row, each pixel's last ground pixel at or before it and first at or after it (fill's left and
//                  right: maximum.accumulate and minimum.accumulate), one workgroup a row
//   fill_lost      whether some hole pixel has no ground to mirror in
//   fill_mix       fill: a hole pixel takes the ground mirrored in from its two sides, crossfaded, or the colour
//                  pushpull spread there when neither mirror reaches, mixed in by the hole's coverage
//   sticker        sticker's composite over its box: shadow, edge ring, the figure's coverage cut out and its
//                  premultiplied picture added
// The ring, the shadow and their blurs are fx_plane.wgsl's (fx/plane.rs).

struct Params {
    w: u32,             // the plane written (the box, for crop and sticker)
    h: u32,
    sw: u32,            // the plane read, where it is another size
    sh: u32,
    x0: u32,            // crop, uncrop, sticker: the box's top left in the full plane
    y0: u32,
    flag: u32,          // sticker: 1 a shadow, 2 an edge ring; gate: the passes
    pad: u32,
    a: f32,             // bbox, fill: the cut; sticker: the shadow's alpha
    b: f32,
    c: f32,
    d: f32,
    colour: vec4<f32>,  // sticker: the shadow's colour
    colour2: vec4<f32>, // sticker: the edge's colour
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> aux: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> aux2: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> lr: array<vec2<i32>>;
@group(0) @binding(7) var<storage, read_write> box: array<atomic<u32>, 4>;
@group(0) @binding(8) var<storage, read> fa: array<f32>;
@group(0) @binding(9) var<storage, read_write> fb: array<f32>;
@group(0) @binding(10) var<storage, read> fc: array<f32>;
@group(0) @binding(11) var<storage, read> grids: array<u32>;
@group(0) @binding(12) var<storage, read_write> args: array<u32>;
@group(0) @binding(13) var<storage, read> fd: array<f32>;

fn here(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

// gate: the indirect dispatches of pushpull's P.flag passes: each its workgroups (`grids`: x, y) when fill_lost found
// a hole pixel with neither mirror (box[0] set), none otherwise.
@compute @workgroup_size(64)
fn gate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.flag) {
        return;
    }
    let on = atomicLoad(&box[0]) != 0u;
    args[3u * id.x] = select(0u, grids[2u * id.x], on);
    args[3u * id.x + 1u] = select(0u, grids[2u * id.x + 1u], on);
    args[3u * id.x + 2u] = 1u;
}

// ---------------------------------------------------------------- bbox: box[0] = w - x0, box[1] = h - y0, box[2] = x1,
// box[3] = y1 (all 0: nothing over the cut), each the largest over the workgroups' own; bbox reads .x of `src`, bbox1
// the coverage `fa`
var<workgroup> wbox: array<atomic<u32>, 4>;

fn bbox_put(id: vec3<u32>, li: u32, over: bool) {
    if (over) {
        atomicMax(&wbox[0], P.w - id.x);
        atomicMax(&wbox[1], P.h - id.y);
        atomicMax(&wbox[2], id.x + 1u);
        atomicMax(&wbox[3], id.y + 1u);
    }
    workgroupBarrier();
    if (li < 4u) {
        let v = atomicLoad(&wbox[li]);
        if (v > 0u) {
            atomicMax(&box[li], v);
        }
    }
}

@compute @workgroup_size(16, 16)
fn bbox(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li < 4u) {
        atomicStore(&wbox[li], 0u);
    }
    workgroupBarrier();
    bbox_put(id, li, id.x < P.w && id.y < P.h && src[id.y * P.w + id.x].x > P.a);
}

@compute @workgroup_size(16, 16)
fn bbox1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li < 4u) {
        atomicStore(&wbox[li], 0u);
    }
    workgroupBarrier();
    bbox_put(id, li, id.x < P.w && id.y < P.h && fa[id.y * P.w + id.x] > P.a);
}

// ---------------------------------------------------------------- crop: fb (w x h) = fa[y0:y0 + h, x0:x0 + w] (fa sw wide)
@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    fb[i] = fa[(P.y0 + id.y) * P.sw + P.x0 + id.x];
}

// uncrop: a box (w x h) of `fa` laid back into `fb` (sw wide) at (x0, y0).
@compute @workgroup_size(16, 16)
fn uncrop(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = here(id);
    if (j < 0) {
        return;
    }
    fb[(P.y0 + id.y) * P.sw + P.x0 + id.x] = fa[j];
}

@compute @workgroup_size(16, 16)
fn cover_of(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    fb[i] = clamp(src[i].w, 0.0, 1.0);
}

// ---------------------------------------------------------------- pushpull
// known: P0 = img * known, known = 1 where the hole (`fa`) is at most the cut, else 0; known in .w.
@compute @workgroup_size(16, 16)
fn known(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let k = select(1.0, 0.0, fa[i] > P.a);
    dst[i] = vec4<f32>(src[i].xyz * k, k);
}

fn reflect101(p: i32, n: i32) -> i32 {
    if (n == 1) {
        return 0;
    }
    let period = 2 * n - 2;
    var q = p % period;
    if (q < 0) {
        q = q + period;
    }
    if (q >= n) {
        q = period - q;
    }
    return q;
}

// One row of pyrDown's horizontal pass at source row `y`, written column x: s[2x] * 6 + (s[2x - 1] + s[2x + 1]) * 4 +
// s[2x - 2] + s[2x + 2].
fn pyr_row(y: i32, x: i32) -> vec4<f32> {
    let w = i32(P.sw);
    let row = u32(reflect101(y, i32(P.sh))) * P.sw;
    let c = 2 * x;
    let m2 = src[row + u32(reflect101(c - 2, w))];
    let m1 = src[row + u32(reflect101(c - 1, w))];
    let z = src[row + u32(reflect101(c, w))];
    let p1 = src[row + u32(reflect101(c + 1, w))];
    let p2 = src[row + u32(reflect101(c + 2, w))];
    return z * 6.0 + (m1 + p1) * 4.0 + m2 + p2;
}

@compute @workgroup_size(16, 16)
fn pyr_down(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let x = i32(id.x);
    let c = 2 * i32(id.y);
    let rm = pyr_row(c - 2, x);
    let r0 = pyr_row(c - 1, x);
    let r1 = pyr_row(c, x);
    let r2 = pyr_row(c + 1, x);
    let r3 = pyr_row(c + 2, x);
    dst[i] = (r1 * 6.0 + (r0 + r2) * 4.0 + rm + r3) * (1.0 / 256.0);
}

@compute @workgroup_size(16, 16)
fn pull_top(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let p = src[i];
    let a = p.w;
    var c = vec3<f32>(0.5);
    if (a > 1e-4) {
        c = p.xyz / max(a, 1e-4);
    }
    dst[i] = vec4<f32>(c, 0.0);
}

// dst = P[j] + (1 - A[j]) * up: `src` the level (its known in .w), `aux` the colour of the level above resized to it.
@compute @workgroup_size(16, 16)
fn pull_up(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let p = src[i];
    dst[i] = vec4<f32>(p.xyz + (1.0 - p.w) * aux[i].xyz, 0.0);
}

// ---------------------------------------------------------------- fill
// One workgroup a row: lr[x] = (last x' <= x outside the hole or -1, first x' >= x outside it or w), the hole being
// fa > P.a. Each of the 256 threads scans its own run of the row; the runs' ends are carried across by a scan of the
// workgroup's.
const FILL_THREADS: u32 = 256u;
var<workgroup> carry_l: array<i32, 256>;
var<workgroup> carry_r: array<i32, 256>;

@compute @workgroup_size(256)
fn fill_scan(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let y = wg.x + wg.y * 65535u;
    let w = i32(P.w);
    let n = (P.w + FILL_THREADS - 1u) / FILL_THREADS;
    let a = min(t * n, P.w);
    let b = min(a + n, P.w);
    let row = y * P.w;
    let inside = y < P.h;
    // the run's own last ground pixel (or -1) and first (or w)
    var last = -1;
    var first = w;
    if (inside) {
        for (var x = a; x < b; x = x + 1u) {
            if (!(fa[row + x] > P.a)) {
                last = i32(x);
                if (first == w) {
                    first = i32(x);
                }
            }
        }
    }
    carry_l[t] = last;
    carry_r[t] = first;
    workgroupBarrier();
    // inclusive scans: carry_l[t] the last ground pixel up to the end of run t; carry_r[t] the first from run t on
    for (var s = 1u; s < FILL_THREADS; s = s * 2u) {
        var l = carry_l[t];
        var r = carry_r[t];
        if (t >= s) {
            l = max(l, carry_l[t - s]);
        }
        if (t + s < FILL_THREADS) {
            r = min(r, carry_r[t + s]);
        }
        workgroupBarrier();
        carry_l[t] = l;
        carry_r[t] = r;
        workgroupBarrier();
    }
    if (!inside) {
        return;
    }
    var left = -1;
    if (t > 0u) {
        left = carry_l[t - 1u];
    }
    for (var x = a; x < b; x = x + 1u) {
        if (!(fa[row + x] > P.a)) {
            left = i32(x);
        }
        lr[row + x].x = left;
    }
    var right = w;
    if (t + 1u < FILL_THREADS) {
        right = carry_r[t + 1u];
    }
    for (var x = b; x > a; x = x - 1u) {
        if (!(fa[row + x - 1u] > P.a)) {
            right = i32(x - 1u);
        }
        lr[row + x - 1u].y = right;
    }
}

// fill_mix: `src` the picture, `fa` the hole, `aux2` what pushpull spread, lr the scan; P.a the cut.
fn ground_at(row: u32, x: i32) -> vec4<f32> {
    return src[row + u32(clamp(x, 0, i32(P.w) - 1))];
}

fn is_hole(row: u32, x: i32) -> bool {
    return fa[row + u32(clamp(x, 0, i32(P.w) - 1))] > P.a;
}

// Where hole pixel i at (x, row) mirrors the ground in from: (src_l, src_r, ok_l, ok_r) as fill works them out.
fn mirrors(i: u32, row: u32, x: i32) -> vec4<i32> {
    let w = i32(P.w);
    let s = lr[i];
    let src_l = 2 * s.x + 1 - x;
    let src_r = 2 * s.y - 1 - x;
    let ok_l = s.x >= 0 && src_l >= 0 && !is_hole(row, src_l);
    let ok_r = s.y < w && src_r < w && !is_hole(row, src_r);
    return vec4<i32>(src_l, src_r, select(0, 1, ok_l), select(0, 1, ok_r));
}

// fill_lost: box[0] = 1 when some hole pixel has neither mirror (pushpull is needed); `fa` the hole.
@compute @workgroup_size(16, 16)
fn fill_lost(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0 || !(fa[i] > P.a)) {
        return;
    }
    let m = mirrors(u32(i), id.y * P.w, i32(id.x));
    if (m.z == 0 && m.w == 0) {
        atomicMax(&box[0], 1u);
    }
}

@compute @workgroup_size(16, 16)
fn fill_mix(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = here(id);
    if (i < 0) {
        return;
    }
    let img = src[i];
    let hole = fa[i];
    if (!(hole > P.a)) {
        dst[i] = img;
        return;
    }
    let row = id.y * P.w;
    let x = i32(id.x);
    let m = mirrors(u32(i), row, x);
    let ok_l = m.z != 0;
    let ok_r = m.w != 0;
    var filled: vec3<f32>;
    if (!ok_l && !ok_r) {
        filled = aux2[i].xyz;
    } else {
        let s = lr[i];
        let dl = f32(x - s.x);
        let dr = f32(s.y - x);
        let u = clamp(0.5 + (dr - dl) / max(0.5 * (dl + dr), 8.0), 0.0, 1.0);
        var wl = select(0.0, 1.0, ok_l);
        if (ok_l && ok_r) {
            wl = u * u * (3.0 - 2.0 * u);
        }
        filled = ground_at(row, m.x).xyz * wl + ground_at(row, m.y).xyz * (1.0 - wl);
    }
    dst[i] = vec4<f32>(img.xyz * (1.0 - hole) + filled * hole, img.w);
}

// ---------------------------------------------------------------- sticker: over its box (w x h) of `dst` (sw wide, at x0, y0)
// fa: the figure's coverage (box), fc: the shadow's shade (box, before its alpha), fd: the edge's ring (box), src: the
// premultiplied figure (full plane).
@compute @workgroup_size(16, 16)
fn sticker(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = here(id);
    if (j < 0) {
        return;
    }
    let i = (P.y0 + id.y) * P.sw + P.x0 + id.x;
    var d = dst[i];
    var rgb = d.xyz;
    if ((P.flag & 1u) != 0u) {
        let a = fc[j] * P.a;
        rgb = rgb + (P.colour.xyz - rgb) * a;
    }
    if ((P.flag & 2u) != 0u) {
        rgb = rgb + (P.colour2.xyz - rgb) * fd[j];
    }
    rgb = rgb * (1.0 - fa[j]);
    rgb = rgb + src[i].xyz;
    dst[i] = vec4<f32>(rgb, d.w);
}
