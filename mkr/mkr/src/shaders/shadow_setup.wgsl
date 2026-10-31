// The shadow maps' windows (after records.wgsl and shadow_lod.wgsl): EEVEE Next draws a light's cube at the level each
// receiver on screen asks for (shadow_lod.wgsl), so a close-up's shadows come from texels a fraction of a millimetre
// wide. mkr keeps every cube face whole at a coarse level (frame.shadow_lod.w) and shares a pool of finer windows
// (frame.shadow_pool) among the lights, placed here on the GPU from the depth of the frame's first sample:
// - shadow_tag: every pixel's receiver P, per light that casts a cube map and reaches P (inside its influence radius,
//   in front of an area light): the face and texel it falls in and the level it asks for; per face and level the box
//   of those texels (finest-level texels; reduced per workgroup first);
// - shadow_windows (one invocation): per face the finest level whose receivers fit in a window (with a margin for the
//   normal offset, the filter and motion; the next level's receivers too when they fit), when it is finer than the
//   whole face's; the pool's windows go to those faces finest level first, then the larger box first. Each placed
//   window is its face's window record (origin in its level's texels, level, its layer of the fine maps) and its
//   matrix: the face's, its uv range scaled to the window. Receivers no window serves read the whole face (and take
//   its level's biases: shadow_map.wgsl).

@group(0) @binding(0) var<storage, read> frame: Frame;
@group(0) @binding(1) var<storage, read> samples: array<SampleData>;
@group(0) @binding(2) var<storage, read> lights: array<Light>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
// per cube face, per level: the box of its receivers' texels, as atomic maxima of (8192 - min x, 8192 - min y,
// max x + 1, max y + 1) (0: none)
@group(0) @binding(4) var<storage, read_write> tags: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> windows: array<vec4<i32>>;
@group(0) @binding(6) var<storage, read_write> faces: array<ShadowFace>;

const LEVELS: u32 = 6u;
const BOUNDS: u32 = 24u;                // per face: LEVELS x 4
const MARGIN: f32 = 16.0;               // window texels kept around the receivers' box on every side

var<workgroup> wg_tags: array<atomic<u32>, 144>;    // 6 faces x BOUNDS

// Whether light `l` (a cube light) lights P at all: inside its influence radius, in front of an area light.
fn shadow_reaches(l: Light, P: vec3<f32>) -> bool {
    let d = P - l.pos.xyz;
    if (l.color.w > 0.0 && dot(d, d) * l.color.w >= 1.0) {
        return false;
    }
    let kind = u32(l.pos.w);
    return !((kind == KIND_RECT || kind == KIND_ELLIPSE) && dot(d, l.fwd.xyz) <= 0.0);
}

@compute @workgroup_size(8, 8)
fn shadow_tag(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    var ok = false;
    var P = vec3<f32>(0.0);
    if (gid.x < u32(frame.size.x) && gid.y < u32(frame.size.y)) {
        let depth = textureLoad(depth_tex, gid.xy, 0);
        if (depth > 0.0) {
            let ndc = vec2<f32>((f32(gid.x) + 0.5) * frame.size.z * 2.0 - 1.0, 1.0 - (f32(gid.y) + 0.5) * frame.size.w * 2.0);
            let ph = samples[0].vp_inv * vec4<f32>(ndc, depth, 1.0);
            P = ph.xyz / ph.w;
            ok = true;
        }
    }
    let fp = shadow_footprint(P);
    for (var i = 0u; i < frame.light_count; i = i + 1u) {
        let l = lights[i];
        if (l.map.x < 0.0 || u32(l.pos.w) == KIND_SUN) {
            continue;                               // uniform: every invocation skips alike
        }
        for (var k = li; k < 144u; k = k + 64u) {
            atomicStore(&wg_tags[k], 0u);
        }
        workgroupBarrier();
        if (ok && shadow_reaches(l, P)) {
            let f = shadow_face_of(l, P - l.pos.xyz);
            if (f.face < shadow_face_count(l)) {
                let lod = u32(shadow_lod(fp, f.a, l.map.w));
                let t = vec2<u32>(clamp(shadow_texel0(f.uv), vec2<f32>(0.0), vec2<f32>(SHADOW_LOD0 - 1.0)));
                let b = f.face * BOUNDS + lod * 4u;
                let n = u32(SHADOW_LOD0);
                atomicMax(&wg_tags[b], n - t.x);
                atomicMax(&wg_tags[b + 1u], n - t.y);
                atomicMax(&wg_tags[b + 2u], t.x + 1u);
                atomicMax(&wg_tags[b + 3u], t.y + 1u);
            }
        }
        workgroupBarrier();
        let first = u32(l.map.x);
        for (var k = li; k < 144u; k = k + 64u) {
            let v = atomicLoad(&wg_tags[k]);
            if (v != 0u) {
                atomicMax(&tags[first * BOUNDS + k], v);
            }
        }
        workgroupBarrier();
    }
}

// A face's receivers at level `lod`: (min x, min y, max x, max y) in finest-level texels; x > z when there are none.
fn tag_box(face: u32, lod: u32) -> vec4<f32> {
    let b = face * BOUNDS + lod * 4u;
    let n = SHADOW_LOD0;
    let lo_x = atomicLoad(&tags[b]);
    if (lo_x == 0u) {
        return vec4<f32>(1.0, 1.0, 0.0, 0.0);
    }
    return vec4<f32>(n - f32(lo_x), n - f32(atomicLoad(&tags[b + 1u])), f32(atomicLoad(&tags[b + 2u])) - 1.0,
                     f32(atomicLoad(&tags[b + 3u])) - 1.0);
}

fn box_union(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    if (a.x > a.z) {
        return b;
    }
    if (b.x > b.z) {
        return a;
    }
    return vec4<f32>(min(a.xy, b.xy), max(a.zw, b.zw));
}

// Whether a box fits a window of `w` texels at level `lod` with the margin.
fn box_fits(bx: vec4<f32>, lod: f32, w: f32) -> bool {
    let ext = (bx.zw - bx.xy + 1.0) / exp2(lod) + 2.0 * MARGIN;
    return ext.x <= w && ext.y <= w;
}

struct Candidate {
    lod: f32,                           // -1: none
    centre: vec2<f32>,                  // finest-level texels
    area: f32,
}

// The window face `face` asks for: the finest level its receivers (from the finest up) fit a window of `w` texels
// at, finer than the whole face's level `base_lod`.
fn face_candidate(face: u32, base_lod: f32, w: f32) -> Candidate {
    var c = Candidate(-1.0, vec2<f32>(0.0), 0.0);
    var ub = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    var started = false;
    for (var lod = 0u; lod < LEVELS; lod = lod + 1u) {
        let bx = tag_box(face, lod);
        if (bx.x > bx.z && !started) {
            continue;
        }
        started = true;
        ub = box_union(ub, bx);
        if (f32(lod) >= base_lod) {
            return c;                       // the whole face's map serves these levels
        }
        if (ub.x <= ub.z && box_fits(ub, f32(lod), w)) {
            var region = ub;
            if (lod + 1u < LEVELS) {
                let more = box_union(ub, tag_box(face, lod + 1u));
                if (box_fits(more, f32(lod), w)) {
                    region = more;
                }
            }
            c.lod = f32(lod);
            c.centre = (region.xy + region.zw + 1.0) * 0.5;
            c.area = (ub.z - ub.x + 1.0) * (ub.w - ub.y + 1.0);
            return c;
        }
    }
    return c;
}

// Whether candidate `a` takes a window before `b`: the finer level first, then the larger box.
fn before(a: Candidate, b: Candidate) -> bool {
    return a.lod < b.lod || (a.lod == b.lod && a.area > b.area);
}

@compute @workgroup_size(1)
fn shadow_windows() {
    let base_lod = frame.shadow_lod.w;
    let w = frame.shadow_pool.w;
    for (var i = 0u; i < frame.light_count; i = i + 1u) {
        let l = lights[i];
        if (l.map.x < 0.0 || u32(l.pos.w) == KIND_SUN) {
            continue;
        }
        for (var f = 0u; f < shadow_face_count(l); f = f + 1u) {
            windows[u32(l.map.x) + f] = vec4<i32>(0, 0, -1, -1);
        }
    }
    let pool = u32(frame.shadow_pool.z);
    for (var s = 0u; s < pool; s = s + 1u) {
        let entry = u32(frame.shadow_pool.y) + s;
        var best = Candidate(-1.0, vec2<f32>(0.0), 0.0);
        var best_face = 0u;
        for (var i = 0u; i < frame.light_count; i = i + 1u) {
            let l = lights[i];
            if (l.map.x < 0.0 || u32(l.pos.w) == KIND_SUN) {
                continue;
            }
            for (var f = 0u; f < shadow_face_count(l); f = f + 1u) {
                let face = u32(l.map.x) + f;
                if (windows[face].w >= 0) {
                    continue;                   // served already
                }
                let c = face_candidate(face, base_lod, w);
                if (c.lod >= 0.0 && (best.lod < 0.0 || before(c, best))) {
                    best = c;
                    best_face = face;
                }
            }
        }
        if (best.lod < 0.0) {
            // unused: every vertex behind the window
            faces[entry].vp = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, -1.0));
            continue;
        }
        let s_l = exp2(best.lod);
        let origin = floor(best.centre / s_l) - vec2<f32>(w * 0.5);
        windows[best_face] = vec4<i32>(i32(origin.x), i32(origin.y), i32(best.lod),
                                       i32(entry) - i32(frame.shadow_pool.x));
        // the face's matrix, its uv range scaled to the window's
        let half = w * s_l / SHADOW_LOD0;
        let cx = (origin.x + w * 0.5) * s_l / SHADOW_LOD0 * 2.0 - 1.0;
        let cy = 1.0 - (origin.y + w * 0.5) * s_l / SHADOW_LOD0 * 2.0;
        var m = faces[best_face].vp;
        for (var j = 0; j < 4; j = j + 1) {
            let col = m[j];
            m[j] = vec4<f32>((col.x - cx * col.w) / half, (col.y - cy * col.w) / half, col.z, col.w);
        }
        faces[entry].vp = m;
    }
}
