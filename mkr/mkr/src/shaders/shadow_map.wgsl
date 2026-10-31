// Shadows as EEVEE Next draws them by default (after shadow_lod.wgsl): per light a shadow map drawn once a frame from
// the light's centre C (a cube of 90 degree faces in the light's own frame for point, spot and area lights, area and
// spot lights without the face behind them; an orthographic map along a sun's direction), and per sample a shadow ray
// (or several: light.wgsl light_visibility) from a random point O of the light toward the shading point traced through
// that map in a few steps.
//
// A cube face is kept whole at a coarse level (frame.shadow_lod.w) and, where receivers need finer texels, in a window
// of the fine maps at their level (shadow_setup.wgsl places them): a receiver the window covers reads its level, at
// the centre of that level's texel, in the window; any other receiver reads the whole face, at its level or the
// face's when that is coarser, and takes the biases of the level it reads. A face's depth is reverse-Z with an
// infinite far plane: depth = near / (distance along the face's axis); the maps are cleared to 0 (nothing). render.rs
// draws them with the same axes (shadows.rs face_axes). Hashed casters (shadows.rs) are in front of a texel's opaque
// depth with the probability of their coverage, drawn per texel, map and sample, as EEVEE hashes them again every
// sample.

const SHADOW_NONE: f32 = 3.0e38;

// Whether a hashed caster stops a lookup at texel `px` of map `layer` (`which`: 0 coarse, 1 fine): the probability
// of the hashed casters' coverage there (`clear`: their transparency), drawn per texel, map and sample.
fn shadow_hashed_hit(px: vec2<i32>, layer: i32, which: u32, clear: f32) -> bool {
    let u = unit(hash3(bitcast<u32>(px.x) ^ (g_sample * 0x9e3779b9u),
                       bitcast<u32>(px.y) ^ ((u32(layer) * 2u + which) * 0x632be5abu), frame.frame_seed ^ 0x85ebca6bu));
    return u >= clear;
}

// The depth a lookup sees at texel `px` of a whole cube face `layer`: the opaque casters' (R), or the nearest hashed
// caster's in front of them (G) with the probability of their coverage there.
fn shadow_texel(layer: i32, px: vec2<i32>) -> f32 {
    let d = textureLoad(shadow_depths, px, layer, 0).xy;
    if (d.y > d.x && shadow_hashed_hit(px, layer, 0u, textureLoad(shadow_clear, px, layer, 0).x)) {
        return d.y;
    }
    return d.x;
}

// The same in the fine maps (a sun's map, a window).
fn shadow_texel_fine(layer: i32, px: vec2<i32>) -> f32 {
    let d = textureLoad(shadow_fine_depths, px, layer, 0).xy;
    if (d.y > d.x && shadow_hashed_hit(px, layer, 1u, textureLoad(shadow_fine_clear, px, layer, 0).x)) {
        return d.y;
    }
    return d.x;
}

// The stored depth of a sun's map (layer of the fine maps) at uv in [-1, 1] (nearest texel).
fn shadow_depth_at(layer: u32, uv: vec2<f32>, size: f32) -> f32 {
    let t = vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5) * size;
    let px = clamp(vec2<i32>(floor(t)), vec2<i32>(0), vec2<i32>(i32(size) - 1));
    return shadow_texel_fine(i32(layer), px);
}

// What a ray's lookups need of its receiver P: the level it reads (its own when a window serves it, else at least the
// whole face's), its face and finest-level texel, the window serving it (w < 0: none), and how far from P
// (finest-level texels) its steps read that window.
struct Recv {
    lod: f32,
    face: u32,
    t0: vec2<f32>,
    reach: f32,
    win: vec4<i32>,
}

// The window (fine layer) holding finest-level texel t0 at level `lod`, (texel, layer); layer < 0: outside.
fn shadow_window_texel(win: vec4<i32>, t0: vec2<f32>, lod: f32) -> vec3<i32> {
    let s = exp2(lod);
    let c0 = (floor(t0 / s) + 0.5) * s;     // the centre of the level's texel, finest-level texels
    let px = vec2<i32>(floor(c0 / exp2(f32(win.z)))) - win.xy;
    let size = i32(frame.shadow_pool.w);
    if (win.w < 0 || any(px < vec2<i32>(0)) || any(px >= vec2<i32>(size))) {
        return vec3<i32>(0, 0, -1);
    }
    return vec3<i32>(px, win.w);
}

// The stored depth of face `face` of a cube light at uv: within the receiver's reach on its face, in its window at
// its level; else from the whole face (at the receiver's level when that is coarser than the face's).
fn shadow_cube_depth(l: Light, rc: Recv, face: u32, uv: vec2<f32>) -> f32 {
    let t0 = shadow_texel0(uv);
    if (face == rc.face && all(abs(t0 - rc.t0) < vec2<f32>(rc.reach))) {
        let wt = shadow_window_texel(rc.win, t0, rc.lod);
        if (wt.z >= 0) {
            return shadow_texel_fine(wt.z, wt.xy);
        }
    }
    let base = frame.shadow_lod.w;
    let s = exp2(rc.lod);
    let q = select(t0, (floor(t0 / s) + 0.5) * s, rc.lod >= base);
    let size = i32(l.map.z);
    let px = clamp(vec2<i32>(floor(q / exp2(base))), vec2<i32>(0), vec2<i32>(size - 1));
    return shadow_texel(i32(u32(l.map.x) + face), px);
}

// Which of a sun's maps (SUN_CASCADES of them from l.map.x, finest first: shadows.rs) holds Q: the finest whose square
// it falls in (a texel inside the edge), else the last, over the whole scene. Returns the entry and Q in its clip space.
struct SunMap {
    entry: u32,
    c: vec4<f32>,
}

fn shadow_sun_map(l: Light, Q: vec3<f32>) -> SunMap {
    let first = u32(l.map.x);
    for (var k = 0u; k + 1u < SUN_CASCADES; k = k + 1u) {
        let c = shadow_faces[first + k].vp * vec4<f32>(Q, 1.0);
        if (all(abs(c.xy) < vec2<f32>(1.0 - 2.0 / l.map.z))) {
            return SunMap(first + k, c);
        }
    }
    let last = first + SUN_CASCADES - 1u;
    return SunMap(last, shadow_faces[last].vp * vec4<f32>(Q, 1.0));
}

// A sun's shadow ray from a finer map runs SUN_RAY times that map's width toward the sun (the scene's map: across its
// whole depth span), so its steps near the receiver fall as near as the receiver's map is fine, as EEVEE's receiver's
// clipmap level sets them.
const SUN_RAY: f32 = 8.0;

// A sun's receiver at P: the texel (m) of the map that holds P, and how far its shadow ray runs. The last map's texel
// is render.rs's fit (l.map.w), its ray the depth span (l.map.y); a finer map's texel from its matrix (its x row is
// the map's axis over its half width). One map per sun (SUN_CASCADES 1: no sun casts shadows) reads as it always did.
fn shadow_sun_texel(l: Light, P: vec3<f32>) -> f32 {
    if (SUN_CASCADES == 1u) {
        return l.map.w;
    }
    let m = shadow_sun_map(l, P);
    if (m.entry + 1u == u32(l.map.x) + SUN_CASCADES) {
        return l.map.w;
    }
    let vp = shadow_faces[m.entry].vp;
    return 2.0 / (l.map.z * length(vec3<f32>(vp[0].x, vp[1].x, vp[2].x)));
}

fn shadow_sun_ray(l: Light, P: vec3<f32>) -> f32 {
    if (SUN_CASCADES == 1u) {
        return l.map.y;
    }
    let m = shadow_sun_map(l, P);
    if (m.entry + 1u == u32(l.map.x) + SUN_CASCADES) {
        return l.map.y;
    }
    let vp = shadow_faces[m.entry].vp;
    return min(l.map.y, SUN_RAY * 2.0 / length(vec3<f32>(vp[0].x, vp[1].x, vp[2].x)));
}

// A sun's map at Q: x Q's distance from the map's near plane, y the nearest caster's along the same line (SHADOW_NONE:
// nothing). Orthographic faces: shadow_faces[layer] maps world to clip (x, y in [-1, 1], depth reverse linear), every
// one of a sun's maps over the same depth span.
fn shadow_sun_sample(l: Light, Q: vec3<f32>) -> vec2<f32> {
    var entry = u32(l.map.x);
    var c = vec4<f32>(0.0);
    if (SUN_CASCADES == 1u) {
        c = shadow_faces[entry].vp * vec4<f32>(Q, 1.0);
    } else {
        let m = shadow_sun_map(l, Q);
        entry = m.entry;
        c = m.c;
    }
    let stored = shadow_depth_at(entry - u32(frame.shadow_pool.x), c.xy, l.map.z);
    // reverse linear depth: 1 at the near plane, 0 at the far one; l.map.y holds the depth span (m)
    let r = (1.0 - c.z) * l.map.y;
    if (stored <= 0.0) {
        return vec2<f32>(r, SHADOW_NONE);
    }
    return vec2<f32>(r, (1.0 - stored) * l.map.y);
}

// A point at d from C in the light's frame (right, up, fwd) against its cube map (`rc`: the ray's receiver): x its
// distance from C, y the nearest caster's distance along the same line over x (SHADOW_NONE: nothing there).
fn shadow_cube_sample(l: Light, rc: Recv, d: vec3<f32>) -> vec2<f32> {
    let r = length(d);
    let f = shadow_face(d.x, d.y, d.z);
    if (f.face >= shadow_face_count(l)) {
        return vec2<f32>(r, SHADOW_NONE);
    }
    let stored = shadow_cube_depth(l, rc, f.face, f.uv);
    if (stored <= 0.0) {
        return vec2<f32>(r, SHADOW_NONE);
    }
    // the caster's distance along the face axis over the point's
    return vec2<f32>(r, l.map.y / (stored * max(f.a, 1e-30)));
}

// Where step i of a ray's `steps` sits (the last one, i = steps, is P' itself): far to near.
fn shadow_step(i: u32, steps: u32, jit: f32) -> f32 {
    if (i >= steps) {
        return 0.0;
    }
    return frame.trace.y * pow((f32(steps - 1u - i) + jit) / f32(steps), frame.shadow_map.x);
}

// Visibility of light `li` from P (geometric normal Ng): EEVEE Next's shadow map tracing as measured on probes
// (subagent ShadowTrace; clean-room, from EEVEE's frames at 64 samples). Every scale follows P's texel radius r
// (shadow_lod.wgsl): the ray runs from P' (P lifted along Ng by r sin(theta) (frame.bias.x + F frame.bias.w), theta
// between Ng and the direction to C, F the light's filter radius: EEVEE's filter shifts a tilted receiver's shadow
// toward the light by about 0.7 to 0.85 F r sin(theta) on its probes, then jittered over a disk of F r x
// frame.shadow_map.z square to that direction) toward a random point O of the light (above P's horizon), sampled at
// t_k = len ((k + u) / S)^p of the way (k < S = frame.trace.x steps, one u per ray, len = frame.trace.y, p =
// frame.shadow_map.x) and at P' itself, far to near, every lookup at
// P's level. In the plane of the ray and the map's origin C (u along the ray, v the distance from it toward C,
// metres), each sample's lookup gives an occluder point X = C + D dir; v > 0 is a sample behind the map's surface (by
// more than the depth tolerance r x frame.bias.y), and one with fewer than two samples before it is a hit. After that
// the two rules part (frame.shadow_map.y):
// - interim (the default): a behind sample is a hit when the line through the two previous occluder points (A, B)
//   crosses the ray between this sample and the previous one; lines closer to parallel than 1e-12 m never cross. Where A
//   saw nothing, that crossing is computed far off its limit (f32 cancellation against A at 1e6) and lands in the window
//   for some rays only: measured, that matches EEVEE's figures against an empty background better than any exact rule
//   tried (never, always, or the limit's line through B toward A's line).
// - v2 (ShadowTrace's probe fit): after a behind sample, the line through its occluder point and this one's; after a
//   sample in front, the line through A and B, or where A or B saw nothing, the line through this sample's point
//   parallel to B - A; each must cross the ray between this sample and the previous one.
// A sun's map lines all run toward it: its points keep v in units of the sun direction's offset from the ray; its
// texel and its ray's length are those of its map around P (shadow_sun_texel, shadow_sun_ray), at no level.
fn shadow_visibility(l: Light, li: u32, P: vec3<f32>, Ng: vec3<f32>) -> f32 {
    if (l.spot.z < 0.5 || frame.shadow.w < 0.5 || l.map.x < 0.0) {
        return 1.0;
    }
    let sun = u32(l.pos.w) == KIND_SUN;
    let v2 = frame.shadow_map.y > 0.5;
    let fp = shadow_footprint(P);
    var rc = Recv(0.0, 0u, vec2<f32>(0.0), 0.0, vec4<i32>(0, 0, -1, -1));
    var r = 0.0;
    if (sun) {
        r = shadow_radius(shadow_sun_texel(l, P), fp, 0.0);
    } else {
        let f = shadow_face_of(l, P - l.pos.xyz);
        var lod = shadow_lod(fp, f.a, l.map.w);
        rc.face = f.face;
        rc.t0 = shadow_texel0(f.uv);
        var served = false;
        if (f.face < shadow_face_count(l)) {
            // a window at the receiver's level or finer, covering its texel, serves it
            let w = shadow_windows[u32(l.map.x) + f.face];
            if (w.w >= 0 && lod >= f32(w.z) && shadow_window_texel(w, rc.t0, lod).z >= 0) {
                rc.win = w;
                served = true;
            }
        }
        if (!served) {
            lod = max(lod, frame.shadow_lod.w);     // the whole face's texels, and their biases
        }
        rc.lod = lod;
        rc.reach = frame.shadow_map.w * exp2(lod);
        r = shadow_radius(2.0 * f.a / SHADOW_LOD0 * exp2(lod), fp, l.map.w);
    }
    let toward = normalize(select(l.pos.xyz - P, -l.fwd.xyz, sun));
    let cos_t = clamp(dot(Ng, toward), -1.0, 1.0);
    let filt = l.factor.w;
    var E = P + Ng * (r * sqrt(1.0 - cos_t * cos_t) * (frame.bias.x + filt * frame.bias.w));
    if (filt > 0.0) {
        // the end jittered over a disk of the filter radius, square to the direction toward the light
        let tx = normalize(select(cross(toward, vec3<f32>(0.0, 1.0, 0.0)), cross(toward, vec3<f32>(1.0, 0.0, 0.0)),
                                  abs(toward.y) > 0.9));
        let ty = cross(toward, tx);
        let j = shadow_disk_u(li);
        E = E + (tx * cos(MKS_TAU * j.y) + ty * sin(MKS_TAU * j.y)) * (sqrt(j.x) * filt * frame.shadow_map.z * r);
    }
    var O = light_point_above(l, li, P, Ng);
    if (sun) {
        // a sun's ray ends where its map does: across the depth span, or SUN_RAY widths of a finer map
        O = E + normalize(O - E) * shadow_sun_ray(l, P);
    }
    let ray = O - E;
    let len_ray = length(ray);
    if (len_ray <= 0.0) {
        return 1.0;
    }
    let e = ray / len_ray;
    // C's place along the ray and its distance from it (suns: the map's lines run toward the sun)
    let u_c = dot(l.pos.xyz - E, e);
    let v_c = length(l.pos.xyz - E - e * u_c);
    let sun_u = dot(-l.fwd.xyz, e);
    let tol = r * frame.bias.y;
    let far = select(1e6, 1e30, v2);
    let steps = max(u32(frame.trace.x), 1u);
    let jit = shadow_steps_u(li);
    // cube maps: the ray in the light's frame, so each sample is one multiply-add away
    let to_light = transpose(mat3x3<f32>(l.right.xyz, l.up.xyz, l.fwd.xyz));
    let E_l = to_light * (E - l.pos.xyz);
    let ray_l = to_light * ray;
    var a = vec2<f32>(0.0);     // the occluder points (u, v) of the sample before the previous one, and of the previous
    var b = vec2<f32>(0.0);
    var a_empty = false;
    var b_empty = false;
    var prev_behind = false;
    var u_prev = 3.0e38;
    for (var i = 0u; i <= steps; i = i + 1u) {
        let t = shadow_step(i, steps, jit);
        let u_k = t * len_ray;
        // the occluder point of this sample's line (nothing there: far beyond, on the other side of the ray), and
        // whether the sample lies behind the map's surface
        var x: vec2<f32>;
        var empty: bool;
        var behind: bool;
        if (sun) {
            let s = shadow_sun_sample(l, E + ray * t);
            empty = s.y >= SHADOW_NONE * 0.5;
            behind = !empty && s.y < s.x - tol;
            let back = select(clamp(s.x - s.y, -far, far), -far, empty);
            x = vec2<f32>(u_k + sun_u * back, back);
        } else {
            let s = shadow_cube_sample(l, rc, E_l + ray_l * t);
            empty = s.y >= SHADOW_NONE * 0.5;
            behind = !empty && s.y * s.x < s.x - tol;
            let f = select(min(s.y, far), far, empty);
            x = vec2<f32>(u_c + f * (u_k - u_c), v_c * (1.0 - f));
        }
        if (behind) {
            if (i < 2u) {
                return 0.0;
            }
            if (v2) {
                var cross_u = a.x + (b.x - a.x) * a.y / (a.y - b.y);
                if (prev_behind) {
                    cross_u = b.x + (x.x - b.x) * b.y / (b.y - x.y);
                } else if (a_empty || b_empty) {
                    cross_u = x.x - x.y * (b.x - a.x) / (b.y - a.y);
                }
                if (cross_u >= u_k && cross_u <= u_prev) {
                    return 0.0;
                }
            } else {
                let dv = b.y - a.y;
                if (abs(dv) > 1e-12) {
                    // this order of operations, exactly: see the interim rule above
                    let u_star = a.x + (b.x - a.x) * (-a.y / dv);
                    if (u_star >= min(u_k, u_prev) && u_star <= max(u_k, u_prev)) {
                        return 0.0;
                    }
                }
            }
        }
        a = b;
        a_empty = b_empty;
        b = x;
        b_empty = empty;
        prev_behind = behind;
        u_prev = u_k;
    }
    return 1.0;
}
