// The scene's lights on a point's closures, as EEVEE Next lights them: per light the unshadowed analytic integral of
// each lobe over the light's shape (Lambert polygon integral for diffuse and translucent, GGX through linearly
// transformed cosines for reflection and coat: ltc.wgsl), times its visibility (shadow_map.wgsl: EEVEE's tracing
// through shadow maps; or shadow_rays.wgsl: ray queries toward a random point of the light), which is 1 on a surface
// whose material receives no shadow (g_unshadowed). The deferred lighting pass (deferred.wgsl) and the forward entries
// (blended materials, Shader to RGB) share it.

const VIS_DIFFUSE: u32 = 1u;
const VIS_GLOSSY: u32 = 2u;
const VIS_TRANSMISSION: u32 = 4u;

const IDENTITY3: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));

// A point's closures ready for lighting, in glTF's axes.
struct Shading {
    P: vec3<f32>,
    V: vec3<f32>,
    Ng: vec3<f32>,
    diffuse: vec3<f32>,
    diffuse_N: vec3<f32>,
    translucent: vec3<f32>,
    translucent_N: vec3<f32>,          // the side translucent light comes from (SLAB)
    translucent_slab: f32,             // 1: thickness mode SLAB (Lambert around translucent_N), 0: SPHERE (all sides)
    reflection: vec3<f32>,
    reflection_N: vec3<f32>,
    reflection_minv: mat3x3<f32>,
    coat: vec3<f32>,
    coat_N: vec3<f32>,
    coat_minv: mat3x3<f32>,
}

fn lobe_minv(N: vec3<f32>, V: vec3<f32>, roughness: f32) -> mat3x3<f32> {
    let c = textureSampleLevel(ltc_tex, s_lut, ltc_matrix_coords(roughness, clamp(dot(N, V), 0.0, 1.0)), 0.0);
    return ltc_minv(c);
}

// The lobes' LTC matrices once the colours, normals and roughnesses are in.
fn shading_finish(h_in: Shading, reflection_roughness: f32, coat_roughness: f32) -> Shading {
    var h = h_in;
    h.diffuse = h.diffuse * frame.scale.x;
    h.reflection = h.reflection * frame.scale.y;
    h.coat = h.coat * frame.scale.y;
    h.reflection_minv = IDENTITY3;
    h.coat_minv = IDENTITY3;
    if (any(h.reflection > vec3<f32>(0.0))) {
        h.reflection_minv = lobe_minv(h.reflection_N, h.V, reflection_roughness);
    }
    if (any(h.coat > vec3<f32>(0.0))) {
        h.coat_minv = lobe_minv(h.coat_N, h.V, coat_roughness);
    }
    return h;
}

// Forward: a material's closures at its surface.
fn shading_of(c: Closures, s: Surface) -> Shading {
    var h: Shading;
    h.P = to_gltf(s.P);
    h.V = to_gltf(s.I);
    h.Ng = to_gltf(s.Ng);
    h.diffuse = c.diffuse;
    h.diffuse_N = to_gltf(c.diffuse_N);
    h.translucent = c.translucent;
    h.translucent_N = -to_gltf(c.translucent_N);
    h.translucent_slab = c.translucent_slab;
    h.reflection = c.reflection;
    h.reflection_N = to_gltf(c.reflection_N);
    h.coat = c.coat;
    h.coat_N = to_gltf(c.coat_N);
    return shading_finish(h, c.reflection_roughness, c.coat_roughness);
}

// EEVEE fades a light out toward the distance where it falls under the scene's light threshold.
fn influence(l: Light, P: vec3<f32>) -> f32 {
    if (l.color.w <= 0.0 || u32(l.pos.w) == KIND_SUN) {
        return 1.0;
    }
    let d = P - l.pos.xyz;
    let f = dot(d, d) * l.color.w;
    let w = clamp(1.0 - f * f, 0.0, 1.0);
    return w * w;
}

// Whether light `l` may light some point of the box lo..hi (world, glTF axes): false only where light_lobes leaves every
// point of it unlit (no colour or no visibility, past the influence radius, behind a one-sided area light, outside a
// spot's cone), each test with a margin over its rounding so that a light that lights a point is never dropped. The
// deferred pass (deferred.wgsl) culls each tile's lights with it.
fn light_reaches_box(l: Light, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let kind = u32(l.pos.w);
    if (all(l.color.rgb == vec3<f32>(0.0)) || u32(l.spot.w) == 0u) {
        return false;
    }
    if (kind == KIND_SUN) {
        return true;
    }
    let c = l.pos.xyz;
    // the box's point nearest the light, past the radius where the influence ends
    let d = clamp(c, lo, hi) - c;
    if (l.color.w > 0.0 && dot(d, d) * l.color.w > 1.0001) {
        return false;
    }
    if (kind == KIND_RECT || kind == KIND_ELLIPSE) {
        // the box's point farthest along the way the light shines, behind it
        let e = select(lo, hi, l.fwd.xyz > vec3<f32>(0.0)) - c;
        let a = abs(e);
        if (dot(e, l.fwd.xyz) < -(1e-5 * (a.x + a.y + a.z) + 1e-6)) {
            return false;
        }
    }
    if (kind == KIND_SPOT) {
        // the box's bounding sphere wholly outside the cone: the centre farther from the cone's side (as a line through
        // the apex, which is never farther than the cone itself up to a half angle of 90 degrees) than the radius
        let v = 0.5 * (lo + hi) - c;
        let along = dot(v, l.fwd.xyz);
        let across = length(v - l.fwd.xyz * along);
        let s = sqrt(max(1.0 - l.spot.x * l.spot.x, 0.0));
        let r = 0.5 * length(hi - lo);
        if (across * l.spot.x - along * s > r * 1.001 + 1e-4 * length(v) + 1e-5) {
            return false;
        }
    }
    return true;
}

// The corners of a rectangle light.
fn rect_corners(l: Light) -> array<vec3<f32>, 4> {
    let c = l.pos.xyz;
    let x = l.right.xyz * l.right.w;
    let y = l.up.xyz * l.up.w;
    return array<vec3<f32>, 4>(c - x - y, c + x - y, c + x + y, c - x + y);
}

// Disk lights: centre, the two half axes, all in world space, facing P.
struct Disk {
    c: vec3<f32>,
    x: vec3<f32>,
    y: vec3<f32>,
}

fn disk_of(l: Light, P: vec3<f32>) -> Disk {
    let kind = u32(l.pos.w);
    if (kind == KIND_ELLIPSE) {
        return Disk(l.pos.xyz, l.right.xyz * l.right.w, l.up.xyz * l.up.w);
    }
    if (kind == KIND_SUN) {
        // a disk one metre away, as wide as the sun's angle
        let r = tan(max(l.fwd.w, 1e-3));
        let w = -l.fwd.xyz;
        let t = normalize(select(cross(w, vec3<f32>(0.0, 1.0, 0.0)), cross(w, vec3<f32>(1.0, 0.0, 0.0)), abs(w.y) > 0.9));
        return Disk(P + w, t * r, cross(w, t) * r);
    }
    // a sphere as EEVEE lights with it: a disk as wide as the sphere through its centre, facing P (on its axis that is
    // P / (4 pi (d² + r²)), not the sphere's P / (4 pi d²): lab point_near, within 1 % from 2 radii away)
    let to = l.pos.xyz - P;
    let d = length(to);
    let w = to / d;
    let r = max(l.fwd.w, 1e-4 * d);
    let t = normalize(select(cross(w, vec3<f32>(0.0, 1.0, 0.0)), cross(w, vec3<f32>(1.0, 0.0, 0.0)), abs(w.y) > 0.9));
    return Disk(l.pos.xyz, t * r, cross(w, t) * r);
}

fn spot_mask(l: Light, P: vec3<f32>) -> f32 {
    let dir = normalize(P - l.pos.xyz);
    let cosine = dot(dir, l.fwd.xyz);
    let smooth_band = (1.0 - l.spot.x) * l.spot.y;
    if (smooth_band <= 0.0) {
        return select(0.0, 1.0, cosine > l.spot.x);
    }
    return smoothstep(0.0, 1.0, (cosine - l.spot.x) / smooth_band);
}

// The lobe integrals of one light over one shading point (no shadow); `sphere`: the sphere approximation of the
// horizon (ltc.wgsl) instead of clipping to N's hemisphere.
fn integrate(l: Light, h: Shading, Nl: vec3<f32>, minv: mat3x3<f32>, sphere: bool) -> f32 {
    let kind = u32(l.pos.w);
    if (kind == KIND_RECT) {
        if (sphere) {
            return ltc_quad_sphere(Nl, h.V, h.P, minv, rect_corners(l));
        }
        return ltc_quad(Nl, h.V, h.P, minv, rect_corners(l));
    }
    let d = disk_of(l, h.P);
    if (sphere) {
        return ltc_disk_sphere(Nl, h.V, h.P, minv, d.c, d.x, d.y);
    }
    return ltc_disk(Nl, h.V, h.P, minv, d.c, d.x, d.y);
}

// How much of light `l` EEVEE lets into a glossy lobe of normal N at P (ltc_horizon_fade): all of it until the sphere
// bounding the light (an area light's half diagonal, a sphere light's radius, the sun's disk 1 m away) sinks under the
// lobe's horizon, then none within a tenth in sine. Part of EEVEE's glossy model (frame.bias.z: params ltc_glossy
// "eevee"); "mkr" keeps mkr's own model as it was.
fn glossy_horizon_fade(l: Light, P: vec3<f32>, N: vec3<f32>) -> f32 {
    let kind = u32(l.pos.w);
    if (kind == KIND_SUN) {
        return ltc_horizon_fade(N, -l.fwd.xyz, tan(max(l.fwd.w, 1e-3)));
    }
    let to = l.pos.xyz - P;
    let d = length(to);
    let area = kind == KIND_RECT || kind == KIND_ELLIPSE;
    let radius = select(l.fwd.w, length(vec2<f32>(l.right.w, l.up.w)), area);
    return ltc_horizon_fade(N, to / d, radius / d);
}

// What one light leaves on a point before shadows: the lobes lit on the surface's own side (diffuse, reflection, coat)
// and its translucency (lit through the surface when the light is behind it).
struct Lit {
    front: vec3<f32>,
    translucent: vec3<f32>,
}

fn light_lobes(l: Light, h: Shading) -> Lit {
    var o = Lit(vec3<f32>(0.0), vec3<f32>(0.0));
    let kind = u32(l.pos.w);
    var w = influence(l, h.P);
    if (w <= 0.0) {
        return o;
    }
    if ((kind == KIND_RECT || kind == KIND_ELLIPSE) && dot(h.P - l.pos.xyz, l.fwd.xyz) <= 0.0) {
        return o;   // behind a one-sided area light
    }
    if (kind == KIND_SPOT) {
        w = w * spot_mask(l, h.P);
        if (w <= 0.0) {
            return o;
        }
    }
    if (kind == KIND_POINT || kind == KIND_SPOT) {
        // a sphere integrated as a disk of another radius (disk_of keeps it at least 1e-4 of the distance across) than
        // its radiance was given for (render.rs: its radius, at least 1e-4 m): the radiance spread over that disk, so
        // that the light falls off with the square of the distance (the ratio is 1 when the two agree)
        let r_rad = max(l.fwd.w, 1e-4);
        let r_disk = max(l.fwd.w, 1e-4 * length(l.pos.xyz - h.P));
        w = w * (r_rad / r_disk) * (r_rad / r_disk);
    }
    let vis = u32(l.spot.w);
    let scale = l.color.rgb * w;
    let sphere_diffuse = frame.trace.z > 0.5;
    let sphere_glossy = frame.trace.w > 0.5;
    if ((vis & VIS_DIFFUSE) != 0u && any(h.diffuse > vec3<f32>(0.0))) {
        o.front = o.front + h.diffuse * (integrate(l, h, h.diffuse_N, IDENTITY3, sphere_diffuse) * l.factor.x);
    }
    if ((vis & VIS_TRANSMISSION) != 0u && any(h.translucent > vec3<f32>(0.0))) {
        // EEVEE (measured) lights translucency with the closure colour squared (Mix weights included); a sphere's
        // (thickness mode SPHERE) from every side alike: the Lambert integral of a surface facing the light, over pi;
        // a slab's as Lambert from behind
        var t = 0.0;
        if (h.translucent_slab > 0.5) {
            t = integrate(l, h, h.translucent_N, IDENTITY3, sphere_diffuse);
        } else {
            let toward = select(normalize(l.pos.xyz - h.P), -l.fwd.xyz, kind == KIND_SUN);
            t = integrate(l, h, toward, IDENTITY3, sphere_diffuse) / MKS_PI;
        }
        o.translucent = h.translucent * h.translucent * (t * l.factor.z * frame.scale.x) * scale;
    }
    if ((vis & VIS_GLOSSY) != 0u) {
        let eevee = frame.bias.z > 0.5;
        if (any(h.reflection > vec3<f32>(0.0))) {
            var fade = 1.0;
            if (eevee) {
                fade = glossy_horizon_fade(l, h.P, h.reflection_N);
            }
            if (fade > 0.0) {
                let r = integrate(l, h, h.reflection_N, h.reflection_minv, sphere_glossy) * fade;
                o.front = o.front + h.reflection * (r * l.factor.y);
            }
        }
        if (any(h.coat > vec3<f32>(0.0))) {
            var fade = 1.0;
            if (eevee) {
                fade = glossy_horizon_fade(l, h.P, h.coat_N);
            }
            if (fade > 0.0) {
                let c = integrate(l, h, h.coat_N, h.coat_minv, sphere_glossy) * fade;
                o.front = o.front + h.coat * (c * l.factor.y);
            }
        }
    }
    o.front = o.front * scale;
    return o;
}

fn light_unshadowed(l: Light, h: Shading) -> vec3<f32> {
    let o = light_lobes(l, h);
    return o.front + o.translucent;
}

// A point of the light's shape from two uniform numbers (rects uniformly; disks, spheres' silhouettes and the sun's
// disk through the concentric mapping), scaled by the shadow spread.
fn light_point(l: Light, u: vec2<f32>, P: vec3<f32>) -> vec3<f32> {
    let kind = u32(l.pos.w);
    let spread = frame.shadow.z;
    if (kind == KIND_RECT) {
        return l.pos.xyz + l.right.xyz * (l.right.w * spread * (2.0 * u.x - 1.0))
             + l.up.xyz * (l.up.w * spread * (2.0 * u.y - 1.0));
    }
    let a = 2.0 * u - 1.0;
    var r = 0.0;
    var phi = 0.0;
    if (a.x != 0.0 || a.y != 0.0) {
        if (abs(a.x) > abs(a.y)) {
            r = a.x;
            phi = (MKS_PI / 4.0) * (a.y / a.x);
        } else {
            r = a.y;
            phi = (MKS_PI / 2.0) - (MKS_PI / 4.0) * (a.x / a.y);
        }
    }
    let q = vec2<f32>(r * cos(phi), r * sin(phi)) * spread;
    let d = disk_of(l, P);
    if (kind == KIND_SUN) {
        // far along the direction through the disk point
        return P + normalize(d.c + d.x * q.x + d.y * q.y - P) * 1e5;
    }
    return d.c + d.x * q.x + d.y * q.y;
}

// ------------------------------------------------------------------------------------------ a shadow ray's numbers
// The random numbers of a shadow ray toward light li (render.rs `sampling`): frame.sampling.x rays per sample and
// light, ray r of sample s drawing point s K + r of the sequences. By default R2 for the light's point and for the
// filter disk's, R1 for the steps' jitter, each rotated per pixel (sample_rand2, sample_rand); with frame.sampling.y,
// one point of Sobol's sequence in five dimensions per ray (shadow_draw), its index shuffled among the frame's rays and
// its numbers Owen-scrambled per pixel and light: both pairs and the jitter stratified apart (R2's two pairs are one
// sequence turned two ways), less noise for the same rays.
var<private> g_ray: vec4<f32>;
var<private> g_ray_jit: f32;

fn shadow_draw(li: u32) {
    if (frame.sampling.y < 0.5) {
        return;
    }
    let seed = hash3(g_pixel.x, g_pixel.y, (li * 0x632be5abu) ^ frame.frame_seed ^ 0x51ed270bu);
    // within the aligned block of a power of two holding the frame's rays: their points a Sobol net
    let rays = max(frame.sample_count * u32(frame.sampling.x), 2u);
    let x = sobol5(owen(g_sample, seed) & ((1u << (32u - countLeadingZeros(rays - 1u))) - 1u));
    var q: array<f32, 5>;
    for (var d = 0u; d < 5u; d = d + 1u) {
        q[d] = unit(owen(x[d], hash_u32(seed ^ ((d + 1u) * 0x68bc21ebu))));
    }
    g_ray = vec4<f32>(q[0], q[1], q[2], q[3]);
    g_ray_jit = q[4];
}

fn shadow_light_u(li: u32) -> vec2<f32> {
    if (frame.sampling.y > 0.5) {
        return g_ray.xy;
    }
    return sample_rand2(32u + li);
}

fn shadow_disk_u(li: u32) -> vec2<f32> {
    if (frame.sampling.y > 0.5) {
        return g_ray.zw;
    }
    return sample_rand2(48u + li);
}

fn shadow_steps_u(li: u32) -> f32 {
    if (frame.sampling.y > 0.5) {
        return g_ray_jit;
    }
    return sample_rand(56u + li);
}

// A point of the light above the horizon of the surface (normal Ng) at P: the part below lights nothing (the lobe
// integrals clip it), so a shadow ray toward it would only darken the terminator twice. A few draws, then the last
// one lifted onto the horizon.
fn light_point_above(l: Light, li: u32, P: vec3<f32>, Ng: vec3<f32>) -> vec3<f32> {
    var u = shadow_light_u(li);
    var q = light_point(l, u, P);
    for (var k = 0u; k < 3u && dot(q - P, Ng) <= 0.0; k = k + 1u) {
        u = fract(u + vec2<f32>(unit(hash3(g_pixel.x, g_pixel.y, li * 7u + k)), unit(hash3(g_pixel.y, k, li * 13u + g_sample))));
        q = light_point(l, u, P);
    }
    let below = dot(q - P, Ng);
    if (below <= 0.0) {
        q = q - Ng * (below - 1e-4 * length(q - P));
    }
    return q;
}

// Light li's visibility from P (shadow_visibility: the maps' tracing, or ray queries), the mean of its rays: Ng the
// geometric normal on the side it lights, N the shading normal there. None of its shadow falls on a surface whose
// material receives none (g_unshadowed).
fn light_visibility(l: Light, li: u32, P: vec3<f32>, Ng: vec3<f32>, N: vec3<f32>) -> f32 {
    if (l.spot.z < 0.5 || frame.shadow.w < 0.5 || g_unshadowed) {
        return 1.0;
    }
    let k = u32(frame.sampling.x);
    let s = g_sample;
    var v = 0.0;
    for (var r = 0u; r < k; r = r + 1u) {
        g_sample = s * k + r;
        shadow_draw(li);
        v = v + shadow_visibility(l, li, P, Ng, N);
    }
    g_sample = s;
    return v / f32(k);
}

// ------------------------------------------------------------------------------------------------- the lit colour
// `out` (the lit colour so far) with light `i`'s share added: its lobes, shadowed. Translucency from a light behind the
// surface is shadowed from that side (a sheet does not shadow its own back); the rest from the surface's own side.
fn light_add(out: vec3<f32>, i: u32, h: Shading) -> vec3<f32> {
    let l = lights[i];
    let o = light_lobes(l, h);
    let front = any(o.front > vec3<f32>(0.0));
    let through = any(o.translucent > vec3<f32>(0.0));
    if (!front && !through) {
        return out;
    }
    let toward = select(l.pos.xyz - h.P, -l.fwd.xyz, u32(l.pos.w) == KIND_SUN);
    let behind = dot(toward, h.Ng) < 0.0;
    var v_front = 1.0;
    if (front || (through && !behind)) {
        // EEVEE lifts its shadow rays along the shading normal (the G-buffer's, kept on the surface's side)
        let n = h.diffuse_N;
        let N = select(h.Ng, normalize(n), dot(n, n) > 0.25 && dot(n, h.Ng) > 0.0);
        v_front = light_visibility(l, i, h.P, h.Ng, N);
    }
    var v_back = v_front;
    if (through && behind) {
        v_back = light_visibility(l, i, h.P, -h.Ng, -h.Ng);
    }
    return out + o.front * v_front + o.translucent * v_back;
}

// Whether any lobe a light lights (light_lobes) is there: emission and transparency alone take no light.
fn takes_light(h: Shading) -> bool {
    return any(h.diffuse > vec3<f32>(0.0)) || any(h.translucent > vec3<f32>(0.0)) || any(h.reflection > vec3<f32>(0.0))
        || any(h.coat > vec3<f32>(0.0));
}

fn light_shading(h: Shading) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    if (!takes_light(h)) {
        return out;
    }
    for (var i = 0u; i < frame.light_count; i = i + 1u) {
        out = light_add(out, i, h);
    }
    return out;
}

fn light_shading_unshadowed(h: Shading) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    for (var i = 0u; i < frame.light_count; i = i + 1u) {
        out = out + light_unshadowed(lights[i], h);
    }
    return out;
}

fn light_closures(c: Closures, s: Surface) -> vec3<f32> {
    let h = shading_of(c, s);
    var o = c.emission + light_shading(h);
    if (frame.world.x > 0.5) {
        o = o + world_light(h, c.reflection_roughness, c.coat_roughness);
    }
    return o;
}

// What a DITHERED surface kept by the prepass shows: its closures over its own coverage (the prepass already dropped it
// with the probability of its transparency), lit.
fn shade_dithered(c: Closures, s: Surface) -> vec3<f32> {
    let a = mks_opacity(c);
    var k = c;
    if (a > 0.0 && a < 1.0) {
        k = mks_closures_scale(c, 1.0 / a);
    }
    return light_closures(k, s);
}
