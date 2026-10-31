// EEVEE Next's ray tracing (scene.eevee.use_raytracing, ray_tracing_method SCREEN; raytrace.rs), as the lab's rt_* and
// gi_* cases show it: a glossy lobe (reflection or coat) up to trace_max_roughness sends one ray per sample, drawn
// from its GGX distribution of visible normals and marched through the sample's depth buffer. A ray that finds a
// surface takes the light the previous sample left there (radiance_in: the lit opaque surfaces, reflections included,
// so light bounces from sample to sample as EEVEE's radiance feedback does); one that leaves the screen, goes behind
// everything or passes farther than the trace's thickness behind a surface takes the world's probe (world_light.wgsl).
// The film averages the samples' rays after EEVEE's temporal accumulation (rt_temporal). Fast GI (rt_scan): EEVEE's
// horizon scan of the depth buffer hides the world's light from diffuse closures (and, GLOBAL_ILLUMINATION, lights
// them with what it finds); rougher glossy lobes fade from their traced ray into it. Only the deferred lighting pass
// (deferred.wgsl) traces: blended surfaces are lit forward with the probe, as EEVEE lights them.

@group(1) @binding(5) var radiance_in: texture_2d<f32>;
@group(1) @binding(6) var radiance_out: texture_storage_2d<rgba16float, write>;
// the traced lobes' light after rt_temporal and the raw light's luminance: the previous sample's, and this one's
@group(1) @binding(7) var history_in: texture_2d<f32>;
@group(1) @binding(8) var history_out: texture_storage_2d<rgba16float, write>;

// Distance along the view axis of the point at reversed depth d (frame.rt2: near and far planes).
fn rt_distance(d: f32) -> f32 {
    let n = frame.rt2.z;
    let f = frame.rt2.w;
    if (frame.cam_pos.w > 0.5) {
        return f - d * (f - n);
    }
    return n * f / max(d * (f - n) + n, 1e-20);
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals": a microfacet normal seen from Ve (the lobe's frame,
// z the normal) for GGX alpha `a`.
fn rt_visible_normal(Ve: vec3<f32>, a: f32, u: vec2<f32>) -> vec3<f32> {
    let Vh = normalize(vec3<f32>(a * Ve.x, a * Ve.y, Ve.z));
    let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    var T1 = vec3<f32>(1.0, 0.0, 0.0);
    if (lensq > 0.0) {
        T1 = vec3<f32>(-Vh.y, Vh.x, 0.0) / sqrt(lensq);
    }
    let T2 = cross(Vh, T1);
    let r = sqrt(u.x);
    let phi = MKS_TAU * u.y;
    let t1 = r * cos(phi);
    let s = 0.5 * (1.0 + Vh.z);
    let t2 = (1.0 - s) * sqrt(max(1.0 - t1 * t1, 0.0)) + s * r * sin(phi);
    let Nh = t1 * T1 + t2 * T2 + sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0)) * Vh;
    return normalize(vec3<f32>(a * Nh.x, a * Nh.y, max(Nh.z, 1e-6)));
}

// The pixel where a ray from P along L first passes behind a surface of the sample's depth buffer, a surface being
// `thickness` deep (metres along the view axis); x -1 when it finds none on the screen. `steps` steps, denser near P, a
// random offset per pixel and sample, then a bisection. A ray going away from the camera hits a surface it crossed since
// the last step, with frame.sampling.w (trace_cross) of the thickness to spare past it; one coming back toward the
// camera, a surface it is less than the thickness behind (lab rt_emit: EEVEE draws the same reflections at thicknesses
// 0.05 and 0.2; this finds the 20 and 60 cm cards' light within 1.5 % of its undenoised frames up to roughness 0.2,
// where the stretch overlapping the whole thickness found 11-15 % more, from rays passing under and behind the cards).
fn rt_trace(P: vec3<f32>, L: vec3<f32>, vp: mat4x4<f32>, jitter: f32, steps: u32, thickness: f32) -> vec2<i32> {
    let size = frame.size.xy;
    // how far the ray may go: to the far plane, or to the near plane when it comes back toward the camera
    var len = frame.rt2.w;
    let along = dot(L, frame.cam_fwd.xyz);
    if (along < 0.0) {
        let ahead = dot(P - frame.cam_pos.xyz, frame.cam_fwd.xyz) - frame.rt2.z * 1.5;
        len = min(len, max(ahead, 0.0) / -along);
    }
    let cs = vp * vec4<f32>(P, 1.0);
    let ce = vp * vec4<f32>(P + L * len, 1.0);
    if (cs.w <= 0.0 || ce.w <= 0.0) {
        return vec2<i32>(-1);
    }
    let ns = cs.xyz / cs.w;
    let ne = ce.xyz / ce.w;
    let ps = vec2<f32>((ns.x * 0.5 + 0.5) * size.x, (0.5 - ns.y * 0.5) * size.y);
    let pe = vec2<f32>((ne.x * 0.5 + 0.5) * size.x, (0.5 - ne.y * 0.5) * size.y);
    let d = pe - ps;
    // the segment clipped to the picture (z/w and the screen position are both affine along it)
    var t_end = 1.0;
    if (d.x > 0.0) { t_end = min(t_end, (size.x - ps.x) / d.x); }
    if (d.x < 0.0) { t_end = min(t_end, -ps.x / d.x); }
    if (d.y > 0.0) { t_end = min(t_end, (size.y - ps.y) / d.y); }
    if (d.y < 0.0) { t_end = min(t_end, -ps.y / d.y); }
    let pixels = length(d) * t_end;
    if (pixels < 1.5) {
        return vec2<i32>(-1);
    }
    let n = max(steps, 1u);
    let t_min = 1.5 / length(d);
    var t_prev = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let s = (f32(i) + jitter) / f32(n);
        let t = max(t_end * s * s, t_min);
        let p = ps + d * t;
        let scene = textureLoad(gb_depth, vec2<i32>(floor(p)), 0);
        let dr = mix(ns.z, ne.z, t);
        // behind the surface now, and less than the slack behind it: a ray going away takes its stretch since the last
        // step on top of a share of the thickness
        let dcur = rt_distance(dr);
        let dstep = dcur - rt_distance(mix(ns.z, ne.z, t_prev));
        if (dr < scene) {
            if (dcur - rt_distance(scene) < select(thickness, frame.sampling.w * thickness + dstep, dstep > 0.0)) {
                var lo = t_prev;
                var hi = t;
                for (var k = 0u; k < 5u; k = k + 1u) {
                    let mid = 0.5 * (lo + hi);
                    let q = ps + d * mid;
                    if (mix(ns.z, ne.z, mid) < textureLoad(gb_depth, vec2<i32>(floor(q)), 0)) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let q = vec2<i32>(floor(ps + d * hi));
                // a bisection that walked off what it found (a thin edge): the step's own pixel
                if (rt_distance(mix(ns.z, ne.z, hi)) - rt_distance(textureLoad(gb_depth, q, 0)) < thickness) {
                    return q;
                }
                return vec2<i32>(floor(p));
            }
        }
        t_prev = t;
    }
    return vec2<i32>(-1);
}

// A frame (tangent, bitangent) around unit N (Duff et al. 2017).
fn rt_basis(N: vec3<f32>) -> mat2x3<f32> {
    let sgn = select(-1.0, 1.0, N.z >= 0.0);
    let a0 = -1.0 / (sgn + N.z);
    let b0 = N.x * N.y * a0;
    return mat2x3<f32>(vec3<f32>(1.0 + sgn * N.x * N.x * a0, sgn * b0, -sgn * N.x), vec3<f32>(b0, sgn + N.y * N.y * a0, -N.y));
}

// Where a ray off the surface at P starts: a hair along the geometric normal, more the farther from the camera.
fn rt_start(h: Shading) -> vec3<f32> {
    return h.P + h.Ng * (frame.rt2.x * (1.0 + abs(dot(h.P - frame.cam_pos.xyz, frame.cam_fwd.xyz))));
}

// EEVEE's clamp of the light a ray brings (scene clamp_surface_indirect, frame.sampling.z; 0: none): scaled down until
// its largest channel is the clamp (lab rt_clamp: that rule 4.6 levels from EEVEE, the luminance's 5.0, none 7.7).
fn rt_clamp(c: vec3<f32>) -> vec3<f32> {
    let m = max(max(c.r, c.g), c.b);
    if (frame.sampling.z > 0.0 && m > frame.sampling.z) {
        return c * (frame.sampling.z / m);
    }
    return c;
}

// The light a glossy lobe (normal N, `roughness`) gets from around the scene: a traced ray's when the ray finds a
// surface, else the world's probe as without tracing, clamped as EEVEE clamps it. `dim`: its random numbers'
// dimensions.
fn rt_traced(h: Shading, N: vec3<f32>, roughness: f32, dim: u32, R: vec3<f32>) -> vec3<f32> {
    let tb = rt_basis(N);
    let Ve = vec3<f32>(dot(h.V, tb[0]), dot(h.V, tb[1]), max(dot(h.V, N), 1e-4));
    let alpha = max(roughness * roughness, 1e-4);
    let m = rt_visible_normal(normalize(Ve), alpha, sample_rand2(dim));
    var L = reflect(-h.V, tb[0] * m.x + tb[1] * m.y + N * m.z);
    if (dot(L, h.Ng) <= 0.0) {
        // into the surface: the mirror direction instead
        L = R;
        if (dot(L, h.Ng) <= 0.0) {
            return rt_clamp(world_glossy(R, roughness));
        }
    }
    let hit = rt_trace(rt_start(h), L, samples[imm.sample].vp, sample_rand(dim + 2u), u32(frame.rt.w), frame.rt.z);
    if (hit.x < 0) {
        // nothing found: the probe as the lobe sees it, or (frame.rt2.y >= 0) along the ray, blurred as a lobe that
        // many times as rough
        if (frame.rt2.y >= 0.0) {
            return rt_clamp(world_glossy(L, roughness * frame.rt2.y));
        }
        return rt_clamp(world_glossy(R, roughness));
    }
    return rt_clamp(textureLoad(radiance_in, hit, 0).rgb);
}

// How far a glossy lobe of `roughness` has faded from its traced ray into fast GI: 0 up to the trace's roughness limit
// (frame.rt.y), 1 from frame.rt4.w on (EEVEE's rough floors darken from 0.5 to 1 as its reflections fade: lab
// gi_rough).
fn rt_fade(roughness: f32) -> f32 {
    if (roughness <= frame.rt.y) {
        return 0.0;
    }
    return clamp((roughness - frame.rt.y) / max(frame.rt4.w - frame.rt.y, 1e-4), 0.0, 1.0);
}

// A glossy lobe's light: its traced ray's, faded into fast GI's (`g`: its bounce and open share, the probe along the
// reflection lighting the open share) by rt_fade.
fn rt_lobe(h: Shading, N: vec3<f32>, roughness: f32, dim: u32, g: vec4<f32>) -> vec3<f32> {
    let R = reflect(-h.V, N);
    let w = rt_fade(roughness);
    var traced = vec3<f32>(0.0);
    if (w < 1.0) {
        traced = rt_traced(h, N, roughness, dim, R);
    }
    if (w <= 0.0) {
        return traced;
    }
    return mix(traced, world_glossy(R, roughness) * g.a + g.rgb, w);
}

// The point at pixel position q (pixels from the top left) and reversed depth z through the sample's camera.
fn rt_unproject(q: vec2<f32>, z: f32, vp_inv: mat4x4<f32>) -> vec3<f32> {
    let ph = vp_inv * vec4<f32>(q.x * frame.size.z * 2.0 - 1.0, 1.0 - q.y * frame.size.w * 2.0, z, 1.0);
    return ph.xyz / ph.w;
}

// The cosine-weighted measure of a slice's directions round the view vector from its lower horizon (n - pi/2) up to
// angle t (from the view vector, toward the slice's tangent): the integral of cos(x - n) |sin x|, n the angle of the
// normal projected into the slice. Its antiderivative over x > 0 is A(x) = -cos(2x - n) / 4 + x sin(n) / 2.
fn rt_slice_cdf(t: f32, n: f32) -> f32 {
    let lo = n - 0.5 * MKS_PI;
    let a_lo = -0.25 * cos(2.0 * lo - n) + 0.5 * lo * sin(n);
    let a_t = -0.25 * cos(2.0 * t - n) + 0.5 * t * sin(n);
    if (t <= 0.0) {
        return a_lo - a_t;
    }
    return a_lo + a_t + 0.5 * cos(n);
}

// Bits b0 .. b1 - 1 of a 32-bit mask.
fn rt_bits(b0: u32, b1: u32) -> u32 {
    if (b1 <= b0) {
        return 0u;
    }
    let n = b1 - b0;
    return select((1u << n) - 1u, 0xffffffffu, n >= 32u) << b0;
}

// EEVEE's fast GI as its horizon scan (frame.rt3: method, rays, steps, near thickness; frame.rt4: reach, angular
// thickness, normal bias): per ray a slice of directions round the view vector along a random screen direction, both
// ways `steps` depth samples out to the screen's edge (or the reach), denser near the pixel; each sample hides the
// sectors between its front and its back (the near thickness farther along its camera ray, and at least the angular
// thickness beyond its front), the slice's sectors equal in the cosine-weighted measure (32 of them). rgb: the light
// of the sectors a sample hides first (the previous sample's picture there: the bounce) when the side of it the camera
// sees faces the pixel, clamped as EEVEE clamps indirect light (lab gi_strands: no bounce on the wall behind thin
// strands from their lit fronts; rt_clamp: the floor's glow round bright strips), a: the open share.
fn rt_scan(h: Shading, N0: vec3<f32>) -> vec4<f32> {
    let sd = samples[imm.sample];
    let size = frame.size.xy;
    let px = vec2<f32>(g_pixel) + 0.5;
    let z0 = textureLoad(gb_depth, vec2<i32>(g_pixel), 0);
    let P = rt_unproject(px, z0, sd.vp_inv);
    let ortho = frame.cam_pos.w > 0.5;
    var V = -frame.cam_fwd.xyz;
    if (!ortho) {
        V = normalize(frame.cam_pos.xyz - P);
    }
    let N = normalize(N0 + V * frame.rt4.z);
    let n_rays = max(u32(frame.rt3.y), 1u);
    let steps = max(u32(frame.rt3.z), 1u);
    let thick = frame.rt3.w;
    let far = frame.rt4.y;
    let reach = frame.rt4.x;
    let bounce_on = frame.rt3.x > 1.001;
    var sum_w = 0.0;
    var sum_open = 0.0;
    var sum_b = vec3<f32>(0.0);
    // the sample's directions and step offsets, the same for every pixel (neighbours read neighbouring texels) and
    // stratified over the samples (R2) and the rays
    let r2 = fract(vec2<f32>((f32(g_sample) + 0.5) * 0.7548776662, (f32(g_sample) + 0.5) * 0.5698402910));
    for (var k = 0u; k < n_rays; k = k + 1u) {
        let phi = MKS_PI * fract(r2.x + f32(k) / f32(n_rays));
        let d = vec2<f32>(cos(phi), sin(phi));
        let T = rt_unproject(px + d, z0, sd.vp_inv) - P;
        let axis = normalize(cross(V, T));
        let B = cross(axis, V);
        let Np = N - axis * dot(N, axis);
        let np_len = length(Np);
        if (np_len < 1e-5) {
            continue;
        }
        let n = atan2(dot(Np, B), dot(Np, V));
        let lo = n - 0.5 * MKS_PI;
        let hi = n + 0.5 * MKS_PI;
        let total = rt_slice_cdf(hi, n);
        var mask = 0u;
        var b = vec3<f32>(0.0);
        for (var side = 0u; side < 2u; side = side + 1u) {
            let sgn = select(-1.0, 1.0, side == 0u);
            let dir = d * sgn;
            var t_end = 1e6;
            if (dir.x > 1e-6) { t_end = min(t_end, (size.x - px.x) / dir.x); }
            if (dir.x < -1e-6) { t_end = min(t_end, -px.x / dir.x); }
            if (dir.y > 1e-6) { t_end = min(t_end, (size.y - px.y) / dir.y); }
            if (dir.y < -1e-6) { t_end = min(t_end, -px.y / dir.y); }
            if (t_end < 1.0) {
                continue;
            }
            let jit = fract(r2.y + 0.6180339887 * f32(2u * k + side));
            for (var j = 0u; j < steps; j = j + 1u) {
                let s = (f32(j) + jit) / f32(steps);
                let q = px + dir * max(t_end * s * s, 1.0);
                let qi = clamp(vec2<i32>(floor(q)), vec2<i32>(0), vec2<i32>(size) - 1);
                let zq = textureLoad(gb_depth, qi, 0);
                if (zq <= 0.0) {
                    continue;
                }
                let S = rt_unproject(vec2<f32>(qi) + 0.5, zq, sd.vp_inv);
                let dF = S - P;
                if (reach > 0.0 && dot(dF, dF) > reach * reach) {
                    continue;
                }
                var back = frame.cam_fwd.xyz;
                if (!ortho) {
                    back = normalize(S - frame.cam_pos.xyz);
                }
                let dB = dF + back * thick;
                let aF = atan2(dot(dF, B), dot(dF, V));
                var aB = atan2(dot(dB, B), dot(dB, V));
                if (aF >= 0.0) {
                    aB = max(aB, aF + far);
                } else {
                    aB = min(aB, aF - far);
                }
                let a0 = clamp(min(aF, aB), lo, hi);
                let a1 = clamp(max(aF, aB), lo, hi);
                if (a1 <= a0) {
                    continue;
                }
                let b0 = u32(round(32.0 * clamp(rt_slice_cdf(a0, n) / total, 0.0, 1.0)));
                let b1 = u32(round(32.0 * clamp(rt_slice_cdf(a1, n) / total, 0.0, 1.0)));
                let m = rt_bits(b0, b1);
                let fresh = m & ~mask;
                if (bounce_on && fresh != 0u
                    && dot(gb_unpack_n8(textureLoad(gb_base, qi, 0).w & 0xffffu), dF) < 0.0) {
                    b = b + rt_clamp(textureLoad(radiance_in, qi, 0).rgb) * f32(countOneBits(fresh));
                }
                mask = mask | m;
            }
        }
        let w = np_len * total;
        sum_w = sum_w + w;
        sum_open = sum_open + w * f32(32u - countOneBits(mask)) / 32.0;
        sum_b = sum_b + w * b / 32.0;
    }
    if (sum_w <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(sum_b, sum_open) / sum_w;
}

// The share of traced_light's light that is fast GI's bounce: the lighting pass keeps the sample's picture without it
// for the next sample's rays (deferred.wgsl), so the bounce is one bounce of each sample's direct and reflected light,
// not a closed space lighting itself over the samples (jwin's seats and hair: 12 and 9 levels too bright).
var<private> rt_bounce: vec3<f32>;

// The world's light (world_light.wgsl) with the glossy lobes traced, and the diffuse closure's through fast GI
// (frame.rt3.x: 0 none, 1 its occlusion of the world's light alone, more: that and the bounce, times rt3.x - 1).
// Glossy lobes rougher than the trace's limit take fast GI's progressively (rt_lobe).
fn traced_light(h: Shading, reflection_roughness: f32, coat_roughness: f32) -> vec3<f32> {
    var o = vec3<f32>(0.0);
    var rest = h;
    rest.reflection = vec3<f32>(0.0);
    rest.coat = vec3<f32>(0.0);
    let gi_on = frame.rt3.x > 0.5;
    let rough_r = gi_on && any(h.reflection > vec3<f32>(0.0)) && reflection_roughness > frame.rt.y;
    let rough_c = gi_on && any(h.coat > vec3<f32>(0.0)) && coat_roughness > frame.rt.y;
    let gi = gi_on && any(h.diffuse > vec3<f32>(0.0));
    var g = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (gi || rough_r || rough_c) {
        g = rt_scan(h, h.diffuse_N);
        g = vec4<f32>(g.rgb * max(frame.rt3.x - 1.0, 0.0), g.a);
    }
    rt_bounce = vec3<f32>(0.0);
    if (gi) {
        rest.diffuse = vec3<f32>(0.0);
        var d = g.rgb;
        if (frame.world.x > 0.5) {
            d = d + world_irradiance(h.diffuse_N) * g.a;
        }
        o = h.diffuse * d;
        rt_bounce = h.diffuse * g.rgb;
    }
    if (frame.world.x > 0.5) {
        o = o + world_light(rest, reflection_roughness, coat_roughness);
    }
    rt_reflected = vec3<f32>(0.0);
    if (any(h.reflection > vec3<f32>(0.0))) {
        let lr = h.reflection * rt_lobe(h, h.reflection_N, reflection_roughness, 1000u, g);
        o = o + lr;
        rt_reflected = rt_reflected + lr * (1.0 - rt_fade(reflection_roughness));
        if (rough_r) {
            rt_bounce = rt_bounce + h.reflection * rt_fade(reflection_roughness) * g.rgb;
        }
    }
    if (any(h.coat > vec3<f32>(0.0))) {
        let lc = h.coat * rt_lobe(h, h.coat_N, coat_roughness, 1004u, g);
        o = o + lc;
        rt_reflected = rt_reflected + lc * (1.0 - rt_fade(coat_roughness));
        if (rough_c) {
            rt_bounce = rt_bounce + h.coat * rt_fade(coat_roughness) * g.rgb;
        }
    }
    return o;
}

// The share of traced_light's light that its traced rays brought (the lobes' light, less what fades into fast GI):
// what EEVEE's denoiser filters (rt_temporal).
var<private> rt_reflected: vec3<f32>;

// EEVEE's temporal accumulation of the traced light T (frame.rt.x 2: the scene's use_denoise and denoise_temporal on)
// over a frame's samples: the pixel's history, clamped in luminance to the range of the previous sample's raw traced
// light over the 3 x 3 pixels round it and of this sample's own, blended with T at 1 / (n + 1) on sample n; history_out
// keeps the result and T's luminance. Where few rays find an emitter (the edges and tails of its reflection, thin
// emitters, rough lobes) the range is often dark and the history loses its light, as EEVEE's does: lab rt_emit 1.31 ->
// 0.55 levels, the 6-60 cm cards' light within 2 % of EEVEE's frames without the bilateral filter (nobilateral) at
// roughness 0.2, the 2 and 6 cm cards 8-15 % brighter at 0.1. The bilateral filter's further loss is not drawn.
fn rt_temporal(T: vec3<f32>) -> vec3<f32> {
    let px = vec2<i32>(g_pixel);
    let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lt = dot(T, lum);
    var O = T;
    if (g_sample > 0u) {
        let size = vec2<i32>(frame.size.xy);
        var lo = lt;
        var hi = lt;
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let a = textureLoad(history_in, clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), size - 1), 0).a;
                lo = min(lo, a);
                hi = max(hi, a);
            }
        }
        let H = textureLoad(history_in, px, 0).rgb;
        let lh = dot(H, lum);
        var Hc = H;
        if (lh > hi) {
            Hc = H * (hi / lh);
        } else if (lh < lo) {
            // up toward T until its luminance is the range's low end (lt >= lo > lh)
            Hc = H + (T - H) * ((lo - lh) / max(lt - lh, 1e-6));
        }
        O = mix(Hc, T, 1.0 / f32(g_sample + 1u));
    }
    textureStore(history_out, px, vec4<f32>(O, lt));
    return O;
}
