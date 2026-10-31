// Shadows by ray queries (`shadow_mode: "rays"`, better than EEVEE: the converged soft shadows of its jittered shadow
// maps, or Cycles'): per sample one ray toward a random point of the light above the horizon; alpha-tested occluders are
// hashed per ray through `material_alpha_lod` (pipelines.rs generates it over every alpha-casting material).

// The opacity of the candidate triangle a shadow ray met, hashed into hit / no hit.
fn shadow_hit(c: RayIntersection, light: u32) -> bool {
    let m = meshes[c.instance_custom_data];
    let d = draws[m.draw_base + c.geometry_index];
    let t = tri_vertices(d, m, c.primitive_index);
    let b = vec3<f32>(1.0 - c.barycentrics.x - c.barycentrics.y, c.barycentrics.x, c.barycentrics.y);
    let p = world_pos(m, t.x) * b.x + world_pos(m, t.y) * b.y + world_pos(m, t.z) * b.z;
    let n = world_nrm(m, t.x) * b.x + world_nrm(m, t.y) * b.y + world_nrm(m, t.z) * b.z;
    var attr: array<vec4<f32>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        attr[i] = attr_of(m, t.x, d.remap[i]) * b.x + attr_of(m, t.y, d.remap[i]) * b.y
                + attr_of(m, t.z, d.remap[i]) * b.z;
    }
    let generated = (attr_of(m, t.x, m.generated) * b.x + attr_of(m, t.y, m.generated) * b.y
                   + attr_of(m, t.z, m.generated) * b.z).xyz;
    // surface_at points g_param_base / g_image_base at the occluder's material: keep the shading point's
    let keep = vec2<u32>(g_param_base, g_image_base);
    let s = surface_at(d, p, n, face_normal(m, t), !c.front_face, attr, generated, vec2<f32>(0.0), MKR_RAY_SHADOW);
    let alpha = material_alpha_lod(d.material, s);
    g_param_base = keep.x;
    g_image_base = keep.y;
    let threshold = unit(hash3(g_pixel.x ^ (light * 0x9e3779b9u), g_pixel.y ^ (g_sample * 0x632be5abu),
                               c.primitive_index ^ (c.instance_custom_data * 0x85ebca6bu) ^ frame.frame_seed));
    return alpha > threshold;
}

// 1 when the ray from P toward a random point of light `li` (above P's horizon) reaches it, 0 when something stops it.
fn shadow_visibility(l: Light, li: u32, P: vec3<f32>, Ng: vec3<f32>) -> f32 {
    if (l.spot.z < 0.5 || frame.shadow.w < 0.5) {
        return 1.0;
    }
    let target_p = light_point_above(l, li, P, Ng);
    let off = frame.shadow.x + frame.shadow.y * distance(P, frame.cam_pos.xyz);
    let origin = P + Ng * off;
    let span = target_p - origin;
    let dist = length(span);
    if (dist <= 0.0) {
        return 1.0;
    }
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xffu, 0.0, dist * 0.9999, origin, span / dist));
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_TRIANGLE && shadow_hit(c, li)) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);
    return select(1.0, 0.0, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
