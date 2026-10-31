// The direction through texel coordinates st (-1..1, rows down) of cube face `face` (+X, -X, +Y, -Y, +Z, -Z), as
// the GPU's cube maps look them up.
fn cube_dir(face: u32, st: vec2<f32>) -> vec3<f32> {
    let sc = st.x;
    let tc = st.y;
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -tc, -sc)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -tc, sc)); }
        case 2u: { return normalize(vec3<f32>(sc, 1.0, tc)); }
        case 3u: { return normalize(vec3<f32>(sc, -1.0, -tc)); }
        case 4u: { return normalize(vec3<f32>(sc, -tc, 1.0)); }
        default: { return normalize(vec3<f32>(-sc, -tc, -1.0)); }
    }
}
