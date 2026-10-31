// The world's module (world.rs): the drawing core, mkr-shade's `mat_world` (its emission the world's radiance looking
// along Surface.P) and these entry points: the background, drawn where the camera sees no surface (the depth still
// cleared), and the faces of the world's probe, a cube map of its radiance the lighting reads (world_light.wgsl).
// WORLD_PARAM_BASE, WORLD_IMAGE_BASE and WORLD_PROBE_RAY come first (world.rs).

// What the world's shader sees looking along `dir` (glTF world, unit): EEVEE gives a world the direction as its
// position and Generated coordinate, its normal and incoming the way back.
fn world_surface(dir: vec3<f32>, ray: u32) -> Surface {
    g_param_base = WORLD_PARAM_BASE;
    g_image_base = WORLD_IMAGE_BASE;
    var s: Surface;
    let d = to_blender(dir);
    s.P = d;
    s.N = -d;
    s.Ng = -d;
    s.I = -d;
    s.backfacing = 0.0;
    s.generated = d;
    let id = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0),
                         vec4<f32>(0.0, 0.0, 0.0, 1.0));
    s.object = id;
    s.object_inv = id;
    s.camera = frame.camera;
    s.camera_inv = frame.camera_inv;
    s.ortho = frame.cam_pos.w;
    s.rand = vec4<f32>(sample_rand(16u), sample_rand(17u), sample_rand(18u), sample_rand(19u));
    s.ray_type = ray;
    return s;
}

// A triangle over the whole target at depth 0: the far plane the depth is cleared to (reverse Z).
@vertex
fn vs_world(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// The background of sample imm.sample: the world along the ray through the pixel (the sample's jittered camera).
@fragment
fn fs_world_background(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let sd = samples[imm.sample];
    g_pixel = vec2<u32>(pos.xy);
    g_sample = sd.info.y;
    let ndc = vec2<f32>(pos.x * frame.size.z * 2.0 - 1.0, 1.0 - pos.y * frame.size.w * 2.0);
    let a = sd.vp_inv * vec4<f32>(ndc, 1.0, 1.0);
    let b = sd.vp_inv * vec4<f32>(ndc, 0.5, 1.0);
    var s = world_surface(normalize(b.xyz / b.w - a.xyz / a.w), MKR_RAY_CAMERA);
    s.screen = vec2<f32>(pos.x * frame.size.z, 1.0 - pos.y * frame.size.w);
    return vec4<f32>(mat_world(s).emission, 1.0);
}

// A texel of the probe's face imm.draw (imm.sample texels across): the world's radiance there, as the rays that light
// the scene see it.
@fragment
fn fs_world_probe(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    g_pixel = vec2<u32>(pos.xy) + vec2<u32>(imm.draw * 4099u, 0u);
    g_sample = 0u;
    let st = pos.xy / f32(imm.sample) * 2.0 - 1.0;
    let s = world_surface(cube_dir(imm.draw, st), WORLD_PROBE_RAY);
    return vec4<f32>(max(mat_world(s).emission, vec3<f32>(0.0)), 1.0);
}
