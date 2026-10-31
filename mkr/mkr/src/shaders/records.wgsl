// The records render.rs writes for the GPU (FrameGpu, SampleGpu, LightGpu, ShadowFaceGpu): every drawing module starts
// with them (pipelines.rs core), and so does the shadow maps' setup (shadow_setup.wgsl), which binds its own buffers.

struct Frame {
    view: mat4x4<f32>,              // glTF world -> Blender camera axes (the frame's own time)
    camera: mat4x4<f32>,            // Blender viewer -> world, scale removed
    camera_inv: mat4x4<f32>,
    cam_pos: vec4<f32>,             // glTF world; w: 1 orthographic
    cam_fwd: vec4<f32>,             // glTF world, the way the camera looks
    size: vec4<f32>,                // W, H, 1 / W, 1 / H
    light_count: u32,
    sample_count: u32,
    flags: u32,                     // 1: motion blur by time samples; 2: the G-buffer writes the velocity (post blur)
    frame_seed: u32,
    shadow: vec4<f32>,              // ray offset (m), offset per metre from the camera (rays), light spread, 1: shadows on
    scale: vec4<f32>,               // diffuse, glossy, mip bias scale, hashed alpha cell (pixel footprints)
    trace: vec4<f32>,               // shadow steps, their length (fraction of the ray), 1: sphere horizon diffuse, glossy
    bias: vec4<f32>,                // shadow maps: normal offset (texel radii x sin), depth tolerance (texel radii); 1:
                                    // glossy horizon fade; the filter's normal offset (per unit of filter radius)
    shadow_map: vec4<f32>,          // shadow tracing: step spacing power, 1: rule v2; the filter disk's radius (per unit
                                    // of filter radius, texel radii), the reach of the receiver's level (its texels)
    shadow_lod: vec4<f32>,          // a pixel's footprint (m per m of view depth; orthographic: m), level bias,
                                    // resolution limit offset (levels), the level of the whole faces' maps
    shadow_pool: vec4<f32>,         // shadow maps' entries (shadows.rs): the first of the fine maps (suns' maps, then
                                    // the windows), the first window, how many windows, a window's texels across
    world: vec4<f32>,               // the world (world.rs): 1 it lights, the probe's last mip, the harmonics' band 0
                                    // and band 1 weights
    world_probe: vec4<f32>,         // glossy lobes: blur (lobe widths per texel), harmonics blend from / to roughness,
                                    // the probe's texels across (mip 0)
    rt: vec4<f32>,                  // EEVEE's screen tracing (raytrace.rs): 1 on, the roughness it traces up to, the
                                    // thickness (m along the view axis), steps per ray
    rt2: vec4<f32>,                 // the rays' start off the surface (m, per m of view distance plus one), what a ray
                                    // that finds nothing reads (raytrace.wgsl rt_traced), the near and far planes (m)
    rt3: vec4<f32>,                 // fast GI (raytrace.wgsl rt_scan): 0 none, 1 occlusion, more: occlusion and
                                    // rt3.x - 1 of the bounce; slices per sample, depth samples each way, thickness (m)
    rt4: vec4<f32>,                 // fast GI: its reach (m; 0: across the screen), angular thickness (rad), how far
                                    // the normal leans toward the camera, the roughness from which lobes are wholly its
    sampling: vec4<f32>,            // shadow rays per sample and light; 1: their numbers from Sobol's sequence
                                    // (light.wgsl shadow_draw), 0: R2's; EEVEE's clamp of a traced ray's light
                                    // (raytrace.wgsl rt_clamp: clamp_surface_indirect, 0: none); the share of the
                                    // trace thickness a ray going away keeps past a surface it crosses (rt_trace)
}

struct SampleData {
    vp: mat4x4<f32>,                // jittered projection x view on the sample's time
    vp_inv: mat4x4<f32>,            // its inverse: the lighting pass rebuilds positions from depth
    info: vec4<u32>,                // x: 0 = between prev and cur, 1 = between cur and next; y: sample number
    t: vec4<f32>,                   // x: how far from the first to the second; yz: jitter (pixels)
    rand: vec4<f32>,                // per-sample uniform numbers (low discrepancy over the samples)
    // the post motion blur's cameras (no jitter, no lens): the sample's time, the shutter's open and its close
    vp_cur: mat4x4<f32>,
    vp_prev: mat4x4<f32>,
    vp_next: mat4x4<f32>,
}

// Light kinds (Light.pos.w).
const KIND_POINT: u32 = 0u;
const KIND_SUN: u32 = 1u;
const KIND_SPOT: u32 = 2u;
const KIND_RECT: u32 = 3u;
const KIND_ELLIPSE: u32 = 4u;

struct Light {
    pos: vec4<f32>,                 // glTF world; w: kind (0 point, 1 sun, 2 spot, 3 rect, 4 ellipse)
    fwd: vec4<f32>,                 // the way it shines (unit); w: radius (point, spot), half angle (sun)
    right: vec4<f32>,               // unit; w: half size x (area)
    up: vec4<f32>,                  // unit; w: half size y (area)
    color: vec4<f32>,               // radiance (area, point, spot) or irradiance (sun); w: 1 / influence radius²
    spot: vec4<f32>,                // x: cos(half spot angle), y: spot blend, z: casts shadows, w: visibility bits
    factor: vec4<f32>,              // diffuse, specular, transmission factors; shadow filter radius
    map: vec4<f32>,                 // its shadow map: first entry (-1: none), near plane (m; sun: depth span), size
                                    // (texels), resolution limit (m; sun: its map's texel)
}

// One entry of the shadow maps (a cube face, a sun's map, a window): world -> clip (cubes: reverse-Z, infinite far;
// suns: reverse linear depth).
struct ShadowFace {
    vp: mat4x4<f32>,
}
