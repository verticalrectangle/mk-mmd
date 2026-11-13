// mkr's drawing core, after mkr-shade's prelude: the bindings every drawing module shares, the vertex shaders (posed
// vertices pulled from storage, jittered and moved to the sample's time; the shadow maps' faces), the Surface a
// material sees, per-pixel randoms and the hashed alpha of the depth prepass. pipelines.rs compiles each material into
// a module of its own (this core, the material, its entry points) so that changing one material recompiles one small
// module; the lighting of opaque surfaces is one compute pass over the G-buffer (deferred.wgsl); blended surfaces and
// Shader to RGB light forward through light.wgsl.
//
// Everything stored is in glTF's axes (the bake's); Surface and Closures are in Blender's: Blender (x, y, z) = glTF
// (x, -z, y).

// Frame, SampleData, Light and ShadowFace are in records.wgsl, which comes first.

struct Posed {
    pos: vec4<f32>,
    nrm: vec4<f32>,
}

struct MeshInfo {
    object: mat4x4<f32>,            // Blender axes: object -> world
    object_inv: mat4x4<f32>,
    m_prev: array<vec4<f32>, 3>,    // glTF: what places its shape's posed vertices in the world, rows, per time
    m_cur: array<vec4<f32>, 3>,
    m_next: array<vec4<f32>, 3>,
    n_cur: array<vec4<f32>, 3>,     // normals: the inverse transpose of m_cur
    object_color: vec4<f32>,
    object_info: vec4<f32>,
    vertex_base: u32,
    attr_base: u32,
    attr_stride: u32,
    index_base: u32,
    draw_base: u32,
    generated: u32,                 // the attribute slot holding Blender's Generated coordinates (0xffffffff: none)
    flags: u32,
    pad: u32,
}

struct Draw {
    mesh: u32,
    material: u32,
    first_tri: u32,
    tri_count: u32,
    remap: array<u32, 8>,           // the mesh attribute slot behind each of the material's attributes
    param_base: u32,                // the material's first slot in mat_params (its mkr_param slots count from 0)
    image_base: u32,                // the material's first entry in image_map (its mkr_image indices count from 0)
    flags: u32,                     // DRAW_UNSHADOWED: no light's shadow falls on it (its material receives none)
    pad: u32,
}

const DRAW_UNSHADOWED: u32 = 1u;

struct Immediate {
    draw: u32,
    sample: u32,                    // the shadow face in shadow passes
}

@group(0) @binding(0) var<storage, read> frame: Frame;
@group(0) @binding(1) var<storage, read> samples: array<SampleData>;
@group(0) @binding(2) var<storage, read> lights: array<Light>;
@group(0) @binding(3) var tlas: acceleration_structure;
@group(0) @binding(4) var<storage, read> posed: array<Posed>;
@group(0) @binding(5) var<storage, read> posed_prev: array<Posed>;
@group(0) @binding(6) var<storage, read> posed_next: array<Posed>;
@group(0) @binding(7) var<storage, read> attrs: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> indices: array<u32>;
@group(0) @binding(9) var<storage, read> meshes: array<MeshInfo>;
@group(0) @binding(10) var<storage, read> draws: array<Draw>;
@group(0) @binding(11) var<storage, read> mat_params: array<vec4<f32>>;
@group(0) @binding(12) var images: binding_array<texture_2d<f32>>;
@group(0) @binding(13) var s_linear_repeat: sampler;
@group(0) @binding(14) var s_linear_extend: sampler;
@group(0) @binding(15) var s_linear_mirror: sampler;
@group(0) @binding(16) var s_nearest_repeat: sampler;
@group(0) @binding(17) var s_nearest_extend: sampler;
@group(0) @binding(18) var s_nearest_mirror: sampler;
@group(0) @binding(19) var shade_lut: texture_2d_array<f32>;
@group(0) @binding(20) var s_lut: sampler;
@group(0) @binding(21) var ltc_tex: texture_2d<f32>;
@group(0) @binding(22) var<storage, read> image_map: array<u32>;
// The shadow maps (shadows.rs): the cube faces whole (coarse) and the suns' maps and windows (fine); per array the
// depths (R the opaque casters', G the nearest partly transparent hashed caster's in front of them) and the product of
// those hashed casters' transparencies.
@group(0) @binding(23) var shadow_depths: texture_2d_array<f32>;
@group(0) @binding(24) var<storage, read> shadow_faces: array<ShadowFace>;
// per cube face, its window (shadow_setup.wgsl): origin in its level's texels, level, layer of the fine maps (-1: none)
@group(0) @binding(25) var<storage, read> shadow_windows: array<vec4<i32>>;
@group(0) @binding(26) var shadow_clear: texture_2d_array<f32>;
@group(0) @binding(28) var shadow_fine_depths: texture_2d_array<f32>;
@group(0) @binding(29) var shadow_fine_clear: texture_2d_array<f32>;
// The world's probe (world.rs): a cube map of its radiance, mips box-filtered, and its L1 spherical harmonics (Y00,
// Y1-1, Y10, Y11; rgb), what world_light.wgsl lights with (stand-ins without a world).
@group(0) @binding(30) var world_probe: texture_cube<f32>;
@group(0) @binding(31) var<storage, read> world_sh: array<vec4<f32>, 4>;

var<immediate> imm: Immediate;

// ------------------------------------------------------------------------------------------------- axes and hashes
fn to_blender(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x, -v.z, v.y);
}

fn to_gltf(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x, v.z, -v.y);
}

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    return hash_u32(a ^ hash_u32(b ^ hash_u32(c)));
}

fn unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Per-pixel state of the point being shaded (randoms hang off it) and of the material evaluating it; g_unshadowed: no
// light's shadow falls on it (the camera's surface: its draw's DRAW_UNSHADOWED, or the G-buffer's GB_UNSHADOWED).
var<private> g_pixel: vec2<u32>;
var<private> g_sample: u32;
var<private> g_param_base: u32;
var<private> g_image_base: u32;
var<private> g_unshadowed: bool;

// The sample's low-discrepancy number for dimension `dim`, rotated per pixel (Cranley-Patterson): each pixel sees the
// samples' stratification, neighbouring pixels decorrelated.
fn sample_rand(dim: u32) -> f32 {
    let base = fract(f32(g_sample) * fract(0.5 + 0.6180339887 * f32(dim + 1u)) + 0.5 * f32(dim & 1u));
    return fract(base + unit(hash3(g_pixel.x, g_pixel.y, dim * 0x9e3779b9u + frame.frame_seed)));
}

fn sample_rand2(dim: u32) -> vec2<f32> {
    // R2 sequence over the samples (Roberts), rotated per pixel and dimension
    let s = f32(g_sample) + 0.5;
    let base = fract(vec2<f32>(s * 0.7548776662, s * 0.5698402910));
    let rot = vec2<f32>(unit(hash3(g_pixel.x, g_pixel.y, dim * 2u + frame.frame_seed)),
                        unit(hash3(g_pixel.y, g_pixel.x, dim * 2u + 1u + frame.frame_seed)));
    return fract(base + rot);
}

// Point `index` of Sobol's sequence in its first five dimensions, 32-bit fixed point (the direction numbers by their
// recurrences, Joe and Kuo's first polynomials: no tables). Each 2^m points of it from a multiple of 2^m fill every
// 2^-m interval of a dimension once, and the pairs (0, 1) and (2, 3) every cell of their 2D grids of 2^m cells.
fn sobol5(index: u32) -> array<u32, 5> {
    var x = array<u32, 5>(reverseBits(index), 0u, 0u, 0u, 0u);
    var i = index;
    var v1 = 0x80000000u;
    var p2 = 0x80000000u;
    var q2 = 0xc0000000u;
    var p3 = 0x80000000u;
    var q3 = 0xc0000000u;
    var r3 = 0x20000000u;
    var p4 = 0x80000000u;
    var q4 = 0x40000000u;
    var r4 = 0x20000000u;
    loop {
        if (i == 0u) {
            break;
        }
        if ((i & 1u) != 0u) {
            x[1] = x[1] ^ v1;
            x[2] = x[2] ^ p2;
            x[3] = x[3] ^ p3;
            x[4] = x[4] ^ p4;
        }
        i = i >> 1u;
        v1 = v1 ^ (v1 >> 1u);
        let n2 = (p2 >> 2u) ^ p2 ^ q2;
        p2 = q2;
        q2 = n2;
        let n3 = q3 ^ p3 ^ (p3 >> 3u);
        p3 = q3;
        q3 = r3;
        r3 = n3;
        let n4 = r4 ^ p4 ^ (p4 >> 3u);
        p4 = q4;
        q4 = r4;
        r4 = n4;
    }
    return x;
}

// Owen's scrambling of a 32-bit fixed-point number in base 2 (Burley 2020: Laine and Karras's hash on the reversed
// bits, each output digit depending on the digits above it and the seed): the strata of a point set stay strata.
// On an index it shuffles the indices, each aligned block of 2^m onto another.
fn owen(x: u32, seed: u32) -> u32 {
    var v = reverseBits(x) + seed;
    v = v ^ (v * 0x6c50b47cu);
    v = v ^ (v * 0xb82f1e52u);
    v = v ^ (v * 0xc7afe638u);
    v = v ^ (v * 0x8d22f6e6u);
    return reverseBits(v);
}

// ---------------------------------------------------------------------------------------------- what mkr-shade needs
fn ext_uv(st: vec2<f32>, ext: u32) -> bool {
    // CLIP: transparent black outside the image
    return ext != 2u || (st.x >= 0.0 && st.x <= 1.0 && st.y >= 0.0 && st.y <= 1.0);
}

fn image_grad(local: u32, st: vec2<f32>, interp: u32, ext: u32, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    if (!ext_uv(st, ext)) {
        return vec4<f32>(0.0);
    }
    let t = images[image_map[g_image_base + local]];
    if (interp == 1u) {
        if (ext == 0u) {
            return textureSampleGrad(t, s_nearest_repeat, st, dx, dy);
        } else if (ext == 3u) {
            return textureSampleGrad(t, s_nearest_mirror, st, dx, dy);
        }
        return textureSampleGrad(t, s_nearest_extend, st, dx, dy);
    }
    if (ext == 0u) {
        return textureSampleGrad(t, s_linear_repeat, st, dx, dy);
    } else if (ext == 3u) {
        return textureSampleGrad(t, s_linear_mirror, st, dx, dy);
    }
    return textureSampleGrad(t, s_linear_extend, st, dx, dy);
}

// `index` is the material's own (its images in the order mkr-shade met them): image_map turns it into the scene's.
fn mkr_image(index: u32, uv: vec2<f32>, interp: u32, ext: u32) -> vec4<f32> {
    let st = vec2<f32>(uv.x, 1.0 - uv.y);
    let dx = dpdx(st);
    let dy = dpdy(st);
    return image_grad(index, st, interp, ext, dx, dy);
}

fn mkr_image_lod(index: u32, uv: vec2<f32>, interp: u32, ext: u32) -> vec4<f32> {
    let st = vec2<f32>(uv.x, 1.0 - uv.y);
    return image_grad(index, st, interp, ext, vec2<f32>(0.0), vec2<f32>(0.0));
}

fn mkr_param(slot: u32) -> vec4<f32> {
    return mat_params[g_param_base + slot];
}

fn mkr_lut(uv: vec2<f32>, layer: u32) -> vec4<f32> {
    return textureSampleLevel(shade_lut, s_lut, uv, layer, 0.0);
}

// ------------------------------------------------------------------------------------------------------ the vertex
struct VsOut {
    @builtin(position) @invariant pos: vec4<f32>,
    @location(0) p: vec3<f32>,                  // glTF world, the frame's own time (shading)
    @location(1) n: vec3<f32>,
    @location(2) a0: vec4<f32>,
    @location(3) a1: vec4<f32>,
    @location(4) a2: vec4<f32>,
    @location(5) a3: vec4<f32>,
    @location(6) a4: vec4<f32>,
    @location(7) a5: vec4<f32>,
    @location(8) a6: vec4<f32>,
    @location(9) a7: vec4<f32>,
    @location(10) generated: vec3<f32>,
    @location(11) ps: vec3<f32>,                // glTF world, the sample's time (where the raster put it)
    @location(12) @interpolate(flat) draw: u32, // the draw (its instance index: draws come as multi-draws)
    @location(13) pv: vec3<f32>,                // glTF world on the shutter's open and close (the post motion blur's
    @location(14) pn: vec3<f32>,                // velocity; the frame's own time without it)
}

fn attr_of(m: MeshInfo, v: u32, slot: u32) -> vec4<f32> {
    if (slot == 0xffffffffu) {
        return vec4<f32>(0.0);
    }
    return attrs[m.attr_base + (v - m.vertex_base) * m.attr_stride + slot];
}

fn apply3(m: array<vec4<f32>, 3>, p: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m[0], p), dot(m[1], p), dot(m[2], p));
}

// A vertex in the world on the frame's own time (shading, shadows).
fn world_pos(m: MeshInfo, v: u32) -> vec3<f32> {
    return apply3(m.m_cur, vec4<f32>(posed[v].pos.xyz, 1.0));
}

fn world_nrm(m: MeshInfo, v: u32) -> vec3<f32> {
    return apply3(m.n_cur, vec4<f32>(posed[v].nrm.xyz, 0.0));
}

// A vertex in the world on the sample's time (the shutter's two halves interpolate the frame's three poses).
fn raster_pos(m: MeshInfo, v: u32) -> vec3<f32> {
    let sd = samples[imm.sample];
    if ((frame.flags & 1u) == 0u) {
        return world_pos(m, v);
    }
    let cur = world_pos(m, v);
    if (sd.info.x == 0u) {
        return mix(apply3(m.m_prev, vec4<f32>(posed_prev[v].pos.xyz, 1.0)), cur, sd.t.x);
    }
    return mix(cur, apply3(m.m_next, vec4<f32>(posed_next[v].pos.xyz, 1.0)), sd.t.x);
}

fn vertex_out(m: MeshInfo, d: Draw, di: u32, v: u32, clip: vec4<f32>, ps: vec3<f32>) -> VsOut {
    var o: VsOut;
    o.pos = clip;
    o.p = world_pos(m, v);
    o.n = world_nrm(m, v);
    o.a0 = attr_of(m, v, d.remap[0]);
    o.a1 = attr_of(m, v, d.remap[1]);
    o.a2 = attr_of(m, v, d.remap[2]);
    o.a3 = attr_of(m, v, d.remap[3]);
    o.a4 = attr_of(m, v, d.remap[4]);
    o.a5 = attr_of(m, v, d.remap[5]);
    o.a6 = attr_of(m, v, d.remap[6]);
    o.a7 = attr_of(m, v, d.remap[7]);
    o.generated = attr_of(m, v, m.generated).xyz;
    o.ps = ps;
    o.draw = di;
    o.pv = o.p;
    o.pn = o.p;
    if ((frame.flags & 2u) != 0u) {
        o.pv = apply3(m.m_prev, vec4<f32>(posed_prev[v].pos.xyz, 1.0));
        o.pn = apply3(m.m_next, vec4<f32>(posed_next[v].pos.xyz, 1.0));
    }
    return o;
}

// The post motion blur's velocity of the surface at `in`: in pixels (x right, y down), the way to where the shutter's
// open shows it (xy) and its close (zw), through the sample's cameras without jitter (motion_blur.wgsl reads it).
fn mb_velocity(in: VsOut) -> vec4<f32> {
    let sd = samples[imm.sample];
    let half = 0.5 * frame.size.xy;
    let c = sd.vp_cur * vec4<f32>(in.p, 1.0);
    let a = sd.vp_prev * vec4<f32>(in.pv, 1.0);
    let b = sd.vp_next * vec4<f32>(in.pn, 1.0);
    let cur = c.xy / c.w;
    let to = vec2<f32>(1.0, -1.0) * half;
    return vec4<f32>((a.xy / a.w - cur) * to, (b.xy / b.w - cur) * to);
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) di: u32) -> VsOut {
    let d = draws[di];
    let m = meshes[d.mesh];
    let ps = raster_pos(m, v);
    return vertex_out(m, d, di, v, samples[imm.sample].vp * vec4<f32>(ps, 1.0), ps);
}

// Depth alone (materials that are opaque everywhere): the same position as vs_main, nothing else.
@vertex
fn vs_depth(@builtin(vertex_index) v: u32, @builtin(instance_index) di: u32) -> @builtin(position) @invariant vec4<f32> {
    let m = meshes[draws[di].mesh];
    return samples[imm.sample].vp * vec4<f32>(raster_pos(m, v), 1.0);
}

// A shadow map's entry (imm.sample): opaque casters on the frame's own time, their depth into the depths' R too
// (shadows.rs: what the lighting reads).
@vertex
fn vs_shadow(@builtin(vertex_index) v: u32, @builtin(instance_index) di: u32) -> @builtin(position) vec4<f32> {
    let m = meshes[draws[di].mesh];
    return shadow_faces[imm.sample].vp * vec4<f32>(world_pos(m, v), 1.0);
}

@fragment
fn fs_shadow_depth(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(pos.z, 0.0, 0.0, 0.0);
}

// A shadow map's face for casters whose opacity is tested: what their material needs.
@vertex
fn vs_shadow_full(@builtin(vertex_index) v: u32, @builtin(instance_index) di: u32) -> VsOut {
    let d = draws[di];
    let m = meshes[d.mesh];
    let p = world_pos(m, v);
    return vertex_out(m, d, di, v, shadow_faces[imm.sample].vp * vec4<f32>(p, 1.0), p);
}

// Depth alone inside the lit pass (a blended surface's front layer): writes no colour.
@fragment
fn fs_none() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

// ----------------------------------------------------------------------------------------------------- the surface
// The triangle `prim` of the draw: its three global vertex indices.
fn tri_vertices(d: Draw, m: MeshInfo, prim: u32) -> vec3<u32> {
    let at = m.index_base + (d.first_tri + prim) * 3u;
    return vec3<u32>(indices[at], indices[at + 1u], indices[at + 2u]) + vec3<u32>(m.vertex_base);
}

fn face_normal(m: MeshInfo, t: vec3<u32>) -> vec3<f32> {
    let a = world_pos(m, t.x);
    let b = world_pos(m, t.y);
    let c = world_pos(m, t.z);
    let n = cross(b - a, c - a);
    let l = length(n);
    if (l <= 0.0) {
        return normalize(world_nrm(m, t.x));
    }
    return n / l;
}

// The viewer's direction at p (glTF world, unit, toward the viewer).
fn view_dir(p: vec3<f32>) -> vec3<f32> {
    if (frame.cam_pos.w > 0.5) {
        return -frame.cam_fwd.xyz;
    }
    return normalize(frame.cam_pos.xyz - p);
}

fn surface_at(d: Draw, p: vec3<f32>, n_in: vec3<f32>, ng_in: vec3<f32>, back: bool, attr: array<vec4<f32>, 8>,
              generated: vec3<f32>, screen: vec2<f32>, ray: u32) -> Surface {
    let m = meshes[d.mesh];
    g_param_base = d.param_base;
    g_image_base = d.image_base;
    var s: Surface;
    var n = normalize(n_in);
    var ng = ng_in;
    if (back) {
        n = -n;
        ng = -ng;
    }
    s.P = to_blender(p);
    s.N = to_blender(n);
    s.Ng = to_blender(ng);
    s.I = to_blender(view_dir(p));
    s.backfacing = select(0.0, 1.0, back);
    s.generated = generated;
    s.attrs = attr;
    s.object = m.object;
    s.object_inv = m.object_inv;
    s.camera = frame.camera;
    s.camera_inv = frame.camera_inv;
    s.ortho = frame.cam_pos.w;
    s.screen = screen;
    s.rand = vec4<f32>(sample_rand(16u), sample_rand(17u), sample_rand(18u), sample_rand(19u));
    s.ray_type = ray;
    s.object_color = m.object_color;
    s.object_info = m.object_info;
    return s;
}

fn fragment_surface(in: VsOut, front: bool, prim: u32) -> Surface {
    let d = draws[in.draw];
    let m = meshes[d.mesh];
    g_pixel = vec2<u32>(in.pos.xy);
    g_sample = samples[imm.sample].info.y;
    g_unshadowed = (d.flags & DRAW_UNSHADOWED) != 0u;
    let ng = face_normal(m, tri_vertices(d, m, prim));
    let attr = array<vec4<f32>, 8>(in.a0, in.a1, in.a2, in.a3, in.a4, in.a5, in.a6, in.a7);
    let screen = vec2<f32>(in.pos.x * frame.size.z, 1.0 - in.pos.y * frame.size.w);
    // face_normal follows the winding (the front side); surface_at turns both normals toward the viewer on back faces
    return surface_at(d, in.p, in.n, ng, !front, attr, in.generated, screen, MKR_RAY_CAMERA);
}

// The surface a shadow map's face sees (its opacity decides whether it casts): Light Path says shadow ray.
fn shadow_surface(in: VsOut, front: bool, prim: u32) -> Surface {
    let d = draws[in.draw];
    let m = meshes[d.mesh];
    g_pixel = vec2<u32>(in.pos.xy);
    g_sample = imm.sample;
    let ng = face_normal(m, tri_vertices(d, m, prim));
    let attr = array<vec4<f32>, 8>(in.a0, in.a1, in.a2, in.a3, in.a4, in.a5, in.a6, in.a7);
    return surface_at(d, in.p, in.n, ng, !front, attr, in.generated, vec2<f32>(0.0), MKR_RAY_SHADOW);
}

// ---------------------------------------------------------------------------------------------------- hashed alpha
// The coverage threshold of a dithered surface at world point p for a sample: Wyman and McGuire's hashed alpha test, as
// EEVEE's frames show it. A hash of p's cell on grids a fraction of a pixel wide (the two power-of-two grids around
// frame.scale.w pixel footprints, blended and remapped to stay uniform), turned by the sample's offset. The same point
// gets the same threshold whatever surface it lies on (coplanar layers cover together), layers a little apart
// independent ones, and the noise left after the samples is EEVEE's, alone and where layers overlap. p is the frame's
// own position, so moving surfaces keep theirs.
fn alpha_threshold(p: vec3<f32>, sample: u32) -> f32 {
    let footprint = max(length(dpdx(p)), length(dpdy(p))) * frame.scale.w;
    let scale_log = -log2(max(footprint, 1e-12));
    let lo = floor(scale_log);
    let fac = scale_log - lo;
    let x = mix(cell_hash(p * exp2(lo)), cell_hash(p * exp2(lo + 1.0)), fac);
    // the blend of two uniform values is not uniform: map it through its distribution
    let a = min(fac, 1.0 - fac);
    var t = (x - 0.5 * a) / (1.0 - a);
    if (x < a) {
        t = x * x / (2.0 * a * (1.0 - a));
    } else if (x > 1.0 - a) {
        t = 1.0 - (1.0 - x) * (1.0 - x) / (2.0 * a * (1.0 - a));
    }
    return clamp(fract(t + f32(sample) * 0.6180339887), 1e-6, 1.0);
}

fn cell_hash(q: vec3<f32>) -> f32 {
    let c = vec3<i32>(clamp(floor(q), vec3<f32>(-2e9), vec3<f32>(2e9)));
    return unit(hash3(bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(c.z)));
}

// What a hashed caster leaves in a shadow map where it is partly transparent: its depth in the depths' G (blended by
// max: the nearest) and its transparency (blended by multiplication). Where its alpha reaches SHADOW_OPAQUE it casts
// as an opaque caster (the depth and the depths' R).
const SHADOW_OPAQUE: f32 = 0.998;

struct ShadowOut {
    @location(0) depths: vec4<f32>,
    @location(1) clear: f32,
}

// ------------------------------------------------------------------------------------------------------- G-buffer
// What a G-buffer pass writes: the base record and the motion offset; the extras go to their storage buffer.
struct GOut {
    @location(0) base: vec4<u32>,
    @location(1) motion: vec4<f32>,
}
