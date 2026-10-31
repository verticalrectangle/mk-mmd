// EEVEE Next's virtual shadow maps as measured on its frames (Blender 4.2, probe scenes of hard shadows: the staircase
// of an occluder's edge, plates a known height over a receiver, edges blurred by the filter), shared by the maps' setup
// (shadow_setup.wgsl) and the shadow tracing (shadow_map.wgsl):
// - a local light's map is a cube of 90 degree faces in the light's own axes (area and spot lights: no face behind
//   them), 8192 texels across a face at the finest level, half as many at each coarser one down to level 5; texel
//   edges fall on uv = -1 + 2 k / texels (the light's axis lies on texel corners);
// - a receiver asks for the level floor(log2(fp / t0) + 0.19) (frame.shadow_lod.y), at least floor(log2(m / t0) -
//   0.6) (frame.shadow_lod.z) when its light's resolution limit m says so, within 0..5: fp is the pixel's footprint
//   at its view depth, t0 = 2 a / 8192 the finest texel at its depth a along the face's axis (the level changes where
//   fp / t0 crosses 1.75 x 2^k, the same whatever the light's and the camera's distances);
// - every scale of the shadow follows the texel radius r = max(t, fp, m / sqrt 2) / sqrt 2 (t the level's texel): the
//   receiver's normal offset is r sin(theta) (theta between its normal and the light), the depth tolerance 0.088 r,
//   the filter a disk of shadow_filter_radius x r.
// Faces (light frame: x right, y up, z fwd; s, t the face's uv axes, a its depth axis):
//   0 +z (s +x, t +y)   1 +x (s -z, t +y)   2 -x (s +z, t +y)   3 +y (s +x, t -z)   4 -y (s +x, t +z)   5 -z (s -x, t +y)

const SHADOW_LOD0: f32 = 8192.0;        // texels across a face at the finest level
const SHADOW_LOD_MAX: f32 = 5.0;
const SHADOW_INV_SQRT2: f32 = 0.70710678;

// For a direction (x, y, z) from C in the light's frame (right, up, fwd): the face, its uv in [-1, 1] and the distance
// along the face's axis.
struct FaceHit {
    face: u32,
    uv: vec2<f32>,
    a: f32,
}

fn shadow_face(x: f32, y: f32, z: f32) -> FaceHit {
    let ax = abs(x);
    let ay = abs(y);
    let az = abs(z);
    var f: FaceHit;
    if (az >= ax && az >= ay) {
        f.a = az;
        if (z > 0.0) {
            f.face = 0u;
            f.uv = vec2<f32>(x, y);
        } else {
            f.face = 5u;
            f.uv = vec2<f32>(-x, y);
        }
    } else if (ax >= ay) {
        f.a = ax;
        if (x > 0.0) {
            f.face = 1u;
            f.uv = vec2<f32>(-z, y);
        } else {
            f.face = 2u;
            f.uv = vec2<f32>(z, y);
        }
    } else {
        f.a = ay;
        if (y > 0.0) {
            f.face = 3u;
            f.uv = vec2<f32>(x, -z);
        } else {
            f.face = 4u;
            f.uv = vec2<f32>(x, z);
        }
    }
    f.uv = f.uv / max(f.a, 1e-30);
    return f;
}

// The same for a direction d in the world.
fn shadow_face_of(l: Light, d: vec3<f32>) -> FaceHit {
    return shadow_face(dot(d, l.right.xyz), dot(d, l.up.xyz), dot(d, l.fwd.xyz));
}

// Faces a light's cube has: six for a point light, five (none behind) for area and spot lights.
fn shadow_face_count(l: Light) -> u32 {
    return select(5u, 6u, u32(l.pos.w) == KIND_POINT);
}

// A face's uv in finest-level texels (x right, y down from the face's top row).
fn shadow_texel0(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5) * SHADOW_LOD0;
}

// The footprint of the pixel that sees P (m): its view depth times the pixel's angular size, or an orthographic
// camera's pixel.
fn shadow_footprint(P: vec3<f32>) -> f32 {
    if (frame.cam_pos.w > 0.5) {
        return frame.shadow_lod.x;
    }
    return max(dot(P - frame.cam_pos.xyz, frame.cam_fwd.xyz), 0.0) * frame.shadow_lod.x;
}

// The level a receiver of footprint fp at depth a along its face's axis gets, its light's resolution limit m (0: none).
fn shadow_lod(fp: f32, a: f32, m: f32) -> f32 {
    let t0 = 2.0 * a / SHADOW_LOD0;
    var lod = floor(log2(max(fp, 1e-12) / t0) + frame.shadow_lod.y);
    if (m > 0.0) {
        lod = max(lod, floor(log2(m / t0) - frame.shadow_lod.z));
    }
    return clamp(lod, 0.0, SHADOW_LOD_MAX);
}

// The texel radius every scale of a receiver's shadow follows: texel t of its level, footprint fp, resolution limit m.
fn shadow_radius(t: f32, fp: f32, m: f32) -> f32 {
    return max(max(t, fp), m * SHADOW_INV_SQRT2) * SHADOW_INV_SQRT2;
}
