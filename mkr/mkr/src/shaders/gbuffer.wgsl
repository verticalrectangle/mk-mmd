// The G-buffer: what a material leaves on a pixel for the deferred lighting pass (deferred.wgsl), one record per sample.
// Material modules write it (gbuffer_write); the lighting pass reads it back (gbuffer_read). Colours are shared-exponent
// RGB (9-bit mantissas, 5-bit exponent: 0.2 % steps), unit vectors octahedral (16 bits per axis, ~0.003 degrees).
//
// base (rgba32uint target, every surface pixel):
//   x  diffuse colour (rgb9e5)            y  reflection colour (rgb9e5)
//   z  shading normal N (oct 2 x 16)      w  geometric normal Ng (oct 2 x 8) | sqrt(reflection roughness) (12 bits)
//                                            | flags (4 bits)
// motion (rgba16float target, with motion blur): the shading position (the frame's own time) minus the position the
//   sample's raster put there (depth rebuilds the latter)
// extras (storage, two vec4<u32> per pixel, written and read only when GB_EXTRAS is set):
//   e0: emission (rgb9e5), translucent (rgb9e5), translucent normal (oct), reflection normal (oct)
//   e1: coat (rgb9e5), coat normal (oct), sqrt(coat roughness) (unorm16), 0
// All normals and positions are in glTF's axes.

const GB_SURFACE: u32 = 1u;     // a surface is there (the cleared record says background)
const GB_EXTRAS: u32 = 2u;      // emission, translucent, coat or a reflection normal of its own: read the extras
const GB_SLAB: u32 = 4u;        // its translucency is a slab's (thickness mode SLAB), not a sphere's

fn gb_rgb9e5(c_in: vec3f) -> u32 {
  let c = clamp(c_in, vec3f(0.0), vec3f(65408.0));
  let m = max(max(c.r, c.g), c.b);
  if (m <= 0.0) {
    return 0u;
  }
  // exponent e such that m / 2^(e - 15 - 9) < 512
  var e = max(i32(floor(log2(m))) + 1, -15) + 15;
  var scale = exp2(f32(e - 15 - 9));
  if (u32(floor(m / scale + 0.5)) >= 512u) {
    e = e + 1;
    scale = scale * 2.0;
  }
  let q = min(vec3<u32>(floor(c / scale + 0.5)), vec3<u32>(511u));
  return q.r | (q.g << 9u) | (q.b << 18u) | (u32(e) << 27u);
}

fn gb_from_rgb9e5(v: u32) -> vec3f {
  let scale = exp2(f32(i32(v >> 27u) - 15 - 9));
  return vec3f(f32(v & 511u), f32((v >> 9u) & 511u), f32((v >> 18u) & 511u)) * scale;
}

// Octahedral mapping of a unit vector to [-1, 1]^2 and back.
fn gb_oct(n: vec3f) -> vec2f {
  let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z < 0.0) {
    return (vec2f(1.0) - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
  }
  return p;
}

fn gb_unoct(e: vec2f) -> vec3f {
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n = vec3f(n.xy + select(vec2f(t), vec2f(-t), n.xy >= vec2f(0.0)), n.z);
  return normalize(n);
}

fn gb_pack_n16(n: vec3f) -> u32 {
  return pack2x16snorm(gb_oct(normalize(n)));
}

fn gb_unpack_n16(v: u32) -> vec3f {
  return gb_unoct(unpack2x16snorm(v));
}

fn gb_pack_n8(n: vec3f) -> u32 {
  let e = clamp(gb_oct(normalize(n)), vec2f(-1.0), vec2f(1.0));
  let q = vec2<u32>(floor((e * 0.5 + 0.5) * 255.0 + 0.5));
  return q.x | (q.y << 8u);
}

fn gb_unpack_n8(v: u32) -> vec3f {
  let e = vec2f(f32(v & 255u), f32((v >> 8u) & 255u)) / 255.0 * 2.0 - 1.0;
  return gb_unoct(e);
}

// Roughness as sqrt in 12 bits: steps of ~0.0005 near 0.02, where MMD's glossy lobes live.
fn gb_pack_rough12(r: f32) -> u32 {
  return u32(floor(sqrt(clamp(r, 0.0, 1.0)) * 4095.0 + 0.5));
}

fn gb_unpack_rough12(v: u32) -> f32 {
  let q = f32(v) / 4095.0;
  return q * q;
}

// What the lighting pass reads back (glTF axes).
struct GClosures {
  diffuse: vec3f,
  N: vec3f,
  Ng: vec3f,
  reflection: vec3f,
  reflection_N: vec3f,
  reflection_roughness: f32,
  emission: vec3f,
  translucent: vec3f,
  translucent_N: vec3f,
  coat: vec3f,
  coat_N: vec3f,
  coat_roughness: f32,
}
