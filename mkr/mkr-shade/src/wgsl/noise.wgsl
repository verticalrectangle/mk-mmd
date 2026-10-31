// mkr-shade: hashing and procedural textures (Noise, Voronoi, Wave, Gradient, Checker, Brick, Magic, White Noise).
// Points are packed in vec4 with a dimension count `d` (1D: (w, 0, 0, 0), 2D: (x, y, 0, 0), 3D: (x, y, z, 0),
// 4D: (x, y, z, w)); `d` is a constant at every call site, so drivers fold the switches away after inlining.

// Bob Jenkins' lookup3 hash (public domain), initval 13, as Blender hashes.
fn mks_rot(x: u32, k: u32) -> u32 {
  return (x << k) | (x >> (32u - k));
}

struct MksAbc {
  a: u32,
  b: u32,
  c: u32,
}

fn mks_final(v: MksAbc) -> u32 {
  var a = v.a;
  var b = v.b;
  var c = v.c;
  c = c ^ b; c = c - mks_rot(b, 14u);
  a = a ^ c; a = a - mks_rot(c, 11u);
  b = b ^ a; b = b - mks_rot(a, 25u);
  c = c ^ b; c = c - mks_rot(b, 16u);
  a = a ^ c; a = a - mks_rot(c, 4u);
  b = b ^ a; b = b - mks_rot(a, 14u);
  c = c ^ b; c = c - mks_rot(b, 24u);
  return c;
}

fn mks_hash_u1(kx: u32) -> u32 {
  let i = 0xdeadbeefu + (1u << 2u) + 13u;
  return mks_final(MksAbc(i + kx, i, i));
}

fn mks_hash_u2(kx: u32, ky: u32) -> u32 {
  let i = 0xdeadbeefu + (2u << 2u) + 13u;
  return mks_final(MksAbc(i + kx, i + ky, i));
}

fn mks_hash_u3(kx: u32, ky: u32, kz: u32) -> u32 {
  let i = 0xdeadbeefu + (3u << 2u) + 13u;
  return mks_final(MksAbc(i + kx, i + ky, i + kz));
}

fn mks_hash_u4(kx: u32, ky: u32, kz: u32, kw: u32) -> u32 {
  let i = 0xdeadbeefu + (4u << 2u) + 13u;
  var a = i + kx;
  var b = i + ky;
  var c = i + kz;
  // lookup3 mix(a, b, c)
  a = a - c; a = a ^ mks_rot(c, 4u); c = c + b;
  b = b - a; b = b ^ mks_rot(a, 6u); a = a + c;
  c = c - b; c = c ^ mks_rot(b, 8u); b = b + a;
  a = a - c; a = a ^ mks_rot(c, 16u); c = c + b;
  b = b - a; b = b ^ mks_rot(a, 19u); a = a + c;
  c = c - b; c = c ^ mks_rot(b, 4u); b = b + a;
  a = a + kw;
  return mks_final(MksAbc(a, b, c));
}

fn mks_u2f(h: u32) -> f32 {
  return f32(h) / 4294967295.0;
}

fn mks_hash_f1(k: f32) -> f32 {
  return mks_u2f(mks_hash_u1(bitcast<u32>(k)));
}

fn mks_hash_f2(k: vec2f) -> f32 {
  return mks_u2f(mks_hash_u2(bitcast<u32>(k.x), bitcast<u32>(k.y)));
}

fn mks_hash_f3(k: vec3f) -> f32 {
  return mks_u2f(mks_hash_u3(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z)));
}

fn mks_hash_f4(k: vec4f) -> f32 {
  return mks_u2f(mks_hash_u4(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w)));
}

// Float key of dimension d -> float.
fn mks_hash_fd(k: vec4f, d: u32) -> f32 {
  switch d {
    case 1u: { return mks_hash_f1(k.x); }
    case 2u: { return mks_hash_f2(k.xy); }
    case 3u: { return mks_hash_f3(k.xyz); }
    default: { return mks_hash_f4(k); }
  }
}

// Float key of dimension d -> colour (Blender's hash_*_to_vec3 for that dimension).
fn mks_hash_color(k: vec4f, d: u32) -> vec3f {
  switch d {
    case 1u: {
      return vec3f(mks_hash_f1(k.x), mks_hash_f2(vec2f(k.x, 1.0)), mks_hash_f2(vec2f(k.x, 2.0)));
    }
    case 2u: {
      return vec3f(mks_hash_f2(k.xy), mks_hash_f3(vec3f(k.xy, 1.0)), mks_hash_f3(vec3f(k.xy, 2.0)));
    }
    case 3u: {
      return vec3f(mks_hash_f3(k.xyz), mks_hash_f4(vec4f(k.xyz, 1.0)), mks_hash_f4(vec4f(k.xyz, 2.0)));
    }
    default: {
      return vec3f(mks_hash_f4(k), mks_hash_f4(k.zxwy), mks_hash_f4(k.wzyx));
    }
  }
}

// Float key of dimension d -> point offset in the cell (Blender's hash_*_to_vec<d>).
fn mks_hash_point(k: vec4f, d: u32) -> vec4f {
  switch d {
    case 1u: {
      return vec4f(mks_hash_f1(k.x), 0.0, 0.0, 0.0);
    }
    case 2u: {
      return vec4f(mks_hash_f2(k.xy), mks_hash_f3(vec3f(k.xy, 1.0)), 0.0, 0.0);
    }
    case 3u: {
      return vec4f(mks_hash_f3(k.xyz), mks_hash_f4(vec4f(k.xyz, 1.0)), mks_hash_f4(vec4f(k.xyz, 2.0)), 0.0);
    }
    default: {
      return vec4f(mks_hash_f4(k), mks_hash_f4(k.wxyz), mks_hash_f4(k.zwxy), mks_hash_f4(k.yzwx));
    }
  }
}

// ------------------------------------------------------------------------------------------------ Perlin noise
fn mks_fade(t: f32) -> f32 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn mks_negate_if(v: f32, c: u32) -> f32 {
  return select(v, -v, c != 0u);
}

fn mks_grad1(h: u32, x: f32) -> f32 {
  let hh = h & 15u;
  let g = 1.0 + f32(hh & 7u);
  return mks_negate_if(g, hh & 8u) * x;
}

fn mks_grad2(h: u32, x: f32, y: f32) -> f32 {
  let hh = h & 7u;
  let u = select(y, x, hh < 4u);
  let v = 2.0 * select(x, y, hh < 4u);
  return mks_negate_if(u, hh & 1u) + mks_negate_if(v, hh & 2u);
}

fn mks_grad3(h: u32, x: f32, y: f32, z: f32) -> f32 {
  let hh = h & 15u;
  let u = select(y, x, hh < 8u);
  let vt = select(z, x, hh == 12u || hh == 14u);
  let v = select(vt, y, hh < 4u);
  return mks_negate_if(u, hh & 1u) + mks_negate_if(v, hh & 2u);
}

fn mks_grad4(h: u32, x: f32, y: f32, z: f32, w: f32) -> f32 {
  let hh = h & 31u;
  let u = select(y, x, hh < 24u);
  let v = select(z, y, hh < 16u);
  let s = select(w, z, hh < 8u);
  return mks_negate_if(u, hh & 1u) + mks_negate_if(v, hh & 2u) + mks_negate_if(s, hh & 4u);
}

fn mks_bi_mix(v0: f32, v1: f32, v2: f32, v3: f32, x: f32, y: f32) -> f32 {
  let x1 = 1.0 - x;
  return (1.0 - y) * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x);
}

fn mks_tri_mix(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32, x: f32, y: f32,
               z: f32) -> f32 {
  let x1 = 1.0 - x;
  let y1 = 1.0 - y;
  let z1 = 1.0 - z;
  return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +
         z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

fn mks_perlin1(p: f32) -> f32 {
  let fl = floor(p);
  let X = bitcast<u32>(i32(fl));
  let fx = p - fl;
  let u = mks_fade(fx);
  return mix(mks_grad1(mks_hash_u1(X), fx), mks_grad1(mks_hash_u1(X + 1u), fx - 1.0), u);
}

fn mks_perlin2(p: vec2f) -> f32 {
  let fl = floor(p);
  let X = bitcast<u32>(i32(fl.x));
  let Y = bitcast<u32>(i32(fl.y));
  let f = p - fl;
  let u = mks_fade(f.x);
  let v = mks_fade(f.y);
  return mks_bi_mix(mks_grad2(mks_hash_u2(X, Y), f.x, f.y),
                    mks_grad2(mks_hash_u2(X + 1u, Y), f.x - 1.0, f.y),
                    mks_grad2(mks_hash_u2(X, Y + 1u), f.x, f.y - 1.0),
                    mks_grad2(mks_hash_u2(X + 1u, Y + 1u), f.x - 1.0, f.y - 1.0), u, v);
}

fn mks_perlin3(p: vec3f) -> f32 {
  let fl = floor(p);
  let X = bitcast<u32>(i32(fl.x));
  let Y = bitcast<u32>(i32(fl.y));
  let Z = bitcast<u32>(i32(fl.z));
  let f = p - fl;
  let u = mks_fade(f.x);
  let v = mks_fade(f.y);
  let w = mks_fade(f.z);
  return mks_tri_mix(mks_grad3(mks_hash_u3(X, Y, Z), f.x, f.y, f.z),
                     mks_grad3(mks_hash_u3(X + 1u, Y, Z), f.x - 1.0, f.y, f.z),
                     mks_grad3(mks_hash_u3(X, Y + 1u, Z), f.x, f.y - 1.0, f.z),
                     mks_grad3(mks_hash_u3(X + 1u, Y + 1u, Z), f.x - 1.0, f.y - 1.0, f.z),
                     mks_grad3(mks_hash_u3(X, Y, Z + 1u), f.x, f.y, f.z - 1.0),
                     mks_grad3(mks_hash_u3(X + 1u, Y, Z + 1u), f.x - 1.0, f.y, f.z - 1.0),
                     mks_grad3(mks_hash_u3(X, Y + 1u, Z + 1u), f.x, f.y - 1.0, f.z - 1.0),
                     mks_grad3(mks_hash_u3(X + 1u, Y + 1u, Z + 1u), f.x - 1.0, f.y - 1.0, f.z - 1.0), u, v, w);
}

fn mks_perlin4_half(X: u32, Y: u32, Z: u32, W: u32, f: vec4f, u: f32, v: f32, t: f32) -> f32 {
  return mks_tri_mix(mks_grad4(mks_hash_u4(X, Y, Z, W), f.x, f.y, f.z, f.w),
                     mks_grad4(mks_hash_u4(X + 1u, Y, Z, W), f.x - 1.0, f.y, f.z, f.w),
                     mks_grad4(mks_hash_u4(X, Y + 1u, Z, W), f.x, f.y - 1.0, f.z, f.w),
                     mks_grad4(mks_hash_u4(X + 1u, Y + 1u, Z, W), f.x - 1.0, f.y - 1.0, f.z, f.w),
                     mks_grad4(mks_hash_u4(X, Y, Z + 1u, W), f.x, f.y, f.z - 1.0, f.w),
                     mks_grad4(mks_hash_u4(X + 1u, Y, Z + 1u, W), f.x - 1.0, f.y, f.z - 1.0, f.w),
                     mks_grad4(mks_hash_u4(X, Y + 1u, Z + 1u, W), f.x, f.y - 1.0, f.z - 1.0, f.w),
                     mks_grad4(mks_hash_u4(X + 1u, Y + 1u, Z + 1u, W), f.x - 1.0, f.y - 1.0, f.z - 1.0, f.w),
                     u, v, t);
}

fn mks_perlin4(p: vec4f) -> f32 {
  let fl = floor(p);
  let X = bitcast<u32>(i32(fl.x));
  let Y = bitcast<u32>(i32(fl.y));
  let Z = bitcast<u32>(i32(fl.z));
  let W = bitcast<u32>(i32(fl.w));
  let f = p - fl;
  let u = mks_fade(f.x);
  let v = mks_fade(f.y);
  let t = mks_fade(f.z);
  let s = mks_fade(f.w);
  let lo = mks_perlin4_half(X, Y, Z, W, f, u, v, t);
  let hi = mks_perlin4_half(X, Y, Z, W + 1u, f - vec4f(0.0, 0.0, 0.0, 1.0), u, v, t);
  return mix(lo, hi, s);
}

fn mks_finite(x: f32) -> f32 {
  // Blender returns 0 where the noise overflows.
  return select(0.0, x, abs(x) <= MKS_FLT_MAX);
}

// Signed noise, -1..1 roughly.
fn mks_snoise(p: vec4f, d: u32) -> f32 {
  switch d {
    case 1u: { return 0.2500 * mks_finite(mks_perlin1(p.x)); }
    case 2u: { return 0.6616 * mks_finite(mks_perlin2(p.xy)); }
    case 3u: { return 0.9820 * mks_finite(mks_perlin3(p.xyz)); }
    default: { return 0.8344 * mks_finite(mks_perlin4(p)); }
  }
}

fn mks_noise01(p: vec4f, d: u32) -> f32 {
  return 0.5 * mks_snoise(p, d) + 0.5;
}

// Blender's random offsets that decorrelate the channels of a noise texture.
fn mks_random_offset(seed: f32, d: u32) -> vec4f {
  switch d {
    case 1u: {
      return vec4f(100.0 + mks_hash_f1(seed) * 100.0, 0.0, 0.0, 0.0);
    }
    default: {
      var o = vec4f(0.0);
      o.x = 100.0 + mks_hash_f2(vec2f(seed, 0.0)) * 100.0;
      o.y = 100.0 + mks_hash_f2(vec2f(seed, 1.0)) * 100.0;
      if (d >= 3u) {
        o.z = 100.0 + mks_hash_f2(vec2f(seed, 2.0)) * 100.0;
      }
      if (d >= 4u) {
        o.w = 100.0 + mks_hash_f2(vec2f(seed, 3.0)) * 100.0;
      }
      return o;
    }
  }
}

fn mks_noise_fbm(co: vec4f, d: u32, detail: f32, roughness: f32, lacunarity: f32, normalize: bool) -> f32 {
  var fscale = 1.0;
  var amp = 1.0;
  var maxamp = 0.0;
  var sum = 0.0;
  let n = i32(detail);
  for (var i = 0; i <= n; i++) {
    let t = mks_snoise(fscale * co, d);
    sum += t * amp;
    maxamp += amp;
    amp *= roughness;
    fscale *= lacunarity;
  }
  let rmd = detail - floor(detail);
  if (rmd != 0.0) {
    let t = mks_snoise(fscale * co, d);
    let sum2 = sum + t * amp;
    if (normalize) {
      return mix(0.5 * sum / maxamp + 0.5, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd);
    }
    return mix(sum, sum2, rmd);
  }
  if (normalize) {
    return 0.5 * sum / maxamp + 0.5;
  }
  return sum;
}

fn mks_noise_multi_fractal(co: vec4f, d: u32, detail: f32, roughness: f32, lacunarity: f32) -> f32 {
  var p = co;
  var value = 1.0;
  var pwr = 1.0;
  let n = i32(detail);
  for (var i = 0; i <= n; i++) {
    value *= (pwr * mks_snoise(p, d) + 1.0);
    pwr *= roughness;
    p *= lacunarity;
  }
  let rmd = detail - floor(detail);
  if (rmd != 0.0) {
    value *= (rmd * pwr * mks_snoise(p, d) + 1.0);
  }
  return value;
}

fn mks_noise_hetero_terrain(co: vec4f, d: u32, detail: f32, roughness: f32, lacunarity: f32, offset: f32) -> f32 {
  var p = co;
  var pwr = roughness;
  var value = offset + mks_snoise(p, d);
  p *= lacunarity;
  let n = i32(detail);
  for (var i = 1; i <= n; i++) {
    let increment = (mks_snoise(p, d) + offset) * pwr * value;
    value += increment;
    pwr *= roughness;
    p *= lacunarity;
  }
  let rmd = detail - floor(detail);
  if (rmd != 0.0) {
    let increment = (mks_snoise(p, d) + offset) * pwr * value;
    value += rmd * increment;
  }
  return value;
}

fn mks_noise_hybrid_multi_fractal(co: vec4f, d: u32, detail: f32, roughness: f32, lacunarity: f32, offset: f32,
                                  gain: f32) -> f32 {
  var p = co;
  var pwr = 1.0;
  var value = 0.0;
  var weight = 1.0;
  let n = i32(detail);
  for (var i = 0; (weight > 0.001) && (i <= n); i++) {
    if (weight > 1.0) {
      weight = 1.0;
    }
    let signal = (mks_snoise(p, d) + offset) * pwr;
    pwr *= roughness;
    value += weight * signal;
    weight *= gain * signal;
    p *= lacunarity;
  }
  let rmd = detail - floor(detail);
  if ((rmd != 0.0) && (weight > 0.001)) {
    if (weight > 1.0) {
      weight = 1.0;
    }
    let signal = (mks_snoise(p, d) + offset) * pwr;
    value += rmd * weight * signal;
  }
  return value;
}

fn mks_noise_ridged_multi_fractal(co: vec4f, d: u32, detail: f32, roughness: f32, lacunarity: f32, offset: f32,
                                  gain: f32) -> f32 {
  var p = co;
  var pwr = roughness;
  var signal = offset - abs(mks_snoise(p, d));
  signal *= signal;
  var value = signal;
  var weight = 1.0;
  let n = i32(detail);
  for (var i = 1; i <= n; i++) {
    p *= lacunarity;
    weight = clamp(signal * gain, 0.0, 1.0);
    signal = offset - abs(mks_snoise(p, d));
    signal *= signal;
    signal *= weight;
    value += signal * pwr;
    pwr *= roughness;
  }
  return value;
}

// Noise texture types (ShaderNodeTexNoise.noise_type).
const MKS_NOISE_MULTIFRACTAL: u32 = 0u;
const MKS_NOISE_RIDGED: u32 = 1u;
const MKS_NOISE_HYBRID: u32 = 2u;
const MKS_NOISE_FBM: u32 = 3u;
const MKS_NOISE_HETERO: u32 = 4u;

fn mks_noise_select(p: vec4f, d: u32, kind: u32, detail: f32, roughness: f32, lacunarity: f32, offset: f32,
                    gain: f32, normalize: bool) -> f32 {
  switch kind {
    case 0u: { return mks_noise_multi_fractal(p, d, detail, roughness, lacunarity); }
    case 1u: { return mks_noise_ridged_multi_fractal(p, d, detail, roughness, lacunarity, offset, gain); }
    case 2u: { return mks_noise_hybrid_multi_fractal(p, d, detail, roughness, lacunarity, offset, gain); }
    case 4u: { return mks_noise_hetero_terrain(p, d, detail, roughness, lacunarity, offset); }
    default: { return mks_noise_fbm(p, d, detail, roughness, lacunarity, normalize); }
  }
}

// Noise Texture node -> (Fac, Color.rgb).
fn mks_tex_noise(co: vec4f, d: u32, kind: u32, scale: f32, detail_in: f32, roughness_in: f32, lacunarity: f32,
                 offset: f32, gain: f32, distortion: f32, normalize: bool) -> vec4f {
  let detail = clamp(detail_in, 0.0, 15.0);
  let roughness = max(roughness_in, 0.0);
  var p = co * scale;
  if (distortion != 0.0) {
    var dp = vec4f(0.0);
    dp.x = mks_snoise(p + mks_random_offset(0.0, d), d) * distortion;
    if (d >= 2u) {
      dp.y = mks_snoise(p + mks_random_offset(1.0, d), d) * distortion;
    }
    if (d >= 3u) {
      dp.z = mks_snoise(p + mks_random_offset(2.0, d), d) * distortion;
    }
    if (d >= 4u) {
      dp.w = mks_snoise(p + mks_random_offset(3.0, d), d) * distortion;
    }
    p += dp;
  }
  let value = mks_noise_select(p, d, kind, detail, roughness, lacunarity, offset, gain, normalize);
  let s1 = f32(d);
  let g = mks_noise_select(p + mks_random_offset(s1, d), d, kind, detail, roughness, lacunarity, offset, gain,
                           normalize);
  let b = mks_noise_select(p + mks_random_offset(s1 + 1.0, d), d, kind, detail, roughness, lacunarity, offset, gain,
                           normalize);
  return vec4f(value, value, g, b);
}

// ------------------------------------------------------------------------------------------------ Voronoi
struct MksVoronoi {
  distance: f32,
  color: vec3f,
  position: vec4f,      // (x, y, z, w) of the feature point (1D: w)
}

fn mks_vor_dist(a: vec4f, b: vec4f, d: u32, metric: u32, exponent: f32) -> f32 {
  var m = vec4f(1.0, 0.0, 0.0, 0.0);
  if (d >= 2u) { m.y = 1.0; }
  if (d >= 3u) { m.z = 1.0; }
  if (d >= 4u) { m.w = 1.0; }
  let dd = abs(a - b) * m;
  switch metric {
    case 1u: { return dd.x + dd.y + dd.z + dd.w; }
    case 2u: { return max(max(dd.x, dd.y), max(dd.z, dd.w)); }
    case 3u: {
      var s = pow(dd.x, exponent);
      if (d >= 2u) { s += pow(dd.y, exponent); }
      if (d >= 3u) { s += pow(dd.z, exponent); }
      if (d >= 4u) { s += pow(dd.w, exponent); }
      return pow(s, 1.0 / exponent);
    }
    default: { return length(dd); }
  }
}

// 1D points live in x of the packed vector; Blender reports them in W.
fn mks_vor_position(p: vec4f, d: u32) -> vec4f {
  if (d == 1u) {
    return vec4f(0.0, 0.0, 0.0, p.x);
  }
  return p;
}

fn mks_vor_extent(d: u32) -> vec4i {
  return vec4i(1, select(0, 1, d >= 2u), select(0, 1, d >= 3u), select(0, 1, d >= 4u));
}

fn mks_voronoi_f1(co: vec4f, d: u32, metric: u32, exponent: f32, randomness: f32) -> MksVoronoi {
  let cell = floor(co);
  let local = co - cell;
  let e = mks_vor_extent(d);
  var min_d = MKS_FLT_MAX;
  var target_off = vec4f(0.0);
  var target_pos = vec4f(0.0);
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          let off = vec4f(f32(i), f32(j), f32(k), f32(l));
          let pt = off + mks_hash_point(cell + off, d) * randomness;
          let dist = mks_vor_dist(pt, local, d, metric, exponent);
          if (dist < min_d) {
            target_off = off;
            min_d = dist;
            target_pos = pt;
          }
        }
      }
    }
  }
  return MksVoronoi(min_d, mks_hash_color(cell + target_off, d), mks_vor_position(target_pos + cell, d));
}

fn mks_voronoi_smooth_f1(co: vec4f, d: u32, metric: u32, exponent: f32, randomness: f32, smoothness: f32)
    -> MksVoronoi {
  let cell = floor(co);
  let local = co - cell;
  let e = 2 * mks_vor_extent(d);
  var sd = 0.0;
  var sc = vec3f(0.0);
  var sp = vec4f(0.0);
  var h = -1.0;
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          let off = vec4f(f32(i), f32(j), f32(k), f32(l));
          let pt = off + mks_hash_point(cell + off, d) * randomness;
          let dist = mks_vor_dist(pt, local, d, metric, exponent);
          if (h == -1.0) {
            h = 1.0;
          } else {
            h = smoothstep(0.0, 1.0, 0.5 + 0.5 * (sd - dist) / smoothness);
          }
          var corr = smoothness * h * (1.0 - h);
          sd = mix(sd, dist, h) - corr;
          corr /= 1.0 + 3.0 * smoothness;
          let cc = mks_hash_color(cell + off, d);
          sc = mix(sc, cc, h) - vec3f(corr);
          sp = mix(sp, pt, h) - vec4f(corr);
        }
      }
    }
  }
  return MksVoronoi(sd, sc, mks_vor_position(cell + sp, d));
}

fn mks_voronoi_f2(co: vec4f, d: u32, metric: u32, exponent: f32, randomness: f32) -> MksVoronoi {
  let cell = floor(co);
  let local = co - cell;
  let e = mks_vor_extent(d);
  var d1 = MKS_FLT_MAX;
  var d2 = MKS_FLT_MAX;
  var o1 = vec4f(0.0);
  var p1 = vec4f(0.0);
  var o2 = vec4f(0.0);
  var p2 = vec4f(0.0);
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          let off = vec4f(f32(i), f32(j), f32(k), f32(l));
          let pt = off + mks_hash_point(cell + off, d) * randomness;
          let dist = mks_vor_dist(pt, local, d, metric, exponent);
          if (dist < d1) {
            d2 = d1;
            d1 = dist;
            o2 = o1;
            o1 = off;
            p2 = p1;
            p1 = pt;
          } else if (dist < d2) {
            d2 = dist;
            o2 = off;
            p2 = pt;
          }
        }
      }
    }
  }
  return MksVoronoi(d2, mks_hash_color(cell + o2, d), mks_vor_position(p2 + cell, d));
}

fn mks_voronoi_distance_to_edge(co: vec4f, d: u32, randomness: f32) -> f32 {
  let cell = floor(co);
  let local = co - cell;
  if (d == 1u) {
    let mid = mks_hash_f1(cell.x) * randomness;
    let left = -1.0 + mks_hash_f1(cell.x - 1.0) * randomness;
    let right = 1.0 + mks_hash_f1(cell.x + 1.0) * randomness;
    let dl = abs((mid + left) / 2.0 - local.x);
    let dr = abs((mid + right) / 2.0 - local.x);
    return min(dl, dr);
  }
  let e = mks_vor_extent(d);
  var to_closest = vec4f(0.0);
  var min_d = 8.0;
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          let off = vec4f(f32(i), f32(j), f32(k), f32(l));
          let to_pt = off + mks_hash_point(cell + off, d) * randomness - local;
          let dist = dot(to_pt, to_pt);
          if (dist < min_d) {
            min_d = dist;
            to_closest = to_pt;
          }
        }
      }
    }
  }
  min_d = 8.0;
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          let off = vec4f(f32(i), f32(j), f32(k), f32(l));
          let to_pt = off + mks_hash_point(cell + off, d) * randomness - local;
          let perp = to_pt - to_closest;
          if (dot(perp, perp) > 0.0001) {
            let de = dot((to_closest + to_pt) / 2.0, normalize(perp));
            min_d = min(min_d, de);
          }
        }
      }
    }
  }
  return min_d;
}

fn mks_voronoi_n_sphere_radius(co: vec4f, d: u32, randomness: f32) -> f32 {
  let cell = floor(co);
  let local = co - cell;
  let e = mks_vor_extent(d);
  var closest = vec4f(0.0);
  var closest_off = vec4f(0.0);
  var min_d = MKS_FLT_MAX;
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          let off = vec4f(f32(i), f32(j), f32(k), f32(l));
          let pt = off + mks_hash_point(cell + off, d) * randomness;
          let dist = distance(pt, local);
          if (dist < min_d) {
            min_d = dist;
            closest = pt;
            closest_off = off;
          }
        }
      }
    }
  }
  min_d = MKS_FLT_MAX;
  var cc = vec4f(0.0);
  for (var l = -e.w; l <= e.w; l++) {
    for (var k = -e.z; k <= e.z; k++) {
      for (var j = -e.y; j <= e.y; j++) {
        for (var i = -e.x; i <= e.x; i++) {
          if (i == 0 && j == 0 && k == 0 && l == 0) {
            continue;
          }
          let off = vec4f(f32(i), f32(j), f32(k), f32(l)) + closest_off;
          let pt = off + mks_hash_point(cell + off, d) * randomness;
          let dist = distance(closest, pt);
          if (dist < min_d) {
            min_d = dist;
            cc = pt;
          }
        }
      }
    }
  }
  return distance(cc, closest) / 2.0;
}

// Voronoi features (ShaderNodeTexVoronoi.feature).
const MKS_VOR_F1: u32 = 0u;
const MKS_VOR_F2: u32 = 1u;
const MKS_VOR_SMOOTH_F1: u32 = 2u;
const MKS_VOR_DISTANCE_TO_EDGE: u32 = 3u;
const MKS_VOR_N_SPHERE_RADIUS: u32 = 4u;

fn mks_voronoi_octave(co: vec4f, d: u32, feature: u32, metric: u32, exponent: f32, randomness: f32,
                      smoothness: f32) -> MksVoronoi {
  if (feature == MKS_VOR_F2) {
    return mks_voronoi_f2(co, d, metric, exponent, randomness);
  }
  if (feature == MKS_VOR_SMOOTH_F1 && smoothness != 0.0) {
    return mks_voronoi_smooth_f1(co, d, metric, exponent, randomness, smoothness);
  }
  return mks_voronoi_f1(co, d, metric, exponent, randomness);
}

// Voronoi Texture node. Returns distance, colour, position (w in .position.w); radius in .color.x for
// N_SPHERE_RADIUS.
fn mks_tex_voronoi(co_in: vec4f, d: u32, feature: u32, metric: u32, scale: f32, detail_in: f32, roughness_in: f32,
                   lacunarity: f32, smoothness_in: f32, exponent: f32, randomness_in: f32, normalize: bool)
    -> MksVoronoi {
  let detail = clamp(detail_in, 0.0, 15.0);
  let roughness = clamp(roughness_in, 0.0, 1.0);
  let smoothness = clamp(smoothness_in / 2.0, 0.0, 0.5);
  let randomness = clamp(randomness_in, 0.0, 1.0);
  let co = co_in * scale;
  if (feature == MKS_VOR_N_SPHERE_RADIUS) {
    let r = mks_voronoi_n_sphere_radius(co, d, randomness);
    return MksVoronoi(0.0, vec3f(r, 0.0, 0.0), vec4f(0.0));
  }
  var maxd = mks_vor_dist(vec4f(0.0), vec4f(0.5 + 0.5 * randomness), d, metric, exponent);
  if (feature == MKS_VOR_F2) {
    maxd = maxd * 2.0;
  }
  let zero_input = detail == 0.0 || roughness == 0.0;
  if (feature == MKS_VOR_DISTANCE_TO_EDGE) {
    let edge_max = 0.5 + 0.5 * randomness;
    var amplitude = 1.0;
    var max_amplitude = edge_max;
    var sc = 1.0;
    var dist = 8.0;
    let n = i32(ceil(detail));
    for (var i = 0; i <= n; i++) {
      let od = mks_voronoi_distance_to_edge(co * sc, d, randomness);
      if (zero_input) {
        dist = od;
        break;
      } else if (f32(i) <= detail) {
        max_amplitude = mix(max_amplitude, edge_max / sc, amplitude);
        dist = mix(dist, min(dist, od / sc), amplitude);
        sc *= lacunarity;
        amplitude *= roughness;
      } else {
        let rem = detail - floor(detail);
        if (rem != 0.0) {
          let la = mix(max_amplitude, edge_max / sc, amplitude);
          max_amplitude = mix(max_amplitude, la, rem);
          let ld = mix(dist, min(dist, od / sc), amplitude);
          dist = mix(dist, min(dist, ld), rem);
        }
      }
    }
    if (normalize) {
      dist /= max_amplitude;
    }
    return MksVoronoi(dist, vec3f(0.0), vec4f(0.0));
  }
  var amplitude = 1.0;
  var max_amplitude = 0.0;
  var sc = 1.0;
  var out = MksVoronoi(0.0, vec3f(0.0), vec4f(0.0));
  let n = i32(ceil(detail));
  for (var i = 0; i <= n; i++) {
    let o = mks_voronoi_octave(co * sc, d, feature, metric, exponent, randomness, smoothness);
    if (zero_input) {
      max_amplitude = 1.0;
      out = o;
      break;
    } else if (f32(i) <= detail) {
      max_amplitude += amplitude;
      out.distance += o.distance * amplitude;
      out.color += o.color * amplitude;
      out.position = mix(out.position, o.position / sc, amplitude);
      sc *= lacunarity;
      amplitude *= roughness;
    } else {
      let rem = detail - floor(detail);
      if (rem != 0.0) {
        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, rem);
        out.distance = mix(out.distance, out.distance + o.distance * amplitude, rem);
        out.color = mix(out.color, out.color + o.color * amplitude, rem);
        out.position = mix(out.position, mix(out.position, o.position / sc, amplitude), rem);
      }
    }
  }
  if (normalize) {
    out.distance /= max_amplitude * maxd;
    out.color /= max_amplitude;
  }
  out.position = vec4f(mks_safe_div(out.position.x, scale), mks_safe_div(out.position.y, scale),
                       mks_safe_div(out.position.z, scale), mks_safe_div(out.position.w, scale));
  return out;
}

// ------------------------------------------------------------------------------------------------ Wave
// wave_type 0 bands 1 rings; direction 0 X 1 Y 2 Z 3 diagonal / spherical; profile 0 sin 1 saw 2 tri.
fn mks_tex_wave(co: vec3f, scale: f32, distortion: f32, detail: f32, detail_scale: f32, detail_roughness: f32,
                phase: f32, wave_type: u32, direction: u32, profile: u32) -> f32 {
  let p = (co * scale + 0.000001) * 0.999999;
  var n = 0.0;
  if (wave_type == 0u) {
    if (direction == 0u) {
      n = p.x * 20.0;
    } else if (direction == 1u) {
      n = p.y * 20.0;
    } else if (direction == 2u) {
      n = p.z * 20.0;
    } else {
      n = (p.x + p.y + p.z) * 10.0;
    }
  } else {
    var rp = p;
    if (direction == 0u) {
      rp *= vec3f(0.0, 1.0, 1.0);
    } else if (direction == 1u) {
      rp *= vec3f(1.0, 0.0, 1.0);
    } else if (direction == 2u) {
      rp *= vec3f(1.0, 1.0, 0.0);
    }
    n = length(rp) * 20.0;
  }
  n += phase;
  if (distortion != 0.0) {
    n += distortion * (mks_noise_fbm(vec4f(p * detail_scale, 0.0), 3u, detail, detail_roughness, 2.0, true) * 2.0 - 1.0);
  }
  if (profile == 0u) {
    return 0.5 + 0.5 * sin(n - MKS_PI * 0.5);
  }
  if (profile == 1u) {
    let m = n / MKS_TAU;
    return m - floor(m);
  }
  let m = n / MKS_TAU;
  return abs(m - floor(m + 0.5)) * 2.0;
}

// ------------------------------------------------------------------------------------------------ Gradient
// 0 linear 1 quadratic 2 easing 3 diagonal 4 radial 5 quadratic sphere 6 spherical.
fn mks_tex_gradient(p: vec3f, kind: u32) -> f32 {
  var f = 0.0;
  switch kind {
    case 0u: { f = p.x; }
    case 1u: {
      let r = max(p.x, 0.0);
      f = r * r;
    }
    case 2u: {
      let r = min(max(p.x, 0.0), 1.0);
      let t = r * r;
      f = 3.0 * t - 2.0 * t * r;
    }
    case 3u: { f = (p.x + p.y) * 0.5; }
    case 4u: { f = atan2(p.y, p.x) / (MKS_PI * 2.0) + 0.5; }
    default: {
      let r = max(0.999999 - sqrt(p.x * p.x + p.y * p.y + p.z * p.z), 0.0);
      f = select(r, r * r, kind == 5u);
    }
  }
  return clamp(f, 0.0, 1.0);
}

// ------------------------------------------------------------------------------------------------ Checker
fn mks_tex_checker(co: vec3f, scale: f32) -> f32 {
  let p = (co * scale + 0.000001) * 0.999999;
  let xi = i32(abs(floor(p.x)));
  let yi = i32(abs(floor(p.y)));
  let zi = i32(abs(floor(p.z)));
  let check = ((xi % 2) == (yi % 2)) == ((zi % 2) != 0);
  return select(0.0, 1.0, check);
}

// ------------------------------------------------------------------------------------------------ Brick
fn mks_integer_noise(n: i32) -> f32 {
  var nn = (bitcast<u32>(n) + 1013u) & 0x7fffffffu;
  nn = (nn >> 13u) ^ nn;
  nn = (nn * (nn * nn * 60493u + 19990303u) + 1376312589u) & 0x7fffffffu;
  return 0.5 * (f32(nn) / 1073741824.0);
}

// -> (tint, mortar factor). The row parity tests are unsigned modulos of the (possibly negative) row number, as
// EEVEE computes them on the GPU (measured: lab brick cases).
fn mks_brick(p: vec3f, mortar_size: f32, mortar_smooth: f32, bias: f32, brick_width_in: f32, row_height: f32,
             offset_amount: f32, offset_frequency: i32, squash_amount: f32, squash_frequency: i32) -> vec2f {
  var brick_width = brick_width_in;
  var offset = 0.0;
  let rownum = i32(floor(p.y / row_height));
  if (offset_frequency != 0 && squash_frequency != 0) {
    let row = bitcast<u32>(rownum);
    brick_width *= select(squash_amount, 1.0, (row % u32(squash_frequency)) != 0u);
    offset = select(brick_width * offset_amount, 0.0, (row % u32(offset_frequency)) != 0u);
  }
  let bricknum = i32(floor((p.x + offset) / brick_width));
  let x = (p.x + offset) - brick_width * f32(bricknum);
  let y = p.y - row_height * f32(rownum);
  let tint = clamp(mks_integer_noise((rownum << 16u) + (bricknum & 0xFFFF)) + bias, 0.0, 1.0);
  var min_dist = min(min(x, y), min(brick_width - x, row_height - y));
  if (min_dist >= mortar_size) {
    return vec2f(tint, 0.0);
  }
  if (mortar_smooth == 0.0) {
    return vec2f(tint, 1.0);
  }
  min_dist = 1.0 - min_dist / mortar_size;
  return vec2f(tint, smoothstep(0.0, mortar_smooth, min_dist));
}

// ------------------------------------------------------------------------------------------------ Magic
fn mks_tex_magic(co: vec3f, scale: f32, distortion_in: f32, depth: i32) -> vec3f {
  let p = (co * scale) % vec3f(MKS_TAU);
  var distortion = distortion_in;
  var x = sin((p.x + p.y + p.z) * 5.0);
  var y = cos((-p.x + p.y - p.z) * 5.0);
  var z = -cos((-p.x - p.y + p.z) * 5.0);
  if (depth > 0) {
    x *= distortion;
    y *= distortion;
    z *= distortion;
    y = -cos(x - y + z);
    y *= distortion;
    if (depth > 1) {
      x = cos(x - y - z);
      x *= distortion;
      if (depth > 2) {
        z = sin(-x - y - z);
        z *= distortion;
        if (depth > 3) {
          x = -cos(-x + y - z);
          x *= distortion;
          if (depth > 4) {
            y = -sin(-x + y + z);
            y *= distortion;
            if (depth > 5) {
              y = -cos(-x + y + z);
              y *= distortion;
              if (depth > 6) {
                x = cos(x + y + z);
                x *= distortion;
                if (depth > 7) {
                  z = sin(x + y - z);
                  z *= distortion;
                  if (depth > 8) {
                    x = -cos(-x - y + z);
                    x *= distortion;
                    if (depth > 9) {
                      y = -sin(x - y + z);
                      y *= distortion;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (distortion != 0.0) {
    distortion *= 2.0;
    x /= distortion;
    y /= distortion;
    z /= distortion;
  }
  return vec3f(0.5 - x, 0.5 - y, 0.5 - z);
}

// ------------------------------------------------------------------------------------------------ White noise
fn mks_tex_white_noise(p: vec4f, d: u32) -> vec4f {
  return vec4f(mks_hash_fd(p, d), mks_hash_color(p, d));
}
