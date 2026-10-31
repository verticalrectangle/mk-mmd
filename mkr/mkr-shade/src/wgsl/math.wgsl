// mkr-shade: Blender's scalar and vector maths (Math, Vector Math, Map Range, Clamp, Mapping, Vector Rotate).

fn mks_safe_div(a: f32, b: f32) -> f32 {
  return select(0.0, a / b, b != 0.0);
}

fn mks_safe_div3(a: vec3f, b: vec3f) -> vec3f {
  return vec3f(mks_safe_div(a.x, b.x), mks_safe_div(a.y, b.y), mks_safe_div(a.z, b.z));
}

fn mks_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  return select(vec3f(0.0), v / l, l != 0.0);
}

// C fmod: a - b * trunc(a / b), 0 when b is 0.
fn mks_fmod(a: f32, b: f32) -> f32 {
  return select(0.0, a - b * trunc(a / b), b != 0.0);
}

fn mks_fmod3(a: vec3f, b: vec3f) -> vec3f {
  return vec3f(mks_fmod(a.x, b.x), mks_fmod(a.y, b.y), mks_fmod(a.z, b.z));
}

fn mks_floored_mod(a: f32, b: f32) -> f32 {
  return select(0.0, a - floor(a / b) * b, b != 0.0);
}

fn mks_round(x: f32) -> f32 {
  return floor(x + 0.5);
}

// x^y with GLSL's undefined cases defined as Blender does.
fn mks_compatible_pow(x: f32, y: f32) -> f32 {
  if (y == 0.0) {
    return 1.0;
  }
  if (x < 0.0) {
    let ay = abs(y);
    if (mks_fmod(-y, 2.0) == 0.0) {
      return pow(-x, y);
    }
    return -pow(-x, y);
  }
  if (x == 0.0) {
    return 0.0;
  }
  return pow(x, y);
}

fn mks_math_power(a: f32, b: f32) -> f32 {
  if (a >= 0.0) {
    return mks_compatible_pow(a, b);
  }
  let fraction = abs(b) - floor(abs(b));
  if (fraction > 0.999 || fraction < 0.001) {
    return mks_compatible_pow(a, floor(b + 0.5));
  }
  return 0.0;
}

fn mks_math_log(a: f32, b: f32) -> f32 {
  return select(0.0, log(a) / log(b), a > 0.0 && b > 0.0);
}

fn mks_math_sqrt(a: f32) -> f32 {
  return select(0.0, sqrt(a), a > 0.0);
}

// EEVEE does not guard it: 0 gives +inf, negatives NaN (measured: lab math_inverse_sqrt).
fn mks_math_inv_sqrt(a: f32) -> f32 {
  if (a > 0.0) {
    return inverseSqrt(a);
  }
  // runtime bit patterns (a const NaN or infinity is a shader-creation error in WGSL)
  let zero_bits = bitcast<u32>(a) & 0u;
  return select(bitcast<f32>(0x7fc00000u | zero_bits), bitcast<f32>(0x7f800000u | zero_bits), a == 0.0);
}

fn mks_math_asin(a: f32) -> f32 {
  return select(0.0, asin(a), a <= 1.0 && a >= -1.0);
}

fn mks_math_acos(a: f32) -> f32 {
  return select(0.0, acos(a), a <= 1.0 && a >= -1.0);
}

// Wrap a into [min, max).
fn mks_wrap(a: f32, b_max: f32, c_min: f32) -> f32 {
  let range = b_max - c_min;
  return select(c_min, a - range * floor((a - c_min) / range), range != 0.0);
}

fn mks_wrap3(a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  return vec3f(mks_wrap(a.x, b.x, c.x), mks_wrap(a.y, b.y, c.y), mks_wrap(a.z, b.z, c.z));
}

fn mks_pingpong(a: f32, b: f32) -> f32 {
  return select(0.0, abs(fract((a - b) / (b * 2.0)) * b * 2.0 - b), b != 0.0);
}

fn mks_snap(a: f32, b: f32) -> f32 {
  return floor(mks_safe_div(a, b)) * b;
}

fn mks_snap3(a: vec3f, b: vec3f) -> vec3f {
  return floor(mks_safe_div3(a, b)) * b;
}

fn mks_smoothmin(a: f32, b: f32, c: f32) -> f32 {
  if (c != 0.0) {
    let h = max(c - abs(a - b), 0.0) / c;
    return min(a, b) - h * h * h * c * (1.0 / 6.0);
  }
  return min(a, b);
}

fn mks_math_compare(a: f32, b: f32, c: f32) -> f32 {
  return select(0.0, 1.0, abs(a - b) <= max(c, 1e-5));
}

fn mks_sign(a: f32) -> f32 {
  return sign(a);
}

fn mks_frac(a: f32) -> f32 {
  return a - floor(a);
}

// Blender's smoothstep: 0 below edge0, 1 at or above edge1.
fn mks_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  if (x < edge0) {
    return 0.0;
  }
  if (x >= edge1) {
    return 1.0;
  }
  let t = (x - edge0) / (edge1 - edge0);
  return (3.0 - 2.0 * t) * (t * t);
}

fn mks_smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp(mks_safe_div(x - edge0, edge1 - edge0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Map Range, float.
fn mks_map_range_linear(v: f32, fmin: f32, fmax: f32, tmin: f32, tmax: f32) -> f32 {
  if (fmax != fmin) {
    return tmin + ((v - fmin) / (fmax - fmin)) * (tmax - tmin);
  }
  return 0.0;
}

fn mks_map_range_stepped(v: f32, fmin: f32, fmax: f32, tmin: f32, tmax: f32, steps: f32) -> f32 {
  if (fmax != fmin) {
    let factor = (v - fmin) / (fmax - fmin);
    let stepped = select(0.0, floor(factor * (steps + 1.0)) / steps, steps > 0.0);
    return tmin + stepped * (tmax - tmin);
  }
  return 0.0;
}

fn mks_map_range_smoothstep(v: f32, fmin: f32, fmax: f32, tmin: f32, tmax: f32) -> f32 {
  var factor = 0.0;
  if (fmin > fmax) {
    factor = 1.0 - mks_smoothstep(fmax, fmin, v);
  } else {
    factor = mks_smoothstep(fmin, fmax, v);
  }
  return tmin + factor * (tmax - tmin);
}

fn mks_map_range_smootherstep(v: f32, fmin: f32, fmax: f32, tmin: f32, tmax: f32) -> f32 {
  var factor = 0.0;
  if (fmin > fmax) {
    factor = 1.0 - mks_smootherstep(fmax, fmin, v);
  } else {
    factor = mks_smootherstep(fmin, fmax, v);
  }
  return tmin + factor * (tmax - tmin);
}

fn mks_clamp_range(v: f32, a: f32, b: f32) -> f32 {
  return min(max(v, min(a, b)), max(a, b));
}

fn mks_clamp_minmax(v: f32, lo: f32, hi: f32) -> f32 {
  return min(max(v, lo), hi);
}

// Euler XYZ rotation (Blender's euler_to_mat3: R = Rz * Ry * Rx).
fn mks_euler_to_mat3(e: vec3f) -> mat3x3f {
  let cx = cos(e.x);
  let cy = cos(e.y);
  let cz = cos(e.z);
  let sx = sin(e.x);
  let sy = sin(e.y);
  let sz = sin(e.z);
  return mat3x3f(
    vec3f(cy * cz, cy * sz, -sy),
    vec3f(sy * sx * cz - cx * sz, sy * sx * sz + cx * cz, cy * sx),
    vec3f(sy * cx * cz + sx * sz, sy * cx * sz - sx * cz, cy * cx));
}

// Mapping node.
fn mks_mapping_point(v: vec3f, loc: vec3f, rot: vec3f, scale: vec3f) -> vec3f {
  return mks_euler_to_mat3(rot) * (v * scale) + loc;
}

fn mks_mapping_texture(v: vec3f, loc: vec3f, rot: vec3f, scale: vec3f) -> vec3f {
  return mks_safe_div3(transpose(mks_euler_to_mat3(rot)) * (v - loc), scale);
}

fn mks_mapping_vector(v: vec3f, rot: vec3f, scale: vec3f) -> vec3f {
  return mks_euler_to_mat3(rot) * (v * scale);
}

fn mks_mapping_normal(v: vec3f, rot: vec3f, scale: vec3f) -> vec3f {
  return mks_safe_normalize(mks_euler_to_mat3(rot) * mks_safe_div3(v, scale));
}

// Rodrigues rotation of v about unit axis by angle.
fn mks_rotate_axis(v: vec3f, axis: vec3f, angle: f32) -> vec3f {
  let c = cos(angle);
  let s = sin(angle);
  let t = 1.0 - c;
  let a = axis;
  let m = mat3x3f(
    vec3f(t * a.x * a.x + c, t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y),
    vec3f(t * a.x * a.y - s * a.z, t * a.y * a.y + c, t * a.y * a.z + s * a.x),
    vec3f(t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c));
  return m * v;
}

// Vector Rotate node.
fn mks_vector_rotate_axis_angle(v: vec3f, center: vec3f, axis: vec3f, angle: f32, invert: f32) -> vec3f {
  let a = select(angle, -angle, invert != 0.0);
  let n = mks_safe_normalize(axis);
  return select(mks_rotate_axis(v - center, n, a) + center, v, length(axis) == 0.0);
}

fn mks_vector_rotate_euler(v: vec3f, center: vec3f, rot: vec3f, invert: f32) -> vec3f {
  var m = mks_euler_to_mat3(rot);
  if (invert != 0.0) {
    m = transpose(m);
  }
  return m * (v - center) + center;
}

// Vector Math helpers.
fn mks_project(a: vec3f, b: vec3f) -> vec3f {
  let l2 = dot(b, b);
  return select(vec3f(0.0), (dot(a, b) / l2) * b, l2 != 0.0);
}

fn mks_reflect(a: vec3f, b: vec3f) -> vec3f {
  let n = mks_safe_normalize(b);
  return a - 2.0 * dot(n, a) * n;
}

fn mks_refract(a: vec3f, b: vec3f, eta: f32) -> vec3f {
  let n = mks_safe_normalize(b);
  let d = dot(n, a);
  let k = 1.0 - eta * eta * (1.0 - d * d);
  return select(eta * a - (eta * d + sqrt(k)) * n, vec3f(0.0), k < 0.0);
}

fn mks_faceforward(a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  return select(-a, a, dot(c, b) < 0.0);
}

fn mks_pow3(a: vec3f, b: vec3f) -> vec3f {
  return vec3f(mks_math_power(a.x, b.x), mks_math_power(a.y, b.y), mks_math_power(a.z, b.z));
}
