// mkr-shade: geometry-dependent nodes (Bump, Normal Map, image projections, environment mapping, EEVEE Specular).

// Bump finite-difference step in pixels: EEVEE 4.2 offsets texture coordinates by one screen derivative.
const MKS_BUMP_FILTER: f32 = 1.0;

// Bump node (Mikkelsen's surface gradient). h, hx, hy: height at the point and one step along screen x / y;
// dPdx / dPdy: the matching position steps.
fn mks_bump(strength_in: f32, dist_in: f32, h: f32, hx: f32, hy: f32, N_in: vec3f, invert: f32, backfacing: f32,
            dPdx: vec3f, dPdy: vec3f) -> vec3f {
  let N = mks_safe_normalize(N_in);
  var dist = dist_in;
  if (invert != 0.0) {
    dist = -dist;
  }
  if (backfacing != 0.0) {
    dist = -dist;
  }
  let Rx = cross(dPdy, N);
  let Ry = cross(N, dPdx);
  let det = dot(dPdx, Rx);
  let surfgrad = (hx - h) * Rx + (hy - h) * Ry;
  let strength = max(strength_in, 0.0);
  let r = mks_safe_normalize(abs(det) * N - dist * sign(det) * surfgrad);
  return mks_safe_normalize(mix(N, r, strength));
}

// Normal Map, tangent space. t: world tangent (xyz) and bitangent sign (w).
fn mks_normal_map_tangent(t: vec4f, tn: vec3f, N: vec3f, backfacing: f32, object: mat4x4f) -> vec3f {
  if (all(t == vec4f(0.0)) || all(t == vec4f(0.0, 0.0, 0.0, 1.0))) {
    return N;
  }
  let flip = select(1.0, -1.0, backfacing != 0.0);
  let T = t.xyz * flip;
  let m = mat3x3f(object[0].xyz, object[1].xyz, object[2].xyz);
  let neg = select(1.0, -1.0, determinant(m) < 0.0);
  let B = (t.w * flip) * cross(N, T) * neg;
  return mks_safe_normalize(tn.x * T + tn.y * B + tn.z * N);
}

// Image Texture BOX projection: blend weights of the three planar projections from the object-space normal.
fn mks_box_weights(n_obj: vec3f, blend: f32) -> vec3f {
  var N = abs(n_obj);
  N = N / max(N.x + N.y + N.z, 1e-20);
  let limit = 0.5 * (1.0 + blend);
  var w = vec3f(0.0);
  if (N.x > limit * (N.x + N.y) && N.x > limit * (N.x + N.z)) {
    w.x = 1.0;
  } else if (N.y > limit * (N.x + N.y) && N.y > limit * (N.y + N.z)) {
    w.y = 1.0;
  } else if (N.z > limit * (N.x + N.z) && N.z > limit * (N.y + N.z)) {
    w.z = 1.0;
  } else if (blend > 0.0) {
    if (N.z < (1.0 - limit) * (N.y + N.x)) {
      w.x = clamp((N.x / (N.x + N.y) - 0.5 * (1.0 - blend)) / blend, 0.0, 1.0);
      w.y = 1.0 - w.x;
    } else if (N.x < (1.0 - limit) * (N.y + N.z)) {
      w.y = clamp((N.y / (N.y + N.z) - 0.5 * (1.0 - blend)) / blend, 0.0, 1.0);
      w.z = 1.0 - w.y;
    } else if (N.y < (1.0 - limit) * (N.x + N.z)) {
      w.x = clamp((N.x / (N.x + N.z) - 0.5 * (1.0 - blend)) / blend, 0.0, 1.0);
      w.z = 1.0 - w.x;
    } else {
      w = ((2.0 - limit) * N + (limit - 1.0)) / (2.0 * limit - 1.0);
    }
  } else {
    w.x = 1.0;
  }
  return w;
}

fn mks_box_uv_x(co: vec3f, n_obj: vec3f) -> vec2f {
  return vec2f(select(co.y, 1.0 - co.y, n_obj.x < 0.0), co.z);
}

fn mks_box_uv_y(co: vec3f, n_obj: vec3f) -> vec2f {
  return vec2f(select(co.x, 1.0 - co.x, n_obj.y > 0.0), co.z);
}

fn mks_box_uv_z(co: vec3f, n_obj: vec3f) -> vec2f {
  return vec2f(select(co.y, 1.0 - co.y, n_obj.z > 0.0), co.x);
}

fn mks_map_to_sphere(v: vec3f) -> vec2f {
  let len = length(v);
  if (len > 0.0) {
    var u = 0.0;
    if (!(v.x == 0.0 && v.y == 0.0)) {
      u = (1.0 - atan2(v.x, v.y) / MKS_PI) / 2.0;
    }
    let w = 1.0 - acos(clamp(v.z / len, -1.0, 1.0)) / MKS_PI;
    return vec2f(u, w);
  }
  return vec2f(0.0);
}

fn mks_map_to_tube(v: vec3f) -> vec2f {
  let len = sqrt(v.x * v.x + v.y * v.y);
  if (len > 0.0) {
    return vec2f((1.0 - atan2(v.x / len, v.y / len) / MKS_PI) * 0.5, (v.z + 1.0) * 0.5);
  }
  return vec2f(0.0);
}

fn mks_env_equirect(co: vec3f) -> vec2f {
  let n = mks_safe_normalize(co);
  return vec2f(-atan2(n.y, n.x) / (2.0 * MKS_PI) + 0.5, atan2(n.z, length(n.xy)) / MKS_PI + 0.5);
}

fn mks_env_mirror_ball(co: vec3f) -> vec2f {
  var n = mks_safe_normalize(co);
  n.y -= 1.0;
  let div = 2.0 * sqrt(max(-0.5 * n.y, 0.0));
  n /= max(1e-8, div);
  return 0.5 * n.xz + 0.5;
}

// EEVEE's Specular BSDF node.
fn mks_eevee_specular(base: vec3f, spec: vec3f, roughness: f32, emissive: vec3f, transp: f32, N_in: vec3f,
                      clearcoat: f32, clearcoat_roughness: f32, CN_in: vec3f, I: vec3f) -> Closures {
  let N = mks_safe_normalize(N_in);
  let CN = mks_safe_normalize(CN_in);
  var c = mks_closures_none();
  c.emission = emissive;
  c.transmittance = vec3f(transp);
  let alpha = 1.0 - transp;
  c.diffuse = alpha * base;
  c.diffuse_N = N;
  let split = mks_brdf_lut(dot(N, I), roughness).xy;
  c.reflection = alpha * mks_F_brdf_single(spec, vec3f(1.0), split);
  c.reflection_N = N;
  c.reflection_roughness = roughness;
  if (clearcoat > 0.0) {
    let csplit = mks_brdf_lut(dot(CN, I), clearcoat_roughness).xy;
    c.coat = alpha * clearcoat * 0.25 * mks_F_brdf_single(vec3f(0.04), vec3f(1.0), csplit);
    c.coat_N = CN;
    c.coat_roughness = clearcoat_roughness;
  }
  return c;
}
