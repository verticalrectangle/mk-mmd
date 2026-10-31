// mkr-shade: BSDF nodes as EEVEE 4.2 turns them into closures (split-sum weighted colours).
//
// LUT (mkr_shade::lut(), sampled through the renderer's mkr_lut): MKS_LUT_SIZE x MKS_LUT_SIZE texels per layer,
// u = roughness, v = sqrt(1 - cos_theta), texel centres at (i + 0.5) / MKS_LUT_SIZE.
//   layer 0                     rgb: GGX split sum (F0 weight, F90 weight, F82-tint term)
//   layers 1 .. N               r: GGX BTDF transmission factor, ior >= 1, z = sqrt((ior - 1) / (ior + 1))
//                               at layer 1 + z (N - 1)
//   layers N + 1 .. 2N          rgb: (split sum F0 weight, F90 weight, transmission) for ior < 1, z = sqrt(1 - ior)

const MKS_LUT_SIZE: f32 = 64.0;
const MKS_LUT_IOR_LAYERS: u32 = 16u;

fn mks_lut_uv(cos_theta: f32, roughness: f32) -> vec2f {
  let c = vec2f(clamp(roughness, 0.0, 1.0), sqrt(clamp(1.0 - cos_theta, 0.0, 1.0)));
  return c * ((MKS_LUT_SIZE - 1.0) / MKS_LUT_SIZE) + 0.5 / MKS_LUT_SIZE;
}

fn mks_brdf_lut(cos_theta: f32, roughness: f32) -> vec3f {
  return mkr_lut(mks_lut_uv(cos_theta, roughness), 0u).rgb;
}

fn mks_lut_layers(uv: vec2f, base: u32, z: f32) -> vec4f {
  let f = clamp(z, 0.0, 1.0) * f32(MKS_LUT_IOR_LAYERS - 1u);
  let i0 = u32(floor(f));
  let i1 = min(i0 + 1u, MKS_LUT_IOR_LAYERS - 1u);
  return mix(mkr_lut(uv, base + i0), mkr_lut(uv, base + i1), f - floor(f));
}

fn mks_F0_from_ior(eta: f32) -> f32 {
  let f = (eta - 1.0) / (eta + 1.0);
  return f * f;
}

fn mks_ior_from_F0(f0: f32) -> f32 {
  let s = sqrt(clamp(f0, 0.0, 0.99));
  return (1.0 + s) / (1.0 - s);
}

fn mks_F_brdf_single(f0: vec3f, f90: vec3f, lut: vec2f) -> vec3f {
  return f0 * lut.x + f90 * lut.y;
}

// Fdez-Agüera's multiple-scattering albedo scaling over the split sum.
fn mks_F_brdf_multi(f0: vec3f, f90: vec3f, lut: vec2f) -> vec3f {
  let fss_ess = lut.y * f90 + lut.x * f0;
  let ess = lut.x + lut.y;
  let ems = 1.0 - ess;
  let favg = f0 + (f90 - f0) / 21.0;
  return fss_ess / (vec3f(1.0) - ems * favg);
}

struct MksRT {
  reflectance: vec3f,
  transmittance: vec3f,
}

// Reflectance and transmittance of a rough dielectric interface (EEVEE's bsdf_lut).
fn mks_bsdf_lut(F0: vec3f, F90_in: vec3f, tint: vec3f, cos_theta: f32, roughness: f32, ior: f32, multi: bool) -> MksRT {
  if (ior == -1.0) {
    return MksRT(vec3f(0.0), tint);
  }
  var F90 = F90_in;
  var split = vec2f(0.0);
  var tf = 0.0;
  let uv = mks_lut_uv(cos_theta, roughness);
  if (ior >= 1.0) {
    split = mkr_lut(uv, 0u).rg;
    tf = mks_lut_layers(uv, 1u, sqrt((ior - 1.0) / (ior + 1.0))).r;
    if (all(F90 == vec3f(1.0))) {
      F90 = vec3f(clamp(2.33 / 0.33 * (ior - 1.0) / (ior + 1.0), 0.0, 1.0));
    }
  } else {
    let t = mks_lut_layers(uv, 1u + MKS_LUT_IOR_LAYERS, sqrt(1.0 - ior));
    split = t.rg;
    tf = t.b;
  }
  var refl = mks_F_brdf_single(F0, F90, split);
  var trans = (vec3f(1.0) - F0) * tf * tint;
  if (multi) {
    let real_F0 = mks_F0_from_ior(ior);
    let ess = real_F0 * split.x + split.y + (1.0 - real_F0) * tf;
    let ems = 1.0 - ess;
    let favg = select(tint, F0 + (F90 - F0) / 21.0, all(tint == vec3f(0.0)));
    let scale = vec3f(1.0) / (vec3f(1.0) - ems * favg);
    refl *= scale;
    trans *= scale;
  }
  return MksRT(refl, trans);
}

// Metallic F82-tint reflectance (Hoffman / Kutz et al.) over the split sum.
fn mks_brdf_f82_tint(F0: vec3f, F82: vec3f, cos_theta: f32, roughness: f32, multi: bool) -> vec3f {
  let split = mks_brdf_lut(cos_theta, roughness);
  var refl = select(mks_F_brdf_single(F0, vec3f(1.0), split.xy), mks_F_brdf_multi(F0, vec3f(1.0), split.xy), multi);
  let f = 6.0 / 7.0;
  let f5 = (f * f) * (f * f) * f;
  let f6 = (f * f) * (f * f) * (f * f);
  let F_schlick = mix(F0, vec3f(1.0), f5);
  let b = F_schlick * (7.0 / f6) * (vec3f(1.0) - F82);
  refl -= b * split.z;
  return refl;
}

fn mks_max3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

fn mks_bsdf_diffuse(color: vec3f, N: vec3f) -> Closures {
  var c = mks_closures_none();
  c.diffuse = color;
  c.diffuse_N = mks_safe_normalize(N);
  return c;
}

fn mks_bsdf_translucent(color: vec3f, N: vec3f) -> Closures {
  var c = mks_closures_none();
  c.translucent = color;
  c.translucent_N = mks_safe_normalize(N);
  return c;
}

fn mks_bsdf_transparent(color: vec3f) -> Closures {
  var c = mks_closures_none();
  c.transmittance = color;
  return c;
}

fn mks_bsdf_holdout() -> Closures {
  var c = mks_closures_none();
  c.holdout = 1.0;
  return c;
}

fn mks_bsdf_emission(color: vec3f, strength: f32) -> Closures {
  var c = mks_closures_none();
  c.emission = color * strength;
  return c;
}

fn mks_bsdf_glossy(color: vec3f, roughness: f32, N_in: vec3f, I: vec3f, multi: bool) -> Closures {
  let N = mks_safe_normalize(N_in);
  let nv = dot(N, I);
  let split = mks_brdf_lut(nv, roughness).xy;
  var c = mks_closures_none();
  c.reflection = select(mks_F_brdf_single(color, color, split), mks_F_brdf_multi(color, color, split), multi);
  c.reflection_N = N;
  c.reflection_roughness = roughness;
  return c;
}

fn mks_bsdf_refraction(color: vec3f, roughness: f32, ior: f32, N: vec3f) -> Closures {
  var c = mks_closures_none();
  c.refraction = color;
  c.refraction_N = mks_safe_normalize(N);
  c.refraction_roughness = roughness;
  c.refraction_ior = ior;
  return c;
}

fn mks_bsdf_glass(color: vec3f, roughness: f32, ior: f32, N_in: vec3f, I: vec3f, multi: bool) -> Closures {
  let N = mks_safe_normalize(N_in);
  let nv = dot(N, I);
  let rt = mks_bsdf_lut(vec3f(mks_F0_from_ior(ior)), vec3f(1.0), color, nv, roughness, ior, multi);
  var c = mks_closures_none();
  c.reflection = rt.reflectance;
  c.reflection_N = N;
  c.reflection_roughness = roughness;
  c.refraction = rt.transmittance;
  c.refraction_N = N;
  c.refraction_roughness = roughness;
  c.refraction_ior = ior;
  return c;
}

// Albedo of Blender's sheen lobe over (cos_theta, roughness): fitted to Blender 4.2 measurements (see lab/).
fn mks_sheen_albedo(cos_theta: f32, roughness: f32) -> f32 {
  return mkr_lut(mks_lut_uv(cos_theta, roughness), 0u).a;
}

struct MksPrincipled {
  base_color: vec3f,
  metallic: f32,
  roughness: f32,
  ior: f32,
  alpha: f32,
  N: vec3f,
  subsurface_weight: f32,
  specular_ior_level: f32,
  specular_tint: vec3f,
  transmission_weight: f32,
  coat_weight: f32,
  coat_roughness: f32,
  coat_ior: f32,
  coat_tint: vec3f,
  coat_N: vec3f,
  sheen_weight: f32,
  sheen_roughness: f32,
  sheen_tint: vec3f,
  emission: vec3f,
  emission_strength: f32,
}

fn mks_bsdf_principled(p: MksPrincipled, I: vec3f, multi: bool) -> Closures {
  let metallic = clamp(p.metallic, 0.0, 1.0);
  let roughness = clamp(p.roughness, 0.0, 1.0);
  let ior = max(p.ior, 1e-5);
  let alpha = clamp(p.alpha, 0.0, 1.0);
  let transmission = clamp(p.transmission_weight, 0.0, 1.0);
  let coat_weight = max(p.coat_weight, 0.0);
  let coat_ior = max(p.coat_ior, 1.0);
  let sheen_weight = max(p.sheen_weight, 0.0);
  let sheen_roughness = clamp(p.sheen_roughness, 0.0, 1.0);
  let specular_ior_level = max(p.specular_ior_level, 0.0);
  let specular_tint = max(p.specular_tint, vec3f(0.0));
  let N = mks_safe_normalize(p.N);
  let CN = mks_safe_normalize(p.coat_N);
  let nv = dot(N, I);

  var c = mks_closures_none();
  c.transmittance = vec3f(1.0 - alpha);
  var weight = alpha;

  c.diffuse_N = N;
  c.reflection_N = N;
  c.reflection_roughness = roughness;
  c.refraction_N = N;
  c.refraction_roughness = roughness;
  c.refraction_ior = ior;
  c.coat_N = CN;
  c.coat_roughness = p.coat_roughness;

  // Sheen (EEVEE 4.2 adds it to the diffuse closure).
  if (sheen_weight > 0.0) {
    let sheen = sheen_weight * p.sheen_tint * mks_sheen_albedo(nv, sheen_roughness);
    c.diffuse = weight * sheen;
    weight *= max(1.0 - mks_max3(sheen), 0.0);
  }

  // Coat.
  var coat_tint = vec3f(1.0);
  if (coat_weight > 0.0) {
    let coat_nv = dot(CN, I);
    let r = mks_bsdf_lut(vec3f(mks_F0_from_ior(coat_ior)), vec3f(1.0), vec3f(0.0), coat_nv, p.coat_roughness, coat_ior,
                         false).reflectance.x;
    c.coat = vec3f(weight * coat_weight * r);
    weight *= max(1.0 - r * coat_weight, 0.0);
    if (!all(p.coat_tint == vec3f(1.0))) {
      let coat_neta = 1.0 / coat_ior;
      let NT = sqrt(max(1.0 - coat_neta * coat_neta * (1.0 - nv * nv), 0.0));
      coat_tint = mix(vec3f(1.0), pow(max(p.coat_tint, vec3f(0.0)), vec3f(1.0 / NT)), clamp(coat_weight, 0.0, 1.0));
    }
  }

  // Emission.
  c.emission = weight * coat_tint * p.emission * p.emission_strength;

  // Metal.
  if (metallic > 0.0) {
    let F0 = p.base_color;
    let F82 = min(specular_tint, vec3f(1.0));
    c.reflection = weight * metallic * mks_brdf_f82_tint(F0, F82, nv, roughness, multi);
    weight *= max(1.0 - metallic, 0.0);
  }

  // Transmission.
  if (transmission > 0.0) {
    var F0t = vec3f(mks_F0_from_ior(ior)) * 2.0 * specular_ior_level * specular_tint;
    F0t = clamp(F0t, vec3f(0.0), vec3f(1.0));
    let rt = mks_bsdf_lut(F0t, vec3f(1.0), p.base_color, nv, roughness, ior, multi);
    c.reflection += weight * transmission * rt.reflectance;
    c.refraction = weight * transmission * rt.transmittance;
    weight *= max(1.0 - transmission, 0.0);
  }

  // Dielectric specular.
  {
    var eta = ior;
    var F0 = mks_F0_from_ior(eta);
    if (specular_ior_level != 0.5) {
      F0 *= 2.0 * specular_ior_level;
      eta = mks_ior_from_F0(F0);
      if (ior < 1.0) {
        eta = 1.0 / eta;
      }
    }
    let F0v = clamp(vec3f(F0) * specular_tint, vec3f(0.0), vec3f(1.0));
    let rt = mks_bsdf_lut(F0v, vec3f(1.0), vec3f(0.0), nv, roughness, eta, multi);
    c.reflection += weight * rt.reflectance;
    c.reflection *= coat_tint;
    weight *= max(1.0 - mks_max3(rt.reflectance), 0.0);
  }

  // Diffuse (subsurface is drawn as diffuse).
  c.diffuse += weight * p.base_color * coat_tint;
  return c;
}

// Principled with every input at its default: what a material without nodes draws is not this (see the renderer's
// fallback); exposed for graphs that link nothing into a Principled BSDF.
fn mks_principled_default() -> MksPrincipled {
  return MksPrincipled(vec3f(0.8), 0.0, 0.5, 1.5, 1.0, vec3f(0.0, 0.0, 1.0), 0.0, 0.5, vec3f(1.0), 0.0, 0.0, 0.03,
                       1.5, vec3f(1.0), vec3f(0.0, 0.0, 1.0), 0.0, 0.5, vec3f(1.0), vec3f(1.0), 0.0);
}

// Fresnel and Layer Weight nodes.
fn mks_fresnel_dielectric_cos(cosi: f32, eta: f32) -> f32 {
  let c = abs(cosi);
  var g = eta * eta - 1.0 + c * c;
  if (g > 0.0) {
    g = sqrt(g);
    let A = (g - c) / (g + c);
    let B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * A * A * (1.0 + B * B);
  }
  return 1.0;
}

fn mks_fresnel(ior: f32, N_in: vec3f, I: vec3f, backfacing: f32) -> f32 {
  let N = mks_safe_normalize(N_in);
  let eta = max(ior, 0.00001);
  return mks_fresnel_dielectric_cos(dot(I, N), select(eta, 1.0 / eta, backfacing != 0.0));
}

// -> (Fresnel, Facing)
fn mks_layer_weight(blend_in: f32, N_in: vec3f, I: vec3f, backfacing: f32) -> vec2f {
  let N = mks_safe_normalize(N_in);
  let eta = max(1.0 - blend_in, 0.00001);
  let fresnel = mks_fresnel_dielectric_cos(dot(I, N), select(1.0 / eta, eta, backfacing != 0.0));
  var facing = abs(dot(I, N));
  if (blend_in != 0.5) {
    var blend = clamp(blend_in, 0.0, 0.99999);
    blend = select(0.5 / (1.0 - blend), 2.0 * blend, blend < 0.5);
    facing = pow(facing, blend);
  }
  return vec2f(fresnel, 1.0 - facing);
}
