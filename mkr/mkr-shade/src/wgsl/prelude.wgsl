// mkr-shade prelude: the WGSL contract between compiled Blender node graphs (mkr-shade) and the renderer (mkr).
//
// A shader module that evaluates materials is: PRELUDE (this file and the mkr-shade WGSL library after it,
// `mkr_shade::PRELUDE`) + the `wgsl` of every CompiledMaterial + the renderer's own code, which MUST define:
//
//   fn mkr_image(index: u32, uv: vec2f, interp: u32, ext: u32) -> vec4f
//       Image `index` of the images list given to `compile` (global order), sampled at Blender's uv (v up: the
//       renderer flips). interp: 0 Linear, 1 Closest, 2 Cubic, 3 Smart. ext: 0 REPEAT, 1 EXTEND, 2 CLIP (transparent
//       black outside), 3 MIRROR. Returns scene-linear colour and alpha as Blender's GPU texture holds them
//       (straight, or premultiplied when ImageDesc.premultiplied). Missing images: magenta (1, 0, 1, 1).
//       mkr-shade never calls it inside a branch, so it may use implicit derivatives in fragment shaders.
//   fn mkr_image_lod(index: u32, uv: vec2f, interp: u32, ext: u32) -> vec4f
//       The same at mip level 0 without derivatives (only mat_<id>_alpha_lod calls it).
//   fn mkr_param(slot: u32) -> vec4f
//       Per-frame animated socket values (CompiledMaterial.params: anim record -> slot), x = a float socket,
//       xyz a vector, rgba a colour.
//   fn mkr_shader_to_rgb(c: Closures, s: Surface) -> vec4f
//       The lit colour of closures `c` at `s` (Shader to RGB); mkr-shade uses .rgb and computes alpha itself.
//   fn mkr_lut(uv: vec2f, layer: u32) -> vec4f
//       textureSampleLevel of the mkr-shade LUT (`mkr_shade::lut()`, rgba32float 2D array), linear filtering,
//       clamp to edge, level 0.
//
// Every compiled material `id` gives
//   fn mat_<id>(s: Surface) -> Closures        fragment stage (may use dpdx/dpdy: CompiledMaterial.uses_derivatives)
//   fn mat_<id>_alpha(s: Surface) -> f32       opacity = 1 - average(transmittance); no derivatives of its own,
//                                              samples through mkr_image (depth prepass)
//   fn mat_<id>_alpha_lod(s: Surface) -> f32   the same through mkr_image_lod: callable from compute / ray-query code

const MKS_PI: f32 = 3.14159265358979323846;
const MKS_TAU: f32 = 6.28318530717958647692;
const MKS_FLT_MAX: f32 = 3.402823466e+38;

// Surface.ray_type (Light Path node).
const MKR_RAY_CAMERA: u32 = 0u;
const MKR_RAY_SHADOW: u32 = 1u;
const MKR_RAY_DIFFUSE: u32 = 2u;
const MKR_RAY_GLOSSY: u32 = 3u;
const MKR_RAY_TRANSMISSION: u32 = 4u;
const MKR_RAY_VOLUME_SCATTER: u32 = 5u;

// What a material sees at one shading point (Blender's axes, world space unless said otherwise).
struct Surface {
  P: vec3f,                     // position
  N: vec3f,                     // shading normal (smooth / custom), unit, flipped toward the viewer on back faces
  Ng: vec3f,                    // true (face) normal, unit, flipped likewise
  I: vec3f,                     // unit vector from P toward the viewer (orthographic: the view's backward axis)
  backfacing: f32,              // 1 on back faces, 0 on front faces
  generated: vec3f,             // Blender's Generated texture coordinate (the bake's "generated" attribute)
  attrs: array<vec4f, 8>,       // CompiledMaterial.attributes in order, as vec4 by Blender's rule: float f ->
                                // (f, f, f, 1), float2 -> (x, y, 0, 1), float3 -> (x, y, z, 1), colour -> rgba;
                                // "tangent:*": world-space unit tangent xyz, w = bitangent sign; missing -> 0
  object: mat4x4f,              // object -> world (the object's Blender matrix)
  object_inv: mat4x4f,          // world -> object
  camera: mat4x4f,              // viewer -> world (camera matrix, scale removed; the light's for shadow maps)
  camera_inv: mat4x4f,          // world -> viewer (Blender's view matrix: the viewer looks down its -Z)
  ortho: f32,                   // 1 orthographic viewer, 0 perspective
  screen: vec2f,                // Window texture coordinate: 0..1 over the render, origin bottom left
  rand: vec4f,                  // per-sample uniform randoms in [0, 1): closure picking (as EEVEE)
  ray_type: u32,                // MKR_RAY_*
  object_color: vec4f,          // Object Info Color / Alpha (object.color, linear)
  object_info: vec4f,           // x: object pass_index, y: object random (0..1), zw: 0
}

// The closures a material leaves at a point: what the renderer lights. Colours are EEVEE's closure colours (node
// weights, mix factors, Fresnel and split-sum albedo applied): light diffuse with Lambert, reflection and coat with a
// unit-albedo GGX lobe (alpha = roughness^2), refraction with a unit GGX BTDF, translucent as its comment says.
// Normals: unit, Blender axes.
struct Closures {
  diffuse: vec3f,               // Lambert (subsurface and Principled sheen fold in here, as EEVEE 4.2)
  diffuse_N: vec3f,
  translucent: vec3f,           // EEVEE 4.2 lights the SQUARE of this colour, Mix weights included ((w c)^2), by the
                                // material's thickness mode (translucent_slab); measured with lamps (lab lit_*,
                                // ParityLab finding 5): SPHERE (Blender's default): from every side alike, no cosine
                                // with the normal: per light += translucent^2 x the light's Lambert integral for a
                                // surface facing it / pi (a sun: translucent^2 x E / pi^2); SLAB: translucent^2 x
                                // Lambert around -translucent_N, lights in front give nothing. Both are shadowed by the
                                // light's visibility. World light (uniform world, SPHERE): translucent^2 x radiance
  translucent_N: vec3f,
  reflection: vec3f,            // GGX reflection
  reflection_N: vec3f,
  reflection_roughness: f32,
  coat: vec3f,                  // second GGX reflection lobe (Principled coat)
  coat_N: vec3f,
  coat_roughness: f32,
  refraction: vec3f,            // GGX refraction
  refraction_N: vec3f,
  refraction_roughness: f32,
  refraction_ior: f32,
  emission: vec3f,              // emitted radiance
  transmittance: vec3f,         // Transparent BSDF share (opacity = 1 - average(transmittance))
  holdout: f32,                 // Holdout share
  translucent_slab: f32,        // 1: the material's thickness_mode is SLAB, 0: SPHERE (see translucent)
}

fn mks_closures_none() -> Closures {
  let z = vec3f(0.0);
  let n = vec3f(0.0, 0.0, 1.0);
  return Closures(z, n, z, n, z, n, 0.0, z, n, 0.0, z, n, 0.0, 1.0, z, z, 0.0, 0.0);
}

fn mks_avg3(v: vec3f) -> f32 {
  return (v.x + v.y + v.z) * (1.0 / 3.0);
}

// Opacity of a closure set (what hashed / blended transparency uses).
fn mks_opacity(c: Closures) -> f32 {
  return clamp(1.0 - mks_avg3(c.transmittance), 0.0, 1.0);
}

fn mks_closures_scale(c: Closures, w: f32) -> Closures {
  var r = c;
  r.diffuse = c.diffuse * w;
  r.translucent = c.translucent * w;
  r.reflection = c.reflection * w;
  r.coat = c.coat * w;
  r.refraction = c.refraction * w;
  r.emission = c.emission * w;
  r.transmittance = c.transmittance * w;
  r.holdout = c.holdout * w;
  return r;
}

// Stochastic choice between lobes a and b of one kind (EEVEE picks one closure per bin per sample, weighted by its
// colour). Returns true to keep b. Consumes and rescales *rng so later picks stay uniform.
fn mks_pick(wa: f32, wb: f32, rng: ptr<function, f32>) -> bool {
  let p = wb / (wa + wb);
  let r = *rng;
  if (r < p) {
    *rng = r / p;
    return true;
  }
  *rng = (r - p) / max(1.0 - p, 1e-8);
  return false;
}

fn mks_lobe_w(c: vec3f) -> f32 {
  return mks_avg3(abs(c));
}

// a + b. Lobes of one kind with equal parameters add exactly; with different normals / roughness one is kept per
// sample, its colour scaled to the summed weight (unbiased over samples, as EEVEE).
fn mks_closures_add(a: Closures, b: Closures, rng: ptr<function, f32>) -> Closures {
  var r = a;
  r.emission = a.emission + b.emission;
  r.transmittance = a.transmittance + b.transmittance;
  r.holdout = a.holdout + b.holdout;

  var wa = mks_lobe_w(a.diffuse);
  var wb = mks_lobe_w(b.diffuse);
  if (wb > 0.0) {
    if (wa <= 0.0) {
      r.diffuse = b.diffuse;
      r.diffuse_N = b.diffuse_N;
    } else if (all(a.diffuse_N == b.diffuse_N)) {
      r.diffuse = a.diffuse + b.diffuse;
    } else if (mks_pick(wa, wb, rng)) {
      r.diffuse = b.diffuse * ((wa + wb) / wb);
      r.diffuse_N = b.diffuse_N;
    } else {
      r.diffuse = a.diffuse * ((wa + wb) / wa);
    }
  }

  wa = mks_lobe_w(a.translucent);
  wb = mks_lobe_w(b.translucent);
  if (wb > 0.0) {
    if (wa <= 0.0) {
      r.translucent = b.translucent;
      r.translucent_N = b.translucent_N;
    } else if (all(a.translucent_N == b.translucent_N)) {
      r.translucent = a.translucent + b.translucent;
    } else if (mks_pick(wa, wb, rng)) {
      r.translucent = b.translucent * ((wa + wb) / wb);
      r.translucent_N = b.translucent_N;
    } else {
      r.translucent = a.translucent * ((wa + wb) / wa);
    }
  }

  // Reflection lobes: a second distinct lobe goes to the coat slot when that is free (EEVEE keeps more than one
  // reflection closure per pixel), else the two are merged stochastically.
  var refl = r.reflection;
  var refl_N = r.reflection_N;
  var refl_r = r.reflection_roughness;
  var coat = a.coat;
  var coat_N = a.coat_N;
  var coat_r = a.coat_roughness;
  for (var k = 0; k < 2; k++) {
    var c = b.reflection;
    var cN = b.reflection_N;
    var cr = b.reflection_roughness;
    if (k == 1) {
      c = b.coat;
      cN = b.coat_N;
      cr = b.coat_roughness;
    }
    let wc = mks_lobe_w(c);
    if (wc <= 0.0) {
      continue;
    }
    let w0 = mks_lobe_w(refl);
    let w1 = mks_lobe_w(coat);
    if (w0 <= 0.0) {
      refl = c;
      refl_N = cN;
      refl_r = cr;
    } else if (all(refl_N == cN) && refl_r == cr) {
      refl = refl + c;
    } else if (w1 <= 0.0) {
      coat = c;
      coat_N = cN;
      coat_r = cr;
    } else if (all(coat_N == cN) && coat_r == cr) {
      coat = coat + c;
    } else if (mks_pick(w1, wc, rng)) {
      coat = c * ((w1 + wc) / wc);
      coat_N = cN;
      coat_r = cr;
    } else {
      coat = coat * ((w1 + wc) / w1);
    }
  }
  r.reflection = refl;
  r.reflection_N = refl_N;
  r.reflection_roughness = refl_r;
  r.coat = coat;
  r.coat_N = coat_N;
  r.coat_roughness = coat_r;

  wa = mks_lobe_w(a.refraction);
  wb = mks_lobe_w(b.refraction);
  if (wb > 0.0) {
    if (wa <= 0.0) {
      r.refraction = b.refraction;
      r.refraction_N = b.refraction_N;
      r.refraction_roughness = b.refraction_roughness;
      r.refraction_ior = b.refraction_ior;
    } else if (all(a.refraction_N == b.refraction_N) && a.refraction_roughness == b.refraction_roughness &&
               a.refraction_ior == b.refraction_ior) {
      r.refraction = a.refraction + b.refraction;
    } else if (mks_pick(wa, wb, rng)) {
      r.refraction = b.refraction * ((wa + wb) / wb);
      r.refraction_N = b.refraction_N;
      r.refraction_roughness = b.refraction_roughness;
      r.refraction_ior = b.refraction_ior;
    } else {
      r.refraction = a.refraction * ((wa + wb) / wa);
    }
  }
  return r;
}

// Mix Shader: (1 - fac) a + fac b, fac clamped to 0..1.
fn mks_closures_mix(a: Closures, b: Closures, fac: f32, rng: ptr<function, f32>) -> Closures {
  let f = clamp(fac, 0.0, 1.0);
  return mks_closures_add(mks_closures_scale(a, 1.0 - f), mks_closures_scale(b, f), rng);
}

// A colour or value linked into a shader socket acts as an emission of that colour.
fn mks_closure_from_color(c: vec3f) -> Closures {
  var r = mks_closures_none();
  r.emission = c;
  return r;
}
