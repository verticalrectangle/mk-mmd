// Area lights through Linearly Transformed Cosines (Heitz, Dupuy, Hill, Neubelt 2016, "Real-Time Polygonal-Light
// Shading with Linearly Transformed Cosines"): the GGX lobe and Lambert integrated over polygonal lights. Two tables
// (tools/ltc_fit.rs, params ltc_glossy picks the one uploaded): EEVEE's, measured from Blender's frames, and mkr's own
// fit. src/ltc.rs documents them and holds a CPU port of this file for its tests: keep the two in sync line for line.
//
// No bindings here. The renderer samples the 64 x 64 rgba32float table (ltc::table(fit), texel (x = j, y = i) = entry
// i * 64 + j) with a linear clamp-to-edge sampler:
//
//   let Minv = ltc_minv(textureSampleLevel(ltc_tex, ltc_sampler, ltc_matrix_coords(roughness, dot(N, V)), 0.0));
//   let glossy = ltc_quad(N, V, P, Minv, corners);          // normalised GGX lobe over the light, 0..1
//   let diffuse = ltc_quad(N, V, P, LTC_LAMBERT, corners);  // cos / pi over the light, 0..1
//
// N: unit shading normal. V: unit vector from P toward the viewer. P: the shading point. Lights in world space.
// ltc_quad / ltc_disk return the integral over the light, as seen from P and clipped to the hemisphere of N, of
// D ~ BRDF(V, L) cos(L) / E(V) (GGX, alpha = roughness^2, F = 1, height-correlated Smith G2, E the directional albedo:
// closure colours carry Fresnel and albedo), so a light covering the hemisphere gives 1. Lambert: cos(L) / pi.
// Winding does not matter and the result is never negative; one-sidedness is the caller's.
//
// D is the clamped cosine carried through M (= Minv^-1), restricted to the hemisphere of N and divided by its mass
// there, (1 + cos b) / 2 (ltc_mass_above): mkr's fit matches that clipped, renormalised lobe.
//
// ltc_quad_sphere / ltc_disk_sphere (the renderer's default, params ltc_sphere_*) take EEVEE's way with the horizon
// instead: no clipping to N's hemisphere and no renormalisation, the polygon's form factor in the cosine's space
// integrated as a sphere over that space's hemisphere. Lambert light wholly above the horizon gives the same. EEVEE's
// table is a cosine normalised over the whole sphere, fitted for this path: what lies under N's horizon is lost, as
// in EEVEE (a light covering the hemisphere gives (1 + cos b) / 2). With it the renderer also fades lights out of the
// glossy lobes once they sink under the horizon, as EEVEE does (ltc_horizon_fade).

const LTC_LUT_SIZE: f32 = 64.0;
const LTC_PI: f32 = 3.14159265358979323846;
const LTC_LAMBERT: mat3x3f = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

// Disk and ellipse lights: the inscribed 32-gon scaled to the ellipse's area (radius x sqrt(2 pi / (32 sin(pi / 16))):
// vertices 0.32 % outside the ellipse, edge midpoints 0.16 % inside, so even a mirror's reflection stays round).
const LTC_DISK_SIDES: u32 = 32u;
const LTC_DISK_SCALE: f32 = 1.00322211;
const LTC_DISK_STEP: vec2f = vec2f(0.980785251, 0.195090324); // cos, sin of 2 pi / 32

// uv of the table entry for (roughness, cos(N, V)): x = sqrt(1 - cos) and y = roughness, each 0..1 over the texel
// centres 0..63.
fn ltc_matrix_coords(roughness: f32, cos_nv: f32) -> vec2f {
  let t = vec2f(sqrt(1.0 - clamp(cos_nv, 0.0, 1.0)), clamp(roughness, 0.0, 1.0));
  return (t * (LTC_LUT_SIZE - 1.0) + 0.5) / LTC_LUT_SIZE;
}

// The table entry (m00, m02, m20, m22) as Minv = [[m00, 0, m02], [0, 1, 0], [m20, 0, m22]] (rows) in the shading
// frame (T1, T2, N).
fn ltc_minv(c: vec4f) -> mat3x3f {
  return mat3x3f(vec3f(c.x, 0.0, c.z), vec3f(0.0, 1.0, 0.0), vec3f(c.y, 0.0, c.w));
}

// World to the shading frame: rows T1 (toward V's part along the surface), T2 = N x T1, N.
fn ltc_frame(N: vec3f, V: vec3f) -> mat3x3f {
  var t1 = V - N * dot(N, V);
  let l2 = dot(t1, t1);
  if (l2 > 1e-12) {
    t1 = t1 * inverseSqrt(l2);
  } else {
    // V along N: the lobe is isotropic, any tangent does (Duff et al. 2017).
    let s = select(-1.0, 1.0, N.z >= 0.0);
    let a = -1.0 / (s + N.z);
    t1 = vec3f(1.0 + s * N.x * N.x * a, s * N.x * N.y * a, -s * N.x);
  }
  return transpose(mat3x3f(t1, cross(N, t1), N));
}

// acos(a.b / |a| |b|) normalize(a x b).z: 2 pi times the z part of the edge's vector form factor.
fn ltc_edge(a: vec3f, b: vec3f) -> f32 {
  let c = cross(a, b);
  let s = length(c);
  return c.z * (atan2(s, dot(a, b)) / max(s, 1e-30));
}

// The polygon's edges, carried into the cosine's space (unit vectors), clipped to N's hemisphere (dot(n, q) >= 0)
// and to the cosine's (q.z >= 0) and summed as they come; a convex polygon leaves each half-space once.
struct LtcAcc {
  n: vec3f,       // N's horizon normal in the cosine's space: dot(n, Minv p) = p.z
  sum: f32,
  exit_n: vec3f,  // where the polygon leaves N's hemisphere / enters it again
  entry_n: vec3f,
  exit_z: vec3f,  // where it leaves the cosine's hemisphere / enters it again
  entry_z: vec3f,
  flags: u32,     // 1 exit_n, 2 entry_n, 4 exit_z, 8 entry_z
}

fn ltc_acc_new(Minv: mat3x3f) -> LtcAcc {
  let z = vec3f(0.0);
  return LtcAcc(cross(Minv[0], Minv[1]) * sign(determinant(Minv)), 0.0, z, z, z, z, 0u);
}

fn ltc_acc_cosine_edge(acc: ptr<function, LtcAcc>, a: vec3f, b: vec3f) {
  if (a.z >= 0.0 && b.z >= 0.0) {
    (*acc).sum += ltc_edge(a, b);
  } else if (a.z >= 0.0) {
    let x = a + (b - a) * (a.z / (a.z - b.z));
    (*acc).sum += ltc_edge(a, x);
    (*acc).exit_z = x;
    (*acc).flags |= 4u;
  } else if (b.z >= 0.0) {
    let x = a + (b - a) * (a.z / (a.z - b.z));
    (*acc).sum += ltc_edge(x, b);
    (*acc).entry_z = x;
    (*acc).flags |= 8u;
  }
}

fn ltc_acc_edge(acc: ptr<function, LtcAcc>, a: vec3f, b: vec3f) {
  let ha = dot((*acc).n, a);
  let hb = dot((*acc).n, b);
  if (ha >= 0.0 && hb >= 0.0) {
    ltc_acc_cosine_edge(acc, a, b);
  } else if (ha >= 0.0) {
    let x = a + (b - a) * (ha / (ha - hb));
    ltc_acc_cosine_edge(acc, a, x);
    (*acc).exit_n = x;
    (*acc).flags |= 1u;
  } else if (hb >= 0.0) {
    let x = a + (b - a) * (ha / (ha - hb));
    ltc_acc_cosine_edge(acc, x, b);
    (*acc).entry_n = x;
    (*acc).flags |= 2u;
  }
}

// Share of the clamped cosine carried through M that lies in N's hemisphere: (1 + cos b) / 2, b the angle between
// the cosine's axis and n.
fn ltc_mass_above(n: vec3f) -> f32 {
  return max(0.5 + 0.5 * n.z / length(n), 1e-6);
}

fn ltc_acc_finish(acc: ptr<function, LtcAcc>) -> f32 {
  if (((*acc).flags & 3u) == 3u) {
    ltc_acc_cosine_edge(acc, (*acc).exit_n, (*acc).entry_n);
  }
  if (((*acc).flags & 12u) == 12u) {
    (*acc).sum += ltc_edge((*acc).exit_z, (*acc).entry_z);
  }
  return abs((*acc).sum) / (2.0 * LTC_PI * ltc_mass_above((*acc).n));
}

// Quad light, corners in order around it (either way round).
fn ltc_quad(N: vec3f, V: vec3f, P: vec3f, Minv: mat3x3f, corners: array<vec3f, 4>) -> f32 {
  let m = Minv * ltc_frame(N, V);
  let q0 = normalize(m * (corners[0] - P));
  let q1 = normalize(m * (corners[1] - P));
  let q2 = normalize(m * (corners[2] - P));
  let q3 = normalize(m * (corners[3] - P));
  var acc = ltc_acc_new(Minv);
  ltc_acc_edge(&acc, q0, q1);
  ltc_acc_edge(&acc, q1, q2);
  ltc_acc_edge(&acc, q2, q3);
  ltc_acc_edge(&acc, q3, q0);
  return ltc_acc_finish(&acc);
}

// Disk or ellipse light: centre and the two half axes (world space; for a sphere light, the disk of its radius facing
// P), as the area-matched 32-gon.
fn ltc_disk(N: vec3f, V: vec3f, P: vec3f, Minv: mat3x3f, center: vec3f, half_x: vec3f, half_y: vec3f) -> f32 {
  let m = Minv * ltc_frame(N, V);
  let c = m * (center - P);
  let x = m * half_x * LTC_DISK_SCALE;
  let y = m * half_y * LTC_DISK_SCALE;
  var acc = ltc_acc_new(Minv);
  let first = normalize(c + x);
  var prev = first;
  var cs = vec2f(1.0, 0.0);
  for (var i = 1u; i < LTC_DISK_SIDES; i++) {
    cs = vec2f(cs.x * LTC_DISK_STEP.x - cs.y * LTC_DISK_STEP.y, cs.y * LTC_DISK_STEP.x + cs.x * LTC_DISK_STEP.y);
    let q = normalize(c + x * cs.x + y * cs.y);
    ltc_acc_edge(&acc, prev, q);
    prev = q;
  }
  ltc_acc_edge(&acc, prev, first);
  return ltc_acc_finish(&acc);
}

// EEVEE's way with the horizon (the sphere approximation of the LTC authors' production notes): the polygon's vector
// form factor F in the cosine's space, and the integral over the cosine's hemisphere of a sphere with that form factor
// and direction, by the exact clipped-sphere formula (Lagarde and de Rousiers 2014, "Moving Frostbite to PBR",
// 4.7.2). N's hemisphere is not clipped and the lobe not renormalised.
fn ltc_edge_vec(a: vec3f, b: vec3f) -> vec3f {
  let c = cross(a, b);
  let s = length(c);
  return c * (atan2(s, dot(a, b)) / max(s, 1e-30));
}

// A sphere of form factor f (sin² of its half angle) whose centre lies at cos z from the axis, over the hemisphere.
fn ltc_sphere_integral(z: f32, f: f32) -> f32 {
  if (z * z >= f) {
    return f * max(z, 0.0);
  }
  let st = sqrt(max(1.0 - z * z, 0.0));
  let x = sqrt(max(1.0 / f - 1.0, 0.0));
  let y = clamp(-x * z / max(st, 1e-12), -1.0, 1.0);
  let s = st * sqrt(max(1.0 - y * y, 0.0));
  return max(((z * acos(y) - x * s) * f + atan2(s, x)) / LTC_PI, 0.0);
}

// F summed over the polygon's edges (2 pi times the vector form factor), `toward` a direction toward the polygon.
fn ltc_sphere_finish(F_in: vec3f, toward: vec3f) -> f32 {
  let F = F_in * select(1.0, -1.0, dot(F_in, toward) < 0.0) / (2.0 * LTC_PI);
  let f = length(F);
  if (f <= 1e-12) {
    return 0.0;
  }
  return ltc_sphere_integral(F.z / f, min(f, 1.0));
}

fn ltc_quad_sphere(N: vec3f, V: vec3f, P: vec3f, Minv: mat3x3f, corners: array<vec3f, 4>) -> f32 {
  let m = Minv * ltc_frame(N, V);
  let q0 = normalize(m * (corners[0] - P));
  let q1 = normalize(m * (corners[1] - P));
  let q2 = normalize(m * (corners[2] - P));
  let q3 = normalize(m * (corners[3] - P));
  let F = ltc_edge_vec(q0, q1) + ltc_edge_vec(q1, q2) + ltc_edge_vec(q2, q3) + ltc_edge_vec(q3, q0);
  return ltc_sphere_finish(F, q0 + q1 + q2 + q3);
}

fn ltc_disk_sphere(N: vec3f, V: vec3f, P: vec3f, Minv: mat3x3f, center: vec3f, half_x: vec3f, half_y: vec3f) -> f32 {
  let m = Minv * ltc_frame(N, V);
  let c = m * (center - P);
  let x = m * half_x * LTC_DISK_SCALE;
  let y = m * half_y * LTC_DISK_SCALE;
  let first = normalize(c + x);
  var prev = first;
  var F = vec3f(0.0);
  var cs = vec2f(1.0, 0.0);
  for (var i = 1u; i < LTC_DISK_SIDES; i++) {
    cs = vec2f(cs.x * LTC_DISK_STEP.x - cs.y * LTC_DISK_STEP.y, cs.y * LTC_DISK_STEP.x + cs.x * LTC_DISK_STEP.y);
    let q = normalize(c + x * cs.x + y * cs.y);
    F += ltc_edge_vec(prev, q);
    prev = q;
  }
  F += ltc_edge_vec(prev, first);
  return ltc_sphere_finish(F, c);
}

// EEVEE's fade of a light out of a glossy lobe once the sphere bounding the light sinks under the lobe's horizon
// (measured on rectangle, disk, sphere and sun lights under a glossy plane): 1 until the top of that sphere reaches the
// horizon, then down to 0 over a tenth in sine. The sphere horizon above clips no tail of the cosine under N's
// horizon, so without it a light wholly behind the surface still lights a lobe seen obliquely. `toward`: unit
// direction toward the light's centre; `r`: its bounding radius over its distance (the sun: tan of its half angle).
fn ltc_horizon_fade(N: vec3f, toward: vec3f, r: f32) -> f32 {
  return clamp(1.0 + 10.0 * (dot(N, toward) + r), 0.0, 1.0);
}
