//! Fits mkr's GGX Linearly Transformed Cosines tables (src/ltc.rs, src/shaders/ltc.wgsl) from the maths. std only:
//!
//!     rustc -O --edition 2021 tools/ltc_fit.rs -o /tmp/ltc_fit
//!     /tmp/ltc_fit [CRATE_DIR]            fit, write CRATE_DIR/src/ltc_ggx.bin and src/ltc_ggx_albedo.bin
//!     /tmp/ltc_fit --check [CRATE_DIR]    re-integrate the written table against brute force on random rectangles
//!     /tmp/ltc_fit --eevee [CRATE_DIR]    fit EEVEE's table (section "EEVEE's table"), write src/ltc_ggx_eevee.bin
//!
//! CRATE_DIR defaults to the current directory (the mkr crate root). Below, up to that section: mkr's own table.
//!
//! Model (Heitz, Dupuy, Hill, Neubelt 2016, "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines").
//! In the shading frame (T1, T2, N), T1 in the plane of V and N so that V = (sin t, 0, cos t), the target is the
//! normalised GGX lobe f(L) = BRDF(V, L) cos(L) / E(V): F = 1, height-correlated Smith G2, alpha = roughness^2, E the
//! directional albedo, so f integrates to 1 over the upper hemisphere. It is approximated by a clamped cosine
//! D_o(w) = max(w.z, 0) / pi carried through a matrix M: D(L) = D_o(Minv L / |Minv L|) |det Minv| / |Minv L|^3.
//! mkr integrates D over the light clipped to the upper hemisphere of N and divides by the LTC's own mass above that
//! horizon, A = (1 + cos b) / 2 with b the angle between the cosine's axis and the third row of M (N's horizon normal
//! in the cosine's space, since (M q).z = M[2] . q). A polygon covering the hemisphere then gives exactly 1, and the
//! fit spends all four parameters on the lobe's shape where light can arrive.
//!
//! Fit: per cell, minimise the symmetric chi-square  integral over z > 0 of (D/A - f)^2 / (D/A + f)  (the lobe-weighted
//! squared relative error), estimated with multiple importance sampling (balance heuristic) of the GGX visible
//! normals and of the LTC on fixed low-discrepancy points, by Nelder-Mead over (lobe axis angle relative to the lobe's
//! mean direction, log of the two scales, skew); the first cell of each roughness row (V = N) is a 1D isotropic fit,
//! the others are warm-started from the previous view angle. M = R(phi) [[sx, 0, k], [0, sy, 0], [0, 0, 1]], R a
//! rotation about T2, phi = axis - atan(k). Of the metrics tried on polygon integrals (chi-square, Hellinger, L1, L2,
//! the paper's L3, relative), chi-square and Hellinger came out best and L2 / L3 worst. No LTC reaches 3 % on small
//! polygons for rough lobes at grazing view (`--check` prints the envelope).
//!
//! Table: 64 x 64 cells, cell (i, j) at index i * 64 + j: roughness = i / 63 (alpha = max(roughness^2, ALPHA_MIN)),
//! sqrt(1 - cos(theta_v)) = j / 63, except the last column (cos 0) which holds the fit at sqrt(1 - cos) = 62.5 / 63
//! (LAST_V): toward 90 degrees the lobe collapses into a sliver no LTC follows (L1 above 0.6 at cos 1e-4).
//! ltc_ggx.bin: 4 little-endian f32 per cell, (m00, m02, m20, m22) of Minv scaled so that m11 = 1, i.e.
//! Minv = [[x, 0, y], [0, 1, 0], [z, 0, w]] in (T1, T2, N); this scaling keeps every coefficient bounded and smooth down
//! to mirror-like lobes, so bilinear filtering is accurate. ltc_ggx_albedo.bin: one little-endian f32 per cell,
//! E(V, alpha).

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const N: usize = 64;
/// GGX alpha of the roughness-0 row: roughness below 0.01 is drawn as 0.01 (a lobe of about 0.01 degree).
const ALPHA_MIN: f64 = 1e-4;
/// sqrt(1 - cos(theta_v)) of the last column's fit (cos 0.0158, 89.1 degrees).
const LAST_V: f64 = 62.5 / 63.0;
/// Low-discrepancy points per sampling strategy in the fit's error estimate.
const FIT_SAMPLES: usize = 1 << 14;
/// Points for E(V, alpha) and the lobe's mean direction.
const ALBEDO_SAMPLES: usize = 1 << 20;
/// Points for the per-cell L1 report and for each brute-force rectangle in --check.
const CHECK_SAMPLES: usize = 1 << 18;

// ---------------------------------------------------------------------------------------------------------------
// Vectors and matrices

#[derive(Clone, Copy, Debug, Default)]
struct V3 {
    x: f64,
    y: f64,
    z: f64,
}

const fn v3(x: f64, y: f64, z: f64) -> V3 {
    V3 { x, y, z }
}

impl std::ops::Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<f64> for V3 {
    type Output = V3;
    fn mul(self, s: f64) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}

fn dot(a: V3, b: V3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a: V3, b: V3) -> V3 {
    v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

fn length(a: V3) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: V3) -> V3 {
    a * (1.0 / length(a))
}

/// Row-major 3x3.
#[derive(Clone, Copy, Debug)]
struct M3([[f64; 3]; 3]);

impl M3 {
    const IDENTITY: M3 = M3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    fn mul(&self, v: V3) -> V3 {
        let m = &self.0;
        v3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }

    fn row(&self, i: usize) -> V3 {
        v3(self.0[i][0], self.0[i][1], self.0[i][2])
    }

    fn col(&self, j: usize) -> V3 {
        v3(self.0[0][j], self.0[1][j], self.0[2][j])
    }

    fn det(&self) -> f64 {
        dot(self.col(0), cross(self.col(1), self.col(2)))
    }

    /// The rows of the inverse are the reciprocal basis of the columns.
    fn inverse(&self) -> M3 {
        let (c0, c1, c2) = (self.col(0), self.col(1), self.col(2));
        let d = 1.0 / self.det();
        let (r0, r1, r2) = (cross(c1, c2) * d, cross(c2, c0) * d, cross(c0, c1) * d);
        M3([[r0.x, r0.y, r0.z], [r1.x, r1.y, r1.z], [r2.x, r2.y, r2.z]])
    }

    fn scale(&self, s: f64) -> M3 {
        M3(self.0.map(|r| r.map(|v| v * s)))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Sampling points

/// Point i of n of a stratified Kronecker lattice in [0, 1)^2, shifted (Cranley-Patterson) by `shift`.
fn lattice(i: usize, n: usize, shift: (f64, f64)) -> (f64, f64) {
    const G: f64 = 0.618_033_988_749_894_9;
    let u = (i as f64 + 0.5) / n as f64 + shift.0;
    let v = i as f64 * G + shift.1;
    (u - u.floor(), v - v.floor())
}

fn cosine_dir(u: f64, v: f64) -> V3 {
    let r = u.sqrt();
    let (s, c) = (2.0 * PI * v).sin_cos();
    v3(r * c, r * s, (1.0 - u).max(0.0).sqrt())
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GGX with height-correlated Smith masking, F = 1

/// Smith Lambda of a direction with cosine `c` to N.
fn lambda(c: f64, a2: f64) -> f64 {
    if c >= 1.0 {
        return 0.0;
    }
    let q2 = a2 * (1.0 - c * c) / (c * c);
    if !(q2 < 1e30) {
        return 0.5 * q2.sqrt();
    }
    q2 / (2.0 * ((1.0 + q2).sqrt() + 1.0))
}

#[derive(Clone, Copy)]
struct Ggx {
    alpha: f64,
    a2: f64,
    v: V3,
    lambda_v: f64,
    g1v: f64,
}

impl Ggx {
    fn new(alpha: f64, cos_v: f64) -> Ggx {
        let a2 = alpha * alpha;
        let v = v3((1.0 - cos_v * cos_v).max(0.0).sqrt(), 0.0, cos_v);
        let lambda_v = lambda(cos_v, a2);
        Ggx { alpha, a2, v, lambda_v, g1v: 1.0 / (1.0 + lambda_v) }
    }

    fn ndf(&self, nh: f64) -> f64 {
        if nh <= 0.0 {
            return 0.0;
        }
        let t = nh * nh * (self.a2 - 1.0) + 1.0;
        self.a2 / (PI * t * t)
    }

    /// (BRDF(V, l) cos(l), pdf of l under visible-normal sampling), both per steradian.
    fn eval(&self, l: V3) -> (f64, f64) {
        let h = self.v + l;
        let hl = length(h);
        if hl <= 0.0 {
            return (0.0, 0.0);
        }
        let d = self.ndf(h.z / hl);
        let pdf = self.g1v * d / (4.0 * self.v.z);
        if l.z <= 0.0 {
            return (0.0, pdf);
        }
        let g2 = 1.0 / (1.0 + self.lambda_v + lambda(l.z, self.a2));
        (d * g2 / (4.0 * self.v.z), pdf)
    }

    /// G2 / G1(V): the weight of a visible-normal sample of BRDF cos.
    fn weight(&self, l: V3) -> f64 {
        if l.z <= 0.0 {
            return 0.0;
        }
        (1.0 + self.lambda_v) / (1.0 + self.lambda_v + lambda(l.z, self.a2))
    }

    /// Reflected direction of a visible normal (spherical-cap sampling of the stretched hemisphere).
    fn sample(&self, u: f64, w: f64) -> V3 {
        let a = self.alpha;
        let vh = normalize(v3(a * self.v.x, a * self.v.y, self.v.z));
        let (sp, cp) = (2.0 * PI * u).sin_cos();
        let z = (1.0 - w) * (1.0 + vh.z) - vh.z;
        let s = (1.0 - z * z).max(0.0).sqrt();
        let hs = v3(s * cp, s * sp, z) + vh;
        let h = normalize(v3(a * hs.x, a * hs.y, hs.z));
        h * (2.0 * dot(self.v, h)) - self.v
    }

    /// (E, angle of the lobe's mean direction from N toward +T1).
    fn albedo(&self, n: usize) -> (f64, f64) {
        let (mut e, mut m) = (0.0, v3(0.0, 0.0, 0.0));
        for i in 0..n {
            let (u, w) = lattice(i, n, (0.0, 0.0));
            let l = self.sample(u, w);
            let g = self.weight(l);
            e += g;
            m = m + l * g;
        }
        (e / n as f64, m.x.atan2(m.z))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// LTC

/// Fit parameters: lobe axis angle relative to the lobe's mean direction, ln sx, ln sy, skew k.
type Params = [f64; 4];

#[derive(Clone, Copy)]
struct Ltc {
    m: M3,
    minv: M3,
    abs_det_m: f64,
    /// Share of the distribution above the horizon of N.
    above: f64,
}

/// (1 + cos b) / 2: the clamped cosine's mass on the side of the plane with normal n (b the angle from its axis).
fn mass_above(n: V3) -> f64 {
    0.5 + 0.5 * n.z / length(n)
}

impl Ltc {
    fn new(p: &Params, axis_ref: f64) -> Ltc {
        let k = p[3];
        let phi = axis_ref + p[0] - k.atan();
        let (sx, sy) = (p[1].exp(), p[2].exp());
        let (s, c) = phi.sin_cos();
        Ltc::from_m(M3([[c * sx, 0.0, c * k + s], [0.0, sy, 0.0], [-s * sx, 0.0, c - s * k]]))
    }

    fn from_m(m: M3) -> Ltc {
        let det = m.det();
        Ltc { m, minv: m.inverse(), abs_det_m: det.abs(), above: mass_above(m.row(2) * det.signum()) }
    }

    /// The (unclipped, sphere-normalised) LTC density.
    fn eval(&self, w: V3) -> f64 {
        let o = self.minv.mul(w);
        if o.z <= 0.0 {
            return 0.0;
        }
        let l2 = dot(o, o);
        o.z / (PI * l2 * l2 * self.abs_det_m)
    }

    /// Direction and density of the image of a cosine-distributed direction `o`.
    fn sample(&self, o: V3) -> (V3, f64) {
        let w = self.m.mul(o);
        let l = length(w);
        (w * (1.0 / l), o.z * l * l * l / (PI * self.abs_det_m))
    }

    /// Minv scaled to m11 = 1, as stored: (m00, m02, m20, m22).
    fn coefficients(&self) -> [f64; 4] {
        let q = self.minv.scale(1.0 / self.minv.0[1][1]);
        [q.0[0][0], q.0[0][2], q.0[2][0], q.0[2][2]]
    }
}

// ---------------------------------------------------------------------------------------------------------------
// One table cell

struct BrdfPoint {
    l: V3,
    f: f64,
    pdf: f64,
}

struct Cell {
    ggx: Ggx,
    e: f64,
    /// Angle of the lobe's mean direction from N (toward +T1).
    axis: f64,
    /// Visible-normal samples above the horizon (the others carry no error), with f normalised by E.
    brdf: Vec<BrdfPoint>,
    /// Cosine-distributed directions, carried through M by each candidate.
    cosine: Vec<V3>,
    n: usize,
}

/// Symmetric chi-square integrand.
fn chi2(d: f64, f: f64) -> f64 {
    let s = d + f;
    if s <= 0.0 { 0.0 } else { (d - f) * (d - f) / s }
}

impl Cell {
    fn new(alpha: f64, cos_v: f64, n: usize, shift: (f64, f64)) -> Cell {
        let ggx = Ggx::new(alpha, cos_v);
        let (e, mut axis) = ggx.albedo(ALBEDO_SAMPLES);
        if cos_v >= 1.0 {
            axis = 0.0;
        }
        Cell::with_points(ggx, e, axis, n, shift)
    }

    /// The same lobe on other points.
    fn resampled(&self, n: usize, shift: (f64, f64)) -> Cell {
        Cell::with_points(self.ggx, self.e, self.axis, n, shift)
    }

    fn with_points(ggx: Ggx, e: f64, axis: f64, n: usize, shift: (f64, f64)) -> Cell {
        let mut brdf = Vec::with_capacity(n);
        let mut cosine = Vec::with_capacity(n);
        for i in 0..n {
            let (u, w) = lattice(i, n, shift);
            let l = ggx.sample(u, w);
            if l.z > 0.0 {
                let (f, pdf) = ggx.eval(l);
                brdf.push(BrdfPoint { l, f: f / e, pdf });
            }
            cosine.push(cosine_dir(u, w));
        }
        Cell { ggx, e, axis, brdf, cosine, n }
    }

    /// MIS estimate of the integral over z > 0 of err(D / A, f).
    fn error(&self, ltc: &Ltc, err: fn(f64, f64) -> f64) -> f64 {
        let mut s = 0.0;
        for b in &self.brdf {
            let pl = ltc.eval(b.l);
            s += err(pl / ltc.above, b.f) / (pl + b.pdf);
        }
        for &o in &self.cosine {
            let (w, pl) = ltc.sample(o);
            if w.z <= 0.0 {
                continue;
            }
            let (f, pb) = self.ggx.eval(w);
            s += err(pl / ltc.above, f / self.e) / (pl + pb);
        }
        s / self.n as f64
    }

    fn cost(&self, p: &Params) -> f64 {
        self.error(&Ltc::new(p, self.axis), chi2)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Minimisers

fn golden(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let g = 0.5 * (5f64.sqrt() - 1.0);
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    while b - a > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

fn nelder_mead(f: &dyn Fn(&Params) -> f64, x0: Params, step: Params, max_evals: usize) -> (Params, f64) {
    let mut s: Vec<(Params, f64)> = Vec::with_capacity(5);
    s.push((x0, f(&x0)));
    for i in 0..4 {
        let mut x = x0;
        x[i] += step[i];
        s.push((x, f(&x)));
    }
    let mut evals = 5;
    let at = |c: &Params, d: &Params, t: f64| -> Params { std::array::from_fn(|i| c[i] + t * (d[i] - c[i])) };
    while evals < max_evals {
        s.sort_by(|a, b| a.1.total_cmp(&b.1));
        if s[4].1 - s[0].1 <= 1e-9 * s[0].1.abs() {
            break;
        }
        let mut c = [0.0; 4];
        for v in &s[..4] {
            for i in 0..4 {
                c[i] += 0.25 * v.0[i];
            }
        }
        let xr = at(&c, &s[4].0, -1.0);
        let fr = f(&xr);
        evals += 1;
        if fr < s[0].1 {
            let xe = at(&c, &s[4].0, -2.0);
            let fe = f(&xe);
            evals += 1;
            s[4] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < s[3].1 {
            s[4] = (xr, fr);
        } else {
            let xc = if fr < s[4].1 { at(&c, &xr, 0.5) } else { at(&c, &s[4].0, 0.5) };
            let fc = f(&xc);
            evals += 1;
            if fc < fr.min(s[4].1) {
                s[4] = (xc, fc);
            } else {
                let best = s[0].0;
                for v in &mut s[1..] {
                    v.0 = at(&best, &v.0, 0.5);
                    v.1 = f(&v.0);
                }
                evals += 4;
            }
        }
    }
    s.sort_by(|a, b| a.1.total_cmp(&b.1));
    s[0]
}

/// V = N: the lobe is isotropic about N, one scale.
fn fit_isotropic(cell: &Cell) -> Params {
    let f = |ls: f64| cell.cost(&[0.0, ls, ls, 0.0]);
    let la = cell.ggx.alpha.ln();
    let (lo, hi, steps) = (la - 1.5, la + 2.5, 80);
    let mut best = (f64::INFINITY, lo);
    for s in 0..=steps {
        let x = lo + (hi - lo) * s as f64 / steps as f64;
        let v = f(x);
        if v < best.0 {
            best = (v, x);
        }
    }
    let h = (hi - lo) / steps as f64;
    let ls = golden(&f, best.1 - h, best.1 + h, 1e-9);
    [0.0, ls, ls, 0.0]
}

fn fit_cell(cell: &Cell, x0: Params) -> Params {
    let f = |p: &Params| cell.cost(p);
    let (mut x, mut fx) = (x0, f(&x0));
    let mut scale = 1.0;
    for round in 0..6 {
        let s_min = x[1].exp().min(x[2].exp()).min(1.0);
        let step = [0.2 * s_min * scale, 0.1 * scale, 0.1 * scale, 0.05 * scale];
        let (xn, fxn) = nelder_mead(&f, x, step, 3000);
        let gain = fx - fxn;
        if fxn < fx {
            x = xn;
            fx = fxn;
        }
        if round > 0 && gain <= 1e-7 * fx.abs() {
            break;
        }
        scale *= 0.5;
    }
    x
}

// ---------------------------------------------------------------------------------------------------------------
// Table

fn grid_roughness(i: usize) -> f64 {
    i as f64 / (N - 1) as f64
}

fn alpha_of(roughness: f64) -> f64 {
    (roughness * roughness).max(ALPHA_MIN)
}

fn grid_cos(j: usize) -> f64 {
    let v = if j == N - 1 { LAST_V } else { j as f64 / (N - 1) as f64 };
    1.0 - v * v
}

#[derive(Clone, Copy, Default)]
struct CellFit {
    coef: [f64; 4],
    e: f64,
    above: f64,
    /// Integral over z > 0 of |D / A - f| (0 perfect, 2 disjoint), on points the fit did not use.
    l1: f64,
}

fn l1(d: f64, f: f64) -> f64 {
    (d - f).abs()
}

fn fit_row(i: usize) -> Vec<CellFit> {
    let alpha = alpha_of(grid_roughness(i));
    let mut out = Vec::with_capacity(N);
    let mut prev: Option<(Params, f64)> = None;
    for j in 0..N {
        let cos_v = grid_cos(j);
        let cell = Cell::new(alpha, cos_v, FIT_SAMPLES, (0.0, 0.0));
        let p = match prev {
            None => fit_isotropic(&cell),
            Some((p, prev_cos)) => {
                // Toward grazing the lobe narrows across the plane of incidence about as cos(theta_v).
                let mut narrowed = p;
                narrowed[2] += (cos_v / prev_cos).ln();
                fit_cell(&cell, if cell.cost(&narrowed) < cell.cost(&p) { narrowed } else { p })
            }
        };
        prev = Some((p, cos_v));
        let ltc = Ltc::new(&p, cell.axis);
        let check = cell.resampled(CHECK_SAMPLES, (0.318, 0.871));
        out.push(CellFit { coef: ltc.coefficients(), e: cell.e, above: ltc.above, l1: check.error(&ltc, l1) });
    }
    out
}

fn table_paths(dir: &Path) -> (PathBuf, PathBuf) {
    let src = dir.join("src");
    (src.join("ltc_ggx.bin"), src.join("ltc_ggx_albedo.bin"))
}

fn write_f32(path: &Path, values: impl Iterator<Item = f64>) {
    let bytes: Vec<u8> = values.flat_map(|v| (v as f32).to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

fn read_f32(path: &Path, count: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    assert_eq!(bytes.len(), count * 4, "{}: {} bytes, expected {}", path.display(), bytes.len(), count * 4);
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn run_fit(dir: &Path) {
    let (mat_path, albedo_path) = table_paths(dir);
    assert!(mat_path.parent().is_some_and(|p| p.is_dir()), "{} is not the mkr crate (no src/)", dir.display());
    let t0 = Instant::now();
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Option<Vec<CellFit>>>> = Mutex::new(vec![None; N]);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    // Rough rows first: they are the slowest to fit.
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= N {
                        break;
                    }
                    let i = N - 1 - k;
                    let row = fit_row(i);
                    let worst = row.iter().map(|c| c.l1).fold(0.0, f64::max);
                    eprintln!("row {i:2} (roughness {:.3}) worst L1 {worst:.4}  {:.0?}", grid_roughness(i), t0.elapsed());
                    rows.lock().unwrap()[i] = Some(row);
                }
            });
        }
    });
    let cells: Vec<CellFit> = rows.into_inner().unwrap().into_iter().flat_map(|r| r.unwrap()).collect();
    write_f32(&mat_path, cells.iter().flat_map(|c| c.coef));
    write_f32(&albedo_path, cells.iter().map(|c| c.e));

    let worst = cells.iter().enumerate().max_by(|a, b| a.1.l1.total_cmp(&b.1.l1)).unwrap();
    let mean = cells.iter().map(|c| c.l1).sum::<f64>() / cells.len() as f64;
    let min_above = cells.iter().map(|c| c.above).fold(1.0, f64::min);
    let max_coef = cells.iter().flat_map(|c| c.coef).map(f64::abs).fold(0.0, f64::max);
    println!("wrote {} and {} in {:.0?}", mat_path.display(), albedo_path.display(), t0.elapsed());
    println!(
        "fit L1 = integral over the hemisphere of |D/A - f|: mean {mean:.4}, max {:.4} at roughness {:.3}, cos {:.4}",
        worst.1.l1,
        grid_roughness(worst.0 / N),
        grid_cos(worst.0 % N)
    );
    println!("LTC mass above the horizon A >= {min_above:.4}; largest |coefficient| {max_coef:.3}");
    for i in [0, 3, 13, 32, 63] {
        let line: Vec<String> = [0, 32, 63]
            .iter()
            .map(|&j| {
                let c = &cells[i * N + j];
                format!("cos {:.3}: L1 {:.4} E {:.4}", grid_cos(j), c.l1, c.e)
            })
            .collect();
        println!("  roughness {:.3}: {}", grid_roughness(i), line.join(" | "));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// EEVEE's table (--eevee)
//
// What EEVEE Next (Blender 4.2) draws, measured on probes (a white Glossy BSDF plane under a small or a hemisphere-wide
// area light, orthographic views, the LTC of each probe fitted to its frame: numbers in src/ltc.rs's tests): the
// split-sum albedo times the integral over the light of an LTC normalised over the whole sphere (its mass under the
// surface's horizon is lost; nothing renormalises it), and that LTC is Heitz et al.'s own fitting method run on this
// grid. Its parts, each confirmed by the probes' coefficients matching this table to about 1e-3:
//   - target BRDF(V, L) cos(L): GGX, F = 1, height-correlated Smith G2, zero under the horizon; its magnitude (the
//     directional albedo E) and mean direction estimated on the fit's own stratified points;
//   - M = [X Y Z] [[m11, 0, m13], [0, m22, 0], [0, 0, 1]]: Z the BRDF's mean direction (its T2 part dropped),
//     X = (Z.z, 0, -Z.x), Y = T2; three free parameters, the lobe's frame is not fitted;
//   - error: the integral over the sphere of |E D - BRDF cos|^3, estimated with multiple importance sampling (balance
//     heuristic) of the LTC and of GGX through half vectors drawn from D (pdf D(h) cos(h) / (4 V.h)), 32 x 32
//     stratified points each;
//   - Nelder-Mead from the previous cell's parameters: simplex step 0.05, stop when the best and the worst point differ
//     by 1e-5 relative or after 100 iterations; m11, m22 >= 1e-7; V = N is isotropic (m22 = m11, m13 = 0);
//   - order: rows from roughness 1 down to 0, each walking from V = N toward grazing; a row's first cell starts from
//     the first cell of the row above (roughness 1: the cosine itself);
//   - grid: roughness = i / 63 (alpha = max(roughness^2, EEVEE_ALPHA_MIN)), sqrt(1 - cos(theta_v)) = j / 63 with
//     theta_v at most EEVEE_THETA_MAX.
// ltc_ggx_eevee.bin stores it like ltc_ggx.bin: (m00, m02, m20, m22) of Minv scaled so that m11 = 1.

/// GGX alpha of EEVEE's roughness-0 row.
const EEVEE_ALPHA_MIN: f64 = 1e-5;
/// Largest view angle of EEVEE's grid (radians): the last column's.
const EEVEE_THETA_MAX: f64 = 1.57;
/// Stratified points per axis, per sampling strategy, of EEVEE's fit.
const EEVEE_POINTS: usize = 32;

/// GGX as EEVEE's fit sees it, in the frame where N = z and V = (sin, 0, cos).
struct EeveeGgx {
    alpha: f64,
    a2: f64,
    v: V3,
    lambda_v: f64,
}

impl EeveeGgx {
    fn new(alpha: f64, theta: f64) -> EeveeGgx {
        let a2 = alpha * alpha;
        let v = v3(theta.sin(), 0.0, theta.cos());
        EeveeGgx { alpha, a2, v, lambda_v: lambda(v.z, a2) }
    }

    /// (BRDF(V, l) cos(l), pdf of l under half-vector sampling), both per steradian.
    fn eval(&self, l: V3) -> (f64, f64) {
        let h = normalize(self.v + l);
        let t = 1.0 / (1.0 + (h.x * h.x + h.y * h.y) / (h.z * h.z * self.a2));
        let d = t * t / (PI * self.a2 * (h.z * h.z) * (h.z * h.z));
        let pdf = (d * h.z / (4.0 * dot(self.v, h))).abs();
        let g2 = if l.z <= 0.0 { 0.0 } else { 1.0 / (1.0 + self.lambda_v + lambda(l.z, self.a2)) };
        (d * g2 / (4.0 * self.v.z), pdf)
    }

    /// V reflected about a half vector drawn from D.
    fn sample(&self, u1: f64, u2: f64) -> V3 {
        let (s, c) = (2.0 * PI * u1).sin_cos();
        let r = self.alpha * (u2 / (1.0 - u2)).sqrt();
        let m = normalize(v3(r * c, r * s, 1.0));
        m * (2.0 * dot(m, self.v)) - self.v
    }

    /// (E, the mean direction of BRDF cos with its T2 part dropped), on the fit's points.
    fn moments(&self, pts: &[(f64, f64)]) -> (f64, V3) {
        let (mut e, mut m) = (0.0, v3(0.0, 0.0, 0.0));
        for &(u1, u2) in pts {
            let l = self.sample(u1, u2);
            let (f, pdf) = self.eval(l);
            if pdf > 0.0 {
                e += f / pdf;
                m = m + l * (f / pdf);
            }
        }
        (e / pts.len() as f64, normalize(v3(m.x, 0.0, m.z)))
    }
}

/// EEVEE's LTC, M = [X Y Z] [[m11, 0, m13], [0, m22, 0], [0, 0, 1]] (X, Y, Z from the frame's axis Z), its density
/// normalised over the whole sphere.
struct EeveeLtc {
    m: M3,
    minv: M3,
    abs_det_m: f64,
}

impl EeveeLtc {
    fn new([m11, m22, m13]: [f64; 3], z: V3) -> EeveeLtc {
        let x = v3(z.z, 0.0, -z.x);
        let (c0, c2) = (x * m11, x * m13 + z);
        let m = M3([[c0.x, 0.0, c2.x], [0.0, m22, 0.0], [c0.z, 0.0, c2.z]]);
        EeveeLtc { m, minv: m.inverse(), abs_det_m: m.det().abs() }
    }

    fn eval(&self, l: V3) -> f64 {
        let o = normalize(self.minv.mul(l));
        let k = length(self.m.mul(o));
        o.z.max(0.0) / PI * k * k * k / self.abs_det_m
    }

    /// The image of a cosine-distributed direction.
    fn sample(&self, u1: f64, u2: f64) -> V3 {
        let (s, c) = (2.0 * PI * u2).sin_cos();
        let st = (1.0 - u1).max(0.0).sqrt();
        normalize(self.m.mul(v3(st * c, st * s, u1.sqrt())))
    }

    /// Minv scaled to m11 = 1, as stored: (m00, m02, m20, m22).
    fn coefficients(&self) -> [f64; 4] {
        let q = self.minv.scale(1.0 / self.minv.0[1][1]);
        [q.0[0][0], q.0[0][2], q.0[2][0], q.0[2][2]]
    }
}

/// The fit's error: the integral over the sphere of |e D - BRDF cos|^3 (MIS of both samplings; the rare point where a
/// density is not finite counts nothing).
fn eevee_error(ltc: &EeveeLtc, ggx: &EeveeGgx, e: f64, pts: &[(f64, f64)]) -> f64 {
    let mut s = 0.0;
    for &(u1, u2) in pts {
        for l in [ltc.sample(u1, u2), ggx.sample(u1, u2)] {
            let (f, pdf_brdf) = ggx.eval(l);
            let pdf_ltc = ltc.eval(l);
            let term = (f - e * pdf_ltc).abs().powi(3) / (pdf_ltc + pdf_brdf);
            if term.is_finite() {
                s += term;
            }
        }
    }
    s / pts.len() as f64
}

/// Nelder-Mead as EEVEE's table was fitted: the simplex is `start` and `start` moved by `delta` along each axis;
/// reflection 1, expansion 2, inside contraction 1/2, shrink 1/2 toward the best; it stops when the best and the worst
/// value differ by `tol` relative, or after `iters` iterations, and returns the best point of its last ranking.
fn eevee_nelder_mead(f: &dyn Fn(&[f64; 3]) -> f64, start: [f64; 3], delta: f64, tol: f64, iters: usize) -> [f64; 3] {
    let mut s = [start; 4];
    for (i, p) in s.iter_mut().enumerate().skip(1) {
        p[i - 1] += delta;
    }
    let mut fs = [f(&s[0]), f(&s[1]), f(&s[2]), f(&s[3])];
    let mut lo = 0;
    for _ in 0..iters {
        let (mut hi, mut nh) = (0, 0);
        lo = 0;
        for i in 1..4 {
            if fs[i] < fs[lo] {
                lo = i;
            }
            if fs[i] > fs[hi] {
                nh = hi;
                hi = i;
            } else if fs[i] > fs[nh] {
                nh = i;
            }
        }
        let (a, b) = (fs[lo].abs(), fs[hi].abs());
        if 2.0 * (a - b).abs() < (a + b) * tol {
            break;
        }
        let mut o = [0.0; 3];
        for p in s.iter().enumerate().filter(|&(i, _)| i != hi).map(|(_, p)| p) {
            for k in 0..3 {
                o[k] += p[k] / 3.0;
            }
        }
        let worst = s[hi];
        let at = |t: f64| -> [f64; 3] { std::array::from_fn(|k| o[k] + t * (o[k] - worst[k])) };
        let r = at(1.0);
        let fr = f(&r);
        if fr < fs[lo] {
            let e = at(2.0);
            let fe = f(&e);
            if fe < fr {
                (s[hi], fs[hi]) = (e, fe);
                continue;
            }
        }
        if fr < fs[nh] {
            (s[hi], fs[hi]) = (r, fr);
            continue;
        }
        let c = at(-0.5);
        let fc = f(&c);
        if fc < fs[hi] {
            (s[hi], fs[hi]) = (c, fc);
            continue;
        }
        let best = s[lo];
        for k in (0..4).filter(|&k| k != lo) {
            s[k] = std::array::from_fn(|d| best[d] + 0.5 * (s[k][d] - best[d]));
            fs[k] = f(&s[k]);
        }
    }
    s[lo]
}

/// One cell of EEVEE's table: the fitted (m11, m22, m13) from `start`.
fn eevee_fit_cell(ggx: &EeveeGgx, z: V3, start: [f64; 3], isotropic: bool, pts: &[(f64, f64)]) -> [f64; 3] {
    let (e, _) = ggx.moments(pts);
    let params = |p: &[f64; 3]| -> [f64; 3] {
        let (m11, m22) = (p[0].max(1e-7), p[1].max(1e-7));
        if isotropic { [m11, m11, 0.0] } else { [m11, m22, p[2]] }
    };
    let f = |p: &[f64; 3]| eevee_error(&EeveeLtc::new(params(p), z), ggx, e, pts);
    params(&eevee_nelder_mead(&f, start, 0.05, 1e-5, 100))
}

fn eevee_alpha(i: usize) -> f64 {
    let r = grid_roughness(i);
    (r * r).max(EEVEE_ALPHA_MIN)
}

fn eevee_theta(j: usize) -> f64 {
    let x = j as f64 / (N - 1) as f64;
    (1.0 - x * x).acos().min(EEVEE_THETA_MAX)
}

/// EEVEE's table, (m00, m02, m20, m22) per cell in ltc_ggx.bin's order.
fn eevee_table() -> Vec<[f64; 4]> {
    let n = EEVEE_POINTS;
    let pts: Vec<(f64, f64)> =
        (0..n * n).map(|k| (((k % n) as f64 + 0.5) / n as f64, ((k / n) as f64 + 0.5) / n as f64)).collect();
    let axis = v3(0.0, 0.0, 1.0);
    // The rows' first cells (V = N), roughest first, each from the one above.
    let mut first = vec![[1.0, 1.0, 0.0]; N];
    let mut start = [1.0, 1.0, 0.0];
    for i in (0..N).rev() {
        first[i] = eevee_fit_cell(&EeveeGgx::new(eevee_alpha(i), eevee_theta(0)), axis, start, true, &pts);
        start = first[i];
    }
    // Each row from there toward grazing, each cell from the one before, in its mean direction's frame.
    let rows = parallel_map(&first, |i, &p0| {
        let mut row = vec![EeveeLtc::new(p0, axis).coefficients()];
        let mut p = p0;
        for j in 1..N {
            let ggx = EeveeGgx::new(eevee_alpha(i), eevee_theta(j));
            let (_, z) = ggx.moments(&pts);
            p = eevee_fit_cell(&ggx, z, p, false, &pts);
            row.push(EeveeLtc::new(p, z).coefficients());
        }
        row
    });
    rows.concat()
}

fn run_eevee(dir: &Path) {
    let path = dir.join("src").join("ltc_ggx_eevee.bin");
    assert!(path.parent().is_some_and(|p| p.is_dir()), "{} is not the mkr crate (no src/)", dir.display());
    let t0 = Instant::now();
    let cells = eevee_table();
    write_f32(&path, cells.iter().flatten().copied());
    let max_coef = cells.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
    // The mass of each cell's LTC above the surface's horizon, (1 + cos b) / 2 (Minv's first column holds it).
    let above = |c: &[f64; 4]| 0.5 + 0.5 * c[0] / (c[0] * c[0] + c[2] * c[2]).sqrt();
    let low = cells.iter().enumerate().min_by(|a, b| above(a.1).total_cmp(&above(b.1))).unwrap();
    println!("wrote {} in {:.0?}; largest |coefficient| {max_coef:.3}", path.display(), t0.elapsed());
    println!(
        "LTC mass above the surface's horizon >= {:.4} (roughness {:.3}, cos {:.4})",
        above(low.1),
        grid_roughness(low.0 / N),
        eevee_theta(low.0 % N).cos()
    );
    for i in [13, 22, 32, 50, 63] {
        let line: Vec<String> = [0, 32, 52, 63]
            .iter()
            .map(|&j| format!("cos {:.3}: above {:.4}", eevee_theta(j).cos(), above(&cells[i * N + j])))
            .collect();
        println!("  roughness {:.3}: {}", grid_roughness(i), line.join(" | "));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Polygon integration (the algorithm of src/shaders/ltc.wgsl, in f64)

/// The table sampled as the renderer does: uv = ltc_matrix_coords, bilinear, clamp to edge.
struct Table {
    coef: Vec<f32>,
}

impl Table {
    fn minv(&self, roughness: f64, cos_v: f64) -> M3 {
        let x = (1.0 - cos_v.clamp(0.0, 1.0)).sqrt() * (N - 1) as f64;
        let y = roughness.clamp(0.0, 1.0) * (N - 1) as f64;
        let (x0, y0) = ((x.floor() as usize).min(N - 1), (y.floor() as usize).min(N - 1));
        let (x1, y1) = ((x0 + 1).min(N - 1), (y0 + 1).min(N - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let at = |yi: usize, xi: usize, k: usize| self.coef[(yi * N + xi) * 4 + k] as f64;
        let c: [f64; 4] = std::array::from_fn(|k| {
            (1.0 - fy) * ((1.0 - fx) * at(y0, x0, k) + fx * at(y0, x1, k)) + fy * ((1.0 - fx) * at(y1, x0, k) + fx * at(y1, x1, k))
        });
        M3([[c[0], 0.0, c[1]], [0.0, 1.0, 0.0], [c[2], 0.0, c[3]]])
    }
}

/// acos(a.b / |a||b|) normalize(a x b).z: the z part of the edge's vector form factor times 2 pi.
fn edge(a: V3, b: V3) -> f64 {
    let c = cross(a, b);
    let s = length(c);
    c.z * (s.atan2(dot(a, b)) / s.max(1e-300))
}

/// The polygon's edges, carried into the cosine's space (unit vectors), clipped to N's hemisphere (n.q >= 0) and to
/// the cosine's (q.z >= 0) and summed as they come.
struct Acc {
    /// N's horizon normal in the cosine's space: n.(Minv p) = p.z.
    n: V3,
    sum: f64,
    exit_n: Option<V3>,
    entry_n: Option<V3>,
    exit_z: Option<V3>,
    entry_z: Option<V3>,
}

impl Acc {
    fn new(minv: &M3) -> Acc {
        let n = cross(minv.col(0), minv.col(1)) * minv.det().signum();
        Acc { n, sum: 0.0, exit_n: None, entry_n: None, exit_z: None, entry_z: None }
    }

    fn cosine_edge(&mut self, a: V3, b: V3) {
        let x = || a + (b - a) * (a.z / (a.z - b.z));
        match (a.z >= 0.0, b.z >= 0.0) {
            (true, true) => self.sum += edge(a, b),
            (true, false) => {
                self.sum += edge(a, x());
                self.exit_z = Some(x());
            }
            (false, true) => {
                self.sum += edge(x(), b);
                self.entry_z = Some(x());
            }
            (false, false) => {}
        }
    }

    fn edge(&mut self, a: V3, b: V3) {
        let (ha, hb) = (dot(self.n, a), dot(self.n, b));
        let x = || a + (b - a) * (ha / (ha - hb));
        match (ha >= 0.0, hb >= 0.0) {
            (true, true) => self.cosine_edge(a, b),
            (true, false) => {
                self.cosine_edge(a, x());
                self.exit_n = Some(x());
            }
            (false, true) => {
                self.cosine_edge(x(), b);
                self.entry_n = Some(x());
            }
            (false, false) => {}
        }
    }

    fn finish(mut self) -> f64 {
        if let (Some(a), Some(b)) = (self.exit_n, self.entry_n) {
            self.cosine_edge(a, b);
        }
        if let (Some(a), Some(b)) = (self.exit_z, self.entry_z) {
            self.sum += edge(a, b);
        }
        self.sum.abs() / (2.0 * PI * mass_above(self.n).max(1e-6))
    }
}

/// Polygon given in the shading frame relative to the shading point.
fn ltc_polygon(minv: M3, verts: &[V3]) -> f64 {
    let q: Vec<V3> = verts.iter().map(|&p| normalize(minv.mul(p))).collect();
    let mut acc = Acc::new(&minv);
    for i in 0..q.len() {
        acc.edge(q[i], q[(i + 1) % q.len()]);
    }
    acc.finish()
}

// ---------------------------------------------------------------------------------------------------------------
// Brute force

#[derive(Clone, Copy)]
struct Rect {
    c: V3,
    ex: V3,
    ey: V3,
}

impl Rect {
    fn corners(&self) -> [V3; 4] {
        let (c, x, y) = (self.c, self.ex, self.ey);
        [c - x - y, c + x - y, c + x + y, c - x + y]
    }

    fn hit(&self, l: V3) -> bool {
        let n = cross(self.ex, self.ey);
        let dn = dot(l, n);
        if dn == 0.0 {
            return false;
        }
        let t = dot(self.c, n) / dn;
        if t <= 0.0 {
            return false;
        }
        let q = l * t - self.c;
        dot(q, self.ex).abs() <= dot(self.ex, self.ex) && dot(q, self.ey).abs() <= dot(self.ey, self.ey)
    }
}

/// Integral of the normalised GGX lobe over the rectangle (seen from the origin, clipped to z > 0).
fn brute_ggx(ggx: &Ggx, e: f64, rect: &Rect, n: usize, shift: (f64, f64)) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        let (u, w) = lattice(i, n, shift);
        let l = ggx.sample(u, w);
        if l.z > 0.0 && rect.hit(l) {
            s += ggx.weight(l);
        }
    }
    s / (n as f64 * e)
}

/// Integral of cos / pi over the rectangle clipped to z > 0.
fn brute_lambert(rect: &Rect, n: usize, shift: (f64, f64)) -> f64 {
    let mut hits = 0usize;
    for i in 0..n {
        let (u, w) = lattice(i, n, shift);
        if rect.hit(cosine_dir(u, w)) {
            hits += 1;
        }
    }
    hits as f64 / n as f64
}

/// A rectangle around the direction (elevation, azimuth) in degrees at distance `d`, half sizes `hx`, `hy`, facing the
/// origin, then turned by `tilt` degrees about its x axis.
fn rect_at(elev: f64, azim: f64, d: f64, hx: f64, hy: f64, tilt: f64) -> Rect {
    let (se, ce) = elev.to_radians().sin_cos();
    let (sa, ca) = azim.to_radians().sin_cos();
    let dir = v3(ce * ca, ce * sa, se);
    let up = if se.abs() > 0.99 { v3(1.0, 0.0, 0.0) } else { v3(0.0, 0.0, 1.0) };
    let x = normalize(cross(up, dir));
    let y = cross(dir, x);
    let (st, ct) = tilt.to_radians().sin_cos();
    let y_t = y * ct + dir * st;
    Rect { c: dir * d, ex: x * hx, ey: y_t * hy }
}

fn random_rect(rng: &mut Rng) -> Rect {
    let elev = -25.0 + 115.0 * rng.uniform();
    let azim = 360.0 * rng.uniform();
    let d = 0.5 + 3.5 * rng.uniform();
    let size = |rng: &mut Rng| d * (0.02 + 1.2 * rng.uniform() * rng.uniform());
    let (hx, hy) = (size(rng), size(rng));
    let tilt = -60.0 + 120.0 * rng.uniform();
    rect_at(elev.min(89.0), azim, d, hx, hy, tilt)
}

/// The rectangles the tests in src/ltc.rs use (same numbers).
fn test_rects() -> Vec<(&'static str, Rect)> {
    vec![
        ("big overhead", Rect { c: v3(0.0, 0.0, 1.0), ex: v3(10.0, 0.0, 0.0), ey: v3(0.0, 10.0, 0.0) }),
        ("medium overhead", rect_at(60.0, 30.0, 2.0, 0.5, 0.3, 20.0)),
        ("small grazing, mirror side", rect_at(10.0, 180.0, 1.0, 0.1, 0.1, 0.0)),
        ("small grazing, view side", rect_at(10.0, 0.0, 1.0, 0.1, 0.1, 0.0)),
        ("small grazing, sideways", rect_at(10.0, 90.0, 1.0, 0.1, 0.1, 0.0)),
        ("straddling the horizon", rect_at(0.0, 200.0, 1.0, 0.3, 0.2, 0.0)),
    ]
}

// ---------------------------------------------------------------------------------------------------------------
// --check

#[derive(Default)]
struct Stats {
    n: usize,
    big: usize,
    rel_max: f64,
    rel_sum: f64,
    abs_max: f64,
    abs_sum: f64,
    fails: usize,
    worst: String,
}

impl Stats {
    fn add(&mut self, got: f64, want: f64, what: impl FnOnce() -> String) {
        self.n += 1;
        let abs = (got - want).abs();
        let fail;
        if want > 0.01 {
            let rel = abs / want;
            self.big += 1;
            self.rel_sum += rel;
            fail = rel > 0.03;
            if rel > self.rel_max {
                self.rel_max = rel;
                self.worst = what();
            }
        } else {
            self.abs_sum += abs;
            self.abs_max = self.abs_max.max(abs);
            fail = abs > 0.003;
        }
        self.fails += fail as usize;
    }

    fn print(&self, name: &str) {
        let small = (self.n - self.big).max(1);
        println!(
            "{name}: {} cases; integral > 0.01 ({}): relative error max {:.2}% mean {:.3}%; <= 0.01: absolute error max \
             {:.5} mean {:.6}; outside 3% / 0.003: {}",
            self.n,
            self.big,
            100.0 * self.rel_max,
            100.0 * self.rel_sum / self.big.max(1) as f64,
            self.abs_max,
            self.abs_sum / small as f64,
            self.fails
        );
        if !self.worst.is_empty() {
            println!("    worst: {}", self.worst);
        }
    }
}

struct Case {
    roughness: f64,
    cos_v: f64,
    rect: Rect,
}

/// (LTC, brute force) for the GGX lobe and for Lambert.
fn evaluate(table: &Table, c: &Case, shift: (f64, f64)) -> [(f64, f64); 2] {
    let ggx = Ggx::new(alpha_of(c.roughness), c.cos_v);
    let (e, _) = ggx.albedo(CHECK_SAMPLES);
    let corners = c.rect.corners();
    let g = (ltc_polygon(table.minv(c.roughness, c.cos_v), &corners), brute_ggx(&ggx, e, &c.rect, CHECK_SAMPLES, shift));
    let l = (ltc_polygon(M3::IDENTITY, &corners), brute_lambert(&c.rect, CHECK_SAMPLES, shift));
    [g, l]
}

fn parallel_map<T: Sync, R: Send + Clone + Default>(items: &[T], f: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let out = Mutex::new(vec![R::default(); items.len()]);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= items.len() {
                        break;
                    }
                    let r = f(k, &items[k]);
                    out.lock().unwrap()[k] = r;
                }
            });
        }
    });
    out.into_inner().unwrap()
}

fn run_check(dir: &Path) {
    let (mat_path, albedo_path) = table_paths(dir);
    let table = Table { coef: read_f32(&mat_path, N * N * 4) };
    let albedo = read_f32(&albedo_path, N * N);
    let t0 = Instant::now();

    // Albedo file against a fresh estimate at a few cells.
    let mut e_err: f64 = 0.0;
    for (i, j) in [(0, 0), (5, 40), (20, 63), (40, 10), (63, 0), (63, 63), (31, 31)] {
        let (e, _) = Ggx::new(alpha_of(grid_roughness(i)), grid_cos(j)).albedo(CHECK_SAMPLES);
        e_err = e_err.max((albedo[i * N + j] as f64 - e).abs());
    }
    println!("albedo file vs a fresh estimate at 7 cells: max |error| {e_err:.2e}");

    // The tests' fixed cases.
    println!("tests' rectangles (LTC vs brute force, normalised GGX lobe):");
    let mut fixed = Vec::new();
    for &roughness in &[0.05, 0.2, 0.5, 1.0] {
        for &deg in &[0.0f64, 45.0, 80.0] {
            for (name, rect) in test_rects() {
                fixed.push((name, Case { roughness, cos_v: deg.to_radians().cos(), rect }));
            }
        }
    }
    let results = parallel_map(&fixed, |_, (_, c)| evaluate(&table, c, (0.0, 0.0)).to_vec());
    let (mut sg, mut sl) = (Stats::default(), Stats::default());
    for ((name, c), r) in fixed.iter().zip(&results) {
        let deg = c.cos_v.acos().to_degrees();
        let [g, l] = [r[0], r[1]];
        let what = || format!("roughness {} view {deg:.0} {name}: LTC {:.5} brute {:.5}", c.roughness, g.0, g.1);
        println!("    {}", what());
        sg.add(g.0, g.1, what);
        sl.add(l.0, l.1, || format!("{name}: {:.5} vs {:.5}", l.0, l.1));
    }
    sg.print("  GGX (tests)");
    sl.print("  Lambert (tests)");

    // Random cases: anywhere in the table (bilinear), and at texel centres only (fit error alone).
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut random = Vec::new();
    for k in 0..12000 {
        let (roughness, cos_v) = if k % 3 == 0 {
            (grid_roughness((rng.uniform() * N as f64) as usize % N), grid_cos((rng.uniform() * N as f64) as usize % N))
        } else {
            (rng.uniform(), (rng.uniform() * 89.5f64).to_radians().cos())
        };
        random.push(Case { roughness, cos_v, rect: random_rect(&mut rng) });
    }
    let shifts: Vec<(f64, f64)> = (0..random.len()).map(|_| (rng.uniform(), rng.uniform())).collect();
    let results = parallel_map(&random, |k, c| evaluate(&table, c, shifts[k]).to_vec());
    let (mut grid_g, mut any_g, mut any_l) = (Stats::default(), Stats::default(), Stats::default());
    let mut envelope = Vec::with_capacity(random.len());
    for (k, (c, r)) in random.iter().zip(&results).enumerate() {
        let what = || {
            let corners = c.rect.corners().map(|p| format!("({:.3}, {:.3}, {:.3})", p.x, p.y, p.z));
            format!(
                "roughness {:.4} view {:.2} deg rect {}: LTC {:.5} brute {:.5}",
                c.roughness,
                c.cos_v.acos().to_degrees(),
                corners.join(" "),
                r[0].0,
                r[0].1
            )
        };
        if k % 3 == 0 {
            grid_g.add(r[0].0, r[0].1, what);
        } else {
            any_g.add(r[0].0, r[0].1, what);
        }
        any_l.add(r[1].0, r[1].1, || format!("{:.5} vs {:.5}", r[1].0, r[1].1));
        envelope.push((c.roughness, c.cos_v.acos().to_degrees(), r[0].0, r[0].1));
    }
    grid_g.print("GGX, random rectangles at texel centres");
    any_g.print("GGX, random rectangles, random roughness and view (bilinear)");
    any_l.print("Lambert, random rectangles");
    print_envelope(&envelope);
    println!("({} brute-force points per integral, {:.0?})", CHECK_SAMPLES, t0.elapsed());
}

/// The random GGX cases by roughness and view angle: share within 3 % / 0.003, and the median, 90th percentile and
/// maximum of the relative error where the integral exceeds 0.01. `cases`: (roughness, view degrees, LTC, brute force).
fn print_envelope(cases: &[(f64, f64, f64, f64)]) {
    const R: [f64; 5] = [0.0, 0.15, 0.35, 0.65, 1.0 + 1e-9];
    const V: [f64; 5] = [0.0, 30.0, 60.0, 75.0, 90.0];
    println!("GGX envelope: within 3 % / 0.003 | relative error where the integral > 0.01: median / p90 / max");
    println!(
        "  {:11}{}",
        "roughness",
        (0..4).map(|j| format!("| view {:2.0}-{:2.0} deg            ", V[j], V[j + 1])).collect::<String>()
    );
    for i in 0..4 {
        let mut line = format!("  {:.2}-{:.2}  ", R[i], R[i + 1].min(1.0));
        for j in 0..4 {
            let bin: Vec<_> =
                cases.iter().filter(|c| c.0 >= R[i] && c.0 < R[i + 1] && c.1 >= V[j] && c.1 < V[j + 1]).collect();
            let ok = bin.iter().filter(|c| (c.2 - c.3).abs() <= if c.3 > 0.01 { 0.03 * c.3 } else { 0.003 }).count();
            let mut rel: Vec<f64> = bin.iter().filter(|c| c.3 > 0.01).map(|c| (c.2 - c.3).abs() / c.3).collect();
            rel.sort_by(f64::total_cmp);
            let q = |p: f64| rel.get(((rel.len() as f64 - 1.0) * p).round() as usize).map_or(0.0, |v| 100.0 * v);
            let share = 100.0 * ok as f64 / bin.len().max(1) as f64;
            line += &format!("| {share:3.0}% {:5.1} /{:5.1} /{:6.1}% ", q(0.5), q(0.9), q(1.0));
        }
        println!("{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = args.iter().any(|a| a == "--check");
    let dir = PathBuf::from(args.iter().find(|a| !a.starts_with("--")).map_or(".", |s| s.as_str()));
    if args.iter().any(|a| a == "--eevee") {
        run_eevee(&dir)
    } else if check {
        run_check(&dir)
    } else {
        run_fit(&dir)
    }
}
