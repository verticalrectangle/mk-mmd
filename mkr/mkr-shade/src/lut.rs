//! The tables closure colours need (GGX split sum, BTDF transmission), integrated numerically (clean-room maths:
//! GGX with separable Smith masking, Schlick and exact dielectric Fresnel, Heitz's visible-normal sampling).
//! Layout: `src/wgsl/bsdf.wgsl` header. The sheen albedo (layer 0 alpha) is a table measured from Blender 4.2.
//! `lut()` reads the precomputed copy in `src/lut.bin`; after changing the integrator or the sheen table, rewrite it
//! with `cargo run --release -p mkr-shade --example write_lut` (a test checks it against `compute()`).

use std::sync::LazyLock;

use rayon::prelude::*;

/// Texels per side of a layer.
pub const SIZE: usize = 64;
/// IOR layers per family (ior >= 1, ior < 1).
pub const IOR_LAYERS: usize = 16;
/// Total layers: split sum + both IOR families.
pub const LAYERS: usize = 1 + 2 * IOR_LAYERS;

/// An rgba32float 2D-array texture: `data[((layer * SIZE + y) * SIZE + x) * 4 + c]`, x = roughness axis,
/// y = sqrt(1 - cos_theta) axis.
pub struct Lut {
    pub size: u32,
    pub layers: u32,
    pub data: Vec<f32>,
}

#[derive(Clone, Copy)]
struct V3(f64, f64, f64);

impl V3 {
    fn dot(self, o: V3) -> f64 {
        self.0 * o.0 + self.1 * o.1 + self.2 * o.2
    }
    fn scale(self, s: f64) -> V3 {
        V3(self.0 * s, self.1 * s, self.2 * s)
    }
    fn add(self, o: V3) -> V3 {
        V3(self.0 + o.0, self.1 + o.1, self.2 + o.2)
    }
    fn sub(self, o: V3) -> V3 {
        V3(self.0 - o.0, self.1 - o.1, self.2 - o.2)
    }
    fn cross(self, o: V3) -> V3 {
        V3(self.1 * o.2 - self.2 * o.1, self.2 * o.0 - self.0 * o.2, self.0 * o.1 - self.1 * o.0)
    }
    fn norm(self) -> V3 {
        let l = self.dot(self).sqrt();
        if l > 0.0 { self.scale(1.0 / l) } else { self }
    }
}

fn radical_inverse(mut i: u32) -> f64 {
    i = i.rotate_right(16);
    i = ((i & 0x5555_5555) << 1) | ((i & 0xAAAA_AAAA) >> 1);
    i = ((i & 0x3333_3333) << 2) | ((i & 0xCCCC_CCCC) >> 2);
    i = ((i & 0x0F0F_0F0F) << 4) | ((i & 0xF0F0_F0F0) >> 4);
    i = ((i & 0x00FF_00FF) << 8) | ((i & 0xFF00_FF00) >> 8);
    i as f64 / 4294967296.0
}

/// Heitz 2018: a GGX microfacet normal distributed as seen from v (tangent space, z up).
fn sample_vndf(v: V3, alpha: f64, u1: f64, u2: f64) -> V3 {
    let vh = V3(alpha * v.0, alpha * v.1, v.2).norm();
    let lensq = vh.0 * vh.0 + vh.1 * vh.1;
    let t1 = if lensq > 0.0 { V3(-vh.1, vh.0, 0.0).scale(1.0 / lensq.sqrt()) } else { V3(1.0, 0.0, 0.0) };
    let t2 = vh.cross(t1);
    let r = u1.sqrt();
    let phi = 2.0 * std::f64::consts::PI * u2;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + vh.2);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;
    let nh = t1.scale(p1).add(t2.scale(p2)).add(vh.scale((1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt()));
    V3(alpha * nh.0, alpha * nh.1, nh.2.max(0.0)).norm()
}

fn g1(nx: f64, a2: f64) -> f64 {
    let c2 = (nx * nx).max(1e-12);
    2.0 / (1.0 + (1.0 + a2 * (1.0 / c2 - 1.0)).sqrt())
}

fn fresnel_dielectric(cosi: f64, eta: f64) -> f64 {
    let c = cosi.abs();
    let g = eta * eta - 1.0 + c * c;
    if g > 0.0 {
        let g = g.sqrt();
        let a = (g - c) / (g + c);
        let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
        0.5 * a * a * (1.0 + b * b)
    } else {
        1.0
    }
}

fn f0_from_ior(eta: f64) -> f64 {
    let f = (eta - 1.0) / (eta + 1.0);
    f * f
}

/// Texel coordinate -> (roughness, cos_theta).
fn texel(x: usize, y: usize) -> (f64, f64) {
    let r = x as f64 / (SIZE - 1) as f64;
    let s = y as f64 / (SIZE - 1) as f64;
    (r, (1.0 - s * s).clamp(1e-4, 0.9999))
}

/// GGX split sum (F0 weight, F90 weight, F82-tint term).
fn split_sum(roughness: f64, nv: f64, samples: u32) -> [f64; 3] {
    let alpha = (roughness * roughness).max(1e-4);
    let a2 = alpha * alpha;
    let v = V3((1.0 - nv * nv).sqrt(), 0.0, nv);
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    for i in 0..samples {
        let u1 = (i as f64 + 0.5) / samples as f64;
        let u2 = radical_inverse(i);
        let h = sample_vndf(v, alpha, u1, u2);
        let vh = v.dot(h);
        let l = h.scale(2.0 * vh).sub(v);
        if l.2 <= 0.0 {
            continue;
        }
        let w = g1(l.2, a2);
        let m = (1.0 - vh.max(0.0)).clamp(0.0, 1.0);
        let fc = m.powi(5);
        x += (1.0 - fc) * w;
        y += fc * w;
        z += vh.max(0.0) * m.powi(6) * w;
    }
    let n = samples as f64;
    [x / n, y / n, z / n]
}

/// Rough dielectric interface of relative IOR `eta` (light arriving from the v side):
/// (reflectance with exact Fresnel, transmitted fraction).
fn dielectric(roughness: f64, nv: f64, eta: f64, samples: u32) -> (f64, f64) {
    let alpha = (roughness * roughness).max(1e-4);
    let a2 = alpha * alpha;
    let v = V3((1.0 - nv * nv).sqrt(), 0.0, nv);
    let (mut r, mut t) = (0.0, 0.0);
    for i in 0..samples {
        let u1 = (i as f64 + 0.5) / samples as f64;
        let u2 = radical_inverse(i);
        let h = sample_vndf(v, alpha, u1, u2);
        let vh = v.dot(h);
        if vh <= 0.0 {
            continue;
        }
        let f = fresnel_dielectric(vh, eta);
        // Reflection.
        let l = h.scale(2.0 * vh).sub(v);
        if l.2 > 0.0 {
            r += f * g1(l.2, a2);
        }
        // Refraction (eta = n_inside / n_outside).
        let inv = 1.0 / eta;
        let k = 1.0 - inv * inv * (1.0 - vh * vh);
        if k > 0.0 {
            let lt = v.scale(-inv).add(h.scale(inv * vh - k.sqrt()));
            if lt.2 < 0.0 {
                t += (1.0 - f) * g1(-lt.2, a2);
            }
        }
    }
    let n = samples as f64;
    (r / n, t / n)
}

/// Albedo of Blender 4.2 EEVEE's Principled sheen lobe over (cos_theta, roughness), bilinear in the table measured
/// from Blender renders of sheen-only Principled BSDFs under a uniform white world (`lab/`, `sheen_table.rs`).
pub fn sheen_albedo(nv: f64, roughness: f64) -> f64 {
    use crate::sheen_table::{NV_AT, ROUGH_AT, SHEEN};
    let (x0, x1, tx) = bracket(&NV_AT, nv as f32);
    let (y0, y1, ty) = bracket(&ROUGH_AT, roughness as f32);
    let at = |x: usize, y: usize| SHEEN[y * NV_AT.len() + x] as f64;
    let a = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
    let b = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
    a * (1.0 - ty) + b * ty
}

/// Neighbours of `v` in the ascending positions `at` and the weight of the upper one (clamped outside).
fn bracket(at: &[f32], v: f32) -> (usize, usize, f64) {
    let last = at.len() - 1;
    let hi = at.iter().position(|&a| a >= v).unwrap_or(last);
    let lo = hi.saturating_sub(1);
    let span = at[hi] - at[lo];
    let t = if span > 0.0 { ((v - at[lo]) / span).clamp(0.0, 1.0) } else { 0.0 };
    (lo, hi, t as f64)
}

/// Integrate the tables (about 2 s of CPU). `lut()` returns the copy precomputed into `src/lut.bin`.
pub fn compute() -> Lut {
    let texels = SIZE * SIZE;
    let mut data = vec![0.0f32; LAYERS * texels * 4];
    data.par_chunks_mut(4).enumerate().for_each(|(i, px)| {
        let layer = i / texels;
        let y = (i % texels) / SIZE;
        let x = i % SIZE;
        let (rough, nv) = texel(x, y);
        if layer == 0 {
            let s = split_sum(rough, nv, 4096);
            px[0] = s[0] as f32;
            px[1] = s[1] as f32;
            px[2] = s[2] as f32;
            px[3] = sheen_albedo(nv, rough) as f32;
        } else if layer <= IOR_LAYERS {
            let z = (layer - 1) as f64 / (IOR_LAYERS - 1) as f64;
            // z = sqrt((ior - 1) / (ior + 1))
            let q = (z * z).min(0.999);
            let ior = (1.0 + q) / (1.0 - q);
            let f0 = f0_from_ior(ior);
            let (_, t) = dielectric(rough, nv, ior.max(1.0 + 1e-6), 1024);
            px[0] = (t / (1.0 - f0).max(1e-6)) as f32;
        } else {
            let z = (layer - 1 - IOR_LAYERS) as f64 / (IOR_LAYERS - 1) as f64;
            // z = sqrt(1 - ior)
            let ior = (1.0 - z * z).max(0.05);
            let f0 = f0_from_ior(ior);
            let (r, t) = dielectric(rough, nv, ior, 1024);
            px[0] = 0.0;
            px[1] = r as f32;
            px[2] = (t / (1.0 - f0).max(1e-6)) as f32;
        }
    });
    Lut { size: SIZE as u32, layers: LAYERS as u32, data }
}

/// `compute()`'s tables as little-endian f32.
static LUT_BIN: &[u8] = include_bytes!("lut.bin");

static LUT: LazyLock<Lut> = LazyLock::new(|| {
    assert_eq!(
        LUT_BIN.len(),
        LAYERS * SIZE * SIZE * 4 * 4,
        "src/lut.bin is stale: run `cargo run --release -p mkr-shade --example write_lut`"
    );
    let data = LUT_BIN.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    Lut { size: SIZE as u32, layers: LAYERS as u32, data }
});

/// The LUT (read once per process from the precomputed table).
pub fn lut() -> &'static Lut {
    &LUT
}
