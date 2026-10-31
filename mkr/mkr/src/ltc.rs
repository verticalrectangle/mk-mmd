//! Linearly Transformed Cosines for area lights (Heitz, Dupuy, Hill, Neubelt 2016) for `shaders/ltc.wgsl`: two GGX
//! tables, both fitted from the maths by `tools/ltc_fit.rs`; `Params::ltc_glossy` picks the one the renderer uploads.
//!
//! [`LtcFit::Eevee`] (the default): the lobe EEVEE Next draws, measured on Blender 4.2 (the probes in `tests`: a white
//! Glossy BSDF plane under a 4 cm area light or a hemisphere-wide one, orthographic views, so each frame maps the lobe
//! over light directions for one view; an LTC fitted to each frame leaves 1e-4 of the peak). A clamped cosine
//! normalised over the whole sphere, fitted to BRDF(V, L) cos(L) (GGX, alpha = roughness², F = 1, height-correlated
//! Smith G2) by Heitz et al.'s method: L3 error, the lobe's frame its mean direction (`tools/ltc_fit.rs --eevee`).
//! EEVEE integrates it over the light with the sphere approximation of the horizon (`ltc_quad_sphere`,
//! `ltc_disk_sphere`) and nothing renormalises it, so the part under the surface's horizon is lost: a light covering
//! the hemisphere gives (1 + cos b) / 2, b the angle between the cosine's axis and N's horizon normal in its space
//! (0.91 at roughness 0.5 seen at 71 degrees, 0.88 at roughness 0.35 at 81). The amplitude is the closure colour's
//! split sum (Smith masking separable there: the probes' amplitude). Since that path clips nothing at N's horizon,
//! EEVEE fades a light out of the glossy lobes once the sphere bounding it sinks under that horizon
//! (`ltc_horizon_fade`, measured on lights under the plane; light.wgsl applies it with this table).
//!
//! [`LtcFit::Mkr`]: mkr's own fit. For view direction V, normal N and GGX alpha = roughness², the clamped cosine
//! carried through M = Minv⁻¹, restricted to the hemisphere of N and divided by its mass there, approximates
//! BRDF(V, L) cos(L) / E(V, alpha) (F = 1, height-correlated Smith G2, E the directional albedo), so a light covering
//! the hemisphere gives 1 (`ltc_quad`, `ltc_disk`; Cycles' lobe within a few percent).
//!
//! Lambert is the same integration with Minv = identity (cos / pi).
//!
//! [`table`]: `LTC_SIZE`² entries, entry `i * LTC_SIZE + j` for roughness index `i` and view index `j`:
//! roughness = i / 63 and sqrt(1 - cos(N, V)) = j / 63. mkr's: alpha = max(roughness², 1e-4) (roughness below 0.01 is
//! drawn as 0.01), and column 63 (cos 0, where the renderer's cos <= 0 also lands) holds the fit at
//! sqrt(1 - cos) = 62.5 / 63 (89.1 degrees): toward 90 degrees the lobe collapses into a sliver no LTC follows.
//! EEVEE's: alpha = max(roughness², 1e-5), column 63 fitted at 1.57 rad.
//! Each entry is (m00, m02, m20, m22) of Minv scaled so that m11 = 1: Minv = [[m00, 0, m02], [0, 1, 0], [m20, 0, m22]]
//! (rows) in the frame (T1, T2, N), T1 = normalize(V - N (N.V)) so that V = (sin, 0, cos); `ltc_minv` builds it. This
//! scaling keeps every coefficient within ±1.7 and smooth down to mirror-like lobes, so bilinear filtering is accurate
//! (EEVEE's frames between the cells match it too).
//!
//! Texel addressing: upload the entries in order as a 64 x 64 rgba32float texture, texel (x = j, y = i), i.e. row
//! y = roughness index, 1024 bytes per row. `ltc_matrix_coords(roughness, cos_nv)` returns
//! uv = ((sqrt(1 - clamp(cos_nv, 0, 1)) * 63 + 0.5) / 64, (clamp(roughness, 0, 1) * 63 + 0.5) / 64): texel centres at
//! (k + 0.5) / 64, so a linear clamp-to-edge sampler interpolates between the fitted cells and never past them
//! (rgba32float needs `Features::FLOAT32_FILTERABLE` for a filtering sampler).
//!
//! [`albedo`]: E(V, alpha) per cell of mkr's fit, same addressing (an r32float texture read with the same uv).

use crate::params::LtcFit;

/// Cells per axis of the tables.
pub const LTC_SIZE: usize = 64;

const GGX: &[u8] = include_bytes!("ltc_ggx.bin");
const GGX_ALBEDO: &[u8] = include_bytes!("ltc_ggx_albedo.bin");
const GGX_EEVEE: &[u8] = include_bytes!("ltc_ggx_eevee.bin");

const _: () = assert!(
    GGX.len() == LTC_SIZE * LTC_SIZE * 16
        && GGX_ALBEDO.len() == LTC_SIZE * LTC_SIZE * 4
        && GGX_EEVEE.len() == LTC_SIZE * LTC_SIZE * 16
);

fn f32s(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// The GGX inverse matrices of `fit`, (m00, m02, m20, m22) per cell (module docs).
pub fn table(fit: LtcFit) -> Vec<[f32; 4]> {
    let bytes = match fit {
        LtcFit::Eevee => GGX_EEVEE,
        LtcFit::Mkr => GGX,
    };
    let v: Vec<f32> = f32s(bytes).collect();
    v.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

/// GGX directional albedo E(V, alpha) (F = 1) per cell of mkr's fit, same addressing as [`table`].
pub fn albedo() -> Vec<f32> {
    f32s(GGX_ALBEDO).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Line-for-line port of `shaders/ltc.wgsl` in f32 (WGSL names; mat3x3f is column-major). Keep in sync.
    #[allow(non_snake_case)]
    mod port {
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct Vec3 {
            pub x: f32,
            pub y: f32,
            pub z: f32,
        }

        pub const fn vec3f(x: f32, y: f32, z: f32) -> Vec3 {
            Vec3 { x, y, z }
        }

        impl std::ops::Add for Vec3 {
            type Output = Vec3;
            fn add(self, o: Vec3) -> Vec3 {
                vec3f(self.x + o.x, self.y + o.y, self.z + o.z)
            }
        }

        impl std::ops::Sub for Vec3 {
            type Output = Vec3;
            fn sub(self, o: Vec3) -> Vec3 {
                vec3f(self.x - o.x, self.y - o.y, self.z - o.z)
            }
        }

        impl std::ops::Mul<f32> for Vec3 {
            type Output = Vec3;
            fn mul(self, s: f32) -> Vec3 {
                vec3f(self.x * s, self.y * s, self.z * s)
            }
        }

        /// Columns.
        #[derive(Clone, Copy, Debug)]
        pub struct Mat3(pub [Vec3; 3]);

        impl std::ops::Mul<Vec3> for Mat3 {
            type Output = Vec3;
            fn mul(self, v: Vec3) -> Vec3 {
                self.0[0] * v.x + self.0[1] * v.y + self.0[2] * v.z
            }
        }

        impl std::ops::Mul<Mat3> for Mat3 {
            type Output = Mat3;
            fn mul(self, m: Mat3) -> Mat3 {
                Mat3(m.0.map(|c| self * c))
            }
        }

        pub const fn mat3x3f(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
            Mat3([c0, c1, c2])
        }

        pub fn dot(a: Vec3, b: Vec3) -> f32 {
            a.x * b.x + a.y * b.y + a.z * b.z
        }

        pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
            vec3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
        }

        pub fn length(a: Vec3) -> f32 {
            dot(a, a).sqrt()
        }

        pub fn normalize(a: Vec3) -> Vec3 {
            a * (1.0 / length(a))
        }

        fn inverse_sqrt(x: f32) -> f32 {
            1.0 / x.sqrt()
        }

        fn sign(x: f32) -> f32 {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }

        fn select(f: f32, t: f32, c: bool) -> f32 {
            if c { t } else { f }
        }

        fn transpose(m: Mat3) -> Mat3 {
            let [a, b, c] = m.0;
            Mat3([vec3f(a.x, b.x, c.x), vec3f(a.y, b.y, c.y), vec3f(a.z, b.z, c.z)])
        }

        fn determinant(m: Mat3) -> f32 {
            dot(m.0[0], cross(m.0[1], m.0[2]))
        }

        pub const LTC_LUT_SIZE: f32 = 64.0;
        pub const LTC_PI: f32 = 3.14159265358979323846;
        pub const LTC_LAMBERT: Mat3 =
            mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));

        pub const LTC_DISK_SIDES: u32 = 32;
        pub const LTC_DISK_SCALE: f32 = 1.00322211;
        pub const LTC_DISK_STEP: (f32, f32) = (0.980785251, 0.195090324);

        pub fn ltc_matrix_coords(roughness: f32, cos_nv: f32) -> (f32, f32) {
            let t = ((1.0 - cos_nv.clamp(0.0, 1.0)).sqrt(), roughness.clamp(0.0, 1.0));
            ((t.0 * (LTC_LUT_SIZE - 1.0) + 0.5) / LTC_LUT_SIZE, (t.1 * (LTC_LUT_SIZE - 1.0) + 0.5) / LTC_LUT_SIZE)
        }

        pub fn ltc_minv(c: [f32; 4]) -> Mat3 {
            mat3x3f(vec3f(c[0], 0.0, c[2]), vec3f(0.0, 1.0, 0.0), vec3f(c[1], 0.0, c[3]))
        }

        pub fn ltc_frame(N: Vec3, V: Vec3) -> Mat3 {
            let mut t1 = V - N * dot(N, V);
            let l2 = dot(t1, t1);
            if l2 > 1e-12 {
                t1 = t1 * inverse_sqrt(l2);
            } else {
                let s = select(-1.0, 1.0, N.z >= 0.0);
                let a = -1.0 / (s + N.z);
                t1 = vec3f(1.0 + s * N.x * N.x * a, s * N.x * N.y * a, -s * N.x);
            }
            transpose(mat3x3f(t1, cross(N, t1), N))
        }

        pub fn ltc_edge(a: Vec3, b: Vec3) -> f32 {
            let c = cross(a, b);
            let s = length(c);
            c.z * (s.atan2(dot(a, b)) / s.max(1e-30))
        }

        pub struct LtcAcc {
            n: Vec3,
            sum: f32,
            exit_n: Vec3,
            entry_n: Vec3,
            exit_z: Vec3,
            entry_z: Vec3,
            flags: u32,
        }

        pub fn ltc_acc_new(Minv: Mat3) -> LtcAcc {
            let z = vec3f(0.0, 0.0, 0.0);
            LtcAcc {
                n: cross(Minv.0[0], Minv.0[1]) * sign(determinant(Minv)),
                sum: 0.0,
                exit_n: z,
                entry_n: z,
                exit_z: z,
                entry_z: z,
                flags: 0,
            }
        }

        pub fn ltc_acc_cosine_edge(acc: &mut LtcAcc, a: Vec3, b: Vec3) {
            if a.z >= 0.0 && b.z >= 0.0 {
                acc.sum += ltc_edge(a, b);
            } else if a.z >= 0.0 {
                let x = a + (b - a) * (a.z / (a.z - b.z));
                acc.sum += ltc_edge(a, x);
                acc.exit_z = x;
                acc.flags |= 4;
            } else if b.z >= 0.0 {
                let x = a + (b - a) * (a.z / (a.z - b.z));
                acc.sum += ltc_edge(x, b);
                acc.entry_z = x;
                acc.flags |= 8;
            }
        }

        pub fn ltc_acc_edge(acc: &mut LtcAcc, a: Vec3, b: Vec3) {
            let ha = dot(acc.n, a);
            let hb = dot(acc.n, b);
            if ha >= 0.0 && hb >= 0.0 {
                ltc_acc_cosine_edge(acc, a, b);
            } else if ha >= 0.0 {
                let x = a + (b - a) * (ha / (ha - hb));
                ltc_acc_cosine_edge(acc, a, x);
                acc.exit_n = x;
                acc.flags |= 1;
            } else if hb >= 0.0 {
                let x = a + (b - a) * (ha / (ha - hb));
                ltc_acc_cosine_edge(acc, x, b);
                acc.entry_n = x;
                acc.flags |= 2;
            }
        }

        pub fn ltc_mass_above(n: Vec3) -> f32 {
            (0.5 + 0.5 * n.z / length(n)).max(1e-6)
        }

        pub fn ltc_acc_finish(acc: &mut LtcAcc) -> f32 {
            if (acc.flags & 3) == 3 {
                let (a, b) = (acc.exit_n, acc.entry_n);
                ltc_acc_cosine_edge(acc, a, b);
            }
            if (acc.flags & 12) == 12 {
                acc.sum += ltc_edge(acc.exit_z, acc.entry_z);
            }
            acc.sum.abs() / (2.0 * LTC_PI * ltc_mass_above(acc.n))
        }

        pub fn ltc_quad(N: Vec3, V: Vec3, P: Vec3, Minv: Mat3, corners: [Vec3; 4]) -> f32 {
            let m = Minv * ltc_frame(N, V);
            let q0 = normalize(m * (corners[0] - P));
            let q1 = normalize(m * (corners[1] - P));
            let q2 = normalize(m * (corners[2] - P));
            let q3 = normalize(m * (corners[3] - P));
            let mut acc = ltc_acc_new(Minv);
            ltc_acc_edge(&mut acc, q0, q1);
            ltc_acc_edge(&mut acc, q1, q2);
            ltc_acc_edge(&mut acc, q2, q3);
            ltc_acc_edge(&mut acc, q3, q0);
            ltc_acc_finish(&mut acc)
        }

        pub fn ltc_disk(N: Vec3, V: Vec3, P: Vec3, Minv: Mat3, center: Vec3, half_x: Vec3, half_y: Vec3) -> f32 {
            let m = Minv * ltc_frame(N, V);
            let c = m * (center - P);
            let x = m * half_x * LTC_DISK_SCALE;
            let y = m * half_y * LTC_DISK_SCALE;
            let mut acc = ltc_acc_new(Minv);
            let first = normalize(c + x);
            let mut prev = first;
            let mut cs = (1.0f32, 0.0f32);
            for _ in 1..LTC_DISK_SIDES {
                cs = (cs.0 * LTC_DISK_STEP.0 - cs.1 * LTC_DISK_STEP.1, cs.1 * LTC_DISK_STEP.0 + cs.0 * LTC_DISK_STEP.1);
                let q = normalize(c + x * cs.0 + y * cs.1);
                ltc_acc_edge(&mut acc, prev, q);
                prev = q;
            }
            ltc_acc_edge(&mut acc, prev, first);
            ltc_acc_finish(&mut acc)
        }

        pub fn ltc_edge_vec(a: Vec3, b: Vec3) -> Vec3 {
            let c = cross(a, b);
            let s = length(c);
            c * (s.atan2(dot(a, b)) / s.max(1e-30))
        }

        pub fn ltc_sphere_integral(z: f32, f: f32) -> f32 {
            if z * z >= f {
                return f * z.max(0.0);
            }
            let st = (1.0 - z * z).max(0.0).sqrt();
            let x = (1.0 / f - 1.0).max(0.0).sqrt();
            let y = (-x * z / st.max(1e-12)).clamp(-1.0, 1.0);
            let s = st * (1.0 - y * y).max(0.0).sqrt();
            (((z * y.acos() - x * s) * f + s.atan2(x)) / LTC_PI).max(0.0)
        }

        pub fn ltc_sphere_finish(F_in: Vec3, toward: Vec3) -> f32 {
            let F = F_in * (select(1.0, -1.0, dot(F_in, toward) < 0.0) / (2.0 * LTC_PI));
            let f = length(F);
            if f <= 1e-12 {
                return 0.0;
            }
            ltc_sphere_integral(F.z / f, f.min(1.0))
        }

        pub fn ltc_quad_sphere(N: Vec3, V: Vec3, P: Vec3, Minv: Mat3, corners: [Vec3; 4]) -> f32 {
            let m = Minv * ltc_frame(N, V);
            let q0 = normalize(m * (corners[0] - P));
            let q1 = normalize(m * (corners[1] - P));
            let q2 = normalize(m * (corners[2] - P));
            let q3 = normalize(m * (corners[3] - P));
            let F = ltc_edge_vec(q0, q1) + ltc_edge_vec(q1, q2) + ltc_edge_vec(q2, q3) + ltc_edge_vec(q3, q0);
            ltc_sphere_finish(F, q0 + q1 + q2 + q3)
        }

        pub fn ltc_disk_sphere(N: Vec3, V: Vec3, P: Vec3, Minv: Mat3, center: Vec3, half_x: Vec3, half_y: Vec3) -> f32 {
            let m = Minv * ltc_frame(N, V);
            let c = m * (center - P);
            let x = m * half_x * LTC_DISK_SCALE;
            let y = m * half_y * LTC_DISK_SCALE;
            let first = normalize(c + x);
            let mut prev = first;
            let mut F = vec3f(0.0, 0.0, 0.0);
            let mut cs = (1.0f32, 0.0f32);
            for _ in 1..LTC_DISK_SIDES {
                cs = (cs.0 * LTC_DISK_STEP.0 - cs.1 * LTC_DISK_STEP.1, cs.1 * LTC_DISK_STEP.0 + cs.0 * LTC_DISK_STEP.1);
                let q = normalize(c + x * cs.0 + y * cs.1);
                F = F + ltc_edge_vec(prev, q);
                prev = q;
            }
            F = F + ltc_edge_vec(prev, first);
            ltc_sphere_finish(F, c)
        }

        pub fn ltc_horizon_fade(N: Vec3, toward: Vec3, r: f32) -> f32 {
            (1.0 + 10.0 * (dot(N, toward) + r)).clamp(0.0, 1.0)
        }

        /// textureSampleLevel of the table (64 x 64, texel (x = j, y = i)) through a linear clamp-to-edge sampler.
        pub fn sample_table(table: &[[f32; 4]], uv: (f32, f32)) -> [f32; 4] {
            let n = super::LTC_SIZE;
            let x = uv.0 * n as f32 - 0.5;
            let y = uv.1 * n as f32 - 0.5;
            let (fx, fy) = (x - x.floor(), y - y.floor());
            let clamp = |k: f32| (k.max(0.0) as usize).min(n - 1);
            let (x0, x1, y0, y1) = (clamp(x.floor()), clamp(x.floor() + 1.0), clamp(y.floor()), clamp(y.floor() + 1.0));
            let at = |xi: usize, yi: usize| table[yi * n + xi];
            std::array::from_fn(|k| {
                (1.0 - fy) * ((1.0 - fx) * at(x0, y0)[k] + fx * at(x1, y0)[k])
                    + fy * ((1.0 - fx) * at(x0, y1)[k] + fx * at(x1, y1)[k])
            })
        }
    }

    use port::{
        LTC_LAMBERT, Vec3, ltc_disk, ltc_disk_sphere, ltc_horizon_fade, ltc_matrix_coords, ltc_minv, ltc_quad,
        ltc_quad_sphere, ltc_sphere_integral, sample_table, vec3f,
    };

    // ---- brute force, f64 ----

    #[derive(Clone, Copy, Debug)]
    struct D3 {
        x: f64,
        y: f64,
        z: f64,
    }

    const fn d3(x: f64, y: f64, z: f64) -> D3 {
        D3 { x, y, z }
    }

    impl std::ops::Add for D3 {
        type Output = D3;
        fn add(self, o: D3) -> D3 {
            d3(self.x + o.x, self.y + o.y, self.z + o.z)
        }
    }

    impl std::ops::Sub for D3 {
        type Output = D3;
        fn sub(self, o: D3) -> D3 {
            d3(self.x - o.x, self.y - o.y, self.z - o.z)
        }
    }

    impl std::ops::Mul<f64> for D3 {
        type Output = D3;
        fn mul(self, s: f64) -> D3 {
            d3(self.x * s, self.y * s, self.z * s)
        }
    }

    fn ddot(a: D3, b: D3) -> f64 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    fn dcross(a: D3, b: D3) -> D3 {
        d3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    fn dnormalize(a: D3) -> D3 {
        a * (1.0 / ddot(a, a).sqrt())
    }

    fn f(a: D3) -> Vec3 {
        vec3f(a.x as f32, a.y as f32, a.z as f32)
    }

    /// Point i of n of a stratified Kronecker lattice in [0, 1)².
    fn lattice(i: usize, n: usize) -> (f64, f64) {
        let v = i as f64 * 0.618_033_988_749_894_9 + 0.137;
        ((i as f64 + 0.5) / n as f64, v - v.floor())
    }

    const POINTS: usize = 1 << 18;

    /// GGX (alpha = roughness²), height-correlated Smith G2, F = 1, in the frame N = z, V = (sin, 0, cos).
    struct Ggx {
        alpha: f64,
        v: D3,
        lambda_v: f64,
        e: f64,
    }

    fn lambda(c: f64, a2: f64) -> f64 {
        if c >= 1.0 {
            return 0.0;
        }
        let q2 = a2 * (1.0 - c * c) / (c * c);
        q2 / (2.0 * ((1.0 + q2).sqrt() + 1.0))
    }

    impl Ggx {
        fn new(roughness: f64, view_deg: f64) -> Ggx {
            let alpha = (roughness * roughness).max(1e-4);
            let (s, c) = view_deg.to_radians().sin_cos();
            let mut g = Ggx { alpha, v: d3(s, 0.0, c), lambda_v: lambda(c, alpha * alpha), e: 1.0 };
            g.e = (0..POINTS).map(|i| g.weight(g.sample(lattice(i, POINTS)))).sum::<f64>() / POINTS as f64;
            g
        }

        /// Reflection of V about a visible normal (spherical-cap sampling of the stretched hemisphere).
        fn sample(&self, (u, w): (f64, f64)) -> D3 {
            let a = self.alpha;
            let vh = dnormalize(d3(a * self.v.x, a * self.v.y, self.v.z));
            let (sp, cp) = (2.0 * std::f64::consts::PI * u).sin_cos();
            let z = (1.0 - w) * (1.0 + vh.z) - vh.z;
            let s = (1.0 - z * z).max(0.0).sqrt();
            let hs = d3(s * cp, s * sp, z) + vh;
            let h = dnormalize(d3(a * hs.x, a * hs.y, hs.z));
            h * (2.0 * ddot(self.v, h)) - self.v
        }

        /// BRDF cos / pdf of a visible-normal sample: G2 / G1(V).
        fn weight(&self, l: D3) -> f64 {
            if l.z <= 0.0 {
                return 0.0;
            }
            (1.0 + self.lambda_v) / (1.0 + self.lambda_v + lambda(l.z, self.alpha * self.alpha))
        }

        /// The directional albedo with separable Smith masking G1(V) G1(L) (the closure colours' split sum, F = 1).
        fn albedo_separable(&self) -> f64 {
            let a2 = self.alpha * self.alpha;
            let s: f64 = (0..POINTS)
                .map(|i| self.sample(lattice(i, POINTS)))
                .filter(|l| l.z > 0.0)
                .map(|l| 1.0 / (1.0 + lambda(l.z, a2)))
                .sum();
            s / POINTS as f64
        }

        /// Integral of BRDF cos / E over the directions `hit` accepts (clipped to z > 0).
        fn integrate(&self, hit: impl Fn(D3) -> bool) -> f64 {
            let s: f64 = (0..POINTS)
                .map(|i| self.sample(lattice(i, POINTS)))
                .filter(|&l| l.z > 0.0 && hit(l))
                .map(|l| self.weight(l))
                .sum();
            s / (POINTS as f64 * self.e)
        }
    }

    /// Integral of cos / pi over the directions `hit` accepts.
    fn integrate_lambert(hit: impl Fn(D3) -> bool) -> f64 {
        let hits = (0..POINTS)
            .filter(|&i| {
                let (u, w) = lattice(i, POINTS);
                let (s, c) = (2.0 * std::f64::consts::PI * w).sin_cos();
                hit(d3(u.sqrt() * c, u.sqrt() * s, (1.0 - u).sqrt()))
            })
            .count();
        hits as f64 / POINTS as f64
    }

    /// A planar parallelogram light around `c` with half axes `ex`, `ey`, seen from the origin.
    #[derive(Clone, Copy)]
    struct Light {
        c: D3,
        ex: D3,
        ey: D3,
    }

    impl Light {
        /// Around the direction (elevation, azimuth) in degrees at distance d, facing the origin, then turned by
        /// `tilt` degrees about its x axis.
        fn at(elev: f64, azim: f64, d: f64, hx: f64, hy: f64, tilt: f64) -> Light {
            let (se, ce) = elev.to_radians().sin_cos();
            let (sa, ca) = azim.to_radians().sin_cos();
            let dir = d3(ce * ca, ce * sa, se);
            let up = if se.abs() > 0.99 { d3(1.0, 0.0, 0.0) } else { d3(0.0, 0.0, 1.0) };
            let x = dnormalize(dcross(up, dir));
            let y = dcross(dir, x);
            let (st, ct) = tilt.to_radians().sin_cos();
            Light { c: dir * d, ex: x * hx, ey: (y * ct + dir * st) * hy }
        }

        fn corners(&self) -> [D3; 4] {
            let (c, x, y) = (self.c, self.ex, self.ey);
            [c - x - y, c + x - y, c + x + y, c - x + y]
        }

        /// (s, t) of the ray from the origin along l in the light's axes (|s|, |t| <= 1 on the rectangle), if it meets
        /// the plane in front.
        fn uv(&self, l: D3) -> Option<(f64, f64)> {
            let n = dcross(self.ex, self.ey);
            let t = ddot(self.c, n) / ddot(l, n);
            if !(t > 0.0) {
                return None;
            }
            let q = l * t - self.c;
            Some((ddot(q, self.ex) / ddot(self.ex, self.ex), ddot(q, self.ey) / ddot(self.ey, self.ey)))
        }

        fn hit_rect(&self, l: D3) -> bool {
            self.uv(l).is_some_and(|(s, t)| s.abs() <= 1.0 && t.abs() <= 1.0)
        }

        fn hit_ellipse(&self, l: D3) -> bool {
            self.uv(l).is_some_and(|(s, t)| s * s + t * t <= 1.0)
        }
    }

    /// The tests' rectangles (`tools/ltc_fit.rs --check` prints the same set).
    fn rects() -> Vec<(&'static str, Light)> {
        vec![
            ("big overhead", Light { c: d3(0.0, 0.0, 1.0), ex: d3(10.0, 0.0, 0.0), ey: d3(0.0, 10.0, 0.0) }),
            ("medium overhead", Light::at(60.0, 30.0, 2.0, 0.5, 0.3, 20.0)),
            ("small grazing, mirror side", Light::at(10.0, 180.0, 1.0, 0.1, 0.1, 0.0)),
            ("small grazing, view side", Light::at(10.0, 0.0, 1.0, 0.1, 0.1, 0.0)),
            ("small grazing, sideways", Light::at(10.0, 90.0, 1.0, 0.1, 0.1, 0.0)),
            ("straddling the horizon", Light::at(0.0, 200.0, 1.0, 0.3, 0.2, 0.0)),
        ]
    }

    const N: Vec3 = vec3f(0.0, 0.0, 1.0);
    const P: Vec3 = vec3f(0.0, 0.0, 0.0);

    fn view(deg: f64) -> Vec3 {
        let (s, c) = deg.to_radians().sin_cos();
        vec3f(s as f32, 0.0, c as f32)
    }

    /// The GPU's path: table lookup, then ltc_quad.
    fn ggx_quad(table: &[[f32; 4]], roughness: f64, view_deg: f64, light: &Light) -> f64 {
        let v = view(view_deg);
        let minv = ltc_minv(sample_table(table, ltc_matrix_coords(roughness as f32, port::dot(N, v))));
        ltc_quad(N, v, P, minv, light.corners().map(f)) as f64
    }

    /// Relative error where the integral exceeds 0.01, absolute error below.
    fn within(got: f64, want: f64, rel: f64, abs: f64) -> bool {
        if want > 0.01 { (got - want).abs() <= rel * want } else { (got - want).abs() <= abs }
    }

    #[test]
    fn table_files_have_the_documented_layout() {
        for fit in [LtcFit::Mkr, LtcFit::Eevee] {
            let t = table(fit);
            assert_eq!(t.len(), LTC_SIZE * LTC_SIZE);
            for (k, c) in t.iter().enumerate() {
                assert!(c.iter().all(|v| v.is_finite()), "{fit:?} entry {k}: {c:?}");
                let valid = c[0] > 0.0 && c[3] > 0.0 && c[0] * c[3] - c[1] * c[2] > 0.0;
                assert!(valid, "{fit:?} entry {k} not a valid Minv: {c:?}");
            }
            for i in 0..LTC_SIZE {
                let c = t[i * LTC_SIZE];
                let isotropic = (c[0] - 1.0).abs() < 1e-3 && c[1].abs() < 1e-3 && c[2].abs() < 1e-3;
                assert!(isotropic, "{fit:?}: V = N is isotropic: {c:?}");
            }
        }
        let e = albedo();
        assert_eq!(e.len(), LTC_SIZE * LTC_SIZE);
        assert!(e.iter().all(|&v| v > 0.0 && v <= 1.0), "albedo out of (0, 1]");
    }

    /// EEVEE Next's glossy lobe as measured (Blender 4.2.3: a white Glossy BSDF (GGX) plane, a 4 cm square area light
    /// 1 m above it, shadows off, orthographic views, scene-linear EXR): the LTC fitted to each frame (residual 3e-5 to
    /// 8e-5 of the peak), (m00, m02, m20, m22), at (63 roughness, 63 sqrt(1 - cos(N, V))): table cells, then points
    /// between them.
    const EEVEE_PROBES: [(f64, f64, [f32; 4]); 44] = [
        (13.0, 0.0, [1.0000, -0.0000, -0.0000, 0.0851]),
        (13.0, 16.0, [0.8750, 0.3305, -0.0281, 0.0745]),
        (13.0, 32.0, [0.5511, 0.4976, -0.0422, 0.0470]),
        (13.0, 44.0, [0.2637, 0.4411, -0.0371, 0.0228]),
        (13.0, 52.0, [0.1029, 0.3045, -0.0255, 0.0093]),
        (13.0, 58.0, [0.0252, 0.1582, -0.0128, 0.0027]),
        (22.0, 0.0, [1.0000, -0.0000, 0.0000, 0.2414]),
        (22.0, 16.0, [0.8779, 0.3307, -0.0773, 0.2131]),
        (22.0, 32.0, [0.5590, 0.5013, -0.1162, 0.1379]),
        (22.0, 44.0, [0.2765, 0.4537, -0.1024, 0.0723]),
        (22.0, 52.0, [0.1247, 0.3439, -0.0731, 0.0369]),
        (22.0, 58.0, [0.0541, 0.2430, -0.0469, 0.0205]),
        (32.0, 0.0, [1.0000, -0.0000, 0.0000, 0.4940]),
        (32.0, 16.0, [0.8915, 0.3276, -0.1445, 0.4412]),
        (32.0, 32.0, [0.6255, 0.5309, -0.2191, 0.3111]),
        (32.0, 44.0, [0.3846, 0.5425, -0.2057, 0.2068]),
        (32.0, 52.0, [0.2485, 0.4794, -0.1711, 0.1503]),
        (32.0, 58.0, [0.1695, 0.3877, -0.1372, 0.1216]),
        (41.0, 0.0, [1.0000, -0.0000, 0.0000, 0.7459]),
        (41.0, 16.0, [0.9382, 0.2976, -0.1748, 0.6917]),
        (41.0, 32.0, [0.7698, 0.5223, -0.2773, 0.5618]),
        (41.0, 44.0, [0.5937, 0.5852, -0.2887, 0.4506]),
        (41.0, 52.0, [0.4823, 0.5493, -0.2634, 0.4039]),
        (41.0, 58.0, [0.3997, 0.4835, -0.2335, 0.3710]),
        (50.0, 0.0, [1.0000, -0.0000, 0.0000, 0.9577]),
        (50.0, 16.0, [0.9756, 0.1964, -0.1356, 0.9310]),
        (50.0, 32.0, [0.9050, 0.3570, -0.2316, 0.8525]),
        (50.0, 44.0, [0.8224, 0.4255, -0.2615, 0.7950]),
        (50.0, 52.0, [0.7554, 0.4346, -0.2610, 0.7656]),
        (50.0, 58.0, [0.7021, 0.4107, -0.2409, 0.7786]),
        (63.0, 0.0, [1.0000, -0.0000, 0.0000, 1.1274]),
        (63.0, 16.0, [1.0000, -0.0000, 0.0003, 1.1333]),
        (63.0, 32.0, [1.0001, 0.0001, 0.0002, 1.1570]),
        (63.0, 44.0, [0.9989, 0.0002, 0.0003, 1.1988]),
        (63.0, 52.0, [0.9933, -0.0005, 0.0029, 1.2622]),
        (63.0, 58.0, [0.9802, -0.0117, 0.0325, 1.3638]),
        (31.50, 40.5, [0.4459, 0.5481, -0.2103, 0.2277]),
        (31.50, 47.5, [0.3138, 0.5180, -0.1877, 0.1717]),
        (22.05, 36.5, [0.4521, 0.5031, -0.1163, 0.1136]),
        (50.40, 44.5, [0.8261, 0.4163, -0.2578, 0.8065]),
        (40.95, 50.5, [0.5023, 0.5597, -0.2693, 0.4118]),
        (27.09, 55.5, [0.1252, 0.3594, -0.0991, 0.0631]),
        (50.40, 27.5, [0.9345, 0.3090, -0.2054, 0.8870]),
        (17.01, 49.5, [0.1509, 0.3594, -0.0504, 0.0240]),
    ];

    /// EEVEE's table as the GPU samples it reproduces the probes' LTCs: within 0.0025 per coefficient, 0.02 where a rough
    /// lobe (roughness 0.65 and up) seen near grazing spans the whole frame (it pins the lobe there, less its numbers).
    #[test]
    fn eevee_table_matches_the_measured_lobes() {
        let table = table(LtcFit::Eevee);
        for (r, x, want) in EEVEE_PROBES {
            let (roughness, cos_nv) = (r / 63.0, 1.0 - (x / 63.0) * (x / 63.0));
            let got = sample_table(&table, ltc_matrix_coords(roughness as f32, cos_nv as f32));
            let err = (0..4).map(|k| (got[k] - want[k]).abs()).fold(0.0, f32::max);
            let allowed = if r >= 41.0 && x >= 52.0 { 0.02 } else { 0.0025 };
            assert!(err <= allowed, "roughness {roughness:.4} cos {cos_nv:.4}: table {got:?}, EEVEE {want:?}");
        }
    }

    /// EEVEE Next under a light covering the hemisphere (a 100 m square area light 0.1 m above the same plane): glossy
    /// over diffuse at (roughness index, view index): the split-sum albedo times the LTC's mass above the horizon.
    const EEVEE_HEMISPHERE: [(usize, usize, f64); 36] = [
        (13, 0, 0.9971), (13, 16, 0.9971), (13, 32, 0.9966), (13, 44, 0.9878), (13, 52, 0.9727), (13, 58, 0.9019),
        (22, 0, 0.9814), (22, 16, 0.9795), (22, 32, 0.9634), (22, 44, 0.9253), (22, 52, 0.8564), (22, 58, 0.7646),
        (32, 0, 0.9092), (32, 16, 0.8979), (32, 32, 0.8569), (32, 44, 0.7983), (32, 52, 0.7534), (32, 58, 0.7427),
        (41, 0, 0.7627), (41, 16, 0.7510), (41, 32, 0.7192), (41, 44, 0.6968), (41, 52, 0.6992), (41, 58, 0.7280),
        (50, 0, 0.5635), (50, 16, 0.5630), (50, 32, 0.5669), (50, 44, 0.5898), (50, 52, 0.6260), (50, 58, 0.6763),
        (63, 0, 0.3066), (63, 16, 0.3167), (63, 32, 0.3516), (63, 44, 0.4053), (63, 52, 0.4648), (63, 58, 0.5312),
    ];

    /// What EEVEE loses under the horizon, mkr's EEVEE path loses: the sphere-horizon integral of the hemisphere with
    /// EEVEE's table, times the split sum (separable Smith masking, as the closure colours), within 0.5 %.
    #[test]
    fn eevee_table_loses_what_eevee_loses_under_the_horizon() {
        let table = table(LtcFit::Eevee);
        let sky = Light { c: d3(0.0, 0.0, 1.0), ex: d3(1e4, 0.0, 0.0), ey: d3(0.0, 1e4, 0.0) };
        for (i, j, want) in EEVEE_HEMISPHERE {
            let roughness = i as f64 / 63.0;
            let x = j as f64 / 63.0;
            let view_deg = (1.0 - x * x).acos().to_degrees();
            let v = view(view_deg);
            let minv = ltc_minv(sample_table(&table, ltc_matrix_coords(roughness as f32, port::dot(N, v))));
            let lobe = ltc_quad_sphere(N, v, P, minv, sky.corners().map(f)) as f64;
            let got = Ggx::new(roughness, view_deg).albedo_separable() * lobe;
            assert!((got - want).abs() <= 0.005 * want, "roughness {roughness:.3} view {view_deg:.1}: {got:.4}, EEVEE {want}");
        }
    }

    /// EEVEE Next with the light wholly or partly under a glossy plane's horizon (GGX roughness 0.35 and 0.5, shadows
    /// off): its frame over the sphere-horizon LTC integral, per pixel, with N.L toward the light's centre and the
    /// light's bounding radius over its distance (rectangle and disk: half the diagonal of their size; sphere: the
    /// radius; sun: tan of its half angle).
    const EEVEE_UNDER_HORIZON: [(&str, f32, f32, f32); 15] = [
        ("1.5 m square 0.5 m under", -0.6766, 1.4352, 0.9963),
        ("0.6 m square 0.25 m under", -0.2899, 0.4921, 0.9980),
        ("1.5 m square 0.5 m under", -0.2770, 0.5875, 0.9980),
        ("0.6 m square 0.5 m under", -0.4470, 0.3793, 0.3207),
        ("0.6 m square 0.5 m under", -0.2492, 0.2115, 0.6221),
        ("0.6 m square 0.5 m under", -0.1761, 0.1494, 0.7339),
        ("0.6 m disk 0.5 m under", -0.5226, 0.4435, 0.2073),
        ("0.6 m disk 0.5 m under", -0.2993, 0.2540, 0.5442),
        ("0.2 m sphere 0.3 m under", -0.2417, 0.1611, 0.1934),
        ("0.2 m sphere 0.3 m under", -0.1471, 0.0981, 0.5089),
        ("0.2 m sphere 0.6 m under", -0.1192, 0.0397, 0.2049),
        ("0.05 m sphere 0.3 m under", -0.0987, 0.0165, 0.1768),
        ("10 degree sun 5 degrees under", -0.0872, 0.0875, 1.0004),
        ("10 degree sun 6 degrees under", -0.1045, 0.0875, 0.8317),
        ("10 degree sun 8 degrees under", -0.1392, 0.0875, 0.4834),
    ];

    #[test]
    fn horizon_fade_matches_eevee_under_the_horizon() {
        for (what, nl, r, want) in EEVEE_UNDER_HORIZON {
            let toward = vec3f((1.0 - nl * nl).sqrt(), 0.0, nl);
            let got = ltc_horizon_fade(N, toward, r);
            assert!((got - want).abs() <= 0.005, "{what}: N.L {nl}, r {r}: fade {got}, EEVEE {want}");
        }
        assert_eq!(ltc_horizon_fade(N, vec3f(0.0, 0.0, 1.0), 0.0), 1.0, "a light overhead");
        assert_eq!(ltc_horizon_fade(N, vec3f(0.0, 0.0, -1.0), 0.5), 0.0, "a light underneath");
    }

    #[test]
    fn albedo_matches_brute_force() {
        for (i, j) in [(0, 0), (3, 30), (13, 63), (32, 45), (63, 0), (63, 57), (47, 12)] {
            let roughness = i as f64 / 63.0;
            // The last column holds the fit at sqrt(1 - cos) = 62.5 / 63 (module docs).
            let v = if j == LTC_SIZE - 1 { 62.5 / 63.0 } else { j as f64 / 63.0 };
            let want = Ggx::new(roughness, (1.0 - v * v).acos().to_degrees()).e;
            let got = albedo()[i * LTC_SIZE + j] as f64;
            assert!((got - want).abs() < 1e-3, "cell ({i}, {j}): albedo {got} vs brute force {want}");
        }
    }

    /// The test cases no 4-coefficient LTC brings within 3 % (every fitting metric tried, see `tools/ltc_fit.rs`):
    /// at roughness 0.5 the lobe at grazing view is a crescent against the horizon, at roughness 1 it is isotropic
    /// about N and nearly uniform at grazing, which a clamped cosine under a linear map cannot be. Relative error
    /// allowed: what the table reaches plus a margin.
    const LTC_LIMITED: [(f64, f64, &str, f64); 5] = [
        (0.5, 0.0, "medium overhead", 0.05),             // 3.8 %
        (0.5, 45.0, "small grazing, mirror side", 0.35), // 28.9 %
        (0.5, 80.0, "small grazing, mirror side", 0.50), // 44.1 %
        (0.5, 80.0, "straddling the horizon", 0.85),     // 76.4 %
        (1.0, 80.0, "medium overhead", 0.06),            // 4.5 %
    ];

    #[test]
    fn ggx_rectangles_match_brute_force() {
        let table = table(LtcFit::Mkr);
        let mut report = Vec::new();
        let mut failed = 0;
        for roughness in [0.05, 0.2, 0.5, 1.0] {
            for view_deg in [0.0, 45.0, 80.0] {
                let ggx = Ggx::new(roughness, view_deg);
                for (name, light) in rects() {
                    let got = ggx_quad(&table, roughness, view_deg, &light);
                    let want = ggx.integrate(|l| light.hit_rect(l));
                    let limit = LTC_LIMITED.iter().find(|c| (c.0, c.1, c.2) == (roughness, view_deg, name));
                    let ok = within(got, want, limit.map_or(0.03, |c| c.3), 0.003);
                    failed += !ok as usize;
                    report.push(format!(
                        "{} roughness {roughness} view {view_deg}: {name}: LTC {got:.5} brute force {want:.5}{}",
                        if ok { "  " } else { "!!" },
                        limit.map_or(String::new(), |c| format!(" (LTC limit, {} % allowed)", 100.0 * c.3))
                    ));
                }
            }
        }
        assert!(failed == 0, "{failed} cases outside 3 % / 0.003 or their LTC limit:\n{}", report.join("\n"));
    }

    #[test]
    fn lambert_rectangles_match_brute_force() {
        let mut lights = rects();
        lights.push(("tilted, half below", Light::at(-5.0, 70.0, 1.5, 1.0, 0.4, 50.0)));
        lights.push(("below the horizon", Light::at(-30.0, 10.0, 1.0, 0.2, 0.2, 0.0)));
        for (name, light) in lights {
            let got = ltc_quad(N, view(30.0), P, LTC_LAMBERT, light.corners().map(f)) as f64;
            let want = integrate_lambert(|l| light.hit_rect(l));
            assert!(within(got, want, 0.003, 0.0005), "{name}: LTC {got:.5} brute force {want:.5}");
        }
    }

    /// Integral over the directions `hit` accepts (clipped to z > 0) of the lobe `minv` describes, renormalised as
    /// ltc_quad does, by sampling the LTC (with N = z and V in the xz plane the shading frame is the world frame).
    fn integrate_ltc(minv: [f32; 4], hit: impl Fn(D3) -> bool) -> f64 {
        let [a, b, c, d] = minv.map(|v| v as f64);
        let det = a * d - b * c;
        // M = Minv⁻¹; its third row (-c, 0, a) / det is N's horizon normal in the cosine's space.
        let m = [[d / det, 0.0, -b / det], [0.0, 1.0, 0.0], [-c / det, 0.0, a / det]];
        let above = 0.5 + 0.5 * a / (a * a + c * c).sqrt();
        let hits = (0..POINTS)
            .filter(|&i| {
                let (u, w) = lattice(i, POINTS);
                let (s, cs) = (2.0 * std::f64::consts::PI * w).sin_cos();
                let o = d3(u.sqrt() * cs, u.sqrt() * s, (1.0 - u).sqrt());
                let l = d3(ddot(d3(m[0][0], 0.0, m[0][2]), o), o.y, ddot(d3(m[2][0], 0.0, m[2][2]), o));
                l.z > 0.0 && hit(l)
            })
            .count();
        hits as f64 / (POINTS as f64 * above)
    }

    /// ltc_disk's area-matched 32-gon against the same lobe over the exact ellipse (the lobe's own accuracy is the
    /// rectangle tests' business).
    #[test]
    fn disks_match_the_exact_ellipse() {
        let table = table(LtcFit::Mkr);
        let disks = [
            ("overhead", Light::at(70.0, 0.0, 2.0, 0.6, 0.6, 0.0)),
            ("ellipse, mirror side", Light::at(35.0, 180.0, 1.5, 0.5, 0.25, 15.0)),
            ("close, wide", Light::at(50.0, 120.0, 0.6, 0.5, 0.5, -20.0)),
            ("straddling the horizon", Light::at(3.0, 160.0, 1.0, 0.3, 0.3, 0.0)),
            // Its rim passes through the mirror direction of a 45 degree view.
            ("rim on the mirror direction", Light::at(55.0, 180.0, 1.0, 0.1763, 0.1763, 0.0)),
        ];
        for (name, light) in disks {
            let got = ltc_disk(N, view(30.0), P, LTC_LAMBERT, f(light.c), f(light.ex), f(light.ey)) as f64;
            let want = integrate_lambert(|l| light.hit_ellipse(l));
            assert!(within(got, want, 0.005, 0.0005), "Lambert {name}: 32-gon {got:.5}, ellipse {want:.5}");
            for (roughness, view_deg) in [(0.5, 30.0), (0.2, 50.0), (0.05, 45.0)] {
                let v = view(view_deg);
                let c = sample_table(&table, ltc_matrix_coords(roughness as f32, port::dot(N, v)));
                let got = ltc_disk(N, v, P, ltc_minv(c), f(light.c), f(light.ex), f(light.ey)) as f64;
                let want = integrate_ltc(c, |l| light.hit_ellipse(l));
                // A near-mirror lobe (0.1 degree) on the rim resolves the 32-gon's +0.32 % / -0.16 % radius: a vertex
                // right on the lobe moves 7 % of it.
                let rel = if roughness < 0.1 { 0.08 } else { 0.01 };
                assert!(
                    within(got, want, rel, 0.001),
                    "GGX roughness {roughness} view {view_deg} {name}: 32-gon {got:.5}, ellipse {want:.5}"
                );
            }
        }
    }

    #[test]
    fn winding_horizon_and_sides() {
        let table = table(LtcFit::Mkr);
        let big = Light { c: d3(0.0, 0.0, 1.0), ex: d3(1e4, 0.0, 0.0), ey: d3(0.0, 1e4, 0.0) };
        let below = Light::at(-40.0, 30.0, 1.0, 0.3, 0.3, 0.0);
        let side = Light::at(20.0, 150.0, 1.0, 0.4, 0.3, 30.0);
        for roughness in [0.0, 0.05, 0.3, 0.7, 1.0] {
            for view_deg in [0.0, 30.0, 60.0, 85.0, 90.0] {
                let v = view(view_deg);
                let minv = ltc_minv(sample_table(&table, ltc_matrix_coords(roughness as f32, port::dot(N, v))));
                let all = ltc_quad(N, v, P, minv, big.corners().map(f));
                assert!((all - 1.0).abs() < 1e-3, "roughness {roughness} view {view_deg}: hemisphere gives {all}");
                assert_eq!(ltc_quad(N, v, P, minv, below.corners().map(f)), 0.0, "below the horizon");
                let mut c = side.corners().map(f);
                let a = ltc_quad(N, v, P, minv, c);
                c.reverse();
                let b = ltc_quad(N, v, P, minv, c);
                assert!(a > 0.0 && (a - b).abs() <= 1e-6 + 1e-5 * a, "winding: {a} vs {b}");
            }
        }
        let all = ltc_quad(N, view(40.0), P, LTC_LAMBERT, big.corners().map(f));
        assert!((all - 1.0).abs() < 1e-3, "Lambert hemisphere gives {all}");
    }

    #[test]
    fn mirror_limit_is_stable() {
        // A 0.2 x 0.2 light 1 away around the mirror direction of a 45 degree view, and one 25 degrees off it; each
        // table on its path (mkr's clipped to the hemisphere, EEVEE's through the sphere horizon).
        let hit = Light::at(45.0, 180.0, 1.0, 0.1, 0.1, 0.0);
        let miss = Light::at(70.0, 180.0, 1.0, 0.1, 0.1, 0.0);
        let v = view(45.0);
        for fit in [LtcFit::Mkr, LtcFit::Eevee] {
            let table = table(fit);
            let lobe = |roughness: f64, light: &Light| -> f64 {
                let minv = ltc_minv(sample_table(&table, ltc_matrix_coords(roughness as f32, port::dot(N, v))));
                let corners = light.corners().map(f);
                let sphere = fit == LtcFit::Eevee;
                (if sphere { ltc_quad_sphere(N, v, P, minv, corners) } else { ltc_quad(N, v, P, minv, corners) }) as f64
            };
            let mut last: Option<f64> = None;
            for roughness in [0.0, 1e-3, 0.005, 0.01, 0.02, 0.04] {
                let (a, b) = (lobe(roughness, &hit), lobe(roughness, &miss));
                assert!(a.is_finite() && b.is_finite(), "{fit:?} roughness {roughness}: {a} {b}");
                assert!(a > 0.995 && a <= 1.0 + 1e-4, "{fit:?} roughness {roughness}: mirror direction inside gives {a}");
                assert!(b < 1e-3, "{fit:?} roughness {roughness}: mirror direction outside gives {b}");
                if let Some(prev) = last {
                    assert!((a - prev).abs() < 5e-3, "{fit:?} roughness {roughness}: {a} jumps from {prev}");
                }
                last = Some(a);
            }
        }
    }

    /// The sphere approximation's clipped-sphere integral against brute force over caps, above, across and below the
    /// horizon.
    #[test]
    fn sphere_integral_matches_clipped_caps() {
        for half_deg in [5.0f64, 20.0, 45.0, 80.0] {
            for elev_deg in [80.0f64, 40.0, 10.0, 0.0, -10.0, -30.0] {
                let (sh, ch) = half_deg.to_radians().sin_cos();
                let (se, ce) = elev_deg.to_radians().sin_cos();
                let c = d3(ce, 0.0, se);
                let want = integrate_lambert(|l| ddot(l, c) >= ch);
                let got = ltc_sphere_integral(se as f32, (sh * sh) as f32) as f64;
                assert!(within(got, want, 0.01, 2e-3), "cap {half_deg} deg at {elev_deg} deg: {got} vs {want}");
            }
        }
    }

    /// Above the horizon the sphere path is the exact form factor (the clipped path's Lambert); across it, an
    /// approximation of the clipped polygon.
    #[test]
    fn sphere_horizon_paths() {
        for light in [Light::at(60.0, 30.0, 2.0, 0.5, 0.3, 20.0), Light::at(80.0, 0.0, 1.0, 0.2, 0.2, 0.0)] {
            let a = ltc_quad(N, view(30.0), P, LTC_LAMBERT, light.corners().map(f));
            let b = ltc_quad_sphere(N, view(30.0), P, LTC_LAMBERT, light.corners().map(f));
            assert!((a - b).abs() <= 1e-4 * a, "{a} vs {b}");
        }
        let disk = Light::at(70.0, 0.0, 2.0, 0.6, 0.6, 0.0);
        let want = integrate_lambert(|l| disk.hit_ellipse(l));
        let got = ltc_disk_sphere(N, view(20.0), P, LTC_LAMBERT, f(disk.c), f(disk.ex), f(disk.ey)) as f64;
        assert!(within(got, want, 0.01, 1e-3), "disk: {got} vs {want}");
        let straddle = Light::at(0.0, 200.0, 1.0, 0.3, 0.2, 0.0);
        let want = integrate_lambert(|l| straddle.hit_rect(l));
        let got = ltc_quad_sphere(N, view(30.0), P, LTC_LAMBERT, straddle.corners().map(f)) as f64;
        assert!(within(got, want, 0.25, 0.01), "straddling: {got} vs {want}");
    }
}
