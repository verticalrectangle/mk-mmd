//! The view transform as the bake captured it (bake v5 `lit.view.lut`): Blender's display-referred output for
//! scene-linear inputs 2^(lo + (hi - lo) i / (n - 1)) per channel, f32 RGB at ((b n + g) n + r) 3. Looked up trilinearly
//! in log2 space; inputs at or below 2^lo map as 2^lo does, black as the capture's `zero`.

use std::path::Path;

use anyhow::{Context, Result, bail};

#[derive(Clone, Debug)]
pub struct ViewLut {
    pub n: usize,
    pub lo: f32,
    pub hi: f32,
    /// n³ RGB triples.
    pub data: Vec<f32>,
    /// The output for scene-linear black.
    pub zero: [f32; 3],
}

impl ViewLut {
    pub fn load(path: &Path, n: usize, lo: f32, hi: f32, zero: Option<[f32; 3]>) -> Result<ViewLut> {
        let bytes = std::fs::read(path).with_context(|| format!("cannot read the view LUT {}", path.display()))?;
        if bytes.len() != n * n * n * 12 {
            bail!("{}: {} bytes, a {n}³ RGB float LUT has {}", path.display(), bytes.len(), n * n * n * 12);
        }
        let data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let first = [data[0], data[1], data[2]];
        Ok(ViewLut { n, lo, hi, data, zero: zero.unwrap_or(first) })
    }

    /// The LUT's continuous index of a scene-linear value.
    pub fn index(&self, v: f32) -> f32 {
        let l = if v > 0.0 { v.log2() } else { self.lo };
        ((l - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0) * (self.n - 1) as f32
    }

    fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        let i = ((b * self.n + g) * self.n + r) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// Display-referred output (0..1) of a scene-linear colour.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        if rgb.iter().all(|&c| c <= 0.0) {
            return self.zero;
        }
        let x = rgb.map(|c| self.index(c));
        let i0 = x.map(|t| (t.floor() as usize).min(self.n - 2));
        let f = [x[0] - i0[0] as f32, x[1] - i0[1] as f32, x[2] - i0[2] as f32];
        let mut out = [0.0f32; 3];
        for corner in 0..8 {
            let (dr, dg, db) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let w = (if dr == 1 { f[0] } else { 1.0 - f[0] })
                * (if dg == 1 { f[1] } else { 1.0 - f[1] })
                * (if db == 1 { f[2] } else { 1.0 - f[2] });
            let c = self.at(i0[0] + dr, i0[1] + dg, i0[2] + db);
            for k in 0..3 {
                out[k] += w * c[k];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A LUT whose output is each channel's position along its axis (0..1) times (1, 2, 3).
    fn ramp(n: usize) -> ViewLut {
        let mut data = Vec::with_capacity(n * n * n * 3);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let t = |i: usize| i as f32 / (n - 1) as f32;
                    data.extend([t(r), 2.0 * t(g), 3.0 * t(b)]);
                }
            }
        }
        ViewLut { n, lo: -12.5, hi: 12.5, data, zero: [0.0; 3] }
    }

    #[test]
    fn looks_up_in_log2_space() {
        let lut = ramp(65);
        let v = |t: f32| (-12.5f32 + 25.0 * t).exp2();
        let out = lut.apply([v(0.37), v(0.5), v(0.9)]);
        assert!((out[0] - 0.37).abs() < 1e-4, "{out:?}");
        assert!((out[1] - 1.0).abs() < 1e-4, "{out:?}");
        assert!((out[2] - 2.7).abs() < 1e-4, "{out:?}");
    }

    #[test]
    fn clamps_outside_the_range() {
        let lut = ramp(9);
        assert_eq!(lut.apply([1e9, 1e9, 1e9]), [1.0, 2.0, 3.0]);
        assert_eq!(lut.apply([0.0, 0.0, 0.0]), [0.0; 3], "black is the capture's zero");
        let out = lut.apply([-1.0, 1.0, 0.0]);
        assert!(out[0].abs() < 1e-6 && (out[1] - 1.0).abs() < 1e-5 && out[2].abs() < 1e-6, "{out:?}");
    }
}
