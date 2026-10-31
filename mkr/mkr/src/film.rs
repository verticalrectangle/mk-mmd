//! The film: N jittered samples of the picture, each spread over the pixels around it with a filter of `filter_size`
//! pixels (EEVEE: a Gaussian fitted to Blackman-Harris), summed with their weights and divided by them; and the
//! per-sample random numbers (low-discrepancy over the samples, rotated per pixel on the GPU), the point of the aperture
//! each sample looks from among them.

use glam::Vec2;

use crate::camera::sunflower;

/// Half the side of the square of pixels a sample is spread over: a 5 x 5 neighbourhood.
pub const TAPS: i32 = 2;
pub const TAP_COUNT: usize = ((2 * TAPS + 1) * (2 * TAPS + 1)) as usize;

#[derive(Clone, Copy, Debug)]
pub struct Filter {
    /// Pixels: the filter is zero beyond it.
    pub radius: f32,
    /// The Gaussian's sigma as a fraction of the radius.
    pub sigma: f32,
}

impl Filter {
    /// The weight of a sample `dist2` square pixels from the pixel centre.
    pub fn weight(&self, dist2: f32) -> f32 {
        if self.radius < 0.01 {
            // no filter: each sample counts for the pixel it falls in
            return if dist2 <= 0.25 { 1.0 } else { 0.0 };
        }
        if dist2 >= self.radius * self.radius {
            return 0.0;
        }
        let s = self.sigma * self.radius;
        (-0.5 * dist2 / (s * s)).exp()
    }

    /// The weights of a sample jittered by `j` (pixels, y down) for the pixels around the one it was drawn on: entry
    /// (dy + TAPS) * (2 TAPS + 1) + dx + TAPS is the weight it gets in the pixel (dx, dy) away from it... seen from that
    /// pixel: the film pixel p takes the sample drawn on p + d with weight `weight(|d + j|²)`.
    pub fn taps(&self, j: Vec2) -> [f32; TAP_COUNT] {
        let mut w = [0.0; TAP_COUNT];
        let side = 2 * TAPS + 1;
        for dy in -TAPS..=TAPS {
            for dx in -TAPS..=TAPS {
                let d = Vec2::new(dx as f32, dy as f32) + j;
                w[((dy + TAPS) * side + dx + TAPS) as usize] = self.weight(d.length_squared());
            }
        }
        w
    }
}

/// The radical inverse of `i` in `base` (Halton's sequence): in [0, 1).
pub fn halton(mut i: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f64, 0.0f64);
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r as f32
}

/// One sample of a frame: where it looks inside the pixel, when inside the shutter, which point of the aperture it looks
/// from with depth of field (its index among the sunflower's points, camera.rs `Dof::aperture`), and its taps.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Pixels, in [-scale/2, scale/2)².
    pub jitter: Vec2,
    /// Frames from the frame's own time (0 without motion blur).
    pub time: f32,
    pub lens: u32,
    pub taps: [f32; TAP_COUNT],
}

/// The `n` samples of a frame: Halton (2, 3) jitter over `jitter_scale` pixels, Halton 5 time over the shutter
/// (frames, centred on the frame), the aperture's points in `lens_order`, the filter's taps for each.
pub fn samples(n: u32, filter: &Filter, jitter_scale: f32, shutter: Option<f32>) -> Vec<Sample> {
    let lens = lens_order(n);
    (0..n)
        .map(|s| {
            let jitter = if n == 1 {
                Vec2::ZERO
            } else {
                (Vec2::new(halton(s + 1, 2), halton(s + 1, 3)) - 0.5) * jitter_scale
            };
            let time = match shutter {
                Some(t) if n > 1 => (halton(s + 1, 5) - 0.5) * t,
                _ => 0.0,
            };
            Sample { jitter, time, lens: lens[s as usize], taps: filter.taps(jitter) }
        })
        .collect()
}

/// Which of the aperture's `n` sunflower points each of `n` samples looks from: of 64 shuffles (hashes of the point's
/// index and the shuffle's), the one whose points' x, y and squared radius are least correlated with the samples'
/// jitter and shutter time (Halton 2, 3 and 5), the offsets an aperture point adds its own to, so that a point's blur,
/// the film's filter and the shutter spread independently (in order, the sunflower's radius grows with the sample and
/// locks onto those sequences at a few dozen samples).
pub fn lens_order(n: u32) -> Vec<u32> {
    let hash = |x: u32| {
        let x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
        let x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    };
    let others: Vec<[f64; 3]> = (0..n)
        .map(|s| [halton(s + 1, 2) as f64, halton(s + 1, 3) as f64, halton(s + 1, 5) as f64])
        .collect();
    let points: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let p = sunflower(i, n);
            [p.x as f64, p.y as f64, p.length_squared() as f64]
        })
        .collect();
    let corr = |a: &dyn Fn(usize) -> f64, b: &dyn Fn(usize) -> f64| {
        let m = n as f64;
        let (ma, mb) = ((0..n as usize).map(a).sum::<f64>() / m, (0..n as usize).map(b).sum::<f64>() / m);
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for i in 0..n as usize {
            let (x, y) = (a(i) - ma, b(i) - mb);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        if aa > 0.0 && bb > 0.0 { (ab / (aa * bb).sqrt()).abs() } else { 0.0 }
    };
    let mut best: Option<(f64, Vec<u32>)> = None;
    for shuffle in 0..64u32 {
        let mut order: Vec<u32> = (0..n).collect();
        order.sort_by_key(|&i| (hash(i.wrapping_mul(0x9e37_79b9).wrapping_add(shuffle)), i));
        let worst = (0..3)
            .flat_map(|d| (0..3).map(move |e| (d, e)))
            .map(|(d, e)| corr(&|s| points[order[s] as usize][d], &|s| others[s][e]))
            .fold(0.0, f64::max);
        if best.as_ref().is_none_or(|(w, _)| worst < *w) {
            best = Some((worst, order));
        }
    }
    best.map(|(_, o)| o).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EEVEE: Filter = Filter { radius: 1.5, sigma: 0.284 };

    #[test]
    fn weight_is_a_truncated_gaussian() {
        assert!((EEVEE.weight(0.0) - 1.0).abs() < 1e-6);
        let s: f32 = 0.284 * 1.5;
        assert!((EEVEE.weight(1.0) - (-0.5 / (s * s)).exp()).abs() < 1e-6);
        assert_eq!(EEVEE.weight(1.5 * 1.5), 0.0);
        assert!(EEVEE.weight(0.25) > EEVEE.weight(0.5));
    }

    #[test]
    fn taps_follow_the_jitter() {
        let w = EEVEE.taps(Vec2::ZERO);
        let c = (TAPS * (2 * TAPS + 1) + TAPS) as usize;
        assert!((w[c] - 1.0).abs() < 1e-6, "the pixel's own sample counts fully");
        assert!((w[c + 1] - w[c - 1]).abs() < 1e-6, "symmetric without jitter");
        assert_eq!(w[0], 0.0, "the corners are beyond 1.5 pixels");
        // a sample jittered right sits closer to the pixel on its left... seen from the film pixel p, the sample drawn on
        // p - 1 (dx = -1) lies at -1 + j from p's centre
        let w = EEVEE.taps(Vec2::new(0.4, 0.0));
        assert!(w[c - 1] > w[c + 1]);
        assert!((w[c - 1] - EEVEE.weight(0.36)).abs() < 1e-6);
    }

    #[test]
    fn no_filter_keeps_samples_in_their_pixel() {
        let f = Filter { radius: 0.0, sigma: 0.284 };
        let w = f.taps(Vec2::new(0.3, -0.2));
        let c = (TAPS * (2 * TAPS + 1) + TAPS) as usize;
        assert_eq!(w.iter().filter(|&&x| x > 0.0).count(), 1);
        assert_eq!(w[c], 1.0);
    }

    #[test]
    fn halton_is_the_radical_inverse() {
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(2, 2), 0.25);
        assert_eq!(halton(3, 2), 0.75);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-7);
        assert!((halton(5, 3) - (2.0 / 3.0 + 1.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn samples_cover_the_pixel_and_the_shutter() {
        let s = samples(64, &EEVEE, 1.0, Some(0.35));
        assert_eq!(s.len(), 64);
        for x in &s {
            assert!(x.jitter.x >= -0.5 && x.jitter.x < 0.5 && x.jitter.y >= -0.5 && x.jitter.y < 0.5);
            assert!(x.time.abs() <= 0.175);
        }
        let mean = s.iter().map(|x| x.jitter).sum::<Vec2>() / 64.0;
        assert!(mean.length() < 0.02, "{mean}");
        let one = samples(1, &EEVEE, 1.0, Some(0.35));
        assert_eq!(one[0].jitter, Vec2::ZERO);
        assert_eq!(one[0].time, 0.0);
    }

    /// Every aperture point is some sample's, once; their x, y and squared radius hardly follow the samples' jitter or
    /// shutter time (in their own order the radius grows with the sample, at 16 samples as Halton's base 17 does).
    #[test]
    fn the_lens_order_stands_apart_from_the_jitter_and_the_time() {
        let corr = |a: &[f64], b: &[f64]| {
            let n = a.len() as f64;
            let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
            let ab: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
            let aa: f64 = a.iter().map(|x| (x - ma) * (x - ma)).sum();
            let bb: f64 = b.iter().map(|y| (y - mb) * (y - mb)).sum();
            (ab / (aa * bb).sqrt()).abs()
        };
        for (n, most) in [(1u32, 0.0), (16, 0.3), (32, 0.2), (64, 0.12), (256, 0.06)] {
            let s = samples(n, &EEVEE, 1.0, Some(0.35));
            let mut seen: Vec<u32> = s.iter().map(|x| x.lens).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>());
            if n == 1 {
                continue;
            }
            let pts: Vec<Vec2> = s.iter().map(|x| sunflower(x.lens, n)).collect();
            let lens: [Vec<f64>; 3] = [pts.iter().map(|p| p.x as f64).collect(), pts.iter().map(|p| p.y as f64).collect(),
                                       pts.iter().map(|p| p.length_squared() as f64).collect()];
            let other: [Vec<f64>; 3] = [s.iter().map(|x| x.jitter.x as f64).collect(),
                                        s.iter().map(|x| x.jitter.y as f64).collect(),
                                        s.iter().map(|x| x.time as f64).collect()];
            for l in &lens {
                for o in &other {
                    assert!(corr(l, o) < most, "{n} samples: correlation {}", corr(l, o));
                }
            }
        }
    }
}
