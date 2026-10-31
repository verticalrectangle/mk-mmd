//! Blender 4.2's 8-bit dither, as measured on its PNGs (lab pairs: an 8-bit frame and the same render saved as 16-bit
//! PNG, which carries no dither). One value per pixel, shared by the three channels, the same on every frame and scene
//! of a size.
//!
//! Blender turns the display-referred float picture into bytes in chunks of 64 rows counted from its bottom row (the
//! top chunk keeps what remains: 28 rows at 540, 56 at 1080). For the pixel in column `x` and chunk row `y` (0 = the
//! chunk's bottom row) of a chunk of `h` rows, in a picture `width` wide, everything in single precision (separate
//! multiplies and adds, no fused multiply-add; the C library's `sinf`):
//!
//! ```text
//! s = x * (1 / width)          t = y * (1 / h)
//! u0 = fract(sin(12.9898 s +  78.233 t) * 43758.5453)
//! u1 = fract(sin(19.9898 s + 119.233 t) * 43798.5453)
//! dither = (u0 + u1 - 0.5) * 0.0033 * dither_intensity
//! byte = 0 if v + dither <= 0, 255 if v + dither > 1 - 0.5 / 255, else floor(255 (v + dither) + 0.5)
//! ```
//!
//! with `v` the display value (0..1): a triangular distribution over [-0.5, 1.5) x 0.0033 (mean 0.42 of a level) added
//! before rounding half up (`resolve.wgsl`'s `to8`). Over every lab pair (51 frames at 960x540, 4 at 1920x1080, 4 at
//! 1080x1920; 129 M channel values) every 8-bit value whose 16-bit twin lies more than 2/65535 from a rounding
//! boundary comes out exactly (0 misses); counting every value, 99.85 % / 99.91 % / 99.95 % (one 16-bit step hides
//! where the float value lay against the boundary). The misses within 2/65535 are AgX frames only: there the 8-bit
//! save's display values sit about 2.5e-5 (relative) below the 16-bit save's; with Standard they agree exactly.
//! Variants that measurably lose: `x / width` or `y / h` instead of the reciprocal products, a fused multiply-add in
//! the sine's argument, chunks counted from the top, one hash used twice.

/// Rows per conversion chunk, counted from the picture's bottom row.
const CHUNK: u32 = 64;

/// Dither of each pixel (row 0 = the picture's top) in display units (0..1 scale), per unit of `dither_intensity`.
pub fn pattern(width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let mut out = vec![0.0f32; w * height as usize];
    let inv_w = 1.0f32 / width as f32;
    let s: Vec<f32> = (0..width).map(|x| x as f32 * inv_w).collect();
    let mut c0 = 0u32; // the chunk's first row, counted from the bottom
    while c0 < height {
        let h = (height - c0).min(CHUNK);
        let inv_h = 1.0f32 / h as f32;
        for y in 0..h {
            let t = y as f32 * inv_h;
            let row = (height - 1 - (c0 + y)) as usize;
            for (o, &s) in out[row * w..(row + 1) * w].iter_mut().zip(&s) {
                *o = value(s, t);
            }
        }
        c0 += h;
    }
    out
}

/// The dither at (s, t): two sine hashes summed into a triangular distribution, scaled to display units.
fn value(s: f32, t: f32) -> f32 {
    let u0 = hash(s, t, 12.9898, 78.233, 43758.5453);
    let u1 = hash(s, t, 19.9898, 119.233, 43798.5453);
    (u0 + u1 - 0.5) * 0.0033
}

/// fract(sin(a s + b t) k) in single precision.
fn hash(s: f32, t: f32, a: f32, b: f32, k: f32) -> f32 {
    let v = (s * a + t * b).sin() * k;
    v - v.floor()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// `resolve.wgsl`'s `to8` without the tunable bias: display value plus dither, 8 bits.
    fn to8(v: f32, d: f32) -> u8 {
        (255.0 * (v + d) + 0.5).floor().clamp(0.0, 255.0) as u8
    }

    /// Lab pixels (Blender 4.2, EEVEE, dither 1): width, height, x, y (0 = top), the 16-bit PNG's RGB and the 8-bit
    /// PNG's RGB of the same render. Each 16-bit value lies more than 2/65535 from a rounding boundary, so the byte
    /// follows from it exactly; together they cover the bottom and the short top chunk of each size.
    #[rustfmt::skip]
    const LAB: [(u32, u32, u32, u32, [u16; 3], [u8; 3]); 14] = [
        (960, 540, 404, 15, [30787, 30787, 30787], [119, 119, 119]),
        (960, 540, 395, 21, [29144, 29144, 29144], [115, 115, 115]),
        (960, 540, 31, 56, [1244, 1244, 1244], [6, 6, 6]),
        (960, 540, 414, 471, [32608, 661, 32503], [127, 3, 127]),
        (960, 540, 371, 532, [22603, 17019, 13813], [89, 67, 54]),
        (960, 540, 499, 539, [48798, 37633, 31232], [190, 147, 122]),
        (1920, 1080, 1292, 49, [53813, 53152, 52473], [210, 207, 204]),
        (1920, 1080, 1271, 30, [54737, 54072, 53419], [213, 211, 208]),
        (1920, 1080, 672, 455, [54694, 52726, 56312], [214, 206, 220]),
        (1920, 1080, 892, 974, [50874, 50079, 49288], [199, 195, 192]),
        (1920, 1080, 757, 1055, [56103, 54348, 57651], [218, 212, 224]),
        (1080, 1920, 635, 571, [8657, 5741, 11636], [34, 23, 45]),
        (1080, 1920, 759, 1058, [27219, 21948, 32683], [106, 86, 128]),
        (1080, 1920, 576, 1592, [52407, 51324, 52130], [205, 200, 203]),
    ];

    #[test]
    fn reproduces_blender_bytes() {
        let mut patterns = HashMap::new();
        for &(w, h, x, y, v16, v8) in &LAB {
            let p = patterns.entry((w, h)).or_insert_with(|| pattern(w, h));
            let d = p[(y * w + x) as usize];
            assert_eq!(v16.map(|v| to8(v as f32 / 65535.0, d)), v8, "{w}x{h} pixel ({x}, {y})");
            // the dither decides these: plain rounding misses at least one channel
            assert_ne!(v16.map(|v| to8(v as f32 / 65535.0, 0.0)), v8, "{w}x{h} pixel ({x}, {y})");
        }
    }

    #[test]
    fn chunks_from_the_bottom() {
        let (w, h) = (960u32, 540u32);
        let p = pattern(w, h);
        let row = |r: u32| &p[(r * w) as usize..((r + 1) * w) as usize];
        // the bottom row starts the first chunk, row 27 the short top one (540 = 8 x 64 + 28): both have t = 0
        assert_eq!(row(539), row(27));
        // full chunks repeat every 64 rows
        assert_eq!(row(100), row(164));
        // the top chunk spreads t over its own 28 rows: its row 7 (t = 7 / 28) is a full chunk's row 16 (t = 1 / 4)
        assert_eq!(row(27 - 7), row(539 - 16));
        assert_ne!(row(0), row(64));
        // with t = 0 the value depends on s = x / width alone
        let x = 17u32;
        assert_eq!(row(539)[x as usize], value(x as f32 * (1.0 / 960.0), 0.0));
    }
}
