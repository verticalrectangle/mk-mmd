//! The scene's images as Blender's GPU holds them (bake v5 `lit.images`): 8-bit sRGB images in sRGB textures (linear
//! when sampled, filtered in linear), 8-bit data images as they are, 16-bit and float images in half floats after their
//! colour space's conversion to scene linear; alpha as the file has it (Blender stores byte images straight unless their
//! alpha mode says premultiplied); mipmaps averaged in linear light. A missing image is magenta, as Blender draws it.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::bake::ImageRec;

/// One image ready to upload: its texture format and the bytes of each mip level.
pub struct Pixels {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub levels: Vec<Vec<u8>>,
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Blender draws an image it cannot load as magenta.
pub fn magenta() -> Pixels {
    Pixels { width: 1, height: 1, format: wgpu::TextureFormat::Rgba8Unorm, levels: vec![vec![255, 0, 255, 255]] }
}

/// How a colour space turns into scene-linear Rec.709, when mkr knows it.
fn linear_fn(colorspace: &str) -> Option<fn(f32) -> f32> {
    match colorspace {
        "sRGB" => Some(srgb_to_linear),
        "Non-Color" | "Linear Rec.709" | "Linear" | "Raw" | "Linear CIE-XYZ D65" => Some(|c| c),
        _ => None,
    }
}

/// Whether mkr can show an image in this colour space.
pub fn supported_colorspace(colorspace: &str) -> bool {
    linear_fn(colorspace).is_some()
}

/// Load an image of the bake (`file` absolute or relative to the bake). The format comes from the file's content, as
/// Blender reads it (MMD models name BMPs `.bm`, `.spa`, `.sph` and PNGs `.pn`); a file the bake says Blender loaded and
/// mkr cannot decode is an error.
pub fn load(bake_dir: &Path, rec: &ImageRec) -> Result<Pixels> {
    if rec.missing {
        return Ok(magenta());
    }
    let path = if Path::new(&rec.file).is_absolute() { rec.file.clone().into() } else { bake_dir.join(&rec.file) };
    let img = image::ImageReader::open(&path)
        .and_then(|r| r.with_guessed_format())
        .with_context(|| format!("cannot read image {}", path.display()))?
        .decode()
        .with_context(|| format!("image {}: Blender reads it, mkr cannot decode it", path.display()))?;
    let lin = linear_fn(&rec.colorspace)
        .with_context(|| format!("image {}: colour space {} is not one mkr converts", path.display(), rec.colorspace))?;
    let (w, h) = (img.width(), img.height());
    let eight_bit = matches!(
        img.color(),
        image::ColorType::L8 | image::ColorType::La8 | image::ColorType::Rgb8 | image::ColorType::Rgba8
    );
    if eight_bit {
        let rgba = img.to_rgba8().into_raw();
        let srgb = rec.colorspace == "sRGB";
        let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        if !srgb && rec.colorspace != "Non-Color" && lin(0.5) != 0.5 {
            bail!("image {}: 8-bit {} is not handled", path.display(), rec.colorspace);
        }
        let levels = mips_u8(&rgba, w, h, srgb);
        return Ok(Pixels { width: w, height: h, format, levels });
    }
    // 16-bit and float: to scene linear (alpha kept), half floats
    let rgba = img.to_rgba32f().into_raw();
    let float_file = matches!(img.color(), image::ColorType::Rgb32F | image::ColorType::Rgba32F);
    let mut lin_px: Vec<f32> = Vec::with_capacity(rgba.len());
    for px in rgba.chunks_exact(4) {
        // float files hold scene-linear values already in their space; 16-bit integer files are encoded like 8-bit ones
        let f = |c: f32| if float_file && rec.colorspace == "sRGB" { c } else { lin(c) };
        lin_px.extend([f(px[0]), f(px[1]), f(px[2]), px[3]]);
    }
    let levels = mips_f32(&lin_px, w, h).into_iter().map(|l| to_f16_bytes(&l)).collect();
    Ok(Pixels { width: w, height: h, format: wgpu::TextureFormat::Rgba16Float, levels })
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend(f32_to_f16(x).to_le_bytes());
    }
    out
}

/// IEEE half from single (round to nearest even), saturating to infinity.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal half h * 2^-24 = m * 2^(exp - 150): h = m >> (14 - e)
        let shift = (14 - e) as u32;
        let m = man | 0x80_0000;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let up = rem > halfway || (rem == halfway && (half & 1) == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    sign | (half + up as u32) as u16
}

/// The mip chain of an 8-bit RGBA image: 2x2 box averages, in linear light for sRGB images (alpha linear).
fn mips_u8(rgba: &[u8], w: u32, h: u32, srgb: bool) -> Vec<Vec<u8>> {
    let mut levels = vec![rgba.to_vec()];
    let to_lin: Vec<f32> = (0..256).map(|i| if srgb { srgb_to_linear(i as f32 / 255.0) } else { i as f32 / 255.0 }).collect();
    let (mut cw, mut ch) = (w, h);
    while cw > 1 || ch > 1 {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let src = levels.last().unwrap();
        let mut dst = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                let mut n = 0.0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (2 * x + dx).min(cw - 1);
                    let sy = (2 * y + dy).min(ch - 1);
                    let i = ((sy * cw + sx) * 4) as usize;
                    for c in 0..3 {
                        acc[c] += to_lin[src[i + c] as usize];
                    }
                    acc[3] += src[i + 3] as f32 / 255.0;
                    n += 1.0;
                }
                let o = ((y * nw + x) * 4) as usize;
                for c in 0..3 {
                    let v = acc[c] / n;
                    let e = if srgb { linear_to_srgb(v) } else { v };
                    dst[o + c] = (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
                dst[o + 3] = (acc[3] / n * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        levels.push(dst);
        cw = nw;
        ch = nh;
    }
    levels
}

fn mips_f32(rgba: &[f32], w: u32, h: u32) -> Vec<Vec<f32>> {
    let mut levels = vec![rgba.to_vec()];
    let (mut cw, mut ch) = (w, h);
    while cw > 1 || ch > 1 {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let src = levels.last().unwrap();
        let mut dst = vec![0.0f32; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let o = ((y * nw + x) * 4) as usize;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (2 * x + dx).min(cw - 1);
                    let sy = (2 * y + dy).min(ch - 1);
                    let i = ((sy * cw + sx) * 4) as usize;
                    for c in 0..4 {
                        dst[o + c] += src[i + c] * 0.25;
                    }
                }
            }
        }
        levels.push(dst);
        cw = nw;
        ch = nh;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        // 2^-24, the smallest subnormal
        assert_eq!(f32_to_f16(5.960_464_5e-8), 1);
    }

    #[test]
    fn srgb_round_trip() {
        for i in 0..=255 {
            let c = i as f32 / 255.0;
            assert!((linear_to_srgb(srgb_to_linear(c)) - c).abs() < 1e-5);
        }
    }

    #[test]
    fn mips_average_in_linear_light() {
        // black and white 2x1 sRGB: the 1x1 mip is linear 0.5, sRGB 188
        let rgba = [0, 0, 0, 255, 255, 255, 255, 255];
        let levels = mips_u8(&rgba, 2, 1, true);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[1][0], 188);
        assert_eq!(levels[1][3], 255);
    }

    #[test]
    fn formats_come_from_the_content_not_the_name() {
        // MMD models name PNGs `.pn` and BMPs `.bm`: Blender reads them by content, so must mkr
        let dir = std::env::temp_dir().join(format!("mkr-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let img = image::RgbaImage::from_raw(2, 1, vec![10, 20, 30, 255, 200, 100, 50, 255]).unwrap();
        img.save_with_format(dir.join("a.pn"), image::ImageFormat::Png).unwrap();
        img.save_with_format(dir.join("b.bm"), image::ImageFormat::Bmp).unwrap();
        for file in ["a.pn", "b.bm"] {
            let rec = ImageRec { file: file.into(), colorspace: "sRGB".into(), alpha_mode: "STRAIGHT".into(), size: [2, 1],
                                 missing: false };
            let px = load(&dir, &rec).unwrap();
            assert_eq!((px.width, px.height), (2, 1), "{file}");
            assert_eq!(&px.levels[0][..4], &[10, 20, 30, 255], "{file}");
        }
        let rec = ImageRec { file: "none.png".into(), colorspace: "sRGB".into(), alpha_mode: String::new(), size: [0, 0],
                             missing: false };
        assert!(load(&dir, &rec).is_err(), "a file Blender loaded and mkr cannot read is an error");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
