//! PNG files as mk post reads them with Pillow (mkmmd.post.read / read_rgba): 8-bit samples (16-bit ones cut to their
//! high byte, as Pillow does), grey and palettes widened to RGB, alpha dropped from a picture read as RGB. Each pixel
//! comes out as one little-endian word r | g << 8 | b << 16 | a << 24, the layout the unpack kernels read; a figure
//! layer's luma (Kind::Luma) as one byte (`read_luma`), widened on the GPU (fx::Ctx::upload).

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// How a file is read: a picture (RGB: alpha 255), a layer (straight RGBA, cut to the size when a pixel or two larger,
/// as read_rgba does), a mask (grey: its first channel) or a luma (Pillow's convert("L"): (19595 r + 38470 g + 7471 b
/// + 0x8000) >> 16, one byte a pixel: `read_luma`; a figure's coverage as mkmmd.cutfx.read_figure reads it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Picture,
    Layer,
    Mask,
    Luma,
}

/// The file at `path` as w * h packed RGBA8 words (`kind` says how: not Luma, which `read_luma` reads), or an error
/// naming the file when it cannot be read or is another size than mk post would read without resampling. Each row is
/// packed as it is decoded (an interlaced file whole first).
pub fn read(path: &Path, w: u32, h: u32, kind: Kind) -> Result<Vec<u32>> {
    if kind == Kind::Luma {
        bail!("{}: a luma is read by read_luma", path.display());
    }
    let mut out = Vec::with_capacity(w as usize * h as usize);
    rows(path, w, h, kind, |row, channels| pack_row(row, &mut out, channels, kind))?;
    Ok(out)
}

/// The file at `path` as w * h luma bytes (Kind::Luma), or an error as `read` says.
pub fn read_luma(path: &Path, w: u32, h: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(w as usize * h as usize);
    rows(path, w, h, Kind::Luma, |row, channels| luma_row(row, &mut out, channels))?;
    Ok(out)
}

/// Each of the first h rows of the file at `path` (its first w pixels, `channels` 8-bit samples each) handed to `each`
/// as it is decoded (an interlaced file whole first), once the file is a PNG of w x h (`kind` Layer: a pixel or two
/// larger) with 8-bit samples after expanding.
fn rows(path: &Path, w: u32, h: u32, kind: Kind, mut each: impl FnMut(&[u8], usize)) -> Result<()> {
    let file = File::open(path).with_context(|| format!("{}: cannot open", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::with_capacity(1 << 20, file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().with_context(|| format!("{}: not a PNG", path.display()))?;
    let (fw, fh) = (reader.info().width, reader.info().height);
    let fits = match kind {
        Kind::Layer => fw >= w && fh >= h && fw - w <= 1 && fh - h <= 1,
        _ => fw == w && fh == h,
    };
    if !fits {
        bail!("{}: {}x{}, not the {}x{} of the output", path.display(), fw, fh, w, h);
    }
    let (color, depth) = reader.output_color_type();
    let channels = match color {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => bail!("{}: a palette left after expanding", path.display()),
    };
    if depth != png::BitDepth::Eight {
        bail!("{}: {:?} samples after expanding", path.display(), depth);
    }
    let (w, h) = (w as usize, h as usize);
    if reader.info().interlaced {
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).with_context(|| format!("{}: cannot decode", path.display()))?;
        for y in 0..h {
            each(&buf[y * info.line_size..y * info.line_size + w * channels], channels);
        }
        return Ok(());
    }
    for _ in 0..h {
        let row = reader.next_row().with_context(|| format!("{}: cannot decode", path.display()))?
            .with_context(|| format!("{}: rows missing", path.display()))?;
        each(&row.data()[..w * channels], channels);
    }
    Ok(())
}

/// Pillow's convert("L") of an 8-bit RGB pixel.
fn luma(r: u8, g: u8, b: u8) -> u32 {
    (r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 0x8000) >> 16
}

/// The first w x h pixels of decoded rows (`stride` bytes apart, `channels` 8-bit samples a pixel) as packed RGBA8
/// words read as `kind` (not Luma) says.
pub fn pack(buf: &[u8], stride: usize, w: usize, h: usize, channels: usize, kind: Kind) -> Vec<u32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        pack_row(&buf[y * stride..y * stride + w * channels], &mut out, channels, kind);
    }
    out
}

/// One row of pixels (`channels` 8-bit samples each) as packed RGBA8 words read as `kind` (not Luma) says, after
/// `out`'s; one loop per layout, so each runs as a straight copy.
fn pack_row(row: &[u8], out: &mut Vec<u32>, channels: usize, kind: Kind) {
    const OPAQUE: u32 = 0xff00_0000;
    let grey = |v: u32| v | v << 8 | v << 16;
    match (channels, kind) {
        (1, _) => out.extend(row.iter().map(|&v| grey(v as u32) | OPAQUE)),
        (2, Kind::Picture) => out.extend(row.chunks_exact(2).map(|s| grey(s[0] as u32) | OPAQUE)),
        (2, _) => out.extend(row.chunks_exact(2).map(|s| grey(s[0] as u32) | (s[1] as u32) << 24)),
        (3, _) => out.extend(row.chunks_exact(3)
            .map(|s| s[0] as u32 | (s[1] as u32) << 8 | (s[2] as u32) << 16 | OPAQUE)),
        (_, Kind::Picture) => out.extend(row.chunks_exact(4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) | OPAQUE)),
        (_, _) => out.extend(row.chunks_exact(4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))),
    }
}

/// One row of pixels (`channels` 8-bit samples each) as luma bytes (alpha dropped, grey as it is), after `out`'s.
fn luma_row(row: &[u8], out: &mut Vec<u8>, channels: usize) {
    match channels {
        1 => out.extend(row.iter().map(|&v| luma(v, v, v) as u8)),
        2 => out.extend(row.chunks_exact(2).map(|s| luma(s[0], s[0], s[0]) as u8)),
        3 => out.extend(row.chunks_exact(3).map(|s| luma(s[0], s[1], s[2]) as u8)),
        _ => out.extend(row.chunks_exact(4).map(|s| luma(s[0], s[1], s[2]) as u8)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pixel as the reading used to work it out, sample by sample.
    fn slow(s: &[u8], kind: Kind) -> u32 {
        let (r, g, b, a) = match s.len() {
            1 => (s[0], s[0], s[0], 255),
            2 => (s[0], s[0], s[0], s[1]),
            3 => (s[0], s[1], s[2], 255),
            _ => (s[0], s[1], s[2], s[3]),
        };
        let (r, g, b, a) = match kind {
            Kind::Picture => (r, g, b, 255),
            Kind::Luma => {
                let l = luma(r, g, b) as u8;
                (l, l, l, 255)
            }
            _ => (r, g, b, a),
        };
        r as u32 | (g as u32) << 8 | (b as u32) << 16 | (a as u32) << 24
    }

    /// A luma byte as the word `slow` gives for it.
    fn grey_word(l: u8) -> u32 {
        let l = l as u32;
        l | l << 8 | l << 16 | 0xff00_0000
    }

    #[test]
    fn every_layout_packs_as_each_pixel_reads() {
        let (w, h, pad) = (7usize, 5usize, 3usize);
        for channels in 1..=4 {
            let stride = (w + 1) * channels + pad;          // a row a pixel wider, as a layer may be, and padded
            let buf: Vec<u8> = (0..stride * h).map(|i| (i * 37 % 251) as u8).collect();
            for kind in [Kind::Picture, Kind::Layer, Kind::Mask, Kind::Luma] {
                let got = if kind == Kind::Luma {
                    let mut bytes = Vec::new();
                    for y in 0..h {
                        luma_row(&buf[y * stride..y * stride + w * channels], &mut bytes, channels);
                    }
                    bytes.into_iter().map(grey_word).collect()
                } else {
                    pack(&buf, stride, w, h, channels, kind)
                };
                for y in 0..h {
                    for x in 0..w {
                        let s = &buf[y * stride + x * channels..y * stride + (x + 1) * channels];
                        assert_eq!(got[y * w + x], slow(s, kind), "{channels} channels, {kind:?}, ({x}, {y})");
                    }
                }
            }
        }
    }

    #[test]
    fn every_layout_reads_from_a_file_as_each_pixel_reads() {
        let dir = std::env::temp_dir().join(format!("mkr_post_pngio_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (7u32, 5u32);
        let colours = [png::ColorType::Grayscale, png::ColorType::GrayscaleAlpha, png::ColorType::Rgb,
                       png::ColorType::Rgba];
        for (i, colour) in colours.into_iter().enumerate() {
            let channels = i + 1;
            for kind in [Kind::Picture, Kind::Layer, Kind::Mask, Kind::Luma] {
                let (fw, fh) = if kind == Kind::Layer { (w + 1, h + 1) } else { (w, h) };   // a layer a pixel larger
                let px: Vec<u8> = (0..(fw * fh) as usize * channels).map(|i| (i * 37 % 251) as u8).collect();
                let path = dir.join(format!("{channels}-{kind:?}.png"));
                let mut enc = png::Encoder::new(File::create(&path).unwrap(), fw, fh);
                enc.set_color(colour);
                enc.set_depth(png::BitDepth::Eight);
                enc.set_filter(png::FilterType::Paeth);
                enc.write_header().unwrap().write_image_data(&px).unwrap();
                let got = if kind == Kind::Luma {
                    read_luma(&path, w, h).unwrap().into_iter().map(grey_word).collect()
                } else {
                    read(&path, w, h, kind).unwrap()
                };
                let stride = fw as usize * channels;
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let s = &px[y * stride + x * channels..y * stride + (x + 1) * channels];
                        assert_eq!(got[y * w as usize + x], slow(s, kind), "{channels} channels, {kind:?}, ({x}, {y})");
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
