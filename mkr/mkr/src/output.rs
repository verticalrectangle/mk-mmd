//! Frames and layers to disk: 8-bit PNGs (RGB as Blender saves frames; grey, RGBA for layers), encoded on a thread
//! pool while the GPU draws the next frame.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};

/// Encode `rgb` (w x h x 3, row 0 the top) as `path`, through a temporary file so a reader never sees half a frame.
pub fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) -> Result<()> {
    write_image(path, w, h, 3, rgb)
}

/// Encode `data` (w x h x `channels`: 1 grey, 2 grey and alpha, 3 RGB, 4 RGBA (straight); row 0 the top) as an 8-bit
/// PNG `path`, through a temporary file; its folder is made.
pub fn write_image(path: &Path, w: u32, h: u32, channels: u32, data: &[u8]) -> Result<()> {
    let color = match channels {
        1 => png::ColorType::Grayscale,
        2 => png::ColorType::GrayscaleAlpha,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        n => anyhow::bail!("{}: {n} channels", path.display()),
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot make {}", dir.display()))?;
    }
    let tmp = path.with_extension("png.part");
    {
        let file = File::create(&tmp).with_context(|| format!("cannot write {}", tmp.display()))?;
        let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Fast);
        enc.set_filter(png::FilterType::Sub);
        let mut wr = enc.write_header()?;
        wr.write_image_data(data)?;
        wr.finish()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// A float32 array of shape `shape` (C order) as a NumPy `.npy` file (format 1.0): the scene-linear film `MKR_LINEAR`
/// dumps beside each frame for parity work.
pub fn write_npy(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let dims = if shape.len() == 1 { format!("{},", dims[0]) } else { dims.join(", ") };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
    // magic and version (8), header length (2), the header padded with spaces to a multiple of 64, then a newline
    header.push_str(&" ".repeat((64 - (10 + header.len() + 1) % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(bytemuck::cast_slice(data));
    std::fs::write(path, out).with_context(|| format!("cannot write {}", path.display()))
}

/// A picture for the writers as the GPU hands it over; the writer's thread lays it out as the file's channels.
pub enum Pixels {
    /// Rows of w x `channels` bytes: the file's own layout.
    Packed { channels: u32, data: Vec<u8> },
    /// RGBA8 per pixel, written as RGB (a frame: its alpha dropped).
    Rgbx(Vec<u8>),
    /// A grey byte per pixel, rows `stride` bytes apart, written as `channels` channels: 3 the grey as RGB, 4 the grey
    /// as RGB under an alpha of 255.
    Grey { stride: usize, channels: u32, data: Vec<u8> },
}

impl Pixels {
    /// Channels of the file.
    pub fn channels(&self) -> u32 {
        match self {
            Pixels::Packed { channels, .. } | Pixels::Grey { channels, .. } => *channels,
            Pixels::Rgbx(_) => 3,
        }
    }

    /// The file's rows: w x h x channels bytes, row 0 the top.
    pub fn rows(self, w: u32, h: u32) -> Result<Vec<u8>> {
        let (w, h) = (w as usize, h as usize);
        let c = self.channels() as usize;
        match self {
            Pixels::Packed { data, .. } => {
                anyhow::ensure!(data.len() == w * h * c, "{} bytes for {w}x{h}x{c}", data.len());
                Ok(data)
            }
            Pixels::Rgbx(data) => {
                anyhow::ensure!(data.len() >= w * h * 4, "{} bytes for {w}x{h} RGBA", data.len());
                let mut out = vec![0u8; w * h * 3];
                for (dst, px) in out.chunks_exact_mut(3).zip(data.chunks_exact(4)) {
                    dst.copy_from_slice(&px[..3]);
                }
                Ok(out)
            }
            Pixels::Grey { stride, data, .. } => {
                anyhow::ensure!(stride >= w && data.len() >= stride * h, "{} bytes for {h} rows of {stride}", data.len());
                let mut out = vec![255u8; w * h * c];
                for (dst, src) in out.chunks_exact_mut(w * c).zip(data.chunks(stride)) {
                    for (px, &v) in dst.chunks_exact_mut(c).zip(&src[..w]) {
                        px[..3].fill(v);
                    }
                }
                Ok(out)
            }
        }
    }
}

/// A pool that writes frames in the background; `finish` waits for all of them and returns the first error.
pub struct Writer {
    pool: rayon::ThreadPool,
    tx: mpsc::Sender<Result<()>>,
    rx: mpsc::Receiver<Result<()>>,
    pending: usize,
}

impl Writer {
    pub fn new(threads: usize) -> Result<Writer> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("mkr-png-{i}"))
            .build()?;
        let (tx, rx) = mpsc::channel();
        Ok(Writer { pool, tx, rx, pending: 0 })
    }

    /// Queue a picture for `path` (laid out as its file's channels and encoded on the pool, write_image); blocks
    /// while more than `limit` pictures wait (bounding the memory they hold).
    pub fn push(&mut self, path: PathBuf, w: u32, h: u32, pixels: Pixels, limit: usize) -> Result<()> {
        self.push_after(None, path, w, h, pixels, limit)
    }

    /// `push`, after `first` (a picture of the same size for another file, if any): both written on one thread, `first`
    /// whole on disk before `path` is begun, and `path` not at all when `first` fails, so that `path`'s presence says
    /// `first` is there (a frame's negative layer and its screen layer, layers.rs).
    pub fn push_after(&mut self, first: Option<(PathBuf, Pixels)>, path: PathBuf, w: u32, h: u32, pixels: Pixels,
                      limit: usize) -> Result<()> {
        while self.pending >= limit {
            self.rx.recv().context("the PNG writers stopped")??;
            self.pending -= 1;
        }
        let tx = self.tx.clone();
        self.pending += 1;
        self.pool.spawn(move || {
            let write = |path: &Path, pixels: Pixels| {
                let channels = pixels.channels();
                pixels.rows(w, h).and_then(|rows| write_image(path, w, h, channels, &rows))
            };
            let first = first.map_or(Ok(()), |(p, px)| write(&p, px));
            let _ = tx.send(first.and_then(|()| write(&path, pixels)));
        });
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        let mut first = Ok(());
        while self.pending > 0 {
            let r = self.rx.recv().context("the PNG writers stopped")?;
            self.pending -= 1;
            if first.is_ok() {
                first = r;
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_rgb_pngs() {
        let dir = std::env::temp_dir().join(format!("mkr-png-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = Writer::new(2).unwrap();
        let rgb: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7) as u8).collect();
        w.push(dir.join("00001.png"), 4, 3, Pixels::Packed { channels: 3, data: rgb.clone() }, 4).unwrap();
        // the same frame as the GPU reads it back: RGBA words, the alpha dropped
        let rgbx: Vec<u8> = rgb.chunks(3).flat_map(|p| [p[0], p[1], p[2], 99]).collect();
        w.push(dir.join("00002.png"), 4, 3, Pixels::Rgbx(rgbx), 4).unwrap();
        w.finish().unwrap();
        for f in ["00001.png", "00002.png"] {
            let img = image::open(dir.join(f)).unwrap().to_rgb8();
            assert_eq!(img.dimensions(), (4, 3));
            assert_eq!(img.as_raw(), &rgb);
            assert!(!dir.join(format!("{f}.part")).exists());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn grey_rows_widen_to_the_files_channels() {
        // 3 x 2 grey with rows padded to 4 bytes
        let data = vec![1, 2, 3, 0, 4, 5, 6, 0];
        let rgb = Pixels::Grey { stride: 4, channels: 3, data: data.clone() }.rows(3, 2).unwrap();
        assert_eq!(rgb, vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
        let rgba = Pixels::Grey { stride: 4, channels: 4, data }.rows(3, 2).unwrap();
        assert_eq!(&rgba[..8], &[1, 1, 1, 255, 2, 2, 2, 255]);
        assert_eq!(rgba.len(), 3 * 2 * 4);
        assert!(Pixels::Packed { channels: 4, data: vec![0; 5] }.rows(1, 1).is_err());
    }

    #[test]
    fn a_picture_pushed_after_another_is_written_after_it_or_not_at_all() {
        let dir = std::env::temp_dir().join(format!("mkr-png-after-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rgba = |v: u8| Pixels::Packed { channels: 4, data: vec![v; 2 * 2 * 4] };
        let mut w = Writer::new(4).unwrap();
        w.push_after(Some((dir.join("negative/00001.png"), rgba(1))), dir.join("screen/00001.png"), 2, 2, rgba(2), 4)
            .unwrap();
        w.finish().unwrap();
        for (f, v) in [("negative/00001.png", 1), ("screen/00001.png", 2)] {
            let img = image::open(dir.join(f)).unwrap().to_rgba8();
            assert!(img.as_raw().iter().all(|&x| x == v), "{f}");
        }
        // a first picture that cannot be written keeps the second off the disk
        let mut w = Writer::new(4).unwrap();
        let bad = Pixels::Packed { channels: 4, data: vec![0; 5] };
        let screen = dir.join("screen/00002.png");
        w.push_after(Some((dir.join("negative/00002.png"), bad)), screen.clone(), 2, 2, rgba(2), 4).unwrap();
        assert!(w.finish().is_err());
        assert!(!screen.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
