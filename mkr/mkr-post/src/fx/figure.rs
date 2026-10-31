//! The sticker work the collages and the masked effects share (shaders/fx_figure.wgsl), each the OpenCV / numpy code of
//! mkmmd.framefx and mkmmd.mandala it stands for: OpenCV's Gaussian blur, a coverage grown into a die-cut ring
//! (`_grow`), a plate printed off register (`plate_shift`), a drop shadow (`_cast`), an affine warp shrunk by area first
//! (`_warp`, `_laid`), INTER_AREA and INTER_LINEAR resizes, and a figure cut out and pasted. Planes are buffers of
//! w x h vec4<f32> (a coverage in .x); a pasted figure's coverage, ring and shadow are single-float planes over the
//! box it reaches (fx/plane.rs).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::plane::{self, Area};
use super::warp::{warp, warp_box};
use super::{Ctx, Decoded, Input};
use crate::gpu::Dispatch;
use crate::pngio::Kind;

const FIGURE: &str = include_str!("../shaders/fx_figure.wgsl");
const UNPACK: &str = include_str!("../shaders/unpack.wgsl");
/// fx_figure.wgsl's blur_h: a run of a row and the reach on either side, in a workgroup's memory.
const BLUR_RUN: u32 = 256;
const BLUR_TILE: u32 = 768;

/// cv2.BORDER_CONSTANT and cv2.BORDER_REPLICATE.
pub const BORDER_CONSTANT: u32 = 0;
pub const BORDER_REPLICATE: u32 = 1;

/// Bytes of a w x h plane.
pub fn plane_bytes(w: u32, h: u32) -> u64 {
    w as u64 * h as u64 * 16
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct Params {
    pub w: u32,
    pub h: u32,
    pub sw: u32,
    pub sh: u32,
    pub n: u32,
    pub flag: u32,
    pub x0: u32,
    pub y0: u32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub colour: [f32; 4],
}

impl Params {
    fn size(w: u32, h: u32) -> Params {
        Params { w, h, ..Default::default() }
    }
}

/// One dispatch of fx_figure.wgsl's `entry` over the plane P.w x P.h with `bind` beside the parameters.
pub fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)]) {
    let (x, y) = crate::gpu::tiles(p.w, p.h);
    run_how(cx, entry, p, bind, Dispatch::Groups(x, y));
}

/// `run`, dispatched as `how` says.
pub fn run_how(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], how: Dispatch) {
    let k = cx.kernel("fx_figure", FIGURE, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    cx.dispatch(&k, &all, how);
}

/// A packed 8-bit input (pngio::read, an upload) widened to a plane: v / 255 per channel as numpy reads it (a figure
/// layer read with Kind::Luma: its coverage in .x).
pub fn unpack(cx: &mut Ctx, packed: &wgpu::Buffer, out: &wgpu::Buffer, w: u32, h: u32) {
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Dims {
        w: u32,
        h: u32,
        a: u32,
        b: u32,
    }
    let k = cx.kernel("unpack", UNPACK, "unpack_picture");
    let u = cx.uniform(&Dims { w, h, a: 0, b: 0 });
    let lut = cx.lut.clone();
    cx.pass(&k, &[(0, &u), (1, packed), (2, &lut), (3, out)], crate::gpu::tiles(w, h));
}

// ---------------------------------------------------------------------------------------------------- blur
/// OpenCV's kernel size for a float picture's Gaussian of `sigma` (ksize (0, 0)): cvRound(sigma * 4 * 2 + 1) | 1.
pub fn gaussian_size(sigma: f64) -> usize {
    ((sigma * 4.0 * 2.0 + 1.0).round_ties_even() as i64 | 1).max(1) as usize
}

/// getGaussianKernel(n, sigma, CV_32F): worked out in float64, normalised, then float32.
pub fn gaussian_taps(sigma: f64) -> Vec<f32> {
    let n = gaussian_size(sigma);
    let scale2x = -0.5 / (sigma * sigma);
    let c: Vec<f64> = (0..n).map(|i| {
        let x = i as f64 - (n as f64 - 1.0) * 0.5;
        (scale2x * x * x).exp()
    }).collect();
    let sum = 1.0 / c.iter().sum::<f64>();
    c.iter().map(|v| (v * sum) as f32).collect()
}

/// cv2.GaussianBlur(src, (0, 0), sigma) of a w x h plane into `dst` (`src` may be `dst`) through `tmp`.
pub fn blur(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, tmp: &wgpu::Buffer, w: u32, h: u32, sigma: f64) {
    blur_xy(cx, src, dst, tmp, w, h, sigma, sigma);
}

/// cv2.GaussianBlur(src, (0, 0), sigmaX=sx, sigmaY=sy) of a w x h plane into `dst` (`src` may be `dst`) through `tmp`.
#[allow(clippy::too_many_arguments)]
pub fn blur_xy(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, tmp: &wgpu::Buffer, w: u32, h: u32, sx: f64,
               sy: f64) {
    let tx = gaussian_taps(sx);
    let bx = cx.table("gauss", &tx);
    let (ny, by) = if sy == sx {
        (tx.len(), bx.clone())
    } else {
        let ty = gaussian_taps(sy);
        (ty.len(), cx.table("gauss", &ty))
    };
    let size = Params::size(w, h);
    let p = Params { n: tx.len() as u32, ..size };
    if BLUR_RUN + p.n - 1 <= BLUR_TILE {
        run_how(cx, "blur_h", p, &[(1, src), (2, tmp), (3, &bx)], Dispatch::Groups(w.div_ceil(BLUR_RUN), h));
    } else {
        run(cx, "blur_h_wide", p, &[(1, src), (2, tmp), (3, &bx)]);
    }
    run(cx, "blur_v", Params { n: ny as u32, ..size }, &[(1, tmp), (2, dst), (3, &by)]);
}

// ---------------------------------------------------------------------------------------------------- grow
/// mkmmd.framefx._grow: coverage `cov` (.x) grown by `width` px all round into `out`: the exact distance to the
/// figure (its pixels covered half or more), clip(r + 0.5 - d, 0, 1), from r = 3 px blurred by r / 4 and sharpened
/// back, never under the coverage itself. `t1`, `t2` planes of scratch.
#[allow(clippy::too_many_arguments)]
pub fn grow(cx: &mut Ctx, cov: &wgpu::Buffer, out: &wgpu::Buffer, t1: &wgpu::Buffer, t2: &wgpu::Buffer, w: u32, h: u32,
            width: f64) {
    let r = width.max(0.5);
    let reach = (r + 0.5).ceil() as u32 + 1;
    run(cx, "edt_cols", Params { n: reach, ..Params::size(w, h) }, &[(1, cov), (2, t1)]);
    run(cx, "edt_rows", Params { n: reach, a: (r + 0.5) as f32, ..Params::size(w, h) }, &[(1, t1), (2, t2)]);
    let mut p = Params::size(w, h);
    if r >= 3.0 {
        let sig = 0.25 * r;
        blur(cx, t2, t2, t1, w, h, sig);
        p.flag = 1;
        p.b = (sig * 2.5066) as f32;
    }
    run(cx, "grow_finish", p, &[(1, t2), (2, out), (4, cov)]);
}

// ---------------------------------------------------------------------------------------------------- warps
/// cv2.invertAffineTransform of a float32 map as OpenCV works it out before warpAffine without WARP_INVERSE_MAP (in
/// float64), as the float32 map the warp samples by.
pub fn inverse(m: &[f32; 6]) -> [f32; 6] {
    inverse64(&m.map(|v| v as f64))
}

/// The same for a map given in float64 (as cv2.warpAffine takes one), inverted as it is.
pub fn inverse64(m: &[f64; 6]) -> [f32; 6] {
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22, a12, a21) = (m[4] * d, m[0] * d, -m[1] * d, -m[3] * d);
    let b1 = -a11 * m[2] - a12 * m[5];
    let b2 = -a21 * m[2] - a22 * m[5];
    [a11 as f32, a12 as f32, b1 as f32, a21 as f32, a22 as f32, b2 as f32]
}

/// mkmmd.framefx.plate_shift: a plane printed `dx`, `dy` px off (warpAffine by [[1, 0, dx], [0, 1, dy]] as float32,
/// INTER_LINEAR, its edge repeated).
pub fn shift(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, w: u32, h: u32, dx: f64, dy: f64) {
    let m = inverse(&[1.0, 0.0, dx as f32, 0.0, 1.0, dy as f32]);
    warp(cx, src, (w, h), dst, (w, h), &m, BORDER_REPLICATE, [0.0; 3]);
}

/// How mkmmd.framefx._warp carries a picture by an affine map, as Python works it out (mkmmd.gpufx.figure.warp_spec):
/// `m` the float32 map from the (shrunk) source's px to the picture's, `shrink` the size (w, h) the source is first
/// resized to by area when the map shrinks it under SHRINK_BY_AREA.
#[derive(Debug, Clone, Deserialize)]
pub struct WarpSpec {
    pub m: [f32; 6],
    #[serde(default)]
    pub shrink: Option<[u32; 2]>,
}

/// mkmmd.framefx._warp: plane `src` (sw x sh) carried by `spec` into `dst` (w x h; nothing outside it), through `tmp`
/// (a plane of the shrunk size) when it is shrunk by area first.
#[allow(clippy::too_many_arguments)]
pub fn affine(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (w, h): (u32, u32),
              tmp: &wgpu::Buffer, spec: &WarpSpec) {
    let m = inverse(&spec.m);
    match spec.shrink {
        Some([tw, th]) => {
            resize_area(cx, src, (sw, sh), tmp, (tw, th));
            warp(cx, tmp, (tw, th), dst, (w, h), &m, BORDER_CONSTANT, [0.0; 3]);
        }
        None => warp(cx, src, (sw, sh), dst, (w, h), &m, BORDER_CONSTANT, [0.0; 3]),
    }
}

/// The box (x0, y0, x1, y1; x1, y1 past it; None: nothing) of what a figure layer (`cover`: its bits, bytes of 1 and
/// more, the coverage over 0) covers at all once `affine` has carried it by `spec` into a w x h frame (shrunk by area
/// when it says, then warped by the float32 map the warp kernel samples by, BORDER_CONSTANT): what a GPU pass over
/// the result finds, worked out from the layer alone. It is exact for a map that keeps the axes (each column of the
/// frame then samples one place along a source row, each row one along a source column, at the very coordinates the
/// kernel's fused multiply-adds give): a sample of non-negative values is over 0 exactly where one of the taps it
/// weighs (the one at or before it while its fraction is under 1, the one after while over 0) is, and a shrunk pixel
/// exactly where its area cell holds a pixel over 0. None when it cannot be told that way: a map that turns or
/// shears, or a fraction rounded to 1.
pub fn carried_support(cover: &super::Cover, spec: &WarpSpec, (w, h): (u32, u32)) -> Option<Option<[u32; 4]>> {
    let m = inverse(&spec.m);
    if m[1] != 0.0 || m[3] != 0.0 {
        return None;
    }
    let (lw, lh) = cover.size();
    let (sw, sh) = spec.shrink.map_or((lw, lh), |[tw, th]| (tw, th));
    // each source column's (row's) span of layer columns (rows): its area cell, or itself
    let cells = |n: u32, s: u32| -> Option<Vec<(u32, u32)>> {
        if spec.shrink.is_none() {
            return Some((0..n).map(|x| (x, x)).collect());
        }
        area_tab(n, s).iter().map(|t| Some((t.first()?.0, t.last()?.0))).collect()
    };
    let (cx_, cy_) = (cells(lw, sw)?, cells(lh, sh)?);
    // the warp's per-row terms: with m01 = 0 the column term is m02 on every row
    let r = super::warp::rows(&m, h);
    if r.chunks(2).any(|t| t[0] != r[0]) {
        return None;
    }
    // the span of layer columns (rows) the sample at `s` reads with any weight, or none; Err: not to be told here
    let span = |s: f32, n: u32, cell: &[(u32, u32)]| -> Result<Option<(u32, u32)>, ()> {
        if !(s.abs() < 1.0e9) {
            return Err(());
        }
        let f = s.floor();
        let t = s - f;
        if t >= 1.0 {
            return Err(());
        }
        let lo = (f as i64).max(0);
        let hi = (if t > 0.0 { f as i64 + 1 } else { f as i64 }).min(n as i64 - 1);
        Ok((lo <= hi).then(|| (cell[lo as usize].0, cell[hi as usize].1)))
    };
    let mut xs = Vec::with_capacity(w as usize);
    for x in 0..w {
        xs.push(span(m[0].mul_add(x as f32, r[0]), sw, &cx_).ok()?);
    }
    let mut ys = Vec::with_capacity(h as usize);
    for y in 0..h as usize {
        ys.push(span(r[2 * y + 1], sh, &cy_).ok()?);
    }
    // the layer rows some frame row reads, the layer columns some frame column reads
    let mut rows_read = vec![false; lh as usize];
    for &(a, b) in ys.iter().flatten() {
        rows_read[a as usize..=b as usize].fill(true);
    }
    let stride = (lw as usize).div_ceil(64);
    let mut cols_read = vec![0u64; stride];
    for &(a, b) in xs.iter().flatten() {
        for c in a..=b {
            cols_read[c as usize / 64] |= 1 << (c % 64);
        }
    }
    // the columns holding a pixel over 0 in a row read; the rows holding one in a column read
    let mut cols_on = vec![0u64; stride];
    let mut rows_on = vec![false; lh as usize];
    for (y, read) in rows_read.iter().enumerate() {
        if *read {
            let bits = cover.row_bits(y);
            for (o, b) in cols_on.iter_mut().zip(bits) {
                *o |= b;
            }
            rows_on[y] = bits.iter().zip(&cols_read).any(|(b, c)| b & c != 0);
        }
    }
    let any_col = |(a, b): (u32, u32)| (a..=b).any(|c| cols_on[c as usize / 64] >> (c % 64) & 1 != 0);
    let any_row = |(a, b): (u32, u32)| rows_on[a as usize..=b as usize].iter().any(|&on| on);
    let x0 = xs.iter().position(|s| s.is_some_and(any_col));
    let y0 = ys.iter().position(|s| s.is_some_and(any_row));
    let (Some(x0), Some(y0)) = (x0, y0) else {
        return Some(None);
    };
    let x1 = xs.iter().rposition(|s| s.is_some_and(any_col))? + 1;
    let y1 = ys.iter().rposition(|s| s.is_some_and(any_row))? + 1;
    Some(Some([x0 as u32, y0 as u32, x1 as u32, y1 as u32]))
}

// ---------------------------------------------------------------------------------------------------- resizes
/// OpenCV's area tables along a side of `ssize` px resized to `dsize` (computeResizeAreaTab, scale = 1 / (dsize /
/// ssize)): for each written px, (source px, weight) pairs.
pub fn area_tab(ssize: u32, dsize: u32) -> Vec<Vec<(u32, f32)>> {
    let scale = 1.0 / (dsize as f64 / ssize as f64);
    (0..dsize).map(|dx| {
        let mut out = Vec::new();
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let sx2 = (fsx2.floor() as i64).min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            out.push(((sx1 - 1) as u32, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for sx in sx1..sx2 {
            out.push((sx as u32, (1.0 / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            out.push((sx2 as u32, ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32));
        }
        out
    }).collect()
}

/// cv2.resize(src, (dw, dh), interpolation=INTER_AREA) of a float plane shrunk (sw x sh -> dw x dh), by OpenCV's
/// general tables (its fast path for whole factors adds in another order: within an ulp).
pub fn resize_area(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (dw, dh): (u32, u32)) {
    let xs = area_tab(sw, dw);
    let ys = area_tab(sh, dh);
    let head = (dw + 1 + dh + 1) as usize;
    let mut tab: Vec<u32> = vec![0; head];
    let put = |list: &[Vec<(u32, f32)>], at: usize, tab: &mut Vec<u32>| {
        for (i, taps) in list.iter().enumerate() {
            tab[at + i] = tab.len() as u32;
            for &(s, a) in taps {
                tab.push(s);
                tab.push(a.to_bits());
            }
        }
        tab[at + list.len()] = tab.len() as u32;
    };
    put(&xs, 0, &mut tab);
    put(&ys, dw as usize + 1, &mut tab);
    let t = cx.table("area", &tab);
    let p = Params { sw, sh, flag: dw + 1, ..Params::size(dw, dh) };
    run(cx, "resize_area", p, &[(1, src), (2, dst), (5, &t)]);
}

/// OpenCV's INTER_LINEAR coefficients along a side of `ssize` -> `dsize`: (source px, its weight, the next one's).
pub fn linear_tab(ssize: u32, dsize: u32) -> Vec<(u32, f32, f32)> {
    let scale = 1.0 / (dsize as f64 / ssize as f64);
    (0..dsize).map(|d| {
        let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = f.floor() as i64;
        let mut f = f - s as f32;
        if s < 0 {
            s = 0;
            f = 0.0;
        }
        if s >= ssize as i64 - 1 {
            s = ssize as i64 - 1;
            f = 0.0;
        }
        (s as u32, 1.0 - f, f)
    }).collect()
}

/// cv2.resize(src, (dw, dh), interpolation=INTER_LINEAR) of a float plane.
pub fn resize_linear(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (dw, dh): (u32, u32)) {
    let (x, y) = crate::gpu::tiles(dw, dh);
    resize_linear_how(cx, src, (sw, sh), dst, (dw, dh), Dispatch::Groups(x, y));
}

/// `resize_linear`, dispatched as `how` says (a pass the GPU may find it does not need).
pub fn resize_linear_how(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer,
                         (dw, dh): (u32, u32), how: Dispatch) {
    let mut tab: Vec<u32> = Vec::with_capacity(3 * (dw + dh) as usize);
    for (s, a0, a1) in linear_tab(sw, dw).into_iter().chain(linear_tab(sh, dh)) {
        tab.extend([s, a0.to_bits(), a1.to_bits()]);
    }
    let t = cx.table("linear", &tab);
    let p = Params { sw, sh, flag: 3 * dw, ..Params::size(dw, dh) };
    run_how(cx, "resize_linear", p, &[(1, src), (2, dst), (5, &t)], how);
}

/// OpenCV's interpolateCubic (A = -0.75) of fraction `x`, in float32 as OpenCV works it out.
pub fn cubic_weights(x: f32) -> [f32; 4] {
    let a = -0.75f32;
    let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
    let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let c2 = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// cv2.resize(src, ..., fx, fy, interpolation=INTER_CUBIC) of a float plane (sw x sh) by `f` = (fx, fy) (OpenCV's
/// inverse scales: the written size over the read one as cv2.resize works them out), its first `cw` x `ch` pixels
/// written into `dst` (a picture cut to a size, as mkmmd.fields._up cuts it).
pub fn resize_cubic(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (cw, ch): (u32, u32),
                    f: (f64, f64)) {
    let mut tab: Vec<u32> = Vec::with_capacity(5 * (cw + ch) as usize);
    for (n, inv) in [(cw, f.0), (ch, f.1)] {
        let scale = 1.0 / inv;
        for d in 0..n {
            let fx = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = fx.floor();
            let w = cubic_weights(fx - s);
            tab.extend([(s as i32) as u32, w[0].to_bits(), w[1].to_bits(), w[2].to_bits(), w[3].to_bits()]);
        }
    }
    let t = cx.table("cubic", &tab);
    let p = Params { sw, sh, flag: 5 * cw, ..Params::size(cw, ch) };
    run(cx, "resize_cubic", p, &[(1, src), (2, dst), (5, &t)]);
}

// ---------------------------------------------------------------------------------------------------- sticker parts
/// A drop shadow as mkmmd.framefx._shadow gives it: its colour, offset (px), blur (a Gaussian's sigma, px) and alpha.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Shadow {
    pub color: [f32; 3],
    pub offset: [f64; 2],
    pub blur: f64,
    pub alpha: f64,
}

/// A die-cut edge as mkmmd.framefx._edge gives it: its colour and width (px).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Edge {
    pub color: [f32; 3],
    pub width: f64,
}

/// `pic` (rgb) times coverage `cov` (.x) into `out`, the coverage in .w: a premultiplied sticker.
pub fn premultiply(cx: &mut Ctx, pic: &wgpu::Buffer, cov: &wgpu::Buffer, out: &wgpu::Buffer, w: u32, h: u32) {
    run(cx, "premultiply", Params::size(w, h), &[(1, pic), (2, out), (4, cov)]);
}

/// An effect's picture `out` (in place) mixed with the picture as it reached it (`img`) by the strength `s`
/// (img * (1 - s) + out * s; numpy's float32 1 - s and s).
pub fn mix(cx: &mut Ctx, img: &wgpu::Buffer, out: &wgpu::Buffer, w: u32, h: u32, s: f64) {
    run(cx, "mix", Params { a: s as f32, b: (1.0 - s) as f32, ..Params::size(w, h) }, &[(1, img), (2, out)]);
}

/// Every pixel of `out` the colour `c`.
pub fn fill(cx: &mut Ctx, out: &wgpu::Buffer, w: u32, h: u32, c: [f32; 3]) {
    run(cx, "fill", Params { colour: [c[0], c[1], c[2], 0.0], ..Params::size(w, h) }, &[(2, out)]);
}

/// A copy of plane `src` into `dst` (w x h).
pub fn copy(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, w: u32, h: u32) {
    cx.enc.copy_buffer_to_buffer(src, 0, dst, 0, plane_bytes(w, h));
}

/// Plane `src` (w x h) reflected left for right about x = `x2` / 2 into `dst`: pixel x shows pixel x2 - x (nothing
/// where that falls outside), as mkmmd.stack.lay reflects a layer.
pub fn flip(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, w: u32, h: u32, x2: i32) {
    run(cx, "flip", Params { n: x2 as u32, ..Params::size(w, h) }, &[(1, src), (2, dst)]);
}

/// A packed 8-bit picture of `sw` x `sh` (an upload) widened into plane `dst` (w x h) with its top left at `at`.
pub fn blit(cx: &mut Ctx, packed: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (w, h): (u32, u32),
            at: [u32; 2]) {
    let k = cx.kernel("fx_figure", FIGURE, "blit");
    let u = cx.uniform(&Params { sw, sh, x0: at[0], y0: at[1], ..Params::size(w, h) });
    let lut = cx.lut.clone();
    cx.pass(&k, &[(0, &u), (2, dst), (6, packed), (7, &lut)], crate::gpu::tiles(sw, sh));
}

// ---------------------------------------------------------------------------------------------------- the figure
/// A collage's figure on this frame as mkmmd.gpufx.figure.cutout gives it: its figure layer (read as
/// mkmmd.cutfx.read_figure reads it), the picture it is cut from (None: the picture as it reaches the op; else a PNG,
/// its own shot's) and the map that sets it into the frame (None: where its frame shows it).
#[derive(Debug, Clone, Deserialize)]
pub struct Cutout {
    pub layer: String,
    #[serde(default)]
    pub plate: Option<String>,
    #[serde(default)]
    pub warp: Option<WarpSpec>,
}

impl Cutout {
    /// Its files, in the order `cut` takes them.
    pub fn inputs(&self) -> Vec<Input> {
        let mut v = vec![Input::Png(self.layer.clone(), Kind::Luma)];
        if let Some(p) = &self.plate {
            v.push(Input::Png(p.clone(), Kind::Picture));
        }
        v
    }
}

/// The figure of `c` cut out on the GPU (FrameFx._collage, _posed): its coverage into plane `cov` (.x) and, given `rgb`,
/// its picture times its coverage into it (_paste's pic * m), both set into the frame by its map; then the picture it
/// was cut from, set into the frame (a scratch plane, or the picture as it reaches the op: read it before the next
/// cut). `files` its inputs.
pub fn cut(cx: &mut Ctx, c: &Cutout, files: &[Arc<Decoded>], cov: &wgpu::Buffer, rgb: Option<&wgpu::Buffer>)
           -> Option<wgpu::Buffer> {
    let (w, h) = (cx.w, cx.h);
    let n = plane_bytes(w, h);
    let a = cx.scratch("figure.a", n);
    let t = cx.scratch("figure.t", n);
    let layer = cx.upload(&Input::Png(c.layer.clone(), Kind::Luma), &files[0]);
    match &c.warp {
        None => unpack(cx, &layer, cov, w, h),
        Some(spec) => {
            unpack(cx, &layer, &a, w, h);
            affine(cx, &a, (w, h), cov, (w, h), &t, spec);
        }
    }
    let rgb = rgb?;
    let pic = match &c.plate {
        None => cx.src.clone(),
        Some(p) => {
            let b = cx.upload(&Input::Png(p.clone(), Kind::Picture), &files[1]);
            unpack(cx, &b, &a, w, h);
            a.clone()
        }
    };
    let pic = match &c.warp {
        None => pic,
        Some(spec) => {
            let b = cx.scratch("figure.b", n);
            affine(cx, &pic, (w, h), &b, (w, h), &t, spec);
            b
        }
    };
    premultiply(cx, &pic, cov, rgb, w, h);
    cx.mark("figure cut");
    Some(pic)
}

/// The box (x0, y0, x1, y1) of what cut-out `c` covers at all once cut (its coverage over 0), from its layer's Cover
/// (`layer`, decoded): the layer's own box unwarped, carried by the warp (`carried_support`) warped. Some(None):
/// nothing; None: not to be told from the layer.
pub fn support(cx: &Ctx, c: &Cutout, layer: &Decoded) -> Option<Option<[u32; 4]>> {
    let cover = layer.cover()?;
    let t = super::first_over(cx.levels, 0.0);
    match &c.warp {
        None => Some(t.and_then(|t| cover.bbox(t))),
        Some(spec) if t == Some(1) => carried_support(cover, spec, (cx.w, cx.h)),
        Some(_) => None,
    }
}

/// `cut` over the box `at` of the frame: the figure's coverage into `cov` (a single-float plane over `at`) and, given
/// `rgb` (a vec4 plane over `at`), its picture times its coverage; each pixel as `cut` works it out over the frame.
pub fn cut_box(cx: &mut Ctx, c: &Cutout, files: &[Arc<Decoded>], at: &Area, cov: &wgpu::Buffer,
               rgb: Option<&wgpu::Buffer>) {
    let (w, h) = (cx.w, cx.h);
    let n = plane_bytes(w, h);
    let frame = Area::whole(w, h);
    let layer = cx.upload(&Input::Png(c.layer.clone(), Kind::Luma), &files[0]);
    match &c.warp {
        None => plane::unpack(cx, &layer, cov, at),
        Some(spec) => {
            let a = cx.scratch("cut.a1", frame.bytes());
            plane::unpack(cx, &layer, &a, &frame);
            let m = inverse(&spec.m);
            match spec.shrink {
                Some([tw, th]) => {
                    let t = cx.scratch("cut.t1", Area::whole(tw, th).bytes());
                    plane::resize_area(cx, &a, (w, h), &t, (tw, th));
                    plane::warp(cx, &t, &Area::whole(tw, th), cov, at, &m, BORDER_CONSTANT);
                }
                None => plane::warp(cx, &a, &frame, cov, at, &m, BORDER_CONSTANT),
            }
        }
    }
    let Some(rgb) = rgb else { return };
    let pic = match &c.plate {
        None => cx.src.clone(),
        Some(p) => {
            let b = cx.upload(&Input::Png(p.clone(), Kind::Picture), &files[1]);
            let a = cx.scratch("figure.a", n);
            unpack(cx, &b, &a, w, h);
            a
        }
    };
    match &c.warp {
        None => plane::premultiply(cx, &pic, &frame, cov, rgb, at),
        Some(spec) => {
            // the picture carried into the box alone, then multiplied there
            let m = inverse(&spec.m);
            let b = cx.scratch("figure.b", at.bytes4());
            let span = [at.x0, at.y0, at.x0 + at.w, at.y0 + at.h];
            match spec.shrink {
                Some([tw, th]) => {
                    let t = cx.scratch("figure.t", plane_bytes(tw, th));
                    resize_area(cx, &pic, (w, h), &t, (tw, th));
                    warp_box(cx, &t, (tw, th), &b, span, &m, BORDER_CONSTANT, [0.0; 3]);
                }
                None => warp_box(cx, &pic, (w, h), &b, span, &m, BORDER_CONSTANT, [0.0; 3]),
            }
            plane::premultiply(cx, &b, at, cov, rgb, at);
        }
    }
    cx.mark("figure cut");
}

/// How far past the figure `paste_figure` draws anything (px): its edge's ring and the shadow cast from it.
pub fn paste_reach(edge: Option<&Edge>, shadow: Option<&Shadow>) -> u32 {
    let ring = edge.filter(|e| e.width > 0.0).map_or(0, |e| plane::grow_reach(e.width));
    let cast = shadow.filter(|s| s.alpha > 0.0).map_or(0, |s| plane::cast_reach(s, 1.0));
    ring + cast + 2
}

/// `dst` (in place) with a figure laid on it as mkmmd.framefx._paste lays one unscaled: its coverage `cov` drawn as its
/// premultiplied picture `rgb` (`fill` None) or in the flat colour `fill`, over an `edge` grown round it, the two
/// casting a `shadow`. Worked out over the box `at` alone (`cov` a single-float plane, `rgb` a vec4 plane over it),
/// which holds everything the figure, its ring and its shadow reach (paste_reach): the whole frame's pixels in it,
/// and the picture as it is outside it, where the frame's work leaves it so.
#[allow(clippy::too_many_arguments)]
pub fn paste_figure(cx: &mut Ctx, dst: &wgpu::Buffer, at: &Area, cov: &wgpu::Buffer, rgb: &wgpu::Buffer,
                    fill: Option<[f32; 3]>, edge: Option<&Edge>, shadow: Option<&Shadow>) {
    let t1 = cx.scratch("figure.t1", at.bytes());
    let t2 = cx.scratch("figure.t2", at.bytes());
    let ring = match edge {
        Some(e) if e.width > 0.0 => {
            let g = cx.scratch("figure.ring", at.bytes());
            plane::grow(cx, cov, &g, &t1, &t2, at, e.width);
            cx.mark("figure edge");
            Some(g)
        }
        _ => None,
    };
    if let Some(s) = shadow.filter(|s| s.alpha > 0.0) {
        plane::cast(cx, dst, ring.as_ref().unwrap_or(cov), &t1, &t2, at, s, 1.0);
        cx.mark("figure shadow");
    }
    if let (Some(g), Some(e)) = (&ring, edge) {
        plane::colour_by(cx, dst, g, at, e.color, None);
    }
    plane::paste(cx, dst, rgb, cov, at, fill);
    cx.mark("figure paste");
}
