//! Single-float planes (a coverage, a ring, a shadow: one f32 per pixel) and the sticker work on them
//! (shaders/fx_plane.wgsl): each function the one of fx/figure.rs it stands for, the same arithmetic on a quarter of the
//! memory, and able to work over a box of a frame (Area) exactly as over the whole frame: coordinates and borders are
//! the frame's and a pixel of the frame outside the box reads 0, so a box holding everything over 0 the work reads
//! gives the whole frame's pixels in it (and the picture's outside it untouched, where its coverages are 0).

use bytemuck::{Pod, Zeroable};

use super::Ctx;
use super::figure::{BORDER_REPLICATE, Shadow, area_tab, gaussian_taps, inverse, linear_tab};

const SRC: &str = include_str!("../shaders/fx_plane.wgsl");
/// fx_plane.wgsl's blur_h: a run of a row and the reach on either side, in a workgroup's memory.
const BLUR_RUN: u32 = 256;
const BLUR_TILE: u32 = 2304;

/// A box of a frame a plane covers: its size, its top left in the frame and the frame's size. A plane of its own is the
/// whole of a frame its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Area {
    pub w: u32,
    pub h: u32,
    pub x0: u32,
    pub y0: u32,
    pub fw: u32,
    pub fh: u32,
}

impl Area {
    pub fn whole(w: u32, h: u32) -> Area {
        Area { w, h, x0: 0, y0: 0, fw: w, fh: h }
    }

    /// The box [x0, x1) x [y0, y1) of a fw x fh frame.
    pub fn of((fw, fh): (u32, u32), [x0, y0, x1, y1]: [u32; 4]) -> Area {
        Area { w: x1 - x0, h: y1 - y0, x0, y0, fw, fh }
    }

    /// Bytes of a single-float plane over it.
    pub fn bytes(&self) -> u64 {
        self.w as u64 * self.h as u64 * 4
    }

    /// Bytes of a vec4 plane over it.
    pub fn bytes4(&self) -> u64 {
        self.w as u64 * self.h as u64 * 16
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    fw: u32,
    fh: u32,
    n: u32,
    flag: u32,
    sw: u32,
    sh: u32,
    sx0: u32,
    sy0: u32,
    sfw: u32,
    sfh: u32,
    pad0: u32,
    pad1: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    colour: [f32; 4],
}

impl Params {
    fn over(at: &Area) -> Params {
        Params { w: at.w, h: at.h, x0: at.x0, y0: at.y0, fw: at.fw, fh: at.fh, ..Default::default() }
    }
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], groups: (u32, u32)) {
    let k = cx.kernel("fx_plane", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    cx.pass(&k, &all, groups);
}

fn tiles(at: &Area) -> (u32, u32) {
    crate::gpu::tiles(at.w, at.h)
}

/// figure::unpack's first channel over `at`: a packed 8-bit input of the frame's size (an upload; a figure layer read
/// with Kind::Luma) widened by the 1/255 table, its coverage.
pub fn unpack(cx: &mut Ctx, packed: &wgpu::Buffer, out: &wgpu::Buffer, at: &Area) {
    let lut = cx.lut.clone();
    run(cx, "unpack", Params::over(at), &[(2, out), (6, packed), (7, &lut)], tiles(at));
}

/// The .x of picture `pic` (the frame's) over `at`.
pub fn crop(cx: &mut Ctx, pic: &wgpu::Buffer, out: &wgpu::Buffer, at: &Area) {
    run(cx, "crop", Params::over(at), &[(2, out), (8, pic)], tiles(at));
}

/// figure::blur: cv2.GaussianBlur(src, (0, 0), sigma) of plane `src` over `at` into `dst` (`src` may be `dst`) through
/// `tmp`.
pub fn blur(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, tmp: &wgpu::Buffer, at: &Area, sigma: f64) {
    let taps = gaussian_taps(sigma);
    let t = cx.table("gauss", &taps);
    let p = Params { n: taps.len() as u32, ..Params::over(at) };
    if BLUR_RUN + p.n - 1 <= BLUR_TILE {
        run(cx, "blur_h", p, &[(1, src), (2, tmp), (4, &t)], (at.w.div_ceil(BLUR_RUN), at.h));
    } else {
        run(cx, "blur_h_wide", p, &[(1, src), (2, tmp), (4, &t)], tiles(at));
    }
    run(cx, "blur_v", p, &[(1, tmp), (2, dst), (4, &t)], tiles(at));
}

/// figure::grow: coverage `cov` (over `at`) grown by `width` px into `out` through `t1`, `t2`. `at` holds the figure
/// and all its ring reaches (the ring's distances are found inside it).
#[allow(clippy::too_many_arguments)]
pub fn grow(cx: &mut Ctx, cov: &wgpu::Buffer, out: &wgpu::Buffer, t1: &wgpu::Buffer, t2: &wgpu::Buffer, at: &Area,
            width: f64) {
    let r = width.max(0.5);
    let reach = (r + 0.5).ceil() as u32 + 1;
    run(cx, "edt_cols", Params { n: reach, ..Params::over(at) }, &[(1, cov), (2, t1)], tiles(at));
    run(cx, "edt_rows", Params { n: reach, a: (r + 0.5) as f32, ..Params::over(at) }, &[(1, t1), (2, t2)], tiles(at));
    let mut p = Params::over(at);
    if r >= 3.0 {
        let sig = 0.25 * r;
        blur(cx, t2, t2, t1, at, sig);
        p.flag = 1;
        p.b = (sig * 2.5066) as f32;
    }
    run(cx, "grow_finish", p, &[(1, t2), (2, out), (3, cov)], tiles(at));
}

/// How far past a coverage `grow` by `width` reaches (px): the ring before its blur and the blur.
pub fn grow_reach(width: f64) -> u32 {
    let r = width.max(0.5);
    let blur = if r >= 3.0 { super::figure::gaussian_size(0.25 * r) as u32 / 2 } else { 0 };
    (r + 0.5).ceil() as u32 + 1 + blur
}

/// warp.rs's warp of plane `src` (`from`: a box of the frame whose border rule `border` applies) by the float32 map `m`
/// (from each pixel written to the place it reads) into `dst` over `at` (coordinates of the frame `at` is a box of),
/// INTER_LINEAR; BORDER_CONSTANT reads 0.
pub fn warp(cx: &mut Ctx, src: &wgpu::Buffer, from: &Area, dst: &wgpu::Buffer, at: &Area, m: &[f32; 6], border: u32) {
    let terms: Vec<u32> = (at.y0..at.y0 + at.h).flat_map(|y| {
        let y = y as f32;
        [(m[1] * y + m[2]).to_bits(), (m[4] * y + m[5]).to_bits()]
    }).collect();
    let t = cx.table("rows", &terms);
    let p = Params {
        n: border, sw: from.w, sh: from.h, sx0: from.x0, sy0: from.y0, sfw: from.fw, sfh: from.fh, a: m[0], b: m[3],
        ..Params::over(at)
    };
    run(cx, "warp", p, &[(1, src), (2, dst), (5, &t)], tiles(at));
}

/// figure::shift: plane `src` over `at` printed `dx`, `dy` px off into `dst` over it (the frame's edge repeated).
pub fn shift(cx: &mut Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, at: &Area, dx: f64, dy: f64) {
    let m = inverse(&[1.0, 0.0, dx as f32, 0.0, 1.0, dy as f32]);
    warp(cx, src, at, dst, at, &m, BORDER_REPLICATE);
}

/// figure::resize_area of a plane of its own (sw x sh) into `dst` (dw x dh).
pub fn resize_area(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (dw, dh): (u32, u32)) {
    let xs = area_tab(sw, dw);
    let ys = area_tab(sh, dh);
    let mut tab: Vec<u32> = vec![0; (dw + 1 + dh + 1) as usize];
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
    let to = Area::whole(dw, dh);
    run(cx, "resize_area", Params { sw, sh, flag: dw + 1, ..Params::over(&to) }, &[(1, src), (2, dst), (5, &t)],
        tiles(&to));
}

/// figure::resize_linear of a plane of its own (sw x sh) into `dst` (dw x dh).
pub fn resize_linear(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (dw, dh): (u32, u32)) {
    let mut tab: Vec<u32> = Vec::with_capacity(3 * (dw + dh) as usize);
    for (s, a0, a1) in linear_tab(sw, dw).into_iter().chain(linear_tab(sh, dh)) {
        tab.extend([s, a0.to_bits(), a1.to_bits()]);
    }
    let t = cx.table("linear", &tab);
    let to = Area::whole(dw, dh);
    run(cx, "resize_linear", Params { sw, sh, flag: 3 * dw, ..Params::over(&to) }, &[(1, src), (2, dst), (5, &t)],
        tiles(&to));
}

/// The picture `pic` (the frame's, in place) over `at` under coverage `cov` in `colour`: by `alpha` times it (a
/// shadow: figure::cast's last step), or as it is (None: an edge's ring, figure::over).
pub fn colour_by(cx: &mut Ctx, pic: &wgpu::Buffer, cov: &wgpu::Buffer, at: &Area, colour: [f32; 3],
                 alpha: Option<f32>) {
    let p = Params {
        flag: alpha.is_none() as u32, a: alpha.unwrap_or(0.0), colour: [colour[0], colour[1], colour[2], 0.0],
        ..Params::over(at)
    };
    run(cx, "colour_by", p, &[(1, cov), (9, pic)], tiles(at));
}

/// figure::premultiply over `at`: picture `pic` (over `from`: the frame, or a box of it holding `at`) times coverage
/// `cov` into `out` (a vec4 plane over `at`), the coverage in .w.
pub fn premultiply(cx: &mut Ctx, pic: &wgpu::Buffer, from: &Area, cov: &wgpu::Buffer, out: &wgpu::Buffer, at: &Area) {
    let p = Params { sw: from.w, sx0: from.x0, sy0: from.y0, ..Params::over(at) };
    run(cx, "premultiply", p, &[(1, cov), (8, pic), (9, out)], tiles(at));
}

/// figure::paste over `at`: on picture `pic` (the frame's, in place) the figure of coverage `cov` drawn as its
/// premultiplied picture `rgb` (a vec4 plane over `at`), or in the flat colour `fill`.
pub fn paste(cx: &mut Ctx, pic: &wgpu::Buffer, rgb: &wgpu::Buffer, cov: &wgpu::Buffer, at: &Area,
             fill: Option<[f32; 3]>) {
    let mut p = Params::over(at);
    if let Some(c) = fill {
        p.flag = 1;
        p.colour = [c[0], c[1], c[2], 0.0];
    }
    run(cx, "paste", p, &[(1, cov), (8, rgb), (9, pic)], tiles(at));
}

/// figure::cast over `at`: picture `pic` (in place) under the drop shadow of coverage `cov`, its offset and blur `s`
/// times as large, through `t1`, `t2`.
#[allow(clippy::too_many_arguments)]
pub fn cast(cx: &mut Ctx, pic: &wgpu::Buffer, cov: &wgpu::Buffer, t1: &wgpu::Buffer, t2: &wgpu::Buffer, at: &Area,
            shadow: &Shadow, s: f64) {
    shift(cx, cov, t1, at, shadow.offset[0] * s, shadow.offset[1] * s);
    if shadow.blur * s > 0.3 {
        blur(cx, t1, t1, t2, at, shadow.blur * s);
    }
    colour_by(cx, pic, t1, at, shadow.color, Some(shadow.alpha as f32));
}

/// How far past a coverage `cast` darkens (px): its offset, the bilinear pixel and the blur.
pub fn cast_reach(shadow: &Shadow, s: f64) -> u32 {
    let (dx, dy) = (shadow.offset[0] * s, shadow.offset[1] * s);
    let blur = if shadow.blur * s > 0.3 { super::figure::gaussian_size(shadow.blur * s) as u32 / 2 } else { 0 };
    dx.abs().max(dy.abs()).ceil() as u32 + 1 + blur
}
