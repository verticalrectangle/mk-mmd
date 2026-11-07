//! mandala: rings of stickers round a point (mkmmd.mandala.MandalaFx._mandala), laid as mkmmd.mandala.lay lays them:
//! each tile warped into the box it covers, its edge under it, its soft shadow under both, then the pupils and glints of
//! a copy with googly eyes (fx/inks.rs) (shaders/fx_mandala.wgsl). Python (mkmmd.gpufx.mandala) lays the rings out
//! (mkmmd.core.mandala.copies), works out each sticker's box and map and its pupils' coverages, and writes the tiles of
//! sprites and drawn shapes (kept from frame to frame) to files; the figure's own tiles are cut here from the figure
//! layer and the picture as it reaches the op. The stamps of a paint app (mkmmd.gpufx.paintapp) are laid the same way,
//! each at its own opacity and with its sparkles as inks.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::fields::PatternSpec;
use super::figure::{self, Shadow, WarpSpec, plane_bytes};
use super::inks::{self, Ink};
use super::spiral;
use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_mandala.wgsl");
/// mkmmd.mandala.soft: a shadow blurred this much or more is blurred at half the size.
const SOFT_HALF: f64 = 4.0;

/// What the stickers are laid on (mkmmd.framefx._flat of the entry's back): a colour, the picture as it reaches the
/// op, or a pattern (mkmmd.fields.Painter).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Back {
    Color([f32; 3]),
    Picture,
    Pattern(PatternSpec),
}

/// A sticker tile: one Python made (a sprite's or a drawn shape's: its picture (premultiplied rgb, coverage) and its
/// ring, w x h vec4 float32 files; `once`: written for this frame alone, in colours only a few frames show (a breathing
/// palette on its way), and removed once read), or the figure's own (mkmmd.mandala.figure_tile: the part of the frame
/// cut, its size once resized (by area when shrunk), its fill (None: its picture), the padding round it and its edge's
/// width).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Tile {
    File {
        rgba: String,
        #[serde(default)]
        ring: Option<String>,
        size: [u32; 2],
        #[serde(default)]
        once: bool,
    },
    Figure {
        crop: [u32; 4],
        size: [u32; 2],
        shrink: bool,
        #[serde(default)]
        fill: Option<[f32; 3]>,
        pad: u32,
        edge: f64,
    },
}

/// One copy laid: its tile, the box it covers (x0, y0, x1, y1), the warp from the tile into the box, its opacity (a
/// paint app's stamp fading: the picture, its edge and its shadow all that much there) and what is laid over it after
/// (its pupils and glints, a stamp's sparkles). Without a tile (None: a stamp whose sticker is off the picture) only its
/// inks are laid.
#[derive(Debug, Clone, Deserialize)]
pub struct Sticker {
    #[serde(default)]
    pub tile: Option<usize>,
    #[serde(rename = "box", default)]
    pub area: [u32; 4],
    #[serde(default)]
    pub warp: Option<WarpSpec>,
    #[serde(default = "opaque")]
    pub opacity: f64,
    #[serde(default)]
    pub inks: Vec<Ink>,
}

fn opaque() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mandala {
    pub back: Back,
    /// The figure layer the figure's tiles are cut from.
    #[serde(default)]
    pub figure: Option<String>,
    pub tiles: Vec<Tile>,
    pub stickers: Vec<Sticker>,
    /// The edges' colour (None: no edge drawn).
    #[serde(default)]
    pub edge: Option<[f32; 3]>,
    #[serde(default)]
    pub shadow: Option<Shadow>,
    pub strength: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    bw: u32,
    bh: u32,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    edged: u32,
    shadow: u32,
    pad: u32,
    fill: u32,
    pad0: u32,
    pad1: u32,
    alpha: f32,
    opacity: f32,
    pad3: f32,
    pad4: f32,
    edge: [f32; 4],
    shade: [f32; 4],
}

/// A tile on the GPU: its picture (rgb, a), its ring (.x) and size.
struct Laid {
    rgba: wgpu::Buffer,
    ring: Option<wgpu::Buffer>,
    w: u32,
    h: u32,
}

/// A tile's file as it is read: kept (cached, read again while it stays in the cache) or, `once`, read and removed.
fn file(path: &str, size: [u32; 2], once: bool) -> Input {
    let count = size[0] as usize * size[1] as usize * 4;
    if once { Input::F32 { path: path.to_string(), count } } else { Input::Kept { path: path.to_string(), count } }
}

impl Mandala {
    fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], (w, h): (u32, u32)) {
        let k = cx.kernel("fx_mandala", SRC, entry);
        let u = cx.uniform(&p);
        let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
        all.extend_from_slice(bind);
        cx.pass(&k, &all, crate::gpu::tiles(w, h));
    }

    /// The figure's tile: cut from the figure plane `fig` and the picture, resized, padded, its ring grown.
    #[allow(clippy::too_many_arguments)]
    fn figure_tile(cx: &mut Ctx, fig: &wgpu::Buffer, crop: [u32; 4], size: [u32; 2], shrink: bool, fill: Option<[f32; 3]>,
                   pad: u32, edge: f64) -> Laid {
        let (w, h) = (cx.w, cx.h);
        let (cw, ch) = (crop[2] - crop[0], crop[3] - crop[1]);
        let [nw, nh] = size;
        let n = plane_bytes(w, h);
        let c1 = cx.scratch("mandala.c1", n);
        let c2 = cx.scratch("mandala.c2", n);
        let src = cx.src.clone();
        let p = Params { bw: cw, bh: ch, x0: crop[0], y0: crop[1], w, h, ..Default::default() };
        Self::run(cx, "figure_cut", p, &[(1, &src), (2, &c1), (3, fig), (4, &c2)], (cw, ch));
        let (r1, r2) = if (nw, nh) == (cw, ch) {
            (c1, c2)
        } else {
            let r1 = cx.scratch("mandala.r1", n);
            let r2 = cx.scratch("mandala.r2", n);
            for (a, b) in [(&c1, &r1), (&c2, &r2)] {
                if shrink {
                    figure::resize_area(cx, a, (cw, ch), b, (nw, nh));
                } else {
                    figure::resize_linear(cx, a, (cw, ch), b, (nw, nh));
                }
            }
            (r1, r2)
        };
        let (tw, th) = (nw + 2 * pad, nh + 2 * pad);
        let tb = plane_bytes(tw, th);
        let rgba = cx.gpu.storage("figure tile", tb);
        let cov = cx.scratch("mandala.cov", tb);
        let c = fill.unwrap_or([0.0; 3]);
        let p = Params { bw: tw, bh: th, pad, fill: fill.is_some() as u32, edge: [c[0], c[1], c[2], 0.0], ..Default::default() };
        Self::run(cx, "figure_tile", p, &[(1, &r1), (2, &rgba), (3, &r2), (4, &cov)], (tw, th));
        let ring = (edge > 0.0).then(|| {
            let g = cx.gpu.storage("figure ring", tb);
            let t1 = cx.scratch("mandala.g1", tb);
            let t2 = cx.scratch("mandala.g2", tb);
            figure::grow(cx, &cov, &g, &t1, &t2, tw, th, edge);
            g
        });
        Laid { rgba, ring, w: tw, h: th }
    }

    /// mkmmd.mandala.soft of the box plane `x` (bw x bh) into itself: blurred `sigma`, a wide blur at half the size.
    fn soft(cx: &mut Ctx, x: &wgpu::Buffer, (bw, bh): (u32, u32), sigma: f64) {
        if sigma <= 0.3 {
            return;
        }
        let n = plane_bytes(cx.w, cx.h);
        let t = cx.scratch("mandala.s2", n);
        if sigma < SOFT_HALF || bw.min(bh) < 16 {
            figure::blur(cx, x, x, &t, bw, bh, sigma);
            return;
        }
        let (hw, hh) = (bw.div_ceil(2), bh.div_ceil(2));
        let half = cx.scratch("mandala.half", n);
        figure::resize_area(cx, x, (bw, bh), &half, (hw, hh));
        figure::blur(cx, &half, &half, &t, hw, hh, sigma / 2.0);
        figure::resize_linear(cx, &half, (hw, hh), x, (bw, bh));
    }

    /// One sticker laid on `out` (mkmmd.mandala.lay), its tile carried into its box by `warp`, its picture, edge and
    /// shadow at the sticker's opacity.
    fn lay(&self, cx: &mut Ctx, out: &wgpu::Buffer, t: &Laid, s: &Sticker, warp: &WarpSpec) {
        let (w, h) = (cx.w, cx.h);
        let [x0, y0, x1, y1] = s.area;
        let (bw, bh) = (x1 - x0, y1 - y0);
        let n = plane_bytes(w, h);
        let b1 = cx.scratch("mandala.b1", n);
        let b2 = cx.scratch("mandala.b2", n);
        let a1 = cx.scratch("mandala.a1", n);
        let s1 = cx.scratch("mandala.s1", n);
        let tmp = cx.scratch("mandala.t", plane_bytes(t.w, t.h));
        figure::affine(cx, &t.rgba, (t.w, t.h), &b1, (bw, bh), &tmp, warp);
        let edged = t.ring.is_some() && self.edge.is_some();
        if let (true, Some(ring)) = (edged, &t.ring) {
            figure::affine(cx, ring, (t.w, t.h), &b2, (bw, bh), &tmp, warp);
        }
        let e = self.edge.unwrap_or([0.0; 3]);
        let mut p = Params { bw, bh, x0, y0, w, h, edged: edged as u32, opacity: s.opacity as f32,
                             edge: [e[0], e[1], e[2], 0.0], ..Default::default() };
        Self::run(cx, "lay_merge", p, &[(2, &b1), (3, &b2), (4, &a1)], (bw, bh));
        if let Some(sh) = self.shadow.as_ref().filter(|sh| sh.alpha * s.opacity > 0.0) {
            figure::shift(cx, &a1, &s1, bw, bh, sh.offset[0], sh.offset[1]);
            Self::soft(cx, &s1, (bw, bh), sh.blur);
            p.shadow = 1;
            p.alpha = (sh.alpha * s.opacity) as f32;
            p.shade = [sh.color[0], sh.color[1], sh.color[2], 0.0];
        }
        Self::run(cx, "lay_put", p, &[(1, &b1), (2, out), (3, &s1)], (bw, bh));
    }
}

impl Effect for Mandala {
    fn inputs(&self) -> Vec<Input> {
        let mut v: Vec<Input> = self.figure.iter().map(|f| Input::Png(f.clone(), Kind::Luma)).collect();
        for t in &self.tiles {
            if let Tile::File { rgba, ring, size, once } = t {
                v.push(file(rgba, *size, *once));
                if let Some(r) = ring {
                    v.push(file(r, *size, *once));
                }
            }
        }
        v.extend(self.stickers.iter().flat_map(|s| s.inks.iter().filter_map(Ink::input)));
        v
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        match &self.back {
            Back::Color(c) => figure::fill(cx, &dst, w, h, *c),
            Back::Picture => figure::copy(cx, &src, &dst, w, h),
            Back::Pattern(p) => spiral::paint(cx, p, &dst, w, h),
        }
        let mut next = 0;
        let fig = match &self.figure {
            Some(f) => {
                let packed = cx.upload(&Input::Png(f.clone(), Kind::Luma), &files[0]);
                let plane = cx.scratch("mandala.fig", plane_bytes(w, h));
                figure::unpack(cx, &packed, &plane, w, h);
                next = 1;
                Some(plane)
            }
            None => None,
        };
        let mut laid = Vec::with_capacity(self.tiles.len());
        for t in &self.tiles {
            laid.push(match t {
                Tile::File { rgba, ring, size, once } => {
                    let a = cx.upload(&file(rgba, *size, *once), &files[next]);
                    next += 1;
                    let r = ring.as_ref().map(|r| {
                        let b = cx.upload(&file(r, *size, *once), &files[next]);
                        next += 1;
                        b
                    });
                    Laid { rgba: a, ring: r, w: size[0], h: size[1] }
                }
                Tile::Figure { crop, size, shrink, fill, pad, edge } => {
                    let Some(fig) = &fig else { bail!("a figure tile without the figure's layer") };
                    Self::figure_tile(cx, fig, *crop, *size, *shrink, *fill, *pad, *edge)
                }
            });
        }
        for s in &self.stickers {
            if let Some(i) = s.tile {
                let Some(t) = laid.get(i) else { bail!("sticker of tile {} (of {})", i, laid.len()) };
                let Some(warp) = &s.warp else { bail!("sticker of tile {} without its warp", i) };
                self.lay(cx, &dst, t, s, warp);
            }
            if !s.inks.is_empty() {
                let n = s.inks.iter().filter(|i| i.mask.is_some()).count();
                inks::lay(cx, &dst, &s.inks, &files[next..next + n]);
                next += n;
            }
        }
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
