//! What the remaps share under a `mask` (mkmmd.remap.RemapFx._remapped, fill, pushpull, sticker, shade;
//! shaders/fx_masked.wgsl), and the pyramid they sample shrunk pictures from (mkmmd.remap.Pyramid). Under a mask the
//! figure's place in the picture is filled from the ground round it, then either the ground is remapped and the figure
//! laid back on top as rendered ("ground"), or the figure alone is remapped over the filled ground or a flat `back`
//! ("figure"); the figure wears its `edge` and `shadow` as a collage's sticker does. Planes are w x h vec4<f32>.

use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, Edge, Shadow, plane_bytes};
use super::plane::{self, Area};
use super::{Ctx, Decoded, Input, first_over};
use crate::gpu::Dispatch;
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_masked.wgsl");

/// mkmmd.remap.HOLE_CUT as numpy compares a float32 coverage with it.
pub const HOLE_CUT: f32 = 1.0 / 255.0;

/// A remap's `mask` (mkmmd.gpufx.masked.masking): its figure layer, which way it masks, how far past the figure its
/// place is filled (px: HOLE_MARGIN and the room an edge and a shadow take), the sticker's edge and shadow, a figure
/// mask's flat back.
#[derive(Debug, Clone, Deserialize)]
pub struct Masking {
    pub figure: String,
    pub mask: String,
    pub reach: f64,
    #[serde(default)]
    pub edge: Option<Edge>,
    #[serde(default)]
    pub shadow: Option<Shadow>,
    #[serde(default)]
    pub back: Option<[f32; 3]>,
}

impl Masking {
    /// The figure layer, read as mkmmd.cutfx.read_figure reads it (Pillow's L).
    pub fn input(&self) -> Input {
        Input::Png(self.figure.clone(), Kind::Luma)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    sw: u32,
    sh: u32,
    x0: u32,
    y0: u32,
    flag: u32,
    pad: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    colour: [f32; 4],
    colour2: [f32; 4],
}

impl Params {
    fn size(w: u32, h: u32) -> Params {
        Params { w, h, ..Default::default() }
    }
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], groups: (u32, u32)) {
    run_how(cx, entry, p, bind, Dispatch::Groups(groups.0, groups.1));
}

/// `run`, dispatched as `how` says.
fn run_how(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], how: Dispatch) {
    let k = cx.kernel("fx_masked", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    cx.dispatch(&k, &all, how);
}

/// Scratch labels of the pyramids' levels (a 4K frame has 12).
const PYRAMID: [&str; 16] = ["pyr1", "pyr2", "pyr3", "pyr4", "pyr5", "pyr6", "pyr7", "pyr8", "pyr9", "pyr10",
                             "pyr11", "pyr12", "pyr13", "pyr14", "pyr15", "pyr16"];
const PULL: [&str; 16] = ["pull0", "pull1", "pull2", "pull3", "pull4", "pull5", "pull6", "pull7", "pull8", "pull9",
                          "pull10", "pull11", "pull12", "pull13", "pull14", "pull15"];

/// cv2.pyrDown of plane `src` (w x h) into `dst`, ((w + 1) / 2, (h + 1) / 2), its size.
pub fn pyr_down(cx: &mut Ctx, src: &wgpu::Buffer, (w, h): (u32, u32), dst: &wgpu::Buffer) -> (u32, u32) {
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    let (x, y) = crate::gpu::tiles(dw, dh);
    pyr_down_how(cx, src, (w, h), dst, Dispatch::Groups(x, y))
}

/// `pyr_down`, dispatched as `how` says.
fn pyr_down_how(cx: &mut Ctx, src: &wgpu::Buffer, (w, h): (u32, u32), dst: &wgpu::Buffer, how: Dispatch) -> (u32, u32) {
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    run_how(cx, "pyr_down", Params { sw: w, sh: h, ..Params::size(dw, dh) }, &[(1, src), (2, dst)], how);
    (dw, dh)
}

/// mkmmd.remap.Pyramid: plane `pic` (w x h) and its copies halved again and again, as many as level `want` needs (no
/// more once one is 2 px or less on a side): [(plane, size)], the picture itself first.
pub fn pyramid(cx: &mut Ctx, pic: &wgpu::Buffer, (w, h): (u32, u32), want: u32) -> Vec<(wgpu::Buffer, (u32, u32))> {
    let mut out = vec![(pic.clone(), (w, h))];
    while out.len() as u32 <= want && out.len() <= PYRAMID.len() {
        let (prev, size) = out.last().unwrap().clone();
        if size.0.min(size.1) <= 2 {
            break;
        }
        let next = cx.scratch(PYRAMID[out.len() - 1], plane_bytes(size.0.div_ceil(2), size.1.div_ceil(2)));
        let s = pyr_down(cx, &prev, size, &next);
        out.push((next, s));
    }
    out
}

/// The box of the pixels of plane `cov` whose .x is over `cut`, left on the GPU in `b` (4 u32: w - x0, h - y0, x1, y1;
/// all 0 for none), for kernels that read it.
pub fn bbox_into(cx: &mut Ctx, cov: &wgpu::Buffer, cut: f32, b: &wgpu::Buffer) {
    let (w, h) = (cx.w, cx.h);
    cx.enc.clear_buffer(b, 0, Some(16));
    run(cx, "bbox", Params { a: cut, ..Params::size(w, h) }, &[(1, cov), (7, b)], crate::gpu::tiles(w, h));
}

/// The box (x, y, w, h) of the pixels of coverage `cov` (single floats, the frame's) over HOLE_CUT
/// (cv2.boundingRect), read back; None when there are none. A figure mask's sticker takes it: the box of the figure as
/// the remap left it, which only the GPU has (the sticker's crop, its shadow's shrink, offset and blur follow from it).
pub fn bbox(cx: &mut Ctx, cov: &wgpu::Buffer) -> Result<Option<(u32, u32, u32, u32)>> {
    let (w, h) = (cx.w, cx.h);
    let b = cx.scratch("mk_box", 16);
    cx.enc.clear_buffer(&b, 0, Some(16));
    run(cx, "bbox1", Params { a: HOLE_CUT, ..Params::size(w, h) }, &[(8, cov), (7, &b)], crate::gpu::tiles(w, h));
    let got: Vec<u32> = bytemuck::pod_collect_to_vec(&cx.read_back(&b, 16)?);
    if got[2] == 0 {
        return Ok(None);
    }
    let (x0, y0) = (w - got[0], h - got[1]);
    Ok(Some((x0, y0, got[2] - x0, got[3] - y0)))
}

/// The box (x, y, w, h) of the pixels of a figure layer (its decoded file) whose coverage is over HOLE_CUT, as `bbox`
/// finds it in the layer unpacked, from the layer's Cover; None when there are none.
pub fn figure_box(cx: &Ctx, figure: &Decoded) -> Result<Option<(u32, u32, u32, u32)>> {
    let cover = figure.cover().context("a mask's figure layer not read as a figure")?;
    let Some(t) = first_over(cx.levels, HOLE_CUT) else { return Ok(None) };
    Ok(cover.bbox(t).map(|[x0, y0, x1, y1]| (x0, y0, x1 - x0, y1 - y0)))
}

/// mkmmd.remap.pushpull: the picture `img` with its colours where `hole` (.x over HOLE_CUT) is spread smoothly in from
/// the rest: the known colours and their weights halved down a pyramid, each level's gaps filled from the one below on
/// the way back up. The plane it is in (full size). Every pass runs only when the flag `lost` (fill_lost's, its first
/// word) is set: a `gate` pass turns it into their indirect dispatches, all their workgroups or none.
fn pushpull(cx: &mut Ctx, img: &wgpu::Buffer, hole: &wgpu::Buffer, lost: &wgpu::Buffer) -> wgpu::Buffer {
    let (w, h) = (cx.w, cx.h);
    let n = plane_bytes(w, h);
    // the levels' sizes; the passes' workgroups in their order: `known`, a pyrDown into each level past the first,
    // `pull_top`, then a resize and a `pull_up` on each level on the way back up
    let mut sizes = vec![(w, h)];
    while sizes.len() < PULL.len() {
        let (lw, lh) = *sizes.last().unwrap();
        if lw.min(lh) <= 2 {
            break;
        }
        sizes.push((lw.div_ceil(2), lh.div_ceil(2)));
    }
    let top = *sizes.last().unwrap();
    let order: Vec<(u32, u32)> = sizes.iter().copied().chain([top])
        .chain(sizes.iter().rev().skip(1).flat_map(|&s| [s, s]))
        .collect();
    let grids: Vec<u32> = order.iter().flat_map(|&(lw, lh)| {
        let (x, y) = crate::gpu::tiles(lw, lh);
        [x, y]
    }).collect();
    let args = cx.dispatches("pull_args", order.len() as u64);
    let g = cx.table("pull_grids", &grids);
    run(cx, "gate", Params { flag: order.len() as u32, ..Params::size(1, 1) }, &[(7, lost), (11, &g), (12, &args)],
        (order.len().div_ceil(64) as u32, 1));
    let mut at = 0u64;
    let mut next = || {
        at += 1;
        at - 1
    };
    let mut levels: Vec<(wgpu::Buffer, (u32, u32))> = Vec::new();
    let first = cx.scratch(PULL[0], n);
    run_how(cx, "known", Params { a: HOLE_CUT, ..Params::size(w, h) }, &[(1, img), (2, &first), (8, hole)],
            Dispatch::Indirect(&args, next()));
    levels.push((first, (w, h)));
    for (i, &s) in sizes.iter().enumerate().skip(1) {
        let (prev, size) = levels.last().unwrap().clone();
        let lvl = cx.scratch(PULL[i], plane_bytes(s.0, s.1));
        pyr_down_how(cx, &prev, size, &lvl, Dispatch::Indirect(&args, next()));
        levels.push((lvl, s));
    }
    let cols = [cx.scratch("pull_col_a", n), cx.scratch("pull_col_b", n)];
    let up = cx.scratch("pull_up", n);
    let (top_plane, (tw, th)) = levels.last().unwrap().clone();
    run_how(cx, "pull_top", Params::size(tw, th), &[(1, &top_plane), (2, &cols[0])], Dispatch::Indirect(&args, next()));
    let (mut cur, mut size) = (0usize, (tw, th));
    for (lvl, s) in levels.iter().rev().skip(1) {
        figure::resize_linear_how(cx, &cols[cur], size, &up, *s, Dispatch::Indirect(&args, next()));
        run_how(cx, "pull_up", Params::size(s.0, s.1), &[(1, lvl), (2, &cols[1 - cur]), (3, &up)],
                Dispatch::Indirect(&args, next()));
        cur = 1 - cur;
        size = *s;
    }
    cols[cur].clone()
}

/// mkmmd.remap.fill into `out`: picture `img` with the place `hole` (.x) covers filled from the ground round it. What
/// pushpull spreads is worked out only when some pixel of the hole has no ground to mirror in, as the GPU finds
/// (`fill_lost`); fill_mix reads it only at such a pixel.
fn fill(cx: &mut Ctx, img: &wgpu::Buffer, hole: &wgpu::Buffer, out: &wgpu::Buffer) {
    let (w, h) = (cx.w, cx.h);
    let lr = cx.scratch("mk_lr", w as u64 * h as u64 * 8);
    let p = Params { a: HOLE_CUT, ..Params::size(w, h) };
    let k = cx.kernel("fx_masked", SRC, "fill_scan");
    let u = cx.uniform(&p);
    cx.pass(&k, &[(0, &u), (8, hole), (6, &lr)], (h.min(65535), h.div_ceil(65535)));
    let lost = cx.scratch("mk_lost", 16);
    cx.enc.clear_buffer(&lost, 0, None);
    run(cx, "fill_lost", p, &[(8, hole), (6, &lr), (7, &lost)], crate::gpu::tiles(w, h));
    let spread = pushpull(cx, img, hole, &lost);
    run(cx, "fill_mix", p, &[(1, img), (2, out), (8, hole), (4, &spread), (6, &lr)], crate::gpu::tiles(w, h));
}

/// framefx._grow of coverage `cov` (a plane of its own, `at`) by `width` px into `out`, through scratch planes.
fn grow(cx: &mut Ctx, cov: &wgpu::Buffer, out: &wgpu::Buffer, at: &Area, width: f64) {
    let (t1, t2) = (cx.scratch("grow_t1", at.bytes()), cx.scratch("grow_t2", at.bytes()));
    plane::grow(cx, cov, out, &t1, &t2, at, width);
}

/// framefx._grow(m, reach) of the figure `m` (the frame's coverage; its box `b`, figure_box's) into `hole`: worked
/// out only over the box grown by how far the ring and its blur reach (zero past it, reflected at the frame's edges as
/// over the whole frame), the coverage itself elsewhere.
fn grow_near(cx: &mut Ctx, m: &wgpu::Buffer, (x, y, bw, bh): (u32, u32, u32, u32), reach: f64, hole: &wgpu::Buffer) {
    let (w, h) = (cx.w, cx.h);
    let r = reach.max(0.5);
    let kr = if r >= 3.0 { figure::gaussian_size(0.25 * r) as u32 / 2 } else { 0 };
    let e = (r + 0.5).ceil() as u32 + kr + 2;
    let (x0, y0) = (x.saturating_sub(e), y.saturating_sub(e));
    let (x1, y1) = ((x + bw + e).min(w), (y + bh + e).min(h));
    let near = Area::whole(x1 - x0, y1 - y0);
    let (c, g) = (cx.scratch("gr_c", near.bytes()), cx.scratch("gr_g", near.bytes()));
    let p = Params { sw: w, sh: h, x0, y0, ..Params::size(near.w, near.h) };
    run(cx, "crop", p, &[(8, m), (9, &c)], crate::gpu::tiles(near.w, near.h));
    grow(cx, &c, &g, &near, reach);
    cx.enc.copy_buffer_to_buffer(m, 0, hole, 0, Area::whole(w, h).bytes());
    run(cx, "uncrop", p, &[(8, &g), (9, hole)], crate::gpu::tiles(near.w, near.h));
}

/// mkmmd.remap.shade: how much of each pixel of a box (w x h) the drop shadow of coverage `cov` darkens, before its
/// alpha: `cov` moved, its edge repeated, blurred, a wide blur worked out at a quarter of the size. The plane it is in.
fn shade(cx: &mut Ctx, cov: &wgpu::Buffer, (w, h): (u32, u32), shadow: &Shadow) -> wgpu::Buffer {
    let q = if shadow.blur >= 6.0 && w.min(h) >= 64 { 4 } else { 1 };
    let small = Area::whole(w / q, h / q);
    let src = if q == 1 {
        cov.clone()
    } else {
        let s = cx.scratch("shade_small", small.bytes());
        plane::resize_area(cx, cov, (w, h), &s, (small.w, small.h));
        s
    };
    let k = small.w as f64 / w as f64;
    let moved = cx.scratch("shade_moved", small.bytes());
    plane::shift(cx, &src, &moved, &small, shadow.offset[0] * k, shadow.offset[1] * k);
    if shadow.blur * k > 0.3 {
        let tmp = cx.scratch("shade_tmp", small.bytes());
        plane::blur(cx, &moved, &moved, &tmp, &small, shadow.blur * k);
    }
    if q == 1 {
        return moved;
    }
    let out = cx.scratch("shade_out", Area::whole(w, h).bytes());
    plane::resize_linear(cx, &moved, (small.w, small.h), &out, (w, h));
    out
}

/// mkmmd.remap.sticker: plane `dst` (in place) with a figure laid on it, its premultiplied picture in `prem` (rgb) and
/// its coverage in `cov` (the frame's, single floats), over an `edge` grown round it, the two casting a `shadow`; only
/// the box the figure, its edge and its shadow reach is worked on. `b` the box of the coverage, None when only the GPU
/// has it (read back: bbox).
pub fn sticker(cx: &mut Ctx, dst: &wgpu::Buffer, prem: &wgpu::Buffer, cov: &wgpu::Buffer,
               b: Option<(u32, u32, u32, u32)>, edge: Option<&Edge>, shadow: Option<&Shadow>) -> Result<()> {
    let (w, h) = (cx.w, cx.h);
    let b = match b {
        Some(b) => Some(b),
        None => bbox(cx, cov)?,
    };
    let Some((x, y, bw, bh)) = b else {
        return Ok(());
    };
    let margin = 2.0 + edge.map_or(0.0, |e| e.width);
    let reach = margin + shadow.map_or(0.0, |s| s.offset[0].hypot(s.offset[1]) + 3.0 * s.blur);
    let x0 = ((x as f64 - reach) as i64).max(0) as u32;
    let y0 = ((y as f64 - reach) as i64).max(0) as u32;
    let x1 = ((x as f64 + bw as f64 + reach).ceil() as i64).min(w as i64) as u32;
    let y1 = ((y as f64 + bh as f64 + reach).ceil() as i64).min(h as i64) as u32;
    let (cw, ch) = (x1 - x0, y1 - y0);
    let crop = Area::whole(cw, ch);
    let c = cx.scratch("st_c", crop.bytes());
    run(cx, "crop", Params { sw: w, sh: h, x0, y0, ..Params::size(cw, ch) }, &[(8, cov), (9, &c)],
        crate::gpu::tiles(cw, ch));
    let ring = match edge {
        Some(e) if e.width > 0.0 => {
            let r = cx.scratch("st_ring", crop.bytes());
            grow(cx, &c, &r, &crop, e.width);
            Some(r)
        }
        _ => None,
    };
    let shaded = match shadow {
        Some(s) if s.alpha > 0.0 => Some(shade(cx, ring.as_ref().unwrap_or(&c), (cw, ch), s)),
        _ => None,
    };
    let mut p = Params { sw: w, sh: h, x0, y0, ..Params::size(cw, ch) };
    if let (Some(s), Some(_)) = (shadow, &shaded) {
        p.flag |= 1;
        p.a = s.alpha as f32;
        p.colour = [s.color[0], s.color[1], s.color[2], 0.0];
    }
    if let (Some(e), Some(_)) = (edge, &ring) {
        p.flag |= 2;
        p.colour2 = [e.color[0], e.color[1], e.color[2], 0.0];
    }
    let sh = shaded.unwrap_or_else(|| c.clone());
    let rg = ring.unwrap_or_else(|| c.clone());
    run(cx, "sticker", p, &[(1, prem), (2, dst), (8, &c), (10, &sh), (13, &rg)], crate::gpu::tiles(cw, ch));
    Ok(())
}

/// RemapFx._remapped into `cx.dst`: the picture `cx.src` through `remap` (a plane remapped into another, 3 or 4
/// channels as given) as `mask` says, the figure layer `figure` (its decoded file) cut out and laid back. With no
/// pixel of the figure over HOLE_CUT nothing is filled nor laid back (as the CPU path: the picture remapped under a
/// ground mask, as it is under a figure mask).
pub fn remapped(cx: &mut Ctx, mask: &Masking, figure: &Arc<Decoded>,
                remap: &mut dyn FnMut(&mut Ctx, &wgpu::Buffer, &wgpu::Buffer) -> Result<()>) -> Result<()> {
    let (w, h) = (cx.w, cx.h);
    let n = plane_bytes(w, h);
    let frame = Area::whole(w, h);
    let (img, out) = (cx.src.clone(), cx.dst.clone());
    cx.mark("masked start");
    let ground_mask = mask.mask == "ground";
    let Some(b) = figure_box(cx, figure)? else {
        if ground_mask {
            remap(cx, &img, &out)?;
        } else {
            figure::copy(cx, &img, &out, w, h);
        }
        return Ok(());
    };
    let packed = cx.upload(&mask.input(), figure);
    let m = cx.scratch("mk_m", frame.bytes());
    plane::unpack(cx, &packed, &m, &frame);
    let prem = cx.scratch("mk_prem", n);
    plane::premultiply(cx, &img, &frame, &m, &prem, &frame);
    cx.mark("figure");
    let ground = cx.scratch("mk_ground", n);
    match (ground_mask, mask.back) {
        (false, Some(c)) => figure::fill(cx, &ground, w, h, c),
        _ => {
            let hole = cx.scratch("mk_hole", frame.bytes());
            grow_near(cx, &m, b, mask.reach, &hole);
            cx.mark("hole grown");
            fill(cx, &img, &hole, &ground);
            cx.mark("ground filled");
        }
    }
    if ground_mask {
        remap(cx, &ground, &out)?;
        cx.mark("remapped");
        let done = sticker(cx, &out, &prem, &m, Some(b), mask.edge.as_ref(), mask.shadow.as_ref());
        cx.mark("sticker");
        return done;
    }
    let both = cx.scratch("mk_both", n);
    remap(cx, &prem, &both)?;
    let cov = cx.scratch("mk_cov", frame.bytes());
    run(cx, "cover_of", Params::size(w, h), &[(1, &both), (9, &cov)], crate::gpu::tiles(w, h));
    figure::copy(cx, &ground, &out, w, h);
    sticker(cx, &out, &both, &cov, None, mask.edge.as_ref(), mask.shadow.as_ref())
}
