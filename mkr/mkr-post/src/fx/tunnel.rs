//! tunnel (mkmmd.remap.RemapFx._tunnel, tunnel, level_cover, warp; mkmmd.core.remap.tunnel_levels, tunnel_queue): the
//! frame nested in itself level after level about a centre, falling inward, each level's frame line drawn round it;
//! under a figure mask a queue of whole copies of the figure (fx/masked.rs), mixed in by the strength
//! (shaders/fx_tunnel.wgsl). mkmmd.gpufx.tunnel works the levels out.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure;
use super::masked::{self, Masking};
use super::warp::rows;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_tunnel.wgsl"));

/// One level as mkmmd.remap.tunnel draws it: its box [x0, y0, x1, y1] (the whole frame for the outermost nested one),
/// the pyramid levels it is warped from (level, float32 weight, the float32 map from the box's px to that level's),
/// level_cover's map `a` (float32), its scale `m` and opacity, and its frame line's colour when it has one.
#[derive(Debug, Clone, Deserialize)]
pub struct Level {
    pub bx: [u32; 4],
    pub lods: Vec<(u32, f32, [f32; 6])>,
    pub a: [f32; 6],
    pub m: f32,
    pub alpha: f32,
    #[serde(default)]
    pub colour: Option<[f32; 4]>,
}

/// `nest`: the frame's levels (else a figure's queue, far to near); `width` the frame lines' (px, float32; 0: none);
/// `wr`, `hr` the frame's w - 0.5 and h - 0.5 (float32); `mix` the strength when under 1.
#[derive(Debug, Clone, Deserialize)]
pub struct Tunnel {
    pub nest: bool,
    pub levels: Vec<Level>,
    pub width: f32,
    pub wr: f32,
    pub hr: f32,
    #[serde(default)]
    pub mix: Option<f64>,
    #[serde(default)]
    pub mask: Option<Masking>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    bw: u32,
    bh: u32,
    lw0: u32,
    lh0: u32,
    lw1: u32,
    lh1: u32,
    two: u32,
    first: u32,
    nest: u32,
    lined: u32,
    border: u32,
    pad0: u32,
    m00_0: f32,
    m10_0: f32,
    m00_1: f32,
    m10_1: f32,
    w0: f32,
    w1: f32,
    alpha: f32,
    m: f32,
    a: [f32; 6],
    width: f32,
    wr: f32,
    hr: f32,
    pad1: [f32; 3],
    col: [f32; 4],
}

const BORDER_CONSTANT: u32 = 0;
const BORDER_REFLECT: u32 = 2;

impl Tunnel {
    /// tunnel(pic, levels, ...): plane `pic` drawn as the tunnel into `dst`.
    fn remap(&self, cx: &mut Ctx, pic: &wgpu::Buffer, dst: &wgpu::Buffer) -> Result<()> {
        let (w, h) = (cx.w, cx.h);
        if self.levels.is_empty() {                       // nothing drawn: the picture as it is
            figure::copy(cx, pic, dst, w, h);
            return Ok(());
        }
        if !self.nest {
            cx.enc.clear_buffer(dst, 0, Some(figure::plane_bytes(w, h)));
        }
        let want = self.levels.iter().flat_map(|l| l.lods.iter().map(|d| d.0)).max().unwrap_or(0);
        let pyr = masked::pyramid(cx, pic, (w, h), want);
        let level = |j: u32| pyr[(j as usize).min(pyr.len() - 1)].clone();
        let k = cx.kernel("fx_tunnel", SRC, "level");
        for (n, lv) in self.levels.iter().enumerate() {
            let [x0, y0, x1, y1] = lv.bx;
            if x1 <= x0 || y1 <= y0 || x1 > w || y1 > h {
                bail!("tunnel: level box {:?} outside a {w}x{h} frame", lv.bx);
            }
            if lv.lods.is_empty() || lv.lods.len() > 2 {
                bail!("tunnel: {} pyramid levels, not 1 or 2", lv.lods.len());
            }
            let (bw, bh) = (x1 - x0, y1 - y0);
            let mut p = Params {
                w, h, x0, y0, bw, bh, first: (self.nest && n == 0) as u32, nest: self.nest as u32,
                lined: lv.colour.is_some() as u32, border: if self.nest { BORDER_REFLECT } else { BORDER_CONSTANT },
                alpha: lv.alpha, m: lv.m, a: lv.a, width: self.width, wr: self.wr, hr: self.hr,
                col: lv.colour.unwrap_or([0.0; 4]), ..Default::default()
            };
            let (j0, w0, m0) = lv.lods[0];
            let (l0, (lw0, lh0)) = level(j0);
            let r0 = cx.table("tunnel_rows", &rows(&m0, bh));
            (p.lw0, p.lh0, p.w0, p.m00_0, p.m10_0) = (lw0, lh0, w0, m0[0], m0[3]);
            let (mut l1, mut r1) = (l0.clone(), r0.clone());
            if let Some(&(j1, w1, m1)) = lv.lods.get(1) {
                let (b, (lw1, lh1)) = level(j1);
                l1 = b;
                r1 = cx.table("tunnel_rows", &rows(&m1, bh));
                (p.two, p.lw1, p.lh1, p.w1, p.m00_1, p.m10_1) = (1, lw1, lh1, w1, m1[0], m1[3]);
            }
            let u = cx.uniform(&p);
            cx.pass(&k, &[(0, &u), (1, &l0), (2, &l1), (3, dst), (4, &r0), (5, &r1)], crate::gpu::tiles(bw, bh));
        }
        Ok(())
    }
}

impl Effect for Tunnel {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Masking::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (img, out) = (cx.src.clone(), cx.dst.clone());
        match &self.mask {
            Some(mask) => masked::remapped(cx, mask, &inputs[0], &mut |cx, pic, dst| self.remap(cx, pic, dst))?,
            None => self.remap(cx, &img, &out)?,
        }
        if let Some(s) = self.mix {
            figure::mix(cx, &img, &out, cx.w, cx.h, s);
        }
        Ok(Drawn::Dst)
    }
}
