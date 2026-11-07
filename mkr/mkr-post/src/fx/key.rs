//! key: a colour of the picture keyed out and pushed too far (mkmmd.colorkey; FrameFx._key), what lies behind laid in
//! plane by plane (shaders/fx_key.wgsl). mkmmd.gpufx.key works out the tolerances (the frame's, and each trail frame's at
//! its own), the float32 constants and what shows through; here each pixel's distance from the key colour becomes a
//! matte for each plane, the `burn` trail's frames (the cut's frames before, read from their PNGs) keyed at their own
//! tolerance and kept as bytes, weighed by their age and the strongest taken, then the back laid in, the spill's stain,
//! the mask and the mix.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::plane_bytes;
use super::footage::Clip;
use super::pixelsort::Mask;
use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_key.wgsl");

/// What shows through the key.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum Back {
    /// The picture as it reaches the op (a `delay` before the cut has a frame).
    Picture,
    Colour { color: [f32; 3] },
    /// The cut's frame `delay` before, a PNG.
    Png { path: String },
    /// A clip's frame decoded and fitted to cover (`Key::clip`).
    Clip,
}

/// A cut frame of the `burn` trail: keyed at its own tolerance `tol` (the spread's centre there `centre`), the matte's
/// bytes weighed by `factor` (fade ** j / 255).
#[derive(Debug, Clone, Deserialize)]
pub struct Trail {
    pub path: String,
    pub tol: f32,
    pub centre: [f32; 2],
    pub factor: f32,
}

/// The spill's stain: its colour and `amount`, fading out at `edge` (soft + reach) past the tolerance over 1 / `inv`.
#[derive(Debug, Clone, Deserialize)]
pub struct Spill {
    pub color: [f32; 3],
    pub amount: f32,
    pub edge: f32,
    pub inv: f32,
}

/// The distance in `space` (0 rgb, 1 chroma, 2 luma) from `color`; `ref` its luma, Cb and Cr (the scales applied) and
/// the scales' reciprocals; each plane keyed within `tol` (less `spread` x the pixel's distance from `centre`, in frame
/// heights) + its `split` and falling to 0 by `soft` more (`inv` = 1 / soft; None: a hard step), crushed into `bands`
/// steps of `bscale` (0 bands: smooth); what shows through times `gain`; `trail` the frames before; the strength `s`
/// and 1 - it.
#[derive(Debug, Clone, Deserialize)]
pub struct Key {
    pub space: u32,
    pub color: [f32; 3],
    #[serde(rename = "ref")]
    pub refs: [f32; 5],
    pub tol: f32,
    pub centre: [f32; 2],
    pub split: [f32; 3],
    pub soft: f32,
    pub spread: f32,
    #[serde(default)]
    pub inv: Option<f32>,
    pub bands: u32,
    pub bscale: f32,
    pub gain: [f32; 3],
    pub back: Back,
    #[serde(default)]
    pub clip: Option<Clip>,
    pub trail: Vec<Trail>,
    #[serde(default)]
    pub spill: Option<Spill>,
    #[serde(default)]
    pub mask: Option<Mask>,
    pub s: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    space: u32,
    bands: u32,
    hard: u32,
    masked: u32,
    back: u32,
    spill: u32,
    mixed: u32,
    pad0: [u32; 3],
    inv: f32,
    bscale: f32,
    factor: f32,
    amount: f32,
    edge: f32,
    sinv: f32,
    s: f32,
    s1: f32,
    icr: f32,
    tol: f32,
    soft: f32,
    spread: f32,
    cx: f32,
    cy: f32,
    pad1: [f32; 2],
    split: [f32; 4],
    color: [f32; 4],
    refs: [f32; 4],
    gain: [f32; 4],
    flat: [f32; 4],
    stain: [f32; 4],
}

fn v4(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 0.0]
}

impl Key {
    /// What shows through as an input: the delay's frame or the clip's.
    fn back_input(&self) -> Option<Input> {
        match &self.back {
            Back::Png { path } => Some(Input::Png(path.clone(), Kind::Picture)),
            Back::Clip => self.clip.as_ref().map(|c| Input::Rgb { path: c.path.clone(), size: c.size, keep: c.keep }),
            _ => None,
        }
    }
}

impl Effect for Key {
    /// The trail's frames, then what shows through (a delay's frame or a clip's), then the mask's figure.
    fn inputs(&self) -> Vec<Input> {
        let mut v: Vec<Input> = self.trail.iter().map(|t| Input::Png(t.path.clone(), Kind::Picture)).collect();
        v.extend(self.back_input());
        v.extend(self.mask.iter().map(|m| Input::Png(m.figure.clone(), Kind::Luma)));
        v
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        if let (Back::Clip, Some(c)) = (&self.back, &self.clip) {
            if c.size != [w, h] {
                bail!("key: a clip frame of {}x{} for a {w}x{h} frame", c.size[0], c.size[1]);
            }
        } else if matches!(self.back, Back::Clip) {
            bail!("key: a clip back without its frame");
        }
        let all = self.inputs();
        let (src, dst, lut) = (cx.src.clone(), cx.dst.clone(), cx.lut.clone());
        let [yc, kb, kr, icb, icr] = self.refs;
        let mut p = Params {
            w, h, space: self.space, bands: self.bands, hard: self.inv.is_none() as u32,
            masked: self.mask.as_ref().map_or(0, |m| if m.inside { 1 } else { 2 }), mixed: (self.s[0] < 1.0) as u32,
            inv: self.inv.unwrap_or(0.0), bscale: self.bscale, s: self.s[0], s1: self.s[1], icr, tol: self.tol,
            soft: self.soft, spread: self.spread, cx: self.centre[0], cy: self.centre[1], split: v4(self.split), color: v4(self.color), refs: [yc, kb, kr, icb], gain: v4(self.gain),
            ..Default::default()
        };
        let acc = cx.scratch("key.acc", plane_bytes(w, h));
        let none = cx.table("pic", &[0u32]);
        let u = cx.uniform(&p);
        let k = cx.kernel("fx_key", SRC, "matte_now");
        cx.pass(&k, &[(0, &u), (1, &src), (3, &acc)], cx.tiles());
        let k = cx.kernel("fx_key", SRC, "matte_trail");
        for (n, t) in self.trail.iter().enumerate() {
            let pic = cx.upload(&all[n], &inputs[n]);
            let q = Params { tol: t.tol, cx: t.centre[0], cy: t.centre[1], factor: t.factor, ..p };
            let u = cx.uniform(&q);
            cx.pass(&k, &[(0, &u), (3, &acc), (4, &pic), (6, &lut)], cx.tiles());
        }
        let mut at = self.trail.len();
        let mut stain = [0.0; 4];
        let pixels = match &self.back {
            Back::Colour { color } => {
                p.flat = v4(*color);
                none.clone()
            }
            Back::Picture => {
                p.back = 2;
                none.clone()
            }
            Back::Png { .. } | Back::Clip => {
                p.back = 1;
                at += 1;
                cx.upload(&all[at - 1], &inputs[at - 1])
            }
        };
        if let Some(s) = &self.spill {
            p.spill = 1;
            p.amount = s.amount;
            p.edge = s.edge;
            p.sinv = s.inv;
            stain = v4(s.color);
        }
        p.stain = stain;
        let fig = if self.mask.is_some() { cx.upload(&all[at], &inputs[at]) } else { cx.table("fig", &[0u32]) };
        let u = cx.uniform(&p);
        let k = cx.kernel("fx_key", SRC, "lay");
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst), (3, &acc), (4, &pixels), (5, &fig), (6, &lut)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
