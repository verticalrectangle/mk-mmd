//! pixelsort (mkmmd.picturefx.sort_key, sort_runs, PictureFx._pixelsort): every run of eligible pixels along a row or
//! a column sorted by luma, hue or saturation, a line a workgroup (shaders/fx_pixelsort.wgsl), mixed in by the
//! strength. mkmmd.gpufx.pixelsort works the strength, the key's quantisation and the mask out. The sort is chaotic in
//! its input (a float's last bits can move a key to the next 16-bit step and pixels along their run): it matches the
//! CPU's only on the same picture, so it is always the first op of its frame (mkmmd.postgpu.first), never chained
//! after other GPU effects.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure;
use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_pixelsort.wgsl");
/// The longest line a workgroup sorts (fx_pixelsort.wgsl's LINE).
pub const LINE: u32 = 4096;

/// A pixel sort's mask: the figure layer, and whether the sort works inside the figure (else on the ground).
#[derive(Debug, Clone, Deserialize)]
pub struct Mask {
    pub figure: String,
    pub inside: bool,
}

/// `axis` 0 rows, 1 columns; `by` 0 luma, 1 hue, 2 saturation; `top` the quantised key's largest value; `tail` the
/// first column OpenCV's RGB2HSV works out one by one; `lo`, `hi` float32; `mix` the strength when under 1.
#[derive(Debug, Clone, Deserialize)]
pub struct Pixelsort {
    pub axis: u32,
    pub by: u32,
    pub reverse: bool,
    pub top: u32,
    pub tail: u32,
    pub lo: f32,
    pub hi: f32,
    #[serde(default)]
    pub mask: Option<Mask>,
    #[serde(default)]
    pub mix: Option<f64>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    axis: u32,
    by: u32,
    reverse: u32,
    masked: u32,
    top: u32,
    tail: u32,
    lo: f32,
    hi: f32,
    zero: u32,
    pad: u32,
}

impl Effect for Pixelsort {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(|m| Input::Png(m.figure.clone(), Kind::Luma)).collect()
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (len, lines) = if self.axis == 0 { (w, h) } else { (h, w) };
        if len > LINE || self.top >= 1 << 20 {
            bail!("pixelsort: lines of {len} px (at most {LINE}) or keys up to {} (under 2^20)", self.top);
        }
        let keys = cx.scratch("ps_keys", w as u64 * h as u64 * 4);
        let fig = match (&self.mask, inputs.first()) {
            (Some(_), Some(d)) => cx.upload(&self.inputs()[0], d),
            _ => cx.table("fig", &[0u32]),
        };
        let masked = match &self.mask {
            None => 0,
            Some(m) if m.inside => 1,
            Some(_) => 2,
        };
        let p = cx.uniform(&Params {
            w, h, axis: self.axis, by: self.by, reverse: self.reverse as u32, masked, top: self.top, tail: self.tail,
            lo: self.lo, hi: self.hi, ..Default::default()
        });
        let (src, dst, lut) = (cx.src.clone(), cx.dst.clone(), cx.lut.clone());
        let k = cx.kernel("fx_pixelsort", SRC, "key");
        cx.pass(&k, &[(0, &p), (1, &src), (2, &keys), (4, &fig), (5, &lut)], crate::gpu::tiles(w, h));
        let k = cx.kernel("fx_pixelsort", SRC, "sort_lines");
        cx.pass(&k, &[(0, &p), (1, &src), (2, &keys), (3, &dst)], (lines.min(65535), lines.div_ceil(65535)));
        if let Some(s) = self.mix {
            figure::mix(cx, &src, &dst, w, h, s);
        }
        Ok(Drawn::Dst)
    }
}
