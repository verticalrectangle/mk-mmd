//! mosh (mkmmd.framefx.mosh_step, mosh_carry, FrameFx._moshed, _mosh): a held picture carried along block moves from
//! frame to frame, mixed over the frame by the strength (shaders/fx_mosh.wgsl). Without a made-up flow, the cut's frame
//! before the span is held and carried along the cut's own motion: mkmmd.gpufx.mosh works the motion out on the CPU
//! (optical flow at a quarter size, its blocks); when what is held is of the frame before, only this frame's step is
//! taken, else the whole way from the start, as FrameFx._moshed does. With a flow, the picture as it reaches the op on
//! the span's first frame is held and carried its `carries` whole-pixel steps along the flow's fixed block moves on
//! every frame. The held picture stays here from one frame to the next (Ctx::held).

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, plane_bytes};
use super::pixelsort::Mask;
use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_mosh.wgsl"));

/// A made-up flow (mkmmd.framefx.mosh_flow): `moves`, its blocks' whole-pixel moves (gw x gh x 2 float32 kept on
/// disk); `fresh` on the span's first frame (the picture as it reaches the op is held), else what is held is of the
/// frame before; `carries` steps along the moves a frame.
#[derive(Debug, Clone, Deserialize)]
pub struct Flow {
    pub moves: String,
    pub fresh: bool,
    pub carries: u32,
}

/// Mosh `id` on `frame`. Without a `flow`: held from `start` (the cut's frame, a PNG; none: the picture as it reaches
/// the op on the span's way in) and carried along `steps` ((frame g, the blocks of the motion into g: gw x gh x 2
/// float32 kept on disk), every one since the start). With a `flow`, carried along its moves (and `steps` empty).
/// `cols` and `rows` the block each pixel's column and row read; mixed in by the strength `s` (0: the held picture
/// goes on, the frame is left as it is), with a mask by the figure.
#[derive(Debug, Clone, Deserialize)]
pub struct Mosh {
    pub id: u32,
    pub frame: i64,
    #[serde(default)]
    pub start: Option<String>,
    pub steps: Vec<(i64, String)>,
    #[serde(default)]
    pub flow: Option<Flow>,
    pub gw: u32,
    pub gh: u32,
    pub cols: Vec<u32>,
    pub rows: Vec<u32>,
    pub s: f64,
    #[serde(default)]
    pub mask: Option<Mask>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    gw: u32,
    masked: u32,
    s: f32,
    one_minus: f32,
    carries: u32,
    pad: f32,
}

impl Effect for Mosh {
    fn inputs(&self) -> Vec<Input> {
        let n = (self.gw * self.gh * 2) as usize;
        let mut out: Vec<Input> = self.start.iter().map(|p| Input::Png(p.clone(), Kind::Picture)).collect();
        out.extend(self.steps.iter().map(|(_, p)| Input::Kept { path: p.clone(), count: n }));
        out.extend(self.flow.iter().map(|f| Input::Kept { path: f.moves.clone(), count: n }));
        out.extend(self.mask.iter().map(|m| Input::Png(m.figure.clone(), Kind::Luma)));
        out
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        if self.cols.len() != w as usize || self.rows.len() != h as usize {
            bail!("mosh {}: {} columns and {} rows of blocks for a {w}x{h} frame", self.id, self.cols.len(),
                  self.rows.len());
        }
        let key = format!("mosh{}", self.id);
        let (held, at) = cx.held(&key, plane_bytes(w, h));
        let all = self.inputs();
        let p = Params {
            w, h, gw: self.gw, masked: self.mask.as_ref().map_or(0, |m| if m.inside { 1 } else { 2 }),
            s: self.s as f32, one_minus: (1.0 - self.s) as f32, carries: self.flow.as_ref().map_or(0, |f| f.carries),
            ..Default::default()
        };
        let u = cx.uniform(&p);
        let (cols, rows) = (cx.table("cols", &self.cols), cx.table("rows", &self.rows));
        let tmp = cx.scratch("mosh_tmp", plane_bytes(w, h));
        let mut k = self.start.is_some() as usize;
        if let Some(flow) = &self.flow {
            if flow.fresh {
                let src = cx.src.clone();
                figure::copy(cx, &src, &held, w, h);
            } else if at != Some(self.frame - 1) {
                bail!("mosh {} on frame {}: its flow goes on from frame {} but what is held is of {at:?}", self.id,
                      self.frame, self.frame - 1);
            }
            let moves = cx.upload(&all[k], &inputs[k]);
            k += 1;
            let carry = cx.kernel("fx_mosh", SRC, "carry");
            cx.pass(&carry, &[(0, &u), (1, &held), (2, &tmp), (3, &moves), (4, &cols), (5, &rows)],
                    crate::gpu::tiles(w, h));
            figure::copy(cx, &tmp, &held, w, h);
        } else {
            let on = at == Some(self.frame - 1);         // what is held is of the frame before: one step on
            if !on {
                match &self.start {
                    Some(_) => {
                        let p = cx.upload(&all[0], &inputs[0]);
                        figure::unpack(cx, &p, &held, w, h);
                    }
                    None => {
                        let src = cx.src.clone();
                        figure::copy(cx, &src, &held, w, h);
                    }
                }
            }
            let step = cx.kernel("fx_mosh", SRC, "step");
            for (g, _) in &self.steps {
                if !(on && *g < self.frame) {
                    let blocks = cx.upload(&all[k], &inputs[k]);
                    cx.pass(&step, &[(0, &u), (1, &held), (2, &tmp), (3, &blocks), (4, &cols), (5, &rows)],
                            crate::gpu::tiles(w, h));
                    figure::copy(cx, &tmp, &held, w, h);
                }
                k += 1;
            }
        }
        cx.hold(&key, self.frame);
        if self.s <= 0.0 {
            return Ok(Drawn::Src);
        }
        let fig = if self.mask.is_some() { cx.upload(&all[k], &inputs[k]) } else { cx.table("fig", &[0u32]) };
        let (img, dst, lut) = (cx.src.clone(), cx.dst.clone(), cx.lut.clone());
        let mix = cx.kernel("fx_mosh", SRC, "mix");
        cx.pass(&mix, &[(0, &u), (1, &held), (2, &dst), (6, &img), (7, &fig), (8, &lut)], crate::gpu::tiles(w, h));
        Ok(Drawn::Dst)
    }
}
