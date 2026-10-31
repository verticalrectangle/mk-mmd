//! trails (mkmmd.mandala.MandalaFx._trails, blend, _box, _shifted): rainbow after-images of the figure, each copy the
//! figure's coverage some frames before (moved, its picture too, when the trail steps) laid in its ramp colour over
//! the box it covers, then the figure of this frame laid back on top (shaders/fx_trails.wgsl). mkmmd.gpufx.trails works
//! the copies out with MandalaFx.trails_at.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, plane_bytes};
use super::masked;
use super::warp::warp;
use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_trails.wgsl");
/// mkmmd.mandala._box's threshold.
const BOX_CUT: f32 = 1.0 / 512.0;
const BORDER_CONSTANT: u32 = 0;
const BORDER_REPLICATE: u32 = 1;

/// A copy: its figure layer, the cut's frame of its picture when its colour is mixed with it (`tint` under 1 and the
/// frame rendered), its opacity times the strength and ramp colour (float32), and how far it is moved (px).
#[derive(Debug, Clone, Deserialize)]
pub struct Copy {
    pub figure: String,
    #[serde(default)]
    pub frame: Option<String>,
    pub alpha: f32,
    pub colour: [f32; 3],
    #[serde(default)]
    pub shift: Option<[f64; 2]>,
}

/// The copies oldest first, `blend` 0 over, 1 screen, 2 add, `tint` (float32), and this frame's figure layer.
#[derive(Debug, Clone, Deserialize)]
pub struct Trails {
    pub copies: Vec<Copy>,
    pub blend: u32,
    pub tint: f32,
    #[serde(default)]
    pub figure: Option<String>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    mode: u32,
    tinted: u32,
    alpha: f32,
    tint: f32,
    pad: [f32; 2],
    colour: [f32; 4],
}

/// Plane `out` (w x h) of a packed 8-bit input, moved `shift` px as cv2.warpAffine moves it (the inverse of the
/// float32 map, INTER_LINEAR) with `border`, through `tmp`.
fn laid(cx: &mut Ctx, packed: &wgpu::Buffer, shift: Option<[f64; 2]>, border: u32, out: &wgpu::Buffer,
        tmp: &wgpu::Buffer) {
    let (w, h) = (cx.w, cx.h);
    match shift {
        None => figure::unpack(cx, packed, out, w, h),
        Some([dx, dy]) => {
            figure::unpack(cx, packed, tmp, w, h);
            let m = figure::inverse(&[1.0, 0.0, dx as f32, 0.0, 1.0, dy as f32]);
            warp(cx, tmp, (w, h), out, (w, h), &m, border, [0.0; 3]);
        }
    }
}

impl Effect for Trails {
    fn inputs(&self) -> Vec<Input> {
        let mut out = Vec::new();
        for c in &self.copies {
            out.push(Input::Png(c.figure.clone(), Kind::Luma));
            if let Some(f) = &c.frame {
                out.push(Input::Png(f.clone(), Kind::Picture));
            }
        }
        if let Some(f) = &self.figure {
            out.push(Input::Png(f.clone(), Kind::Luma));
        }
        out
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let n = plane_bytes(w, h);
        let (img, out) = (cx.src.clone(), cx.dst.clone());
        figure::copy(cx, &img, &out, w, h);
        let (cov, pic, tmp) = (cx.scratch("tr_cov", n), cx.scratch("tr_pic", n), cx.scratch("tr_tmp", n));
        let b = cx.scratch("tr_box", 16);
        let all = self.inputs();
        let mut at = 0;
        for c in &self.copies {
            let fig = cx.upload(&all[at], &inputs[at]);
            at += 1;
            laid(cx, &fig, c.shift, BORDER_CONSTANT, &cov, &tmp);
            let mut tinted = 0;
            if c.frame.is_some() {
                let frame = cx.upload(&all[at], &inputs[at]);
                at += 1;
                laid(cx, &frame, c.shift, BORDER_REPLICATE, &pic, &tmp);
                tinted = 1;
            }
            masked::bbox_into(cx, &cov, BOX_CUT, &b);
            let p = cx.uniform(&Params {
                w, h, mode: self.blend, tinted, alpha: c.alpha, tint: self.tint,
                colour: [c.colour[0], c.colour[1], c.colour[2], 0.0], ..Default::default()
            });
            let k = cx.kernel("fx_trails", SRC, "copy");
            cx.pass(&k, &[(0, &p), (1, &out), (2, &cov), (3, &pic), (4, &b)], crate::gpu::tiles(w, h));
        }
        if self.figure.is_some() {
            let fig = cx.upload(&all[at], &inputs[at]);
            figure::unpack(cx, &fig, &cov, w, h);
            masked::bbox_into(cx, &cov, BOX_CUT, &b);
            let p = cx.uniform(&Params { w, h, ..Default::default() });
            let k = cx.kernel("fx_trails", SRC, "figure_back");
            cx.pass(&k, &[(0, &p), (1, &out), (2, &cov), (3, &img), (4, &b)], crate::gpu::tiles(w, h));
        }
        Ok(Drawn::Dst)
    }
}
