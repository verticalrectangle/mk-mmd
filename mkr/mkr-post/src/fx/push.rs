//! push (mkmmd.framefx.push): the picture zooming into a disc that shows another frame of the cut, the seam of a loop
//! (shaders/fx_push.wgsl); mkmmd.gpufx.push works the zoom out with FrameFx.push_at.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_push.wgsl"));

/// float32 as numpy uses them: `c` the zoom's centre in the picture, `p` the disc's centre on screen, `m` the frame's
/// middle, `z` the zoom, `s` the inner frame's scale, `rr` the disc's radius + 0.5; `inner` the frame inside and
/// `border` how it goes on past its edges (cv2.BORDER_REFLECT or cv2.BORDER_REPLICATE).
#[derive(Debug, Clone, Deserialize)]
pub struct Push {
    pub inner: String,
    pub c: [f32; 2],
    pub p: [f32; 2],
    pub m: [f32; 2],
    pub z: f32,
    pub s: f32,
    pub rr: f32,
    pub border: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    border: u32,
    pad0: u32,
    c: [f32; 2],
    p: [f32; 2],
    m: [f32; 2],
    z: f32,
    s: f32,
    rr: f32,
    pad1: [f32; 3],
}

impl Effect for Push {
    fn inputs(&self) -> Vec<Input> {
        vec![Input::Png(self.inner.clone(), Kind::Picture)]
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let k = cx.kernel("fx_push", SRC, "push");
        let inner = cx.upload(&self.inputs()[0], &inputs[0]);
        let p = cx.uniform(&Params {
            w: cx.w, h: cx.h, border: self.border, pad0: 0, c: self.c, p: self.p, m: self.m, z: self.z, s: self.s,
            rr: self.rr, pad1: [0.0; 3],
        });
        let (src, dst, lut) = (cx.src.clone(), cx.dst.clone(), cx.lut.clone());
        let groups = cx.tiles();
        cx.pass(&k, &[(0, &p), (1, &src), (2, &dst), (3, &inner), (4, &lut)], groups);
        Ok(Drawn::Dst)
    }
}
