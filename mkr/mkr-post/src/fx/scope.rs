//! scope (mkmmd.framefx.draw_trace): a stem drawn as an oscilloscope trace and added to the picture with a glow
//! (shaders/fx_scope.wgsl). mkmmd.gpufx.scope draws the line on the CPU (OpenCV's antialiased polyline, mkmmd.framefx
//! .trace_mask) and hands over its coverage over the box it touches.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, plane_bytes};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_scope.wgsl");

/// `a` the line's coverage (float32, mask / 255 as numpy makes it) over box `bx` [x0, y0, w, h] (empty: no line drawn,
/// no file); `glow` (float32, 0: none) blurred `sigma` px (GLOW_SIGMA at the frame's scale); `opacity` and `colour`
/// float32.
#[derive(Debug, Clone, Deserialize)]
pub struct Scope {
    pub a: String,
    pub bx: [u32; 4],
    pub glow: f32,
    pub sigma: f64,
    pub opacity: f32,
    pub colour: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    glowed: u32,
    pad0: u32,
    opacity: f32,
    glow: f32,
    pad1: [f32; 2],
    colour: [f32; 4],
}

impl Effect for Scope {
    fn inputs(&self) -> Vec<Input> {
        if self.bx[2] * self.bx[3] == 0 {
            return Vec::new();
        }
        vec![Input::F32 { path: self.a.clone(), count: (self.bx[2] * self.bx[3]) as usize }]
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let [x0, y0, cw, ch] = self.bx;
        if x0 + cw > w || y0 + ch > h {
            bail!("scope: box {:?} outside a {w}x{h} frame", self.bx);
        }
        let n = plane_bytes(w, h);
        let crop = match inputs.first() {
            Some(d) => cx.upload(&self.inputs()[0], d),
            None => cx.table("crop", &[0.0f32]),
        };
        let (a, b) = (cx.scratch("sc_a", n), cx.scratch("sc_b", n));
        let glowed = self.glow > 0.0;
        let c = self.colour;
        let p = cx.uniform(&Params {
            w, h, x0, y0, cw, ch, glowed: glowed as u32, opacity: self.opacity, glow: self.glow,
            colour: [c[0], c[1], c[2], 0.0], ..Default::default()
        });
        let k = cx.kernel("fx_scope", SRC, "place");
        cx.pass(&k, &[(0, &p), (1, &crop), (2, &a)], crate::gpu::tiles(w, h));
        if glowed {
            let tmp = cx.scratch("sc_tmp", n);
            figure::blur(cx, &a, &b, &tmp, w, h, self.sigma);
        }
        let pic = cx.src.clone();
        let k = cx.kernel("fx_scope", SRC, "add");
        cx.pass(&k, &[(0, &p), (3, &b), (4, &pic), (5, &a)], crate::gpu::tiles(w, h));
        Ok(Drawn::Src)
    }
}
