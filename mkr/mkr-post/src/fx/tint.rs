//! tint: the Dark Hour's green over a stop (mkmmd.framefx.gradient_map; FrameFx._tint), a gradient map mixed in by the
//! strength (shaders/fx_tint.wgsl); Python (mkmmd.gpufx.tint) gives the strength and the colours.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect};

const SRC: &str = include_str!("../shaders/fx_tint.wgsl");

#[derive(Debug, Clone, Deserialize)]
pub struct Tint {
    pub dark: [f32; 3],
    pub color: [f32; 3],
    pub light: [f32; 3],
    /// The strength and 1 - it, float32 as numpy makes them.
    pub s: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
    s: f32,
    s1: f32,
    pad2: f32,
    pad3: f32,
    dark: [f32; 4],
    colour: [f32; 4],
    light: [f32; 4],
}

fn v4(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 0.0]
}

impl Effect for Tint {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let k = cx.kernel("fx_tint", SRC, "tint");
        let u = cx.uniform(&Params {
            w: cx.w, h: cx.h, pad0: 0, pad1: 0, s: self.s[0], s1: self.s[1], pad2: 0.0, pad3: 0.0,
            dark: v4(self.dark), colour: v4(self.color), light: v4(self.light),
        });
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
