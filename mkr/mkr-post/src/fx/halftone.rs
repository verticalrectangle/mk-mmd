//! halftone: comic print (mkmmd.framefx.halftone; FrameFx._halftone): Ben-Day dots in the darks, the red and blue
//! plates off register, mixed in by the strength (shaders/fx_halftone.wgsl). Python (mkmmd.gpufx.halftone) gives the
//! constants.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, BORDER_REPLICATE, plane_bytes};
use super::warp::warp;
use super::{Ctx, Decoded, Drawn, Effect};

const SRC: &str = include_str!("../shaders/fx_halftone.wgsl");

#[derive(Debug, Clone, Deserialize)]
pub struct Halftone {
    /// float32 of the level, cos and sin of the angle, the cell (px), DOT x cell and the lift.
    pub k: [f32; 6],
    pub ink: [f32; 3],
    /// The red plate's offset (px; the blue's the other way), None: on register.
    #[serde(default)]
    pub offset: Option<[f64; 2]>,
    /// The strength and 1 - it, float32.
    pub s: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    shifted: u32,
    mix: u32,
    level: f32,
    ca: f32,
    sa: f32,
    cell: f32,
    dot: f32,
    lift: f32,
    s: f32,
    s1: f32,
    ink: [f32; 4],
}

impl Effect for Halftone {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let [level, ca, sa, cell, dot, lift] = self.k;
        let p = Params {
            w, h, shifted: self.offset.is_some() as u32, mix: (self.s[0] < 1.0) as u32, level, ca, sa, cell, dot, lift,
            s: self.s[0], s1: self.s[1], ink: [self.ink[0], self.ink[1], self.ink[2], 0.0],
        };
        let u = cx.uniform(&p);
        let k = cx.kernel("fx_halftone", SRC, "halftone");
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst)], cx.tiles());
        let n = plane_bytes(w, h);
        let red = cx.scratch("halftone.red", n);
        let blue = cx.scratch("halftone.blue", n);
        if let Some([ox, oy]) = self.offset {
            for (plane, (dx, dy)) in [(&red, (ox, oy)), (&blue, (-ox, -oy))] {
                let m = figure::inverse(&[1.0, 0.0, dx as f32, 0.0, 1.0, dy as f32]);
                warp(cx, &dst, (w, h), plane, (w, h), &m, BORDER_REPLICATE, [0.0; 3]);
            }
        }
        let k = cx.kernel("fx_halftone", SRC, "halftone_plates");
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst), (3, &red), (4, &blue)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
