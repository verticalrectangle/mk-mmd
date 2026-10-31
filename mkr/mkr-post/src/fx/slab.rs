//! slab (mkmmd.framefx.slab_cover, FrameFx._slab): Persona's slanted bar slamming in and flung off, drawn at the
//! places it stands at over its shutter (shaders/fx_slab.wgsl); mkmmd.gpufx.slab works them out with FrameFx.slab_at.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect};

const SRC: &str = include_str!("../shaders/fx_slab.wgsl");

/// float32 as numpy uses them: the places (cx, cy, and the box [x0, x1, y0, y1] slab_cover sums over), the turn's
/// cosine and sine, the slant's tangent and cosine, the half height and width, the opacity and the colour.
#[derive(Debug, Clone, Deserialize)]
pub struct Slab {
    pub places: Vec<([f32; 2], [i32; 4])>,
    pub ca: f32,
    pub sa: f32,
    pub cs: f32,
    pub t: f32,
    pub hh: f32,
    pub hw: f32,
    /// How many places the coverage is averaged over (those off the frame count).
    pub n: u32,
    pub alpha: f32,
    pub colour: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    n: u32,
    pad0: u32,
    ca: f32,
    sa: f32,
    cs: f32,
    t: f32,
    hh: f32,
    hw: f32,
    alpha: f32,
    pad1: f32,
    colour: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Place {
    c: [f32; 2],
    pad: [f32; 2],
    bx: [i32; 4],
}

impl Effect for Slab {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        // the places off the frame add nothing but still count in the average: they stay in the table, boxes empty
        let mut places: Vec<Place> = self.places.iter().map(|(c, b)| Place { c: *c, pad: [0.0; 2], bx: *b }).collect();
        while places.len() < self.n as usize {
            places.push(Place { c: [0.0; 2], pad: [0.0; 2], bx: [0, 0, 0, 0] });
        }
        if places.is_empty() {
            places.push(Place { c: [0.0; 2], pad: [0.0; 2], bx: [0, 0, 0, 0] });
        }
        let k = cx.kernel("fx_slab", SRC, "slab");
        let t = cx.table("places", &places);
        let c = self.colour;
        let p = cx.uniform(&Params {
            w: cx.w, h: cx.h, n: self.n, pad0: 0, ca: self.ca, sa: self.sa, cs: self.cs, t: self.t, hh: self.hh,
            hw: self.hw, alpha: self.alpha, pad1: 0.0, colour: [c[0], c[1], c[2], 0.0],
        });
        let pic = cx.src.clone();
        let groups = cx.tiles();
        cx.pass(&k, &[(0, &p), (1, &pic), (2, &t)], groups);
        Ok(Drawn::Src)
    }
}
