//! invert: the world flipped for a window or on hits (mkmmd.eralook.invert; EraFx._invert), its hues turned back
//! with `hue`, mixed in by its level and kept to its region (shaders/fx_invert.wgsl). Python (mkmmd.gpufx.invert) works
//! out the level (eased, thinned hits) and the strength.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::fields::Region;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_invert.wgsl");

#[derive(Debug, Clone, Deserialize)]
pub struct Invert {
    /// Degrees the hues are turned (None: not turned).
    #[serde(default)]
    pub hue: Option<f32>,
    /// The level times the strength, float32.
    pub v: f32,
    #[serde(default)]
    pub mask: Option<Region>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    turn: u32,
    mask: u32,
    hue: f32,
    v: f32,
    pad0: f32,
    pad1: f32,
}

impl Effect for Invert {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Region::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let region = match &self.mask {
            Some(r) => r.plane(cx, &files[0]),
            None => src.clone(),
        };
        let k = cx.kernel("fx_invert", SRC, "invert");
        let u = cx.uniform(&Params {
            w: cx.w, h: cx.h, turn: self.hue.is_some() as u32,
            mask: self.mask.as_ref().map_or(0, |r| if r.ground { 2 } else { 1 }), hue: self.hue.unwrap_or(0.0),
            v: self.v, pad0: 0.0, pad1: 0.0,
        });
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst), (6, &region)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
