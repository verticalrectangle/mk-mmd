//! shock (mkmmd.framefx.shock): pressure rings running out from a point on the hits, bending the picture and lit at
//! their fronts (shaders/fx_shock.wgsl); mkmmd.gpufx.shock works the waves out with FrameFx.shock_at.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect};

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_shock.wgsl"));

/// float32 as numpy uses them: `c` the centre (px), `width` the front's, `waves` (R px, A = amount * s * 2.33, s),
/// `rim` SHOCK_RIM, `glow` and `colour` the front's light (none when glow is 0).
#[derive(Debug, Clone, Deserialize)]
pub struct Shock {
    pub c: [f32; 2],
    pub width: f32,
    pub rim: f32,
    pub waves: Vec<[f32; 3]>,
    pub glow: f32,
    pub colour: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    n: u32,
    lit: u32,
    cx: f32,
    cy: f32,
    width: f32,
    rim_k: f32,
    glow: f32,
    pad: [f32; 3],
    colour: [f32; 4],
}

impl Effect for Shock {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        if self.waves.is_empty() {
            return Ok(Drawn::Src);
        }
        let k = cx.kernel("fx_shock", SRC, "shock");
        let waves: Vec<[f32; 4]> = self.waves.iter().map(|w| [w[0], w[1], w[2], 0.0]).collect();
        let t = cx.table("waves", &waves);
        let c = self.colour;
        let p = cx.uniform(&Params {
            w: cx.w, h: cx.h, n: waves.len() as u32, lit: (self.glow > 0.0) as u32, cx: self.c[0], cy: self.c[1],
            width: self.width, rim_k: self.rim, glow: self.glow, pad: [0.0; 3], colour: [c[0], c[1], c[2], 0.0],
        });
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let groups = cx.tiles();
        cx.pass(&k, &[(0, &p), (1, &src), (2, &dst), (3, &t)], groups);
        Ok(Drawn::Dst)
    }
}
