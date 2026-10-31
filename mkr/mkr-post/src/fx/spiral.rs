//! spiral: an op-art pattern laid on the picture (mkmmd.fields.pattern; FieldFx._spiral), and the collages' pattern
//! backs (mkmmd.fields.Painter): `paint` fills a plane with one. Python (mkmmd.gpufx.spiral, mkmmd.gpufx.fields) works
//! out its phase and constants.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::fields::{PatternSpec, Region};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/fields.wgsl"), "\n", include_str!("../shaders/fx_spiral.wgsl"));

#[derive(Debug, Clone, Deserialize)]
pub struct Spiral {
    pub pattern: PatternSpec,
    /// fields.wgsl's blend modes: 0 over, 1 screen, 2 multiply, 3 soft light.
    pub blend: u32,
    pub strength: f32,
    #[serde(default)]
    pub mask: Option<Region>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    blend: u32,
    mask: u32,
    cx: f32,
    cy: f32,
    strength: f32,
    pad: f32,
}

/// Plane `out` (w x h) filled with pattern `p` (about its centre, in its unit).
pub fn paint(cx: &mut Ctx, p: &PatternSpec, out: &wgpu::Buffer, w: u32, h: u32) {
    let k = cx.kernel("fx_spiral", SRC, "pattern_fill");
    let u = cx.uniform(&Params { w, h, blend: 0, mask: 0, cx: p.center[0], cy: p.center[1], strength: 0.0, pad: 0.0 });
    let pat = cx.uniform(&p.gpu());
    cx.pass(&k, &[(0, &u), (2, out), (3, &pat)], crate::gpu::tiles(w, h));
}

impl Effect for Spiral {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Region::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let region = match &self.mask {
            Some(r) => r.plane(cx, &files[0]),
            None => src.clone(),
        };
        let k = cx.kernel("fx_spiral", SRC, "spiral");
        let u = cx.uniform(&Params {
            w, h, blend: self.blend, mask: self.mask.as_ref().map_or(0, |r| if r.ground { 2 } else { 1 }),
            cx: self.pattern.center[0], cy: self.pattern.center[1], strength: self.strength, pad: 0.0,
        });
        let pat = cx.uniform(&self.pattern.gpu());
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst), (3, &pat), (6, &region)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
