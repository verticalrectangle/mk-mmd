//! lens (mkmmd.framefx.lens): a black hole's lensing about a point, the picture remapped (shaders/fx_lens.wgsl);
//! mkmmd.gpufx.lens works its centre and Einstein radius out.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect};

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_lens.wgsl"));

/// `c` the centre (px, float32), `theta2` the Einstein radius squared (float32), `shadow` the bound r2 is compared
/// with when a shadow is painted (the float32 the comparison of a float32 with numpy's float64 bound comes to).
#[derive(Debug, Clone, Deserialize)]
pub struct Lens {
    pub c: [f32; 2],
    pub theta2: f32,
    #[serde(default)]
    pub shadow: Option<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    shadow: u32,
    pad: u32,
    cx: f32,
    cy: f32,
    theta2: f32,
    lim: f32,
}

impl Effect for Lens {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        let k = cx.kernel("fx_lens", SRC, "lens");
        let p = cx.uniform(&Params {
            w: cx.w, h: cx.h, shadow: self.shadow.is_some() as u32, pad: 0, cx: self.c[0], cy: self.c[1],
            theta2: self.theta2, lim: self.shadow.unwrap_or(0.0),
        });
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let groups = cx.tiles();
        cx.pass(&k, &[(0, &p), (1, &src), (2, &dst)], groups);
        Ok(Drawn::Dst)
    }
}
