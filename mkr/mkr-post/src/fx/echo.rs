//! echo (mkmmd.framefx.echo): copies of the frames before laid on the picture, each at its opacity, keeping the
//! brighter (`lighten`) or the darker of the two, tinted where they differ from the ground (shaders/fx_echo.wgsl).
//! mkmmd.gpufx.echo names the frames and works the opacities out.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_echo.wgsl");

/// `copies` (the cut's frame, its opacity as float32), laid in order.
#[derive(Debug, Clone, Deserialize)]
pub struct Echo {
    pub copies: Vec<(String, f32)>,
    pub lighten: bool,
    #[serde(default)]
    pub tint: Option<[f32; 3]>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    lighten: u32,
    tinted: u32,
    alpha: f32,
    pad: [f32; 3],
    tint: [f32; 4],
}

impl Effect for Echo {
    fn inputs(&self) -> Vec<Input> {
        self.copies.iter().map(|(p, _)| Input::Png(p.clone(), Kind::Picture)).collect()
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let k = cx.kernel("fx_echo", SRC, "echo");
        let pic = cx.src.clone();
        let lut = cx.lut.clone();
        let tint = self.tint.unwrap_or([1.0; 3]);
        for ((input, (_, alpha)), decoded) in self.inputs().iter().zip(&self.copies).zip(inputs) {
            let frame = cx.upload(input, decoded);
            let p = cx.uniform(&Params {
                w: cx.w, h: cx.h, lighten: self.lighten as u32, tinted: self.tint.is_some() as u32, alpha: *alpha,
                pad: [0.0; 3], tint: [tint[0], tint[1], tint[2], 0.0],
            });
            let groups = cx.tiles();
            cx.pass(&k, &[(0, &p), (1, &pic), (2, &frame), (3, &lut)], groups);
        }
        Ok(Drawn::Src)
    }
}
