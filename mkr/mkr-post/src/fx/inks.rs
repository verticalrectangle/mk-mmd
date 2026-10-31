//! inks: flat colours laid by small coverages over the picture, in order, then mixed in by the strength (a label's box,
//! text and leader: mkmmd.labels.LabelFx._label_inks, laid as mkmmd.labels._ink lays them; shaders/fx_inks.wgsl).
//! Python draws the coverages (type set by Pillow, dots by OpenCV) and writes them for this frame.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_inks.wgsl");

/// One coverage laid: its top left (px; it may stick out of the picture), its size, its values (w x h float32 0..1,
/// removed once read; None: a solid box), its colour and opacity.
#[derive(Debug, Clone, Deserialize)]
pub struct Ink {
    pub at: [i32; 2],
    pub size: [u32; 2],
    #[serde(default)]
    pub mask: Option<String>,
    pub color: [f32; 3],
    pub alpha: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inks {
    pub inks: Vec<Ink>,
    pub strength: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    cw: u32,
    ch: u32,
    solid: u32,
    pad0: u32,
    alpha: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
    colour: [f32; 4],
}

impl Ink {
    pub fn input(&self) -> Option<Input> {
        self.mask.as_ref().map(|p| Input::F32 { path: p.clone(), count: (self.size[0] * self.size[1]) as usize })
    }
}

/// `inks` laid over the picture `dst` (in place), in order; `files` the inputs of those with a coverage (Ink::input), in
/// the same order.
pub fn lay(cx: &mut Ctx, dst: &wgpu::Buffer, inks: &[Ink], files: &[Arc<Decoded>]) {
    let (w, h) = (cx.w, cx.h);
    let k = cx.kernel("fx_inks", SRC, "ink");
    let mut next = 0;
    for ink in inks {
        let cov = match ink.input() {
            Some(input) => {
                let b = cx.upload(&input, &files[next]);
                next += 1;
                b
            }
            None => cx.scratch("inks.none", 16),
        };
        let c = ink.color;
        let u = cx.uniform(&Params {
            w, h, x0: ink.at[0], y0: ink.at[1], cw: ink.size[0], ch: ink.size[1], solid: ink.mask.is_none() as u32,
            pad0: 0, alpha: ink.alpha, pad1: 0.0, pad2: 0.0, pad3: 0.0, colour: [c[0], c[1], c[2], 0.0],
        });
        cx.pass(&k, &[(0, &u), (2, dst), (3, &cov)], crate::gpu::tiles(ink.size[0], ink.size[1]));
    }
}

impl Effect for Inks {
    fn inputs(&self) -> Vec<Input> {
        self.inks.iter().filter_map(Ink::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        figure::copy(cx, &src, &dst, w, h);
        lay(cx, &dst, &self.inks, files);
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
