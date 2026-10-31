//! slitscan (mkmmd.framefx.slitscan): each column (or row) of the picture from the frame of its own age; mkmmd.gpufx
//! .slitscan works the ages out (mkmmd.framefx.slit_ages) and hands over the bands of columns that take one frame
//! (shaders/fx_slitscan.wgsl). Columns of age 0, or older than any frame there is, keep the picture.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_slitscan.wgsl");

/// `bands`: (first, past the last column or row, the cut's frame they show); `across`: columns (else rows).
#[derive(Debug, Clone, Deserialize)]
pub struct Slitscan {
    pub across: bool,
    pub bands: Vec<(u32, u32, String)>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    across: u32,
    i0: u32,
    i1: u32,
    pad: [u32; 3],
}

impl Effect for Slitscan {
    fn inputs(&self) -> Vec<Input> {
        self.bands.iter().map(|(_, _, p)| Input::Png(p.clone(), Kind::Picture)).collect()
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let k = cx.kernel("fx_slitscan", SRC, "slitscan");
        let (pic, lut) = (cx.src.clone(), cx.lut.clone());
        for ((input, &(i0, i1, _)), decoded) in self.inputs().iter().zip(&self.bands).zip(inputs) {
            let n = if self.across { cx.w } else { cx.h };
            let (i0, i1) = (i0.min(n), i1.min(n));
            if i1 <= i0 {
                continue;
            }
            let frame = cx.upload(input, decoded);
            let p = cx.uniform(&Params { w: cx.w, h: cx.h, across: self.across as u32, i0, i1, pad: [0; 3] });
            let groups = if self.across {
                ((i1 - i0).div_ceil(16), cx.h.div_ceil(16))
            } else {
                (cx.w.div_ceil(16), (i1 - i0).div_ceil(16))
            };
            cx.pass(&k, &[(0, &p), (1, &pic), (2, &frame), (3, &lut)], groups);
        }
        Ok(Drawn::Src)
    }
}
