//! split (mkmmd.framefx.split): a broken digital signal, bands torn sideways and the colour planes pulled apart
//! (shaders/fx_split.wgsl); mkmmd.gpufx.split works each row's shifts out with the CPU path's own code
//! (mkmmd.framefx.split_shifts).

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect};

const SRC: &str = include_str!("../shaders/fx_split.wgsl");

/// `bands`: (first row, past the last, red, green and blue shifts in px), covering the rows in order.
#[derive(Debug, Clone, Deserialize)]
pub struct Split {
    pub bands: Vec<(u32, u32, [i32; 3])>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    pad: [u32; 2],
}

impl Effect for Split {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        let mut rows = vec![[0i32; 4]; cx.h as usize];
        let mut next = 0u32;
        for &(a, b, s) in &self.bands {
            if a != next || b <= a || b > cx.h {
                bail!("split: band {a}..{b} does not follow row {next} in a frame of {} rows", cx.h);
            }
            for r in &mut rows[a as usize..b as usize] {
                *r = [s[0], s[1], s[2], 0];
            }
            next = b;
        }
        if next != cx.h {
            bail!("split: the bands end at row {next}, not {}", cx.h);
        }
        let k = cx.kernel("fx_split", SRC, "split");
        let t = cx.table("rows", &rows);
        let p = cx.uniform(&Params { w: cx.w, h: cx.h, pad: [0; 2] });
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let groups = cx.tiles();
        cx.pass(&k, &[(0, &p), (1, &src), (2, &dst), (3, &t)], groups);
        Ok(Drawn::Dst)
    }
}
