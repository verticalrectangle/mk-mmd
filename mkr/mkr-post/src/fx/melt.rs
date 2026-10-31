//! melt (mkmmd.core.remap.melt_field, RemapFx._melt): the picture breathing, swirling, rippling, dripping or wobbling,
//! under a `mask` the ground or the figure alone (fx/masked.rs) (shaders/fx_melt.wgsl); mkmmd.gpufx.melt works the
//! mode's numbers and tables out with the CPU path's own code.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::masked::{self, Masking};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_melt.wgsl"));

/// float32 as numpy uses them (fx_melt.wgsl says which a mode reads): `mode` 0 breathe, 1 swirl, 2 ripple, 3 drip,
/// 4 noise; `c` the centre; `table` a ripple's amplitudes along r (`step` px apart; empty: `k0` everywhere), a drip's
/// (drip, col) per column, or the noise's row taps, column taps and lattices (`n` lattice columns).
#[derive(Debug, Clone, Deserialize)]
pub struct Melt {
    pub mode: u32,
    pub c: [f32; 2],
    pub k0: f32,
    #[serde(default)]
    pub su: f32,
    #[serde(default)]
    pub tau: f32,
    #[serde(default)]
    pub phase: f32,
    #[serde(default)]
    pub step: f32,
    #[serde(default)]
    pub spread: f32,
    #[serde(default)]
    pub n: u32,
    #[serde(default)]
    pub table: Vec<f32>,
    #[serde(default)]
    pub mask: Option<Masking>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    mode: u32,
    n: u32,
    cx: f32,
    cy: f32,
    k0: f32,
    su: f32,
    tau: f32,
    phase: f32,
    step: f32,
    spread: f32,
}

impl Effect for Melt {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Masking::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w as usize, cx.h as usize);
        let want = match self.mode {
            0 | 1 => 0,
            2 => self.n as usize,
            3 => 2 * w,
            4 => 5 * (w + h) + 2 * self.n as usize * self.lattice_rows(w, h),
            m => bail!("melt: no mode {m}"),
        };
        if self.table.len() != want || (self.mode == 2 && self.n == 1) {
            bail!("melt mode {}: a table of {} values, not {want}", self.mode, self.table.len());
        }
        let table = if self.table.is_empty() { vec![0.0f32] } else { self.table.clone() };
        let t = cx.table("melt", &table);
        let reach = cx.scratch("melt_reach", 4);
        cx.enc.clear_buffer(&reach, 0, None);
        let p = cx.uniform(&Params {
            w: cx.w, h: cx.h, mode: self.mode, n: self.n, cx: self.c[0], cy: self.c[1], k0: self.k0, su: self.su,
            tau: self.tau, phase: self.phase, step: self.step, spread: self.spread,
        });
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let groups = cx.tiles();
        let k = cx.kernel("fx_melt", SRC, "melt_reach");
        cx.pass(&k, &[(0, &p), (3, &t), (4, &reach)], groups);
        let k = cx.kernel("fx_melt", SRC, "melt");
        let Some(mask) = &self.mask else {
            cx.pass(&k, &[(0, &p), (1, &src), (2, &dst), (3, &t), (4, &reach)], groups);
            return Ok(Drawn::Dst);
        };
        masked::remapped(cx, mask, &inputs[0], &mut |cx, pic, out| {
            cx.pass(&k, &[(0, &p), (1, pic), (2, out), (3, &t), (4, &reach)], groups);
            Ok(())
        })?;
        // under a mask the CPU path leaves the picture as it is, unfilled, when the field moves nothing by 0.01 px:
        // `melt_keep` puts it back over what the masked work drew then (the reach read on the GPU)
        let k = cx.kernel("fx_melt", SRC, "melt_keep");
        cx.pass(&k, &[(0, &p), (1, &src), (2, &dst), (4, &reach)], groups);
        Ok(Drawn::Dst)
    }
}

impl Melt {
    /// The noise lattices' rows: what the table holds past the taps, over 2 x n per row.
    fn lattice_rows(&self, w: usize, h: usize) -> usize {
        let past = self.table.len().saturating_sub(5 * (w + h));
        if self.n == 0 { 0 } else { past / (2 * self.n as usize) }
    }
}
