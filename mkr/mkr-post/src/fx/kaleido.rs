//! kaleido (mkmmd.remap.RemapFx._kaleido, kaleido, half_map, sample, seam_blend; mkmmd.core.remap.kaleido_coords): the
//! frame folded into mirrored wedges (or plain copies) about a centre, under a `mask` the ground or the figure alone
//! (fx/masked.rs), mixed in by the strength (shaders/fx_kaleido.wgsl). mkmmd.gpufx.kaleido works the wedges, turns,
//! zoom and levels out with RemapFx.kaleido_at.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, plane_bytes};
use super::masked::{self, Masking};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_kaleido.wgsl"));

/// float32 as numpy uses them: the centre `c` and the source's `s` (px), `off1` and `off2` (the first mirror line's and
/// the source wedge's offsets, radians), the wedge `a` and 2 a, 1 / zoom, the source's `radius` (px; none: no fold
/// back), the pyramid levels sampled with their weights (remap.lods); mirrored wedges: the half map's size and scales;
/// plain copies: a / 2, pi / 2 and the offsets of the copies across a seam; `mix` the strength (s, 1 - s) when under 1.
#[derive(Debug, Clone, Deserialize)]
pub struct Kaleido {
    pub mirror: bool,
    pub c: [f32; 2],
    pub s: [f32; 2],
    pub off1: f32,
    pub off2: f32,
    pub a: f32,
    pub a2: f32,
    pub inv_zoom: f32,
    #[serde(default)]
    pub radius: Option<f32>,
    pub lods: Vec<(u32, f32)>,
    #[serde(default)]
    pub half: [u32; 2],
    #[serde(default)]
    pub kx: f32,
    #[serde(default)]
    pub ky: f32,
    #[serde(default)]
    pub half_a: f32,
    #[serde(default)]
    pub pi2: f32,
    #[serde(default)]
    pub off2p: f32,
    #[serde(default)]
    pub off2m: f32,
    #[serde(default)]
    pub mix: Option<f64>,
    #[serde(default)]
    pub mask: Option<Masking>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    hw: u32,
    hh: u32,
    lw0: u32,
    lh0: u32,
    lw1: u32,
    lh1: u32,
    two: u32,
    has_radius: u32,
    pad0: u32,
    pad1: u32,
    c: [f32; 2],
    s: [f32; 2],
    off1: f32,
    off2: f32,
    a: f32,
    a2: f32,
    inv_zoom: f32,
    radius: f32,
    kx: f32,
    ky: f32,
    w0: f32,
    w1: f32,
    k0: f32,
    k1: f32,
    mirror: u32,
    pad2: u32,
    half_a: f32,
    pi2: f32,
    off2p: f32,
    off2m: f32,
    pad3: f32,
    pad4: f32,
}

impl Kaleido {
    fn params(&self, w: u32, h: u32) -> Params {
        Params {
            w, h, hw: self.half[0], hh: self.half[1], has_radius: self.radius.is_some() as u32, c: self.c, s: self.s,
            off1: self.off1, off2: self.off2, a: self.a, a2: self.a2, inv_zoom: self.inv_zoom,
            radius: self.radius.unwrap_or(0.0), kx: self.kx, ky: self.ky, mirror: self.mirror as u32,
            half_a: self.half_a, pi2: self.pi2, off2p: self.off2p, off2m: self.off2m, ..Default::default()
        }
    }

    /// kaleido(pic, coords, shrink, seams): plane `pic` through the kaleidoscope into `dst`.
    fn remap(&self, cx: &mut Ctx, pic: &wgpu::Buffer, dst: &wgpu::Buffer) -> Result<()> {
        let (w, h) = (cx.w, cx.h);
        if self.lods.is_empty() || self.lods.len() > 2 {
            bail!("kaleido: {} pyramid levels, not 1 or 2", self.lods.len());
        }
        let want = self.lods.iter().map(|l| l.0).max().unwrap();
        let pyr = masked::pyramid(cx, pic, (w, h), want);
        let mut p = self.params(w, h);
        let level = |j: u32| pyr[(j as usize).min(pyr.len() - 1)].clone();
        let (l0, (lw0, lh0)) = level(self.lods[0].0);
        p.lw0 = lw0;
        p.lh0 = lh0;
        p.w0 = self.lods[0].1;
        p.k0 = 0.5f64.powi(self.lods[0].0 as i32) as f32;
        let mut l1 = l0.clone();
        if let Some(&(j, wt)) = self.lods.get(1) {
            let (b, (lw1, lh1)) = level(j);
            l1 = b;
            p.two = 1;
            p.lw1 = lw1;
            p.lh1 = lh1;
            p.w1 = wt;
            p.k1 = 0.5f64.powi(j as i32) as f32;
        }
        let u = cx.uniform(&p);
        if self.mirror {
            let (hw, hh) = (self.half[0], self.half[1]);
            let half = cx.scratch("kal_half", plane_bytes(hw, hh));
            let k = cx.kernel("fx_kaleido", SRC, "coords_half");
            cx.pass(&k, &[(0, &u), (3, &half)], crate::gpu::tiles(hw, hh));
            let map = cx.scratch("kal_map", plane_bytes(w, h));
            figure::resize_linear(cx, &half, (hw, hh), &map, (w, h));
            let k = cx.kernel("fx_kaleido", SRC, "sample_map");
            cx.pass(&k, &[(0, &u), (1, &l0), (2, &l1), (3, dst), (4, &map)], crate::gpu::tiles(w, h));
        } else {
            let k = cx.kernel("fx_kaleido", SRC, "sample_plain");
            cx.pass(&k, &[(0, &u), (1, &l0), (2, &l1), (3, dst)], crate::gpu::tiles(w, h));
        }
        Ok(())
    }
}

impl Effect for Kaleido {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Masking::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (img, out) = (cx.src.clone(), cx.dst.clone());
        match &self.mask {
            Some(mask) => masked::remapped(cx, mask, &inputs[0], &mut |cx, pic, dst| self.remap(cx, pic, dst))?,
            None => self.remap(cx, &img, &out)?,
        }
        if let Some(s) = self.mix {
            figure::mix(cx, &img, &out, cx.w, cx.h, s);
        }
        Ok(Drawn::Dst)
    }
}
