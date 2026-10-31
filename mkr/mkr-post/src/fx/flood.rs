//! flood: ink dropped from a point, filling the frame and drawing back (mkmmd.fields.flood; FieldFx._flood). Python
//! (mkmmd.gpufx.flood) works out the drops on the frame (CF.flood_drops), their noise calls and the constants; the
//! fronts, the colour and the lay are drawn here (shaders/fx_flood.wgsl).

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::fields::{NoiseCall, Region, noise_tables};
use super::figure::plane_bytes;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/fields.wgsl"), "\n", include_str!("../shaders/fx_flood.wgsl"));

/// One drop on the frame: how far its front is (share), its opacity and colour, its three noise calls (lobes, edge,
/// mottle).
#[derive(Debug, Clone, Deserialize)]
pub struct Drop {
    pub share: f32,
    pub alpha: f32,
    pub color: [f32; 3],
    pub calls: [NoiseCall; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Flood {
    pub center: [f32; 2],
    pub unit: f32,
    pub wobble: f32,
    pub wob2: f32,
    /// float32 of max(soft x px, 1), and 3, 0.8 and 1.4 times it.
    pub soft: [f32; 4],
    pub drops: Vec<Drop>,
    pub strength: f32,
    #[serde(default)]
    pub mask: Option<Region>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    drop: u32,
    first: u32,
    mask: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    cx: f32,
    cy: f32,
    unit: f32,
    wob: f32,
    soft: f32,
    soft3: f32,
    soft08: f32,
    soft14: f32,
    share: f32,
    alpha: f32,
    wob2: f32,
    strength: f32,
    colour: [f32; 4],
}

impl Effect for Flood {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Region::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let n = plane_bytes(w, h);
        let front = cx.scratch("flood.front", n);
        let acc = cx.scratch("flood.acc", n);
        let calls: Vec<NoiseCall> = self.drops.iter().flat_map(|d| d.calls).collect();
        let (cubic, table) = noise_tables(cx, &calls);
        let maxes = cx.table("flood.max", &vec![0u32; self.drops.len().max(1)]);
        let region = match &self.mask {
            Some(r) => r.plane(cx, &files[0]),
            None => front.clone(),
        };
        let base = Params {
            w, h, drop: 0, first: 1, mask: self.mask.as_ref().map_or(0, |r| if r.ground { 2 } else { 1 }), pad0: 0,
            pad1: 0, pad2: 0, cx: self.center[0], cy: self.center[1], unit: self.unit, wob: self.wobble,
            soft: self.soft[0], soft3: self.soft[1], soft08: self.soft[2], soft14: self.soft[3], share: 0.0, alpha: 0.0,
            wob2: self.wob2, strength: self.strength, colour: [0.0; 4],
        };
        let k_front = cx.kernel("fx_flood", SRC, "flood_front");
        let k_drop = cx.kernel("fx_flood", SRC, "flood_drop");
        let k_lay = cx.kernel("fx_flood", SRC, "flood_lay");
        let groups = cx.tiles();
        for (j, d) in self.drops.iter().enumerate() {
            let p = cx.uniform(&Params {
                drop: j as u32, first: (j == 0) as u32, share: d.share, alpha: d.alpha,
                colour: [d.color[0], d.color[1], d.color[2], 0.0], ..base
            });
            cx.pass(&k_front, &[(0, &p), (3, &front), (5, &maxes), (10, &cubic), (11, &table)], groups);
            cx.pass(&k_drop, &[(0, &p), (3, &front), (4, &acc), (5, &maxes)], groups);
        }
        let p = cx.uniform(&base);
        cx.pass(&k_lay, &[(0, &p), (1, &src), (2, &dst), (4, &acc), (6, &region)], groups);
        Ok(Drawn::Dst)
    }
}
