//! shapes: the picture as flat shapes (mkmmd.shapes.picture; FrameFx._shapes), a cell each (shaders/fx_shapes.wgsl).
//! Python (mkmmd.gpufx.shapes) writes the frame's tables for the layout (and, while its shapes pop in, the layout before:
//! it shows where a cell's new shape has not begun): each cell's shape, turn, size and ink, and each pixel's cell and
//! place in it; here each cell's colour is the picture's box blur at its middle (its ink, or skipped near a flat back),
//! then every pixel draws its cell's shape as a signed distance, over the picture or the flat back, kept to the mask's
//! region and mixed in by the strength.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::pixelsort::Mask;
use super::{Ctx, Decoded, Drawn, Effect, Input};
use crate::pngio::Kind;

const SRC: &str = include_str!("../shaders/fx_shapes.wgsl");

/// A workgroup a cell (256 threads sum its box) from this box side on; under it a thread a cell.
const GROUP_FROM: u32 = 16;
/// Workgroups along x of a 2D dispatch of cells (the kernel's own `wg.y * 65535 + wg.x`).
const ROW: u32 = 65535;

/// A layout of cells: `cells` of them in rows `columns` wide, the picture's box blur of side `blur` read at each cell's
/// middle.
#[derive(Debug, Clone, Deserialize)]
pub struct Layer {
    pub cells: u32,
    pub columns: u32,
    pub blur: u32,
}

/// One frame of shapes: `table` (`count` float32 Python wrote for it) holds each of the `layers` (the new layout, and
/// while it pops in the one before: it shows where a cell's new shape has not begun) with its cells, every picture row
/// and column's place, then the `inks`. `back`: the flat colour the shapes sit on (None: the picture), a cell within
/// `skip` of it draws nothing. Mixed in by the strength `s` (and 1 - it), kept to the mask's region.
#[derive(Debug, Clone, Deserialize)]
pub struct Shapes {
    pub table: String,
    pub count: usize,
    pub layers: Vec<Layer>,
    pub inks: u32,
    #[serde(default)]
    pub back: Option<[f32; 3]>,
    #[serde(default)]
    pub skip: Option<f32>,
    #[serde(default)]
    pub mask: Option<Mask>,
    pub s: [f32; 2],
}

/// shaders/fx_shapes.wgsl's Layer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct LayerParams {
    cells: u32,
    columns: u32,
    k: u32,
    at: u32,
    rows_at: u32,
    cols_at: u32,
    paint_at: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    inks: u32,
    inks_at: u32,
    masked: u32,
    flat: u32,
    mixed: u32,
    layers: u32,
    which: u32,
    pad0: [u32; 3],
    skip: f32,
    s: f32,
    s1: f32,
    pad1: f32,
    back: [f32; 4],
    a: LayerParams,
    b: LayerParams,
}

impl Effect for Shapes {
    fn inputs(&self) -> Vec<Input> {
        let mut v = vec![Input::F32 { path: self.table.clone(), count: self.count }];
        v.extend(self.mask.iter().map(|m| Input::Png(m.figure.clone(), Kind::Luma)));
        v
    }

    fn draw(&self, cx: &mut Ctx, inputs: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        if self.layers.is_empty() || self.layers.len() > 2 || self.layers.iter().any(|l| l.cells == 0 || l.blur == 0) {
            bail!("shapes: {} layouts", self.layers.len());
        }
        let mut placed = [LayerParams::default(); 2];
        let (mut at, mut paint_at) = (0u32, 0u32);
        for (l, p) in self.layers.iter().zip(&mut placed) {
            let rows_at = at + 8 * l.cells;
            let cols_at = rows_at + 4 * h;
            *p = LayerParams { cells: l.cells, columns: l.columns, k: l.blur, at, rows_at, cols_at, paint_at, pad: 0 };
            at = cols_at + 4 * w;
            paint_at += l.cells;
        }
        if self.count != (at + 4 * self.inks) as usize {
            bail!("shapes: a table of {} floats for {} layouts, {} inks and a {w}x{h} frame", self.count,
                  self.layers.len(), self.inks);
        }
        let all = self.inputs();
        let tab = cx.upload(&all[0], &inputs[0]);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let flat = self.back.is_some();
        let b = self.back.unwrap_or([0.0; 3]);
        let mut p = Params {
            w, h, inks: self.inks, inks_at: at, masked: self.mask.as_ref().map_or(0, |m| if m.inside { 1 } else { 2 }),
            flat: flat as u32, mixed: (self.s[0] < 1.0) as u32, layers: self.layers.len() as u32, which: 0,
            pad0: [0; 3], skip: self.skip.unwrap_or(0.0), s: self.s[0], s1: self.s[1], pad1: 0.0,
            back: [b[0], b[1], b[2], 0.0], a: placed[0], b: placed[1],
        };
        let paint = cx.scratch("shapes.paint", 16 * paint_at as u64);
        for (j, l) in self.layers.iter().enumerate() {
            p.which = j as u32;
            let u = cx.uniform(&p);
            if l.blur < GROUP_FROM {
                let k = cx.kernel("fx_shapes", SRC, "cells_one");
                cx.pass(&k, &[(0, &u), (1, &src), (3, &tab), (4, &paint)], (l.cells.div_ceil(64), 1));
            } else {
                let k = cx.kernel("fx_shapes", SRC, "cells_group");
                cx.pass(&k, &[(0, &u), (1, &src), (3, &tab), (4, &paint)], (l.cells.min(ROW), l.cells.div_ceil(ROW)));
            }
        }
        let fig = if self.mask.is_some() { cx.upload(&all[1], &inputs[1]) } else { cx.table("fig", &[0u32]) };
        let lut = cx.lut.clone();
        let u = cx.uniform(&p);
        let k = cx.kernel("fx_shapes", SRC, "draw");
        cx.pass(&k, &[(0, &u), (1, &src), (2, &dst), (3, &tab), (4, &paint), (5, &fig), (6, &lut)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
