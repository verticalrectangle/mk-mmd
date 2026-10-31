//! grid: the frame shrunk into boxes (mkmmd.framefx.cells; FrameFx._grid), each showing the whole frame contain-fitted
//! and centred in it, the figure on the box's back (shaders/fx_grid.wgsl). Python (mkmmd.gpufx.grid) works out the
//! boxes (mkmmd.core.framefx.grid_cells), their backs and fills, the shrunk size and the tile's padding; the tile (the
//! figure's coverage and picture shrunk by area, its ring, its shadow, the plates of its silkscreen prints: fx/silk.rs)
//! is made once here and laid into every box.

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, Cutout, Edge, Shadow, plane_bytes};
use super::silk::{self, Bands};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_grid.wgsl");
/// The tiles of a grid's silkscreen prints, by their number of inks (2 to 4): the plates (.x the key, .y .z the colour
/// steps), the key plate printed off register.
const SILK_TILES: [&str; 3] = ["grid.silk2", "grid.silk3", "grid.silk4"];
const SILK_KEYS: [&str; 3] = ["grid.silkkey2", "grid.silkkey3", "grid.silkkey4"];

/// A box's back: a colour, or the frame as it reaches the op, shrunk as the figure is: its part (x, y, w, h of the
/// shrunk frame) put with its top left at `at` (px of the picture) and reflected past its edges to fill the box.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellBack {
    Color([f32; 3]),
    Picture { part: [u32; 4], at: [i32; 2] },
}

/// A box's figure printed in silkscreen: the print (one of the grid's `silks`) and its inks (the key's first).
#[derive(Debug, Clone, Deserialize)]
pub struct SilkFill {
    pub band: usize,
    pub inks: Vec<[f32; 3]>,
}

/// One box (x0, y0, x1, y1 px), in order: its back, where the tile's top left lands (px of the picture), its fill
/// (None: the figure's picture, or its silkscreen print).
#[derive(Debug, Clone, Deserialize)]
pub struct Cell {
    #[serde(rename = "box")]
    pub area: [u32; 4],
    pub back: CellBack,
    pub at: [i32; 2],
    #[serde(default)]
    pub fill: Option<[f32; 3]>,
    #[serde(default)]
    pub silk: Option<SilkFill>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Grid {
    pub figure: Cutout,
    /// The frame shrunk into a box: (w, h).
    pub small: [u32; 2],
    /// Nothing round the shrunk figure in its tile, px.
    pub pad: u32,
    pub cells: Vec<Cell>,
    /// The silkscreen prints the boxes' fills ask for, one per number of inks.
    #[serde(default)]
    pub silks: Vec<Bands>,
    #[serde(default)]
    pub edge: Option<Edge>,
    /// Its offset and blur already scaled with the figure.
    #[serde(default)]
    pub shadow: Option<Shadow>,
    pub strength: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    tw: u32,
    th: u32,
    tx: i32,
    ty: i32,
    px: u32,
    py: u32,
    sw: u32,
    fill: u32,
    shadow: u32,
    edged: u32,
    silk: u32,
    steps: u32,
    pad0: u32,
    pad1: u32,
    colour: [f32; 4],
    edge: [f32; 4],
    shade: [f32; 4],
    inks: [[f32; 4]; 4],
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], (w, h): (u32, u32)) {
    let k = cx.kernel("fx_grid", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    cx.pass(&k, &all, crate::gpu::tiles(w, h));
}

impl Effect for Grid {
    fn inputs(&self) -> Vec<Input> {
        self.figure.inputs()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let n = plane_bytes(w, h);
        let [sw, sh] = self.small;
        let pad = self.pad;
        let (tw, th) = (sw + 2 * pad, sh + 2 * pad);
        let tn = plane_bytes(tw, th);
        let picture = self.cells.iter().any(|c| c.fill.is_none() && c.silk.is_none());
        // the figure's tile
        let cov = cx.scratch("figure.cov", n);
        let rgb = cx.scratch("figure.rgb", n);
        let pic = figure::cut(cx, &self.figure, files, &cov, (picture || !self.silks.is_empty()).then_some(&rgb));
        let mut prints = Vec::with_capacity(self.silks.len());
        for b in &self.silks {
            let Some(i) = (b.n as usize).checked_sub(2).filter(|i| *i < SILK_TILES.len()) else {
                bail!("a silkscreen in {} inks (2 to 4)", b.n)
            };
            let Some(pic) = &pic else { bail!("a silkscreen without the figure's picture") };
            let full = cx.scratch("grid.silkfull", n);
            silk::bands(cx, pic, &cov, &full, b)?;
            let small = if (sw, sh) == (w, h) {
                full.clone()
            } else {
                let s = cx.scratch("grid.silksmall", n);
                figure::resize_area(cx, &full, (w, h), &s, (sw, sh));
                s
            };
            let tile = cx.scratch(SILK_TILES[i], tn);
            run(cx, "tile_pad", Params { tw, th, px: pad, ..Default::default() }, &[(1, &small), (2, &tile)], (tw, th));
            let key = if b.offset != [0.0, 0.0] {
                let k = cx.scratch(SILK_KEYS[i], tn);
                figure::shift(cx, &tile, &k, tw, th, b.offset[0], b.offset[1]);
                k
            } else {
                tile.clone()
            };
            prints.push((tile, key));
        }
        let (c1, r1) = if (sw, sh) == (w, h) {
            (cov.clone(), rgb.clone())
        } else {
            let c1 = cx.scratch("grid.c1", n);
            let r1 = cx.scratch("grid.r1", n);
            figure::resize_area(cx, &cov, (w, h), &c1, (sw, sh));
            if picture {
                figure::resize_area(cx, &rgb, (w, h), &r1, (sw, sh));
            }
            (c1, r1)
        };
        let tc = cx.scratch("grid.tc", tn);
        let tr = cx.scratch("grid.tr", tn);
        let p = Params { tw, th, px: pad, ..Default::default() };
        run(cx, "tile_pad", p, &[(1, &c1), (2, &tc)], (tw, th));
        if picture {
            run(cx, "tile_pad", p, &[(1, &r1), (2, &tr)], (tw, th));
        }
        let t1 = cx.scratch("grid.t1", tn);
        let t2 = cx.scratch("grid.t2", tn);
        let ring = cx.scratch("grid.ring", tn);
        let edged = self.edge.as_ref().filter(|e| e.width > 0.0);
        if let Some(e) = edged {
            figure::grow(cx, &tc, &ring, &t1, &t2, tw, th, e.width);
        }
        let dark = cx.scratch("grid.dark", tn);
        let shadow = self.shadow.as_ref().filter(|s| s.alpha > 0.0);
        if let Some(s) = shadow {
            figure::shift(cx, if edged.is_some() { &ring } else { &tc }, &dark, tw, th, s.offset[0], s.offset[1]);
            if s.blur > 0.3 {
                figure::blur(cx, &dark, &dark, &t1, tw, th, s.blur);
            }
            run(cx, "tile_dark", Params { tw, th, shade: [0.0, 0.0, 0.0, s.alpha as f32], ..Default::default() },
                &[(2, &dark)], (tw, th));
        }
        // the backs, then the figures
        figure::fill(cx, &dst, w, h, [0.0; 3]);
        let mut small: Option<wgpu::Buffer> = None;
        for c in &self.cells {
            let [x0, y0, x1, y1] = c.area;
            let base = Params { w, h, x0, y0, x1, y1, ..Default::default() };
            match &c.back {
                CellBack::Color(col) => {
                    run(cx, "cell_fill", Params { colour: [col[0], col[1], col[2], 0.0], ..base }, &[(2, &dst)],
                        (x1 - x0, y1 - y0));
                }
                CellBack::Picture { part, at } => {
                    let s = match &small {
                        Some(s) => s.clone(),
                        None => {
                            let s = if (sw, sh) == (w, h) {
                                src.clone()
                            } else {
                                let s = cx.scratch("grid.small", n);
                                figure::resize_area(cx, &src, (w, h), &s, (sw, sh));
                                s
                            };
                            small = Some(s.clone());
                            s
                        }
                    };
                    let p = Params { tw: part[2], th: part[3], tx: at[0], ty: at[1], px: part[0], py: part[1], sw, ..base };
                    run(cx, "cell_back", p, &[(1, &s), (2, &dst)], (x1 - x0, y1 - y0));
                }
            }
        }
        let e = edged.map_or([0.0; 3], |e| e.color);
        let sc = shadow.map_or([0.0; 3], |s| s.color);
        for c in &self.cells {
            let [x0, y0, x1, y1] = c.area;
            let f = c.fill.unwrap_or([0.0; 3]);
            let mut p = Params {
                w, h, x0, y0, x1, y1, tw, th, tx: c.at[0] - pad as i32, ty: c.at[1] - pad as i32,
                fill: c.fill.is_some() as u32, shadow: shadow.is_some() as u32, edged: edged.is_some() as u32,
                colour: [f[0], f[1], f[2], 0.0], edge: [e[0], e[1], e[2], 0.0], shade: [sc[0], sc[1], sc[2], 0.0],
                ..Default::default()
            };
            let (plates, key) = match &c.silk {
                Some(s) => {
                    let Some((tile, key)) = prints.get(s.band) else { bail!("silkscreen print {} (of {})", s.band,
                                                                            prints.len()) };
                    p.silk = 1;
                    p.steps = s.inks.len().saturating_sub(2) as u32;
                    for (k, ink) in s.inks.iter().take(4).enumerate() {
                        p.inks[k] = [ink[0], ink[1], ink[2], 0.0];
                    }
                    (tile.clone(), key.clone())
                }
                None => (tc.clone(), tc.clone()),
            };
            run(cx, "cell_put", p, &[(1, &tr), (2, &dst), (3, &tc), (4, &ring), (5, &dark), (6, &plates), (7, &key)],
                (x1 - x0, y1 - y0));
        }
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
