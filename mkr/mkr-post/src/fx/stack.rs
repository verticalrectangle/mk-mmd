//! stack: a banded echo of the figure (mkmmd.stack; StackFx._stack), its copies laid farthest first and the figure over
//! them; with a mirror, all of them again reflected about a vertical line. Each layer is laid as mkmmd.stack.lay lays it,
//! within the box it covers, its warp's map worked out from that box as the CPU's is, so it samples where the CPU's
//! does (a coverage's half, where its edge is grown from, is sensitive to that). Each figure's support (what it covers
//! at all) comes from its layer's Cover (figure::support; found on the GPU and read back only for a warp that turns),
//! the figure is cut over that box alone (its coverage a single-float plane), then each layer is cut from it, warped
//! into its box, edged, shadowed (fx/plane.rs) and laid (shaders/fx_stack.wgsl). Python (mkmmd.gpufx.stack) works out
//! the copies (mkmmd.core.stack.copies), their frames, fills, maps and opacities.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, BORDER_CONSTANT, Cutout, Shadow};
use super::mandala::Back;
use super::plane::{self, Area};
use super::spiral;
use super::warp::warp;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_stack.wgsl");

/// One layer: the figure on its frame (a copy's, delay frames back), its fill (None: its picture), the map that lays it
/// (float64, with a mirror's reflection in it), x2 of the mirror's line (x of the picture shows x2 - x of the figure),
/// its opacity, the map's scale (sqrt |det|: its shadow's offset and blur scale with it) and its edge's width (px).
#[derive(Debug, Clone, Deserialize)]
pub struct Layer {
    pub figure: Cutout,
    #[serde(default)]
    pub fill: Option<[f32; 3]>,
    pub map: [f64; 6],
    #[serde(default)]
    pub flip: Option<i32>,
    pub alpha: f64,
    pub s: f64,
    pub edge: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stack {
    pub back: Back,
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub edge: Option<[f32; 3]>,
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
    cw: u32,
    ch: u32,
    flip: i32,
    slot: u32,
    fill: u32,
    edged: u32,
    shadow: u32,
    ax0: u32,
    ay0: u32,
    aw: u32,
    ah: u32,
    alpha: f32,
    colour: [f32; 4],
    edge: [f32; 4],
    shade: [f32; 4],
}

impl Params {
    /// The cut planes' box (`ax0`, ...).
    fn cut_over(self, a: &Area) -> Params {
        Params { ax0: a.x0, ay0: a.y0, aw: a.w, ah: a.h, ..self }
    }
}

/// Where a layer goes, as mkmmd.stack.lay works it out: the part of the figure it is cut from (x0, y0, x1, y1, as laid:
/// after the mirror), its box in the picture (X0, Y0, X1, Y1) and the float64 map from the part to the box.
struct Place {
    part: [i64; 4],
    area: [i64; 4],
    map: [f64; 6],
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], (w, h): (u32, u32)) {
    let k = cx.kernel("fx_stack", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    cx.pass(&k, &all, crate::gpu::tiles(w, h));
}

impl Stack {
    /// mkmmd.stack.lay's box of layer `l` whose figure covers `support` (x0, y0, x1, y1) in a w x h picture; None when
    /// it covers nothing there.
    fn place(&self, l: &Layer, support: [i64; 4], w: i64, h: i64) -> Option<Place> {
        let [mut x0, y0, mut x1, y1] = support;
        if let Some(x2) = l.flip {                                // the reflection: its pixels flipped, cut at the frame
            let (a, b) = (x2 as i64 - (x1 - 1), x2 as i64 - x0 + 1);
            x0 = a.max(0);
            x1 = b.min(w);
            if x1 <= x0 {
                return None;
            }
        }
        let a = l.map;
        let reach = match self.shadow.as_ref().filter(|s| s.alpha > 0.0) {
            Some(s) => (s.offset[0].hypot(s.offset[1]) + 4.0 * s.blur) * l.s,
            None => 0.0,
        };
        let pad = 2.0 * l.edge + reach + 3.0;
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (cx, cy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let (px, py) = (cx as f64 - 0.5, cy as f64 - 0.5);
            let p = [px * a[0] + py * a[1] + a[2], px * a[3] + py * a[4] + a[5]];
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let (bx0, by0) = (((lo[0] - pad).floor() as i64).max(0), ((lo[1] - pad).floor() as i64).max(0));
        let (bx1, by1) = (((hi[0] + pad).ceil() as i64 + 1).min(w), ((hi[1] + pad).ceil() as i64 + 1).min(h));
        if bx1 <= bx0 || by1 <= by0 {
            return None;
        }
        // mkmmd.stack.warped: the map cast to float32, then moved to the part's and the box's corners in float64
        let m = a.map(|v| v as f32 as f64);
        let t0 = (m[0] * x0 as f64 + m[1] * y0 as f64) - bx0 as f64;
        let t1 = (m[3] * x0 as f64 + m[4] * y0 as f64) - by0 as f64;
        Some(Place {
            part: [x0, y0, x1, y1],
            area: [bx0, by0, bx1, by1],
            map: [m[0], m[1], m[2] + t0, m[3], m[4], m[5] + t1],
        })
    }

    /// One layer laid on `dst` at `at`, cut from its figure's planes: the coverage `cov` (single floats) and its picture
    /// times it `pic`, both over the box `area` of the picture.
    #[allow(clippy::too_many_arguments)]
    fn lay(&self, cx: &mut Ctx, dst: &wgpu::Buffer, l: &Layer, at: &Place, cov: &wgpu::Buffer, pic: &wgpu::Buffer,
           area: &Area) {
        let (w, h) = (cx.w, cx.h);
        let [x0, y0, x1, y1] = at.part;
        let [bx0, by0, bx1, by1] = at.area;
        let (cw, ch) = ((x1 - x0) as u32, (y1 - y0) as u32);
        let (bw, bh) = ((bx1 - bx0) as u32, (by1 - by0) as u32);
        let (part, laid) = (Area::whole(cw, ch), Area::whole(bw, bh));
        let picture = l.fill.is_none();
        let flip = l.flip.unwrap_or(-1);
        let c0 = cx.scratch("stack.c0", part.bytes());
        let cut = Params { w, h, x0: x0 as u32, y0: y0 as u32, cw, ch, flip, ..Default::default() }.cut_over(area);
        run(cx, "cut1", cut, &[(3, cov), (8, &c0)], (cw, ch));
        let m = figure::inverse64(&at.map);
        let bc = cx.scratch("stack.cov", laid.bytes());
        plane::warp(cx, &c0, &part, &bc, &laid, &m, BORDER_CONSTANT);
        let br = cx.scratch("stack.rgb", laid.bytes4());
        if picture {
            let r0 = cx.scratch("stack.r0", part.bytes4());
            run(cx, "cut", cut, &[(1, pic), (2, &r0)], (cw, ch));
            warp(cx, &r0, (cw, ch), &br, (bw, bh), &m, BORDER_CONSTANT, [0.0; 3]);
        }
        let t1 = cx.scratch("stack.t1", laid.bytes());
        let t2 = cx.scratch("stack.t2", laid.bytes());
        let ring = cx.scratch("stack.ring", laid.bytes());
        let dark = cx.scratch("stack.dark", laid.bytes());
        if l.edge > 0.0 {
            plane::grow(cx, &bc, &ring, &t1, &t2, &laid, l.edge);
        }
        let mut p = Params { w, h, x0: bx0 as u32, y0: by0 as u32, cw: bw, ch: bh, ..Default::default() };
        if let Some(s) = self.shadow.as_ref().filter(|s| s.alpha > 0.0) {
            let dx = if l.flip.is_some() { -s.offset[0] } else { s.offset[0] };
            plane::shift(cx, if l.edge > 0.0 { &ring } else { &bc }, &dark, &laid, dx * l.s, s.offset[1] * l.s);
            if s.blur * l.s > 0.3 {
                plane::blur(cx, &dark, &dark, &t1, &laid, s.blur * l.s);
            }
            p.shadow = 1;
            p.shade = [s.color[0], s.color[1], s.color[2], (s.alpha * l.alpha) as f32];
        }
        let e = self.edge.unwrap_or([0.0; 3]);
        let f = l.fill.unwrap_or([0.0; 3]);
        p.fill = (!picture) as u32;
        p.edged = (l.edge > 0.0) as u32;
        p.alpha = l.alpha as f32;
        p.colour = [f[0], f[1], f[2], 0.0];
        p.edge = [e[0], e[1], e[2], 0.0];
        run(cx, "layer_box", p, &[(2, dst), (3, &bc), (4, &ring), (5, &dark), (6, &br)], (bw, bh));
    }
}

/// The key a cut-out is shared by (the same figure on the same frame, set in the same way).
fn key(c: &Cutout) -> String {
    format!("{}|{:?}|{:?}", c.layer, c.plate, c.warp.as_ref().map(|s| (s.m, s.shrink)))
}

/// A figure cut once for the layers that show it: its support (None: nothing covered), the box of the picture its
/// planes cover and the planes (its coverage, its picture times it).
struct Cut {
    support: Option<[u32; 4]>,
    area: Area,
    cov: wgpu::Buffer,
    pic: wgpu::Buffer,
}

impl Effect for Stack {
    fn inputs(&self) -> Vec<Input> {
        self.layers.iter().flat_map(|l| l.figure.inputs()).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        match &self.back {
            Back::Color(c) => figure::fill(cx, &dst, w, h, *c),
            Back::Picture => figure::copy(cx, &src, &dst, w, h),
            Back::Pattern(p) => spiral::paint(cx, p, &dst, w, h),
        }
        // each figure cut once over its support (its coverage and its picture times it): the support from its layer,
        // or (a warp that turns) cut over the whole picture and found there on the GPU, read back
        let mut cuts: HashMap<String, usize> = HashMap::new();
        let mut planes: Vec<Cut> = Vec::new();
        let mut unknown: Vec<usize> = Vec::new();
        let mut which = Vec::with_capacity(self.layers.len());
        let mut at = 0;
        for l in &self.layers {
            let k = l.figure.inputs().len();
            let picture = self.layers.iter().any(|o| o.fill.is_none() && key(&o.figure) == key(&l.figure));
            let i = match cuts.get(&key(&l.figure)) {
                Some(&i) => i,
                None => {
                    let i = planes.len();
                    let support = figure::support(cx, &l.figure, &files[at]);
                    let area = match support {
                        Some(Some(b)) => Area::of((w, h), b),
                        Some(None) => Area::whole(0, 0),
                        None => Area::whole(w, h),
                    };
                    let cov = cx.scratch_at("stack.cut", i, area.bytes().max(16));
                    let pic = cx.scratch_at("stack.cutpic", i, area.bytes4().max(16));
                    if support.is_none() {
                        unknown.push(i);
                    }
                    if area.w > 0 && area.h > 0 {
                        figure::cut_box(cx, &l.figure, &files[at..at + k], &area, &cov, picture.then_some(&pic));
                    }
                    planes.push(Cut { support: support.flatten(), area, cov, pic });
                    cuts.insert(key(&l.figure), i);
                    i
                }
            };
            which.push(i);
            at += k;
        }
        if !unknown.is_empty() {
            let init: Vec<u32> = unknown.iter().flat_map(|_| [u32::MAX, u32::MAX, 0, 0]).collect();
            let boxes = cx.scratch("stack.boxes", 16 * unknown.len() as u64);
            cx.gpu.queue.write_buffer(&boxes, 0, bytemuck::cast_slice(&init));
            for (slot, &i) in unknown.iter().enumerate() {
                let p = Params { w, h, slot: slot as u32, ..Default::default() }.cut_over(&planes[i].area);
                run(cx, "support", p, &[(3, &planes[i].cov), (7, &boxes)], (w, h));
            }
            let got: Vec<u32> = bytemuck::pod_collect_to_vec(&cx.read_back(&boxes, 16 * unknown.len() as u64)?);
            for (b, &i) in got.chunks(4).zip(&unknown) {
                planes[i].support = (b[2] > b[0] && b[3] > b[1]).then(|| [b[0], b[1], b[2], b[3]]);
            }
        }
        for (l, &i) in self.layers.iter().zip(&which) {
            let Some(b) = planes[i].support else {
                continue;                                        // nothing covered: nothing laid
            };
            if l.alpha <= 0.0 {
                continue;
            }
            let support = [b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64];
            if let Some(p) = self.place(l, support, w as i64, h as i64) {
                let c = &planes[i];
                let (cov, pic, area) = (c.cov.clone(), c.pic.clone(), c.area);
                self.lay(cx, &dst, l, &p, &cov, &pic, &area);
            }
        }
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
