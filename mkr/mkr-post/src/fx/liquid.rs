//! liquid: the 60s oil-and-water light show (mkmmd.fields.liquid; FieldFx._liquid), drawn as the CPU draws it: on a work
//! grid, resized up by bicubic to the render picture and the render picture to the frame (shaders/fx_liquid.wgsl).
//! Python (mkmmd.gpufx.liquid) works out the sizes, the dishes' turns and balls at the frame's second, the bubbles, the
//! noise calls and the constants.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::fields::{NoiseCall, Region, noise_tables};
use super::figure::{blur, plane_bytes, resize_cubic};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/fields.wgsl"), "\n", include_str!("../shaders/fx_liquid.wgsl"));
const SWIRL_BLUR: f64 = 2.0;        // the swirl's noise is blurred this much (mkmmd.fields.liquid)

/// An ink's dish of dye: its turn (cos, sin) and pivot, its colour.
#[derive(Debug, Clone, Deserialize)]
pub struct Dye {
    pub turn: [f32; 4],
    pub ink: [f32; 3],
}

/// A dish of oil (the clear one first): its turn and pivot, 1 - its ink (None: clear oil), its balls (x, y, radius^2).
#[derive(Debug, Clone, Deserialize)]
pub struct Dish {
    pub turn: [f32; 4],
    #[serde(default)]
    pub unink: Option<[f32; 3]>,
    pub balls: Vec<[f32; 3]>,
}

/// The oil cells' constants (mkmmd.fields._cell_walls).
#[derive(Debug, Clone, Deserialize)]
pub struct Cells {
    pub seed: i32,
    /// float32 of 0.06 x scale, half of it, 0.45 x scale, the patch cut, 1.5 px, the wall's reach, 2.4 px, 1.4 px.
    pub k: [f32; 8],
}

/// A bubble: its centre and radius, its glint's radius, its glint's centre, its box.
#[derive(Debug, Clone, Deserialize)]
pub struct BubbleSpec {
    pub c: [f32; 4],
    pub g: [f32; 2],
    #[serde(rename = "box")]
    pub area: [u32; 4],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Liquid {
    pub work: [u32; 2],
    pub render: [u32; 2],
    pub f: u32,
    pub div: u32,
    pub rc: [f32; 2],
    pub runit: f32,
    pub z: f32,
    pub lw: f32,
    pub lc: f32,
    pub amp: f32,
    pub gdx: f32,
    pub level: f32,
    pub flare: f32,
    pub aa: f32,
    /// float32 of 1.7, 4 and 2.8 render px and 0.06 render units.
    pub rpx: [f32; 4],
    pub rough: f32,
    pub marble: f32,
    pub light: [f32; 3],
    pub dyes: Vec<Dye>,
    pub dishes: Vec<Dish>,
    #[serde(default)]
    pub cells: Option<Cells>,
    pub bubbles: Vec<BubbleSpec>,
    /// float32 of 1.4, 1.8, 1.5 and 1.6 px of the frame.
    pub bub: [f32; 4],
    /// The noise calls: the swirl's, five per dye, one per dish of oil, then the cells' patches.
    pub calls: Vec<NoiseCall>,
    pub blend: u32,
    pub strength: f32,
    #[serde(default)]
    pub mask: Option<Region>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    ww: u32,
    hh: u32,
    rw: u32,
    rh: u32,
    w: u32,
    h: u32,
    f: u32,
    nink: u32,
    dish: u32,
    clear: u32,
    mask: u32,
    blend: u32,
    nballs: u32,
    nbubbles: u32,
    cells_call: u32,
    cells_seed: i32,
    rcx: f32,
    rcy: f32,
    runit: f32,
    z: f32,
    lw: f32,
    lc: f32,
    amp: f32,
    gdx: f32,
    level: f32,
    flare: f32,
    aa: f32,
    rpx17: f32,
    rpx4: f32,
    rpx28: f32,
    runit06: f32,
    fi: f32,
    rough: f32,
    marble: f32,
    sizeu: f32,
    sizeu05: f32,
    patchk: f32,
    cut: f32,
    px15: f32,
    wallk: f32,
    px24: f32,
    px14: f32,
    strength: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    ca: f32,
    sa: f32,
    pvx: f32,
    pvy: f32,
    pad3: f32,
    pad4: f32,
    light: [f32; 4],
    unink: [f32; 4],
    bub: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DyeGpu {
    turn: [f32; 4],
    ink: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BubbleGpu {
    c: [f32; 4],
    g: [f32; 4],
    area: [u32; 4],
}

impl Effect for Liquid {
    fn inputs(&self) -> Vec<Input> {
        self.mask.iter().map(Region::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let [ww, hh] = self.work;
        let [rw, rh] = self.render;
        let nink = self.dyes.len() as u32;
        let cells = self.cells.as_ref();
        let ck = cells.map_or([0.0; 8], |c| c.k);
        let base = Params {
            ww, hh, rw, rh, w, h, f: self.f, nink, dish: 0, clear: 0,
            mask: self.mask.as_ref().map_or(0, |r| if r.ground { 2 } else { 1 }), blend: self.blend, nballs: 0,
            nbubbles: self.bubbles.len() as u32, cells_call: 1 + 5 * nink + self.dishes.len() as u32,
            cells_seed: cells.map_or(0, |c| c.seed), rcx: self.rc[0], rcy: self.rc[1], runit: self.runit, z: self.z,
            lw: self.lw, lc: self.lc, amp: self.amp, gdx: self.gdx, level: self.level, flare: self.flare, aa: self.aa,
            rpx17: self.rpx[0], rpx4: self.rpx[1], rpx28: self.rpx[2], runit06: self.rpx[3], fi: self.f as f32,
            rough: self.rough, marble: self.marble, sizeu: ck[0], sizeu05: ck[1], patchk: ck[2], cut: ck[3],
            px15: ck[4], wallk: ck[5], px24: ck[6], px14: ck[7], strength: self.strength, pad0: 0.0, pad1: 0.0,
            pad2: 0.0, ca: 0.0, sa: 0.0, pvx: 0.0, pvy: 0.0, pad3: 0.0, pad4: 0.0,
            light: [self.light[0], self.light[1], self.light[2], 0.0], unink: [0.0; 4], bub: self.bub,
        };
        let (cubic, calls) = noise_tables(cx, &self.calls);
        let dyes: Vec<DyeGpu> = self.dyes.iter()
            .map(|d| DyeGpu { turn: d.turn, ink: [d.ink[0], d.ink[1], d.ink[2], 0.0] }).collect();
        let dyes = cx.table("liquid.dyes", &dyes);
        let mut bubbles: Vec<BubbleGpu> = self.bubbles.iter()
            .map(|b| BubbleGpu { c: b.c, g: [b.g[0], b.g[1], 0.0, 0.0], area: b.area }).collect();
        if bubbles.is_empty() {
            bubbles.push(BubbleGpu::zeroed());
        }
        let bubbles = cx.table("liquid.bubbles", &bubbles);
        let wn = plane_bytes(ww, hh);
        let rn = plane_bytes(rw, rh);
        let psi = cx.scratch("liquid.psi", wn);
        let psib = cx.scratch("liquid.psib", wn);
        let tmp = cx.scratch("liquid.tmp", wn);
        let warp = cx.scratch("liquid.warp", wn);
        let dye = cx.scratch("liquid.dye", wn);
        let field = cx.scratch("liquid.field", wn);
        let slope = cx.scratch("liquid.slope", wn);
        let pic = cx.scratch("liquid.pic", rn);
        let up = cx.scratch("liquid.up", rn);
        let out = cx.scratch("liquid.out", plane_bytes(w, h));
        let region = match &self.mask {
            Some(r) => r.plane(cx, &files[0]),
            None => src.clone(),
        };
        let work = crate::gpu::tiles(ww, hh);
        let render = crate::gpu::tiles(rw, rh);
        let k = |cx: &Ctx, e: &'static str| cx.kernel("fx_liquid", SRC, e);
        let u = cx.uniform(&base);
        // the water's swirl and the dye
        let kp = k(cx, "liquid_psi");
        cx.pass(&kp, &[(0, &u), (2, &psi), (10, &cubic), (11, &calls)], work);
        blur(cx, &psi, &psib, &tmp, ww, hh, SWIRL_BLUR);
        let kw = k(cx, "liquid_warp");
        cx.pass(&kw, &[(0, &u), (1, &psib), (2, &warp), (4, &dye), (6, &dyes), (10, &cubic), (11, &calls)], work);
        let f = (self.f as f64, self.f as f64);
        resize_cubic(cx, &dye, (ww, hh), &up, (rw, rh), f);
        let kd = k(cx, "liquid_dye");
        cx.pass(&kd, &[(0, &u), (2, &pic), (3, &up)], render);
        // the oil, the clear dish first
        let (kf, ks, ko) = (k(cx, "liquid_field"), k(cx, "liquid_slope"), k(cx, "liquid_oil"));
        for (j, d) in self.dishes.iter().enumerate() {
            let balls: Vec<[f32; 4]> = d.balls.iter().map(|b| [b[0], b[1], b[2], 0.0]).collect();
            let balls = cx.table("liquid.balls", if balls.is_empty() { &[[0.0f32; 4]][..] } else { &balls[..] });
            let un = d.unink.unwrap_or([0.0; 3]);
            let p = cx.uniform(&Params {
                dish: 1 + 5 * nink + j as u32, clear: d.unink.is_none() as u32, nballs: d.balls.len() as u32,
                ca: d.turn[0], sa: d.turn[1], pvx: d.turn[2], pvy: d.turn[3], unink: [un[0], un[1], un[2], 0.0], ..base
            });
            cx.pass(&kf, &[(0, &p), (2, &field), (3, &warp), (5, &balls), (10, &cubic), (11, &calls)], work);
            cx.pass(&ks, &[(0, &p), (1, &field), (2, &slope)], work);
            resize_cubic(cx, &slope, (ww, hh), &up, (rw, rh), f);
            cx.pass(&ko, &[(0, &p), (2, &pic), (3, &up)], render);
        }
        if cells.is_some() {
            let kc = k(cx, "liquid_cells");
            cx.pass(&kc, &[(0, &u), (2, &field), (3, &warp), (6, &dyes), (10, &cubic), (11, &calls)], work);
            resize_cubic(cx, &field, (ww, hh), &up, (rw, rh), f);
            let kl = k(cx, "liquid_walls");
            cx.pass(&kl, &[(0, &u), (2, &pic), (3, &up)], render);
        }
        // the frame: the render picture resized up, the bubbles, the lay
        if self.div > 1 {
            resize_cubic(cx, &pic, (rw, rh), &out, (w, h), (self.div as f64, self.div as f64));
        } else {
            super::figure::copy(cx, &pic, &out, w, h);
        }
        if !self.bubbles.is_empty() {
            let kb = k(cx, "liquid_bubbles");
            cx.pass(&kb, &[(0, &u), (2, &out), (7, &bubbles)], cx.tiles());
        }
        let kl = k(cx, "liquid_lay");
        cx.pass(&kl, &[(0, &u), (1, &src), (2, &dst), (3, &out), (8, &region)], cx.tiles());
        Ok(Drawn::Dst)
    }
}
