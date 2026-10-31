//! camcorder: a home video tape (mkmmd.picturefx.tape) with its stamp (picturefx.stamp_inks, laid as fx/inks.rs lays a
//! label's), mixed in by the strength (shaders/fx_camcorder.wgsl). Python draws the tape's random parts as
//! picturefx.tape_draws draws them (the noise before it is stretched over the picture, each line's tear, the tracking
//! band's snow) and works out each line's wash and scanline; mkr-post plays the tape as OpenCV does: YCrCb, the soft
//! luma, the colour bled (a box along the rows, a Gaussian, shifted right as OpenCV's old warp does), the noise stretched
//! (INTER_LINEAR), the lines torn (remap), the band's snow (INTER_NEAREST).

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, plane_bytes};
use super::inks::{self, Ink};
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = concat!(include_str!("../shaders/sample.wgsl"), "\n", include_str!("../shaders/fx_camcorder.wgsl"));

/// The colour bled to the right: a box `n` px wide along the rows, a Gaussian of `sigma` (x, y), shifted `shift` px.
#[derive(Debug, Clone, Deserialize)]
pub struct Bleed {
    pub n: u32,
    pub sigma: [f64; 2],
    pub shift: f64,
}

/// The noise before it is stretched over the picture (float32 normals, removed once read): the luma's (`ny_size`
/// w x h) and the colour's (`nc_size`, two a pixel), and the float32 shares of each added.
#[derive(Debug, Clone, Deserialize)]
pub struct Noise {
    pub ny: String,
    pub ny_size: [u32; 2],
    pub nc: String,
    pub nc_size: [u32; 2],
    pub amount: [f32; 2],
}

/// Which streaks across the tracking band are snow (float32 0 or 1, `size` w x h; removed once read), before they are
/// stretched over the picture.
#[derive(Debug, Clone, Deserialize)]
pub struct Snow {
    pub path: String,
    pub size: [u32; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Camcorder {
    /// The luma's Gaussian (sigma x, y); None: sharp.
    #[serde(default)]
    pub soft: Option<[f64; 2]>,
    #[serde(default)]
    pub chroma: Option<Bleed>,
    pub saturation: f32,
    #[serde(default)]
    pub noise: Option<Noise>,
    /// Four float32 a line (removed once read): its tear (px), wash, snow's strength and scanline factor; `lines` of
    /// them, the picture's height.
    pub rows: String,
    pub lines: u32,
    /// Whether the lines are torn (a tear over 0.01 px somewhere).
    pub tear: bool,
    #[serde(default)]
    pub snow: Option<Snow>,
    /// The stamp.
    pub inks: Vec<Ink>,
    pub strength: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    n: u32,
    noise: u32,
    tear: u32,
    snow: u32,
    nyw: u32,
    ncw: u32,
    nch: u32,
    sw: u32,
    nyh: u32,
    qx: i32,
    sat: f32,
    ny_amt: f32,
    nc_amt: f32,
    pad2: f32,
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)]) {
    let k = cx.kernel("fx_camcorder", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    let tiles = cx.tiles();
    cx.pass(&k, &all, tiles);
}

/// OpenCV's INTER_LINEAR weights along a side of `s` -> `d` as fx_camcorder.wgsl reads them, onto `tab`.
fn linear(tab: &mut Vec<u32>, s: u32, d: u32) {
    for (a, b, c) in figure::linear_tab(s, d) {
        tab.extend([a, b.to_bits(), c.to_bits()]);
    }
}

impl Camcorder {
    /// Its inputs: the lines' table, the noise (luma, colour), the snow, then the stamp's coverages.
    fn files(&self) -> Vec<Input> {
        let mut v = vec![Input::F32 { path: self.rows.clone(), count: 4 * self.lines as usize }];
        if let Some(n) = &self.noise {
            v.push(Input::F32 { path: n.ny.clone(), count: (n.ny_size[0] * n.ny_size[1]) as usize });
            v.push(Input::F32 { path: n.nc.clone(), count: (2 * n.nc_size[0] * n.nc_size[1]) as usize });
        }
        if let Some(s) = &self.snow {
            v.push(Input::F32 { path: s.path.clone(), count: (s.size[0] * s.size[1]) as usize });
        }
        v
    }
}

impl Effect for Camcorder {
    fn inputs(&self) -> Vec<Input> {
        let mut v = self.files();
        v.extend(self.inks.iter().filter_map(Ink::input));
        v
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        if self.lines != h {
            bail!("camcorder: {} lines for a picture {h} tall", self.lines);
        }
        let bytes = plane_bytes(w, h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        let ya = cx.scratch("camcorder.y", bytes);
        let ca = cx.scratch("camcorder.c", bytes);
        let t1 = cx.scratch("camcorder.t1", bytes);
        let t2 = cx.scratch("camcorder.t2", bytes);
        let none = cx.scratch("camcorder.none", 16);
        let own = self.files();
        let up: Vec<wgpu::Buffer> = own.iter().zip(files).map(|(i, d)| cx.upload(i, d)).collect();
        let n_own = up.len();
        let mut p = Params { w, h, sat: self.saturation, tear: self.tear as u32, ..Default::default() };
        let mut tab: Vec<u32> = Vec::with_capacity(7 * (w + h) as usize);
        let (ny, nc) = match &self.noise {
            Some(n) => {
                p.noise = 1;
                [p.nyw, p.nyh, p.ncw, p.nch] = [n.ny_size[0], n.ny_size[1], n.nc_size[0], n.nc_size[1]];
                [p.ny_amt, p.nc_amt] = n.amount;
                linear(&mut tab, n.ny_size[0], w);
                linear(&mut tab, n.ny_size[1], h);
                linear(&mut tab, n.nc_size[0], w);
                linear(&mut tab, n.nc_size[1], h);
                (up[1].clone(), up[2].clone())
            }
            None => {
                tab.resize(6 * (w + h) as usize, 0);
                (none.clone(), none.clone())
            }
        };
        let snow = match &self.snow {
            Some(s) => {
                p.snow = 1;
                p.sw = s.size[0];
                // cv2.resize INTER_NEAREST: source column floor(x / (w / sw)), in float64 as OpenCV works it out
                let ifx = 1.0 / (w as f64 / s.size[0] as f64);
                tab.extend((0..w).map(|x| ((x as f64 * ifx).floor() as u32).min(s.size[0] - 1)));
                up[n_own - 1].clone()
            }
            None => none.clone(),
        };
        let rows = up[0].clone();
        let tab = cx.table("camcorder.tab", &tab);
        run(cx, "split", p, &[(1, &src), (3, &ya), (4, &ca)]);
        if let Some([sx, sy]) = self.soft {
            figure::blur_xy(cx, &ya, &ya, &t1, w, h, sx, sy);
        }
        if let Some(b) = &self.chroma {
            run(cx, "bleed", Params { n: b.n, ..p }, &[(1, &ca), (2, &t1)]);
            figure::blur_xy(cx, &t1, &t1, &t2, w, h, b.sigma[0], b.sigma[1]);
            // the warp's inverse map moves by -dx, the float32 shift as float64; OpenCV rounds it to 1/1024 px
            let qx = (-(b.shift as f32 as f64) * 1024.0).round_ties_even() as i32 + 16;
            run(cx, "shift", Params { qx, ..p }, &[(1, &t1), (2, &ca)]);
        }
        run(cx, "combine", p, &[(2, &t1), (3, &ya), (4, &ca), (5, &ny), (6, &nc), (7, &tab)]);
        run(cx, "lines", p, &[(1, &t1), (2, &dst), (7, &tab), (8, &rows), (9, &snow)]);
        inks::lay(cx, &dst, &self.inks, &files[n_own..]);
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
