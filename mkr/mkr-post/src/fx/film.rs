//! film: an old film print (mkmmd.picturefx.film; PictureFx._film). The print of a film frame is made once, on the
//! first clip frame of its step, from the picture as it reaches the op there, and kept (Ctx::held) for the step's
//! other frames; each frame mixes it over its own picture by the strength. Python (mkmmd.gpufx.film) works out the
//! step, the jolt (gain and weave), the dust, scratches and hairs (drawn with OpenCV into masks, sent as their parts),
//! the grain's normals (numpy's generator) and the constants; the print is made here (shaders/fx_film.wgsl). A film
//! limited to a `rect` prints the rect's box as a frame of its own (PictureFx._region) and is kept inside the rect by
//! its ramps (FrameFx._apply).

use std::sync::Arc;

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{self, plane_bytes};
use super::warp::warp;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_film.wgsl");
/// The grain's spread in 64-bit floats (compiled only on a device that has them).
const SPREAD: &str = include_str!("../shaders/fx_film_spread.wgsl");
const BORDER_REFLECT_101: u32 = 4;

/// A damage mask's part: where it lies (x0, y0, x1, y1) and its values (v / 255, float32, removed once read).
#[derive(Debug, Clone, Deserialize)]
pub struct Mask {
    #[serde(rename = "box")]
    pub area: [u32; 4],
    pub part: String,
}

impl Mask {
    fn input(&self) -> Input {
        let [x0, y0, x1, y1] = self.area;
        Input::F32 { path: self.part.clone(), count: ((x1 - x0) * (y1 - y0)) as usize }
    }
}

/// The halation: its amount and colour, its two blurs' sigmas (px).
#[derive(Debug, Clone, Deserialize)]
pub struct Halation {
    pub amount: f32,
    pub color: [f32; 3],
    pub near: f64,
    pub far: f64,
}

/// The grain: its normals (gw x gh float32, numpy's, removed once read), resized to the frame by bicubic, its amount.
#[derive(Debug, Clone, Deserialize)]
pub struct Grain {
    pub normals: String,
    pub size: [u32; 2],
    pub amount: f32,
}

/// How a step's print is made (mkmmd.picturefx.film), every constant float32 as numpy uses it.
#[derive(Debug, Clone, Deserialize)]
pub struct Print {
    pub mono: bool,
    pub dye_on: bool,
    pub sat: f32,
    /// float32 of 0.45 x dye, 0.3 x dye and dye.
    pub dye: [f32; 3],
    /// The S curve: float32(c / 2), float32(tanh(c / 4)); None: none.
    #[serde(default)]
    pub curve: Option<[f32; 2]>,
    /// The weave: float32 of dx x px and dy x px; None: none.
    #[serde(default)]
    pub weave: Option<[f32; 2]>,
    #[serde(default)]
    pub light: Option<Mask>,
    #[serde(default)]
    pub dark: Option<Mask>,
    /// The masks' blur (px), 0: none.
    pub soft: f64,
    pub gain: f32,
    #[serde(default)]
    pub halation: Option<Halation>,
    #[serde(default)]
    pub grain: Option<Grain>,
    #[serde(default)]
    pub vignette: Option<f32>,
}

/// A film limited to a `rect`: the box it prints as a frame of its own (mkmmd.picturefx.rect_box: x0, y0, x1, y1 px)
/// and how much of each pixel of it the rect covers (rect_ramps: `ramps` float32, the box's rows' then its columns';
/// removed once read).
#[derive(Debug, Clone, Deserialize)]
pub struct Region {
    #[serde(rename = "box")]
    pub area: [u32; 4],
    pub ramps: String,
}

impl Region {
    fn input(&self) -> Input {
        let [x0, y0, x1, y1] = self.area;
        Input::F32 { path: self.ramps.clone(), count: ((x1 - x0) + (y1 - y0)) as usize }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Film {
    /// The entry (its print is kept by it) and the film frame shown.
    pub entry: u32,
    pub step: i64,
    /// On the step's first frame: how its print is made; None on the others (the print kept then).
    #[serde(default)]
    pub print: Option<Print>,
    pub strength: f64,
    #[serde(default)]
    pub region: Option<Region>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    mono: u32,
    dye_on: u32,
    curve: u32,
    light: u32,
    dark: u32,
    grain: u32,
    vig: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    mixed: u32,
    pad1: u32,
    pad2: u32,
    sat: f32,
    dye45: f32,
    dye3: f32,
    dye: f32,
    c2: f32,
    tc4: f32,
    gain: f32,
    halo_amt: f32,
    grain_amt: f32,
    pad4: f32,
    vig_amt: f32,
    pad3: f32,
    halo_rgb: [f32; 4],
    mix: [f32; 4],
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)], groups: (u32, u32)) {
    let k = cx.kernel("fx_film", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    cx.pass(&k, &all, groups);
}

impl Print {
    fn inputs(&self) -> Vec<Input> {
        let mut v: Vec<Input> = self.light.iter().chain(&self.dark).map(Mask::input).collect();
        if let Some(g) = &self.grain {
            v.push(Input::F32 { path: g.normals.clone(), count: (g.size[0] * g.size[1]) as usize });
        }
        v
    }

    /// The print of `src` into `out`.
    fn make(&self, cx: &mut Ctx, src: &wgpu::Buffer, out: &wgpu::Buffer, files: &[Arc<Decoded>]) -> Result<()> {
        let (w, h) = (cx.w, cx.h);
        let n = plane_bytes(w, h);
        let tiles = cx.tiles();
        let c = self.curve.unwrap_or([0.0; 2]);
        let mut p = Params {
            w, h, mono: self.mono as u32, dye_on: self.dye_on as u32, curve: self.curve.is_some() as u32,
            light: self.light.is_some() as u32, dark: self.dark.is_some() as u32, grain: self.grain.is_some() as u32,
            vig: self.vignette.is_some() as u32, sat: self.sat, dye45: self.dye[0], dye3: self.dye[1], dye: self.dye[2],
            c2: c[0], tc4: c[1], gain: self.gain, vig_amt: self.vignette.unwrap_or(0.0), ..Default::default()
        };
        let a = cx.scratch("film.a", n);
        run(cx, "film_tone", p, &[(1, src), (2, &a)], tiles);
        let cur = match self.weave {
            Some([dx, dy]) => {
                let m = figure::inverse(&[1.0, 0.0, dx, 0.0, 1.0, dy]);
                warp(cx, &a, (w, h), out, (w, h), &m, BORDER_REFLECT_101, [0.0; 3]);
                out.clone()
            }
            None => a,
        };
        // the damage, then the gain
        let mut next = 0;
        let t = cx.scratch("film.t", n);
        let mut planes = Vec::new();
        for (label, m) in [("film.light", &self.light), ("film.dark", &self.dark)] {
            let plane = cx.scratch(label, n);
            if let Some(m) = m {
                let [x0, y0, x1, y1] = m.area;
                let part = cx.upload(&m.input(), &files[next]);
                next += 1;
                let q = Params { x0, y0, cw: x1 - x0, ch: y1 - y0, ..p };
                run(cx, "film_put", q, &[(2, &plane), (7, &part)], tiles);
                if self.soft >= 0.3 {
                    figure::blur(cx, &plane, &plane, &t, w, h, self.soft);
                }
            }
            planes.push(plane);
        }
        run(cx, "film_damage", p, &[(2, &cur), (3, &planes[0]), (6, &planes[1])], tiles);
        // the halation
        if let Some(hl) = &self.halation {
            let lights = cx.scratch("film.lights", n);
            let near = cx.scratch("film.near", n);
            let far = cx.scratch("film.far", n);
            run(cx, "film_lights", p, &[(1, &cur), (2, &lights)], tiles);
            figure::blur(cx, &lights, &near, &t, w, h, hl.near);
            figure::blur(cx, &lights, &far, &t, w, h, hl.far);
            run(cx, "film_pair", p, &[(1, &near), (2, &lights), (3, &far)], tiles);
            p.halo_amt = hl.amount;
            p.halo_rgb = [hl.color[0], hl.color[1], hl.color[2], 0.0];
            run(cx, "film_halo", p, &[(2, &cur), (4, &lights)], tiles);
        }
        // the grain, the vignette, the clip
        let noise = cx.scratch("film.noise", n);
        let mut spread = cx.scratch("film.spread", 16);
        if let Some(g) = &self.grain {
            let [gw, gh] = g.size;
            let normals = cx.upload(&Input::F32 { path: g.normals.clone(), count: (gw * gh) as usize }, &files[next]);
            let small = if (gw, gh) == (w, h) { noise.clone() } else { cx.scratch("film.small", n) };
            run(cx, "film_put", Params { w: gw, h: gh, cw: gw, ch: gh, ..Default::default() },
                &[(2, &small), (7, &normals)], crate::gpu::tiles(gw, gh));
            if (gw, gh) != (w, h) {
                figure::resize_cubic(cx, &small, (gw, gh), &noise, (w, h), (w as f64 / gw as f64, h as f64 / gh as f64));
            }
            let sums = cx.scratch("film.sums", 8 * h as u64);
            run(cx, "film_rows", p, &[(5, &noise), (8, &sums)], (h.div_ceil(64), 1));
            p.grain_amt = g.amount;
            spread = grain_spread(cx, &sums, (w, h))?;
        }
        run(cx, "film_grain", p, &[(2, &cur), (5, &noise), (11, &spread)], tiles);
        if cur != *out {
            figure::copy(cx, &cur, out, w, h);
        }
        Ok(())
    }
}

/// The grain field's spread from its rows' sums (film_rows: Σ v and Σ v² along each row, float32): the sums folded in
/// order in float64, std = sqrt(max(Σ v² / n - mean², 0)), then float32(max(std, 1e-6)), the buffer holding it (its
/// first float, what film_grain divides by). On the GPU in 64-bit floats (fx_film_spread.wgsl) where the device has
/// them, the frame's encoding going straight on; otherwise the sums are read back and folded here.
fn grain_spread(cx: &mut Ctx, sums: &wgpu::Buffer, (w, h): (u32, u32)) -> Result<wgpu::Buffer> {
    if cx.gpu.f64 {
        let out = cx.scratch("film.spread", 16);
        let k = cx.kernel("fx_film_spread", SPREAD, "film_spread");
        let u = cx.uniform(&[w, h, 0u32, 0u32]);
        cx.pass(&k, &[(0, &u), (1, sums), (2, &out)], (1, 1));
        return Ok(out);
    }
    let rows: Vec<f32> = bytemuck::pod_collect_to_vec(&cx.read_back(sums, 8 * h as u64)?);
    let count = (w as f64) * (h as f64);
    let (s, q) = rows.chunks(2).fold((0.0f64, 0.0f64), |(s, q), r| (s + r[0] as f64, q + r[1] as f64));
    let mean = s / count;
    let std = (q / count - mean * mean).max(0.0).sqrt();
    Ok(cx.table("film.spread", &[std.max(1e-6) as f32, 0.0, 0.0, 0.0]))
}

impl Effect for Film {
    fn inputs(&self) -> Vec<Input> {
        let mut v = self.print.as_ref().map_or_else(Vec::new, Print::inputs);
        v.extend(self.region.iter().map(Region::input));
        v
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        // the print of the whole picture, or of the rect's box as a picture of its own
        let [x0, y0, x1, y1] = self.region.as_ref().map_or([0, 0, w, h], |r| r.area);
        let (bw, bh) = (x1 - x0, y1 - y0);
        let key = format!("film.{}", self.entry);
        let (print, held) = cx.held(&key, plane_bytes(bw, bh));
        let made = self.print.as_ref().map_or(0, |p| p.inputs().len());
        match &self.print {
            Some(p) if self.region.is_none() => {
                p.make(cx, &src, &print, files)?;
                cx.hold(&key, self.step);
            }
            Some(p) => {
                let part = cx.scratch("film.box", plane_bytes(bw, bh));
                let at = Params { w, h, x0, y0, cw: bw, ch: bh, ..Default::default() };
                run(cx, "film_crop", at, &[(1, &src), (2, &part)], crate::gpu::tiles(bw, bh));
                let stamps = cx.stamps.as_mut().map(|(set, marks)| (*set, &mut **marks));
                let mut boxed = Ctx { gpu: cx.gpu, enc: &mut *cx.enc, w: bw, h: bh, src: &part, dst: &print, lut: cx.lut,
                                      levels: cx.levels, res: cx.res, stamps };
                p.make(&mut boxed, &part, &print, &files[..made])?;
                cx.hold(&key, self.step);
            }
            None if held != Some(self.step) => bail!("film {}: the print of film frame {} is not kept", self.entry,
                                                     self.step),
            None => {}
        }
        let Some(r) = &self.region else {
            figure::copy(cx, &print, &dst, w, h);
            if self.strength < 1.0 {
                figure::mix(cx, &src, &dst, w, h, self.strength);
            }
            return Ok(Drawn::Dst);
        };
        let ramps = cx.upload(&r.input(), &files[made]);
        let p = Params {
            w, h, x0, y0, cw: bw, ch: bh, mixed: (self.strength < 1.0) as u32,
            mix: [(1.0 - self.strength) as f32, self.strength as f32, 0.0, 0.0], ..Default::default()
        };
        let tiles = cx.tiles();
        run(cx, "film_region", p, &[(1, &src), (2, &dst), (9, &print), (10, &ramps)], tiles);
        Ok(Drawn::Dst)
    }
}
