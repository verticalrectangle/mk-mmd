//! A silkscreen print of a collage's figure (mkmmd.framefx.silk_bands, silk_print; shaders/fx_silk.wgsl): its luma
//! smoothed inside the figure, cut into flat bands at quantiles of the luma the figure covers (worked out here from a
//! read back of it, as np.quantile works them out), the key plate with the lines where the luma dips under its mean.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::Ctx;
use super::figure::{self, plane_bytes};

const SRC: &str = include_str!("../shaders/fx_silk.wgsl");
/// mkmmd.framefx.SILK_KEY, SILK_DEPTH.
const SILK_KEY: f64 = 0.2;
const SILK_DEPTH: f32 = 0.06;
/// silk_levels takes its quantiles of at most about this many of the figure's pixels (every k-th).
const SAMPLE: usize = 200_000;

/// How a collage's figure is printed (FrameFx._fill's silkscreen): the number of its inks (2 to 4), the luma smoothed
/// `soft` px, the lines over `line` px (0: none), and the key plate printed `offset` (dx, dy) px off register.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Bands {
    pub n: u32,
    pub soft: f64,
    pub line: f64,
    pub offset: [f64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    steps: u32,
    line: u32,
    t0: f32,
    t1: f32,
    t2: f32,
    depth: f32,
    nan: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

fn run(cx: &mut Ctx, entry: &'static str, p: Params, bind: &[(u32, &wgpu::Buffer)]) {
    let k = cx.kernel("fx_silk", SRC, entry);
    let u = cx.uniform(&p);
    let mut all: Vec<(u32, &wgpu::Buffer)> = vec![(0, &u)];
    all.extend_from_slice(bind);
    let tiles = cx.tiles();
    cx.pass(&k, &all, tiles);
}

/// mkmmd.framefx.silk_levels: the n - 1 thresholds of luma `vals` (the figure's pixels covered over half, in order),
/// np.quantile (linear) of every k-th of them at the key's share and the others' equal shares, half a code value up,
/// as float32; None for no pixel.
pub fn levels(vals: &[f32], n: u32) -> Option<Vec<f32>> {
    if vals.is_empty() {
        return None;
    }
    let step = (vals.len() / SAMPLE).max(1);
    let mut v: Vec<f32> = vals.iter().step_by(step).copied().collect();
    v.sort_unstable_by(f32::total_cmp);
    let last = v.len() - 1;
    let mut qs = vec![SILK_KEY];
    qs.extend((1..n as usize - 1).map(|j| SILK_KEY + (1.0 - SILK_KEY) * j as f64 / (n as f64 - 1.0)));
    Some(qs.into_iter().map(|q| {
        let at = last as f64 * q;
        let lo = (at.floor() as usize).min(last);
        let hi = (lo + 1).min(last);
        let g = at - lo as f64;
        let (a, b) = (v[lo], v[hi]);
        let d = (b - a) as f64;                              // numpy: b - a in float32, then times gamma in float64
        let x = if g >= 0.5 { b as f64 - d * (1.0 - g) } else { a as f64 + d * g };
        (x + 0.5 / 255.0) as f32
    }).collect())
}

/// mkmmd.framefx.silk_bands of the figure (its picture `pic`, coverage `cov` .x; w x h planes) printed as `b` says,
/// into plane `out`: .x the key plate times the coverage, .y and .z the colour steps (their ink over the one before).
/// Waits for the GPU (the quantiles are worked out from what it gives back).
pub fn bands(cx: &mut Ctx, pic: &wgpu::Buffer, cov: &wgpu::Buffer, out: &wgpu::Buffer, b: &Bands) -> Result<()> {
    let (w, h) = (cx.w, cx.h);
    let n = plane_bytes(w, h);
    let a = cx.scratch("silk.a", n);
    let l = cx.scratch("silk.l", n);
    let t = cx.scratch("silk.t", n);
    let p = Params { w, h, steps: b.n.saturating_sub(2), depth: SILK_DEPTH, nan: f32::NAN, ..Default::default() };
    run(cx, "silk_luma", p, &[(1, pic), (2, &a), (3, cov)]);
    if b.soft > 0.0 {
        figure::blur(cx, &a, &a, &t, w, h, b.soft);
        run(cx, "silk_mean", p, &[(1, &a), (2, &l), (3, cov)]);
    } else {
        run(cx, "silk_plain", p, &[(1, &a), (2, &l), (3, cov)]);
    }
    let vals = cx.scratch("silk.vals", w as u64 * h as u64 * 4);
    run(cx, "silk_vals", p, &[(1, &l), (3, cov), (5, &vals)]);
    let got = cx.read_back(&vals, w as u64 * h as u64 * 4)?;
    let kept: Vec<f32> = bytemuck::pod_collect_to_vec::<u8, f32>(&got).into_iter().filter(|v| !v.is_nan()).collect();
    let Some(lv) = levels(&kept, b.n) else {
        figure::fill(cx, out, w, h, [0.0; 3]);
        return Ok(());
    };
    let mut q = Params { t0: lv[0], t1: lv.get(1).copied().unwrap_or(0.0), t2: lv.get(2).copied().unwrap_or(0.0), ..p };
    if b.line > 0.0 {
        q.line = 1;
        run(cx, "silk_masked", q, &[(1, &l), (2, &a)]);
        figure::blur(cx, &a, &a, &t, w, h, b.line);
    }
    run(cx, "silk_plates", q, &[(1, &l), (2, out), (4, &a)]);
    Ok(())
}
