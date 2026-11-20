//! The vector look's construction-paper cut-out finished in mkr (docs/design.md: mkr's looks; the look itself is
//! mkmmd.core.cutout, which stays the reference): what the GPU stage (cutout_gpu.rs) needs from a job's passes and its
//! `finish` record, worked out on the CPU exactly as numpy works it out there.
//!
//! `mk render` (mkmmd/render/mkr.py, mkmmd.render.compose.gpu_finish) gives a look job it can finish here a record in the
//! request's looks (`outputs.<output>.finish.<job>`):
//!
//!   {files: [PNG path], step: the scissors' cut (cutout.boil_step), background: [r, g, b],
//!    cutout: {rough, grain, lift, seed, rim: {color, width} | null, shadow: {color, offset, blur, alpha} | null},
//!    shade_level: the least `shade` byte the look's light reaches (256: none),
//!    ids: [[group, tone]] per material id (id 0 the background: [0, -1]),
//!    tones: [{kind: flat | lit | drawn, cel, colors: [[r, g, b]], at: [brightness steps], grey}]}
//!
//! The random fields (the scissors' wander, the sheets' fibres, each piece's paper and lift) come from numpy's own
//! generators by the same seeds and draws (nprng.rs); the pieces are laid far to near by the median depth of each.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::Deserialize;

use crate::looks::PassData;
use crate::nprng::Rng;

/// mkmmd.core.cutout's constants.
pub const PIECE_SHADOW: f64 = 0.45;
pub const ROUGH_CELLS: [(f64, f32); 2] = [(7.0, 0.7), (1.8, 0.3)];
/// The margin of the paper sheet the pieces' papers are cut from (cutout.compose's `pad`).
pub const PAD: usize = 64;
/// The vector passes are drawn at twice the frame's size (shotstyle.VECTOR_SCALE).
pub const SCALE: usize = 2;
/// A tone's kind on the GPU.
pub const KIND_FLAT: u32 = 0;
pub const KIND_LIT: u32 = 1;
pub const KIND_DRAWN: u32 = 2;

#[derive(Deserialize, Debug, Clone)]
pub struct Rim {
    pub color: [f64; 3],
    pub width: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Shadow {
    pub color: [f64; 3],
    pub offset: [f64; 2],
    pub blur: f64,
    pub alpha: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Spec {
    pub rough: f64,
    pub grain: f64,
    pub lift: f64,
    pub seed: u64,
    pub rim: Option<Rim>,
    pub shadow: Option<Shadow>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Tone {
    pub kind: String,
    #[serde(default)]
    pub cel: bool,
    pub colors: Vec<[f64; 3]>,
    #[serde(default)]
    pub at: Vec<f64>,
    #[serde(default = "one")]
    pub grey: f64,
}

fn one() -> f64 {
    1.0
}

/// A look job mkr finishes itself (the module docstring).
#[derive(Deserialize, Debug, Clone)]
pub struct Finish {
    pub files: Vec<PathBuf>,
    pub step: u64,
    pub background: [f64; 3],
    pub cutout: Spec,
    pub shade_level: u32,
    pub ids: Vec<[i64; 2]>,
    pub tones: Vec<Tone>,
}

/// A vector job's passes at twice the frame's size (looks.rs), top row first.
pub struct Passes<'a> {
    pub w: usize,
    pub h: usize,
    pub id: &'a [u8],
    pub depth: &'a [u8],
    pub shade: &'a [u8],
    pub tex: Option<&'a [u8]>,
    /// The type over the look, at the frame's size: straight RGBA.
    pub typ: Option<&'a [u8]>,
}

impl<'a> Passes<'a> {
    pub fn of(passes: &'a [PassData]) -> Result<Passes<'a>> {
        let get = |name: &str| passes.iter().find(|p| p.name == name);
        let id = get("id").context("no id pass")?;
        let (h, w) = (id.shape[0], id.shape[1]);
        let sized = |name: &str, per: usize| -> Result<Option<&'a [u8]>> {
            match get(name) {
                Some(p) if p.bytes.len() == w * h * per => Ok(Some(&p.bytes[..])),
                Some(p) => bail!("the {name} pass has {} bytes, not {} x {h} x {w}", p.bytes.len(), per),
                None => Ok(None),
            }
        };
        let typ = match get("type") {
            Some(p) if p.bytes.len() == (w / SCALE) * (h / SCALE) * 4 => Some(&p.bytes[..]),
            Some(_) => bail!("the type pass is not the frame's size"),
            None => None,
        };
        if passes.iter().any(|p| p.name == "matte") {
            bail!("a matte's passes are composed on the CPU");
        }
        Ok(Passes {
            w,
            h,
            id: sized("id", 2)?.context("id")?,
            depth: sized("depth", 4)?.context("no depth pass")?,
            shade: sized("shade", 1)?.context("no shade pass")?,
            tex: sized("tex", 3)?,
            typ,
        })
    }

    fn id_at(&self, i: usize) -> usize {
        u16::from_le_bytes([self.id[2 * i], self.id[2 * i + 1]]) as usize
    }

    fn depth_at(&self, i: usize) -> f32 {
        f32::from_le_bytes([self.depth[4 * i], self.depth[4 * i + 1], self.depth[4 * i + 2], self.depth[4 * i + 3]])
    }
}

/// Python's round(): to the nearest whole number, halves to the even one.
pub fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// One of crayon.noise's fields: its grid of random values and, per row and per column of the picture, the grid cell
/// and the smoothstep weight (numpy's: the position in float64, the fraction and its weight in float32).
pub struct Noise {
    pub grid: Vec<f32>,
    pub cols: usize,
    pub ys: Vec<(u32, f32)>,
    pub xs: Vec<(u32, f32)>,
}

impl Noise {
    /// crayon.noise(rng, (rows, cols), (cy, cx)).
    pub fn new(rng: &mut Rng, rows: usize, cols: usize, cy: f64, cx: f64) -> Noise {
        let (cy, cx) = (cy.max(1.0), cx.max(1.0));
        let (gh, gw) = ((rows as f64 / cy) as usize + 2, (cols as f64 / cx) as usize + 2);
        let axis = |n: usize, c: f64| -> Vec<(u32, f32)> {
            (0..n).map(|k| {
                let u = k as f64 / c;
                let i = u.floor();
                let f = (u - i) as f32;
                (i as u32, f * f * (3.0f32 - 2.0f32 * f))
            }).collect()
        };
        Noise { grid: rng.grid(gh, gw), cols: gw, ys: axis(rows, cy), xs: axis(cols, cx) }
    }

    /// The field at (row, col), as numpy blends it (for tests).
    pub fn at(&self, y: usize, x: usize) -> f32 {
        let ((iy, fy), (ix, fx)) = (self.ys[y], self.xs[x]);
        let g = |r: u32, c: u32| self.grid[r as usize * self.cols + c as usize];
        let r0 = g(iy, ix) * (1.0 - fy) + g(iy + 1, ix) * fy;
        let r1 = g(iy, ix + 1) * (1.0 - fy) + g(iy + 1, ix + 1) * fy;
        r0 * (1.0 - fx) + r1 * fx
    }
}

/// crayon.fibres' four fields (mottle, the fibres' two, the fine grain) of a `rows` x `cols` sheet.
pub fn fibres(rows: usize, cols: usize, px: f64, seed: u64) -> [Noise; 4] {
    let mut rng = Rng::new(&[seed, 1515]);
    let a = Noise::new(&mut rng, rows, cols, 90.0 * px, 90.0 * px);
    let b = Noise::new(&mut rng, rows, cols, 1.4 * px, 10.0 * px);
    let c = Noise::new(&mut rng, rows, cols, 30.0 * px, 30.0 * px);
    let d = Noise::new(&mut rng, rows, cols, 1.2 * px, 1.2 * px);
    [a, b, c, d]
}

/// The median of `v` as np.median takes it from float32 values (the two middle ones' mean in float32).
fn median(v: &mut [f32]) -> f32 {
    let n = v.len();
    let (_, hi, _) = v.select_nth_unstable_by(n / 2, f32::total_cmp);
    let hi = *hi;
    if n % 2 == 1 {
        return hi;
    }
    let lo = v[..n / 2].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (lo + hi) / 2.0
}

/// A dilation by shotstyle.dilate_round's disc of radius `r`: a square of `square` px, then `steps` steps of a cross.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Round {
    pub square: u32,
    pub steps: u32,
}

impl Round {
    /// None when dilate_round leaves the mask as it is (r under half a pixel).
    pub fn of(r: f64) -> Option<Round> {
        if r < 0.5 {
            return None;
        }
        let a = pyround(0.414 * r).max(0);
        Some(Round { square: a as u32, steps: (pyround(r) - a).max(0) as u32 })
    }
}

/// A shadow as cutout.compose's `shadowed` casts it: shifted by (dx, dy) whole pixels, boxed `r` px each way twice,
/// `alpha` of it laid in `color`.
#[derive(Clone, Copy, Debug)]
pub struct Cast {
    pub dx: i32,
    pub dy: i32,
    pub r: u32,
    pub alpha: f32,
}

/// A shift held to this many pixels either way: past every window (a pass is at most 16384 px a side), so it casts
/// nothing there, as cutout._shifted gives an empty mask once a shift passes the window; and the shader's sums of a
/// pixel and a shift stay well inside i32.
pub const SHIFT_MAX: i64 = 1 << 24;

/// Pixels held to what a window can be (any margin past this takes the whole pass, as on the CPU).
const MARGIN_MAX: f64 = 1e9;

fn margin(v: f64) -> usize {
    v.ceil().clamp(0.0, MARGIN_MAX) as usize
}

impl Cast {
    fn new(sh: &Shadow, h: usize, px: f64, scale: f64, alpha: f64) -> Cast {
        let r = pyround(0.5 * sh.blur * px * scale);
        let shift = |o: f64| pyround(o * h as f64 * scale).clamp(-SHIFT_MAX, SHIFT_MAX) as i32;
        Cast { dx: shift(sh.offset[0]), dy: shift(sh.offset[1]), r: if r < 1 { 0 } else { r as u32 }, alpha: alpha as f32 }
    }
}

/// The colours of the look's fills, as the GPU reads them.
pub struct Palette {
    /// Per material id: its line group (the piece) and its tone (u32::MAX: none).
    pub ids: Vec<[u32; 2]>,
    /// Per tone: kind, cel, number of brightness steps, first colour, first step table, grey (bits), last colour.
    pub tones: Vec<[u32; 8]>,
    pub colors: Vec<[f32; 4]>,
    /// Per brightness step of a drawn tone, 256 x 256 entries: for a texel's red and green bytes, the least blue byte
    /// whose brightness reaches the step (256: none).
    pub steps: Vec<u32>,
}

/// The texture pass's byte `v` as compose.look_passes reads it (float32 v / 255).
pub fn unit(v: u32) -> f32 {
    v as f32 / 255.0
}

/// For a step `at` of a drawn tone, per (red, green) byte the least blue byte whose brightness (the texel's float32
/// channels dotted with the luma weights in float64, as _vector_slots takes it) is at or over it.
fn step_table(at: f64) -> Vec<u32> {
    let c: Vec<f64> = (0..256).map(|v| unit(v) as f64).collect();
    let mut out = vec![256u32; 65536];
    for r in 0..256 {
        for g in 0..256 {
            let base = c[r] * 0.299 + c[g] * 0.587;
            // brightness grows with blue: the first that reaches `at`
            let (mut lo, mut hi) = (0usize, 256usize);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if base + c[mid] * 0.114 >= at { hi = mid } else { lo = mid + 1 }
            }
            out[r * 256 + g] = lo as u32;
        }
    }
    out
}

impl Palette {
    pub fn new(fin: &Finish, cache: &mut BTreeMap<u64, Vec<u32>>) -> Result<Palette> {
        let mut ids = Vec::with_capacity(fin.ids.len());
        for (i, &[g, t]) in fin.ids.iter().enumerate() {
            if g < 0 || (i > 0 && (t < 0 || t as usize >= fin.tones.len())) {
                bail!("finish: material id {i} has group {g}, tone {t}");
            }
            ids.push([g as u32, if t < 0 { u32::MAX } else { t as u32 }]);
        }
        if ids.is_empty() {
            ids.push([0, u32::MAX]);
        }
        let (mut tones, mut colors, mut steps) = (Vec::new(), Vec::new(), Vec::new());
        for t in &fin.tones {
            let kind = match t.kind.as_str() {
                "flat" => KIND_FLAT,
                "lit" => KIND_LIT,
                "drawn" => KIND_DRAWN,
                k => bail!("finish: tone kind {k}"),
            };
            let need = match kind {
                KIND_FLAT => 1,
                KIND_LIT => 2,
                _ => t.at.len() + 1,
            };
            if t.colors.len() != need {
                bail!("finish: a {} tone with {} colours", t.kind, t.colors.len());
            }
            let first = colors.len() as u32;
            colors.extend(t.colors.iter().map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, 1.0]));
            let table = steps.len() as u32;
            if kind == KIND_DRAWN {
                for &a in &t.at {
                    steps.extend_from_slice(cache.entry(a.to_bits()).or_insert_with(|| step_table(a)));
                }
            }
            tones.push([kind, t.cel as u32, if kind == KIND_DRAWN { t.at.len() as u32 } else { 0 }, first, table,
                        (t.grey as f32).to_bits(), need as u32 - 1, 0]);
        }
        Ok(Palette { ids, tones, colors, steps })
    }
}

/// One piece of paper as the GPU lays it: its group, where its paper is cut from the sheet and how far it is lifted.
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub group: u32,
    pub paper: [u32; 2],
    pub lift: [i32; 2],
}

/// Everything about a frame of the cut-out the CPU works out (cutout.compose's numbers, its order and its fields).
pub struct Plan {
    /// The passes' size.
    pub w: usize,
    pub h: usize,
    pub px: f64,
    /// The box the figure is composed in (x0, y0, width, height), None when there is no figure.
    pub bbox: Option<[usize; 4]>,
    /// The background's sheet: four fields over the whole pass.
    pub sheet_seed: u64,
    /// The scissors' wander (dy's two fields, dx's two) and the pieces' sheet (four fields over the box and the pad).
    pub rough: Vec<Noise>,
    pub tex: Vec<Noise>,
    /// The rim's paper (piece 0's place on the sheet).
    pub rim_paper: [u32; 2],
    pub figure: Option<Round>,
    pub figure_cast: Option<Cast>,
    /// The pieces far to near (each drawn where its group is once cut).
    pub pieces: Vec<Piece>,
    pub grow: Option<Round>,
    pub spread_steps: u32,
    pub piece_cast: Option<Cast>,
    /// How far round its own pixels a piece's window reaches (its growth, lift and shadow).
    pub margin: usize,
}

impl Plan {
    pub fn new(p: &Passes, fin: &Finish, groups: &[[u32; 2]]) -> Result<Plan> {
        let (w, h) = (p.w, p.h);
        let px = w.min(h) as f64 / 1080.0;
        let s = &fin.cutout;
        let ngroups = groups.iter().map(|g| g[0] as usize).max().unwrap_or(0) + 1;
        // the figure's box, its groups and their depths, a band of rows at a time; and the largest id the passes hold
        let bands: Vec<([usize; 4], Vec<Vec<f32>>, usize)> = (0..h).collect::<Vec<_>>().par_chunks(32).map(|rows| {
            let mut b = [usize::MAX, 0, usize::MAX, 0];
            let mut depths = vec![Vec::new(); ngroups];
            let mut top = 0;
            for &y in rows {
                for x in 0..w {
                    let i = y * w + x;
                    let id = p.id_at(i);
                    top = top.max(id);
                    let g = groups.get(id).map_or(0, |g| g[0] as usize);
                    if g == 0 {
                        continue;
                    }
                    b = [b[0].min(y), b[1].max(y), b[2].min(x), b[3].max(x)];
                    depths[g].push(p.depth_at(i));
                }
            }
            (b, depths, top)
        }).collect();
        let (mut y0, mut y1, mut x0, mut x1, mut top) = (usize::MAX, 0, usize::MAX, 0, 0);
        let mut depths: Vec<Vec<f32>> = vec![Vec::new(); ngroups];
        for (b, d, t) in bands {
            (y0, y1, x0, x1) = (y0.min(b[0]), y1.max(b[1]), x0.min(b[2]), x1.max(b[3]));
            top = top.max(t);
            for (all, v) in depths.iter_mut().zip(d) {
                all.extend(v);
            }
        }
        // the CPU look indexes its tables with the ids and stops at one they do not hold
        if top >= groups.len() {
            bail!("the passes hold material id {top}, and the finish record's tables hold {} (ids 0 to {})",
                  groups.len(), groups.len() - 1);
        }
        // numpy seeds its generators with whole numbers of any size; these are carried exactly in 64 bits or not at all
        let seeded = |k: u64, add: u64| s.seed.checked_mul(k).and_then(|v| v.checked_add(add))
            .with_context(|| format!("cutout seed {} does not fit the 64 bits mkr seeds with", s.seed));
        let mut plan = Plan {
            w, h, px, bbox: None, sheet_seed: seeded(7919, 1)?, rough: Vec::new(), tex: Vec::new(), rim_paper: [0, 0],
            figure: None, figure_cast: None, pieces: Vec::new(), grow: None, spread_steps: 0, piece_cast: None, margin: 0,
        };
        let tex_seed = seeded(104729, 3)?;
        if depths.iter().all(|v| v.is_empty()) {
            return Ok(plan);
        }
        let rim = s.rim.as_ref().map_or(0.0, |r| r.width * px);
        let reach = s.shadow.as_ref().map_or(0.0, |sh| (sh.offset[0].abs() + sh.offset[1].abs()) * h as f64 + 3.0 * sh.blur * px);
        if !reach.is_finite() {
            // cutout.compose's int(math.ceil(...)) raises on it too
            bail!("a cutout shadow offset past any float's reach");
        }
        let m = margin(rim + 2.0 * s.rough * px + 2.0 * s.lift * px + reach) + 4;
        let (by0, by1) = (y0.saturating_sub(m), (y1 + m + 1).min(h));
        let (bx0, bx1) = (x0.saturating_sub(m), (x1 + m + 1).min(w));
        let (bh, bw) = (by1 - by0, bx1 - bx0);
        plan.bbox = Some([bx0, by0, bw, bh]);
        let mut rng = Rng::new(&[s.seed, fin.step, 4242]);
        for _ in 0..2 {
            for (c, _) in ROUGH_CELLS {
                plan.rough.push(Noise::new(&mut rng, bh, bw, c * px, c * px));
            }
        }
        plan.tex = fibres(bh + 2 * PAD, bw + 2 * PAD, px, tex_seed).into();
        let paper = |j: u64| {
            let mut r = Rng::new(&[s.seed, j, 99]);
            let oy = r.below(2 * PAD as u64) as u32;
            [oy, r.below(2 * PAD as u64) as u32]
        };
        plan.rim_paper = paper(0);
        if s.rim.is_some() || s.shadow.is_some() {
            plan.figure = if rim > 0.5 { Round::of(rim) } else { None };
            plan.figure_cast = s.shadow.as_ref().map(|sh| Cast::new(sh, h, px, 1.0, sh.alpha));
        }
        // far to near by the median depth of each piece's own pixels, then by group
        let mut far: Vec<(f64, u32)> = depths.par_iter_mut().enumerate().filter(|(_, v)| !v.is_empty())
            .map(|(g, v)| (median(v) as f64, g as u32)).collect();
        far.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        let lift = s.lift * px;
        for &(_, g) in &far {
            let off = if lift >= 0.5 {
                let turn = Rng::new(&[s.seed, g as u64, 31]).uniform(0.0, 2.0 * std::f64::consts::PI);
                // `lift` is at most 40 px at 1080 (cutout.normalize): a few hundred pixels at the most
                [pyround(lift * turn.cos()) as i32, pyround(lift * turn.sin()) as i32]
            } else {
                [0, 0]
            };
            plan.pieces.push(Piece { group: g, paper: paper(g as u64), lift: off });
        }
        if lift >= 0.5 {
            plan.grow = Round::of(lift);
            plan.spread_steps = lift.ceil() as u32;
        }
        plan.piece_cast = s.shadow.as_ref().map(|sh| Cast::new(sh, h, px, PIECE_SHADOW, 0.8 * sh.alpha));
        let shade = s.shadow.as_ref().map_or(0.0, |sh| PIECE_SHADOW * (reach + 2.0 * sh.blur * px));
        plan.margin = margin(2.0 * lift + shade) + 2;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_as_python() {
        assert_eq!([pyround(2.5), pyround(3.5), pyround(-2.5), pyround(2.25)], [2, 4, -2, 2]);
        // dilate_round's octagon of a 1080 frame's 10 px rim at twice the size, of a 3 px lift, of the lit mask's 2 px
        assert_eq!(Round::of(20.0), Some(Round { square: 8, steps: 12 }));
        assert_eq!(Round::of(6.0), Some(Round { square: 2, steps: 4 }));
        assert_eq!(Round::of(2.0), Some(Round { square: 1, steps: 1 }));
        assert_eq!(Round::of(0.4), None);
    }

    #[test]
    fn medians_as_numpy() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut [1.0]), 1.0);
    }

    #[test]
    fn noise_cells_as_numpy() {
        // crayon.noise(np.random.default_rng([3, 1515]), (5, 7), 2.5): its grid is 4 x 4 and row 3 falls in cell 1
        let n = Noise::new(&mut Rng::new(&[3, 1515]), 5, 7, 2.5, 2.5);
        assert_eq!((n.grid.len(), n.cols), (16, 4));
        assert_eq!(n.ys[3].0, 1);
        assert!((n.ys[3].1 - 0.104).abs() < 1e-6);
        assert_eq!(n.at(0, 0), n.grid[0]);
    }
}
