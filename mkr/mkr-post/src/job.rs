//! What `mkr-post serve` reads: the output's setup (once, a JSON file) and one job per frame (a JSON line on stdin).
//! Python (mkmmd/postgpu.py) writes both from the same objects mk post grades with, so every number here is what numpy
//! uses: float32 constants as numpy rounds them, tables (the 1/255 table, the vignette, the grain) as numpy made them.

use serde::Deserialize;

pub use crate::fx::Op;

/// One output's grade, streaks and screen mode, for every frame of it.
#[derive(Debug, Clone, Deserialize)]
pub struct Setup {
    /// (w, h) of the pictures (even, as mk post encodes them).
    pub size: [u32; 2],
    /// "graded": the screen type is laid on before the grade; "over": over the graded picture.
    pub screen: String,
    /// 256 floats: v / 255 as numpy makes it (mkmmd.post.read).
    pub lut: Vec<f32>,
    /// w * h float32 (little endian): mkmmd.post.Grade.vig.
    pub vignette: String,
    pub grain: Grain,
    pub grade: Grade,
    /// No screen type and no grade: the picture the ops leave, quantised (kernel parity tests).
    #[serde(default)]
    pub plain: bool,
    /// With `plain`: each frame's record carries the picture as float32 RGBA (w * h * 16 bytes), not quantised.
    #[serde(default)]
    pub floats: bool,
}

/// mkmmd.post.Grade.grain: `count` frames of w * h float32 in one file, frame k uses k % count.
#[derive(Debug, Clone, Deserialize)]
pub struct Grain {
    pub file: String,
    pub count: u32,
    pub amount: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Grade {
    pub saturation: f32,
    /// keep saturation - saturation, worked out in float64 as Python does.
    pub keep_delta: f32,
    pub pivot: f32,
    pub contrast: f32,
    pub split: Option<Split>,
    pub floor: Option<Floor>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Split {
    /// shadows - 0.5 and highlights - 0.5 (float32 arrays in numpy).
    pub shadows: [f32; 3],
    pub highlights: [f32; 3],
    pub amount: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Floor {
    pub rgb: [f32; 3],
    pub yf: f32,
    pub lim: f32,
    pub lim_yf: f32,
    /// The final clip's floor: yf * 0.98.
    pub lo: f32,
}

/// One frame: where its picture comes from, the frame effects still to draw on it (in table order), its screen type
/// and its keep mask; `k` its index in the clip (the grain frame). Or a frame Python finished itself (`finished`),
/// handed on as it is in the jobs' order.
#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    pub k: u32,
    /// The cut's frame on disk (an 8-bit PNG of the output's size) ...
    #[serde(default)]
    pub png: Option<String>,
    /// ... or a picture a worker made (w * h * 3 float32, removed once read).
    #[serde(default)]
    pub raw: Option<String>,
    #[serde(default)]
    pub layer: Option<String>,
    /// The keep mask, when it applies to this frame.
    #[serde(default)]
    pub keep: Option<String>,
    #[serde(default)]
    pub ops: Vec<Op>,
    #[serde(default)]
    pub finished: Option<Finished>,
}

/// A frame Python finished (the loop's wrap): its w * h * 3 bytes as the encoder takes them in a file (removed once
/// read) and the darkest luma of its graded picture.
#[derive(Debug, Clone, Deserialize)]
pub struct Finished {
    pub path: String,
    pub darkest: f32,
}
