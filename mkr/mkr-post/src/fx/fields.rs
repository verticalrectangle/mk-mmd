//! What the colour fields share (shaders/fields.wgsl; mkmmd.fields): the noise's bicubic weights and its calls, the
//! patterns' parameters, the region a masked field is kept to.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure::{cubic_weights, plane_bytes, unpack};
use super::{Ctx, Decoded, Input};
use crate::pngio::Kind;

/// OpenCV's interpolateCubic of the 32 phases of a picture resized 32 times (cv2.resize, INTER_CUBIC): fine sample j
/// of a lattice cell lies at (j + 0.5) / 32 - 0.5, its weights those of its fraction past the lattice point before it,
/// in float32 as OpenCV works them out.
pub fn cubic32() -> [[f32; 4]; 32] {
    let mut out = [[0.0f32; 4]; 32];
    for (p, o) in out.iter_mut().enumerate() {
        let fx = ((p as f64 + 0.5) / 32.0 - 0.5) as f32;
        *o = cubic_weights(fx - fx.floor());
    }
    out
}

/// One call of mkmmd.fields.noise: its time slice k, seed and Catmull-Rom weights (float32), as Python works them out.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Deserialize)]
pub struct NoiseCall {
    pub k: i32,
    pub seed: i32,
    #[serde(skip)]
    pub pad: [u32; 2],
    pub w: [f32; 4],
}

/// The tables a kernel that draws noise binds: (binding 10: the bicubic weights, binding 11: the calls).
pub fn noise_tables(cx: &Ctx, calls: &[NoiseCall]) -> (wgpu::Buffer, wgpu::Buffer) {
    (cx.table("field.cubic", &cubic32()), cx.table("field.calls", calls))
}

/// A pattern as mkmmd.gpufx.fields works it out from mkmmd.fields.pattern's arguments: every constant float32 as
/// numpy makes it.
#[derive(Debug, Clone, Deserialize)]
pub struct PatternSpec {
    pub kind: u32,
    pub colors: Vec<[f32; 3]>,
    pub mean: [f32; 3],
    #[serde(default)]
    pub moire: u32,
    #[serde(default)]
    pub n: f32,
    #[serde(default)]
    pub tw: f32,
    #[serde(default)]
    pub size: f32,
    pub q0: f32,
    #[serde(default)]
    pub ca: f32,
    #[serde(default)]
    pub sa: f32,
    #[serde(default)]
    pub o: f32,
    #[serde(default)]
    pub warp: f32,
    #[serde(default)]
    pub g: f32,
    #[serde(default)]
    pub c2k: f32,
    /// Its centre (px) and unit (px) in the picture it is painted over.
    pub center: [f32; 2],
    pub unit: f32,
}

/// fields.wgsl's Pattern.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Pattern {
    kind: u32,
    m: u32,
    moire: u32,
    pad0: u32,
    n: f32,
    tw: f32,
    size: f32,
    q0: f32,
    ca: f32,
    sa: f32,
    o: f32,
    warp: f32,
    g: f32,
    unit: f32,
    c2k: f32,
    pad2: f32,
    colors: [[f32; 4]; 4],
    mean: [f32; 4],
}

impl PatternSpec {
    pub fn gpu(&self) -> Pattern {
        let mut colors = [[0.0f32; 4]; 4];
        for (o, c) in colors.iter_mut().zip(&self.colors) {
            *o = [c[0], c[1], c[2], 0.0];
        }
        Pattern {
            kind: self.kind, m: self.colors.len() as u32, moire: self.moire, pad0: 0, n: self.n, tw: self.tw,
            size: self.size, q0: self.q0, ca: self.ca, sa: self.sa, o: self.o, warp: self.warp, g: self.g,
            unit: self.unit, c2k: self.c2k, pad2: 0.0, colors, mean: [self.mean[0], self.mean[1], self.mean[2], 0.0],
        }
    }
}

/// The part of the picture a masked field works on (FrameFx._masked): the figure's coverage (`ground` false) or what
/// it leaves (true), its figure layer read as mkmmd.cutfx.read_figure reads it.
#[derive(Debug, Clone, Deserialize)]
pub struct Region {
    pub layer: String,
    pub ground: bool,
}

impl Region {
    pub fn input(&self) -> Input {
        Input::Png(self.layer.clone(), Kind::Luma)
    }

    /// The figure's coverage as a plane (.x), from its decoded layer.
    pub fn plane(&self, cx: &mut Ctx, file: &Arc<Decoded>) -> wgpu::Buffer {
        let (w, h) = (cx.w, cx.h);
        let packed = cx.upload(&self.input(), file);
        let out = cx.scratch("field.region", plane_bytes(w, h));
        unpack(cx, &packed, &out, w, h);
        out
    }
}
