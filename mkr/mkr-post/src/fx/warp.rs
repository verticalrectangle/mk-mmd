//! An affine warp: cv2.warpAffine(picture, m, size, INTER_LINEAR | WARP_INVERSE_MAP, border) as OpenCV 5 draws it
//! (shaders/warp.wgsl, bit for bit). mkmmd.gpufx.zoom's punch on the hits.

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect};

const WARP: &str = include_str!("../shaders/warp.wgsl");

/// `m` the 2 x 3 map from each pixel of the result to the picture's, as float32; `border` one of OpenCV's (0 constant,
/// 1 replicate, 2 reflect, 4 reflect 101), `cval` the colour outside with a constant border.
#[derive(Debug, Clone, Deserialize)]
pub struct Warp {
    pub m: [f32; 6],
    pub border: u32,
    #[serde(default)]
    pub cval: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WarpParams {
    w: u32,
    h: u32,
    sw: u32,
    sh: u32,
    border: u32,
    x0: u32,
    m00: f32,
    m10: f32,
    cval: [f32; 4],
}

/// The per-row terms of `m` (m01 * y + m02, m11 * y + m12 as float32 products and sums, not fused) for `h` rows: the
/// warp kernel's `rows`.
pub fn rows(m: &[f32; 6], h: u32) -> Vec<f32> {
    (0..h).flat_map(|y| {
        let y = y as f32;
        [m[1] * y + m[2], m[4] * y + m[5]]
    }).collect()
}

/// Warp `src` (a sw x sh picture of vec4) into `dst` (dw x dh) by `m`.
#[allow(clippy::too_many_arguments)]
pub fn warp(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, (dw, dh): (u32, u32),
            m: &[f32; 6], border: u32, cval: [f32; 3]) {
    warp_box(cx, src, (sw, sh), dst, [0, 0, dw, dh], m, border, cval);
}

/// The box [x0, x1) x [y0, y1) of `src` warped by `m` into `dst` (a picture of the box's size): its pixels as `warp`
/// works them out over the whole picture.
#[allow(clippy::too_many_arguments)]
pub fn warp_box(cx: &mut Ctx, src: &wgpu::Buffer, (sw, sh): (u32, u32), dst: &wgpu::Buffer, [x0, y0, x1, y1]: [u32; 4],
                m: &[f32; 6], border: u32, cval: [f32; 3]) {
    let (dw, dh) = (x1 - x0, y1 - y0);
    let terms: Vec<f32> = rows(m, y1).split_off(2 * y0 as usize);
    let rows = cx.table("rows", &terms);
    let p = cx.uniform(&WarpParams {
        w: dw, h: dh, sw, sh, border, x0, m00: m[0], m10: m[3], cval: [cval[0], cval[1], cval[2], 0.0],
    });
    let k = cx.kernel("warp", WARP, "warp");
    cx.pass(&k, &[(0, &p), (1, src), (2, dst), (3, &rows)], crate::gpu::tiles(dw, dh));
}

impl Effect for Warp {
    fn draw(&self, cx: &mut Ctx, _: &[Arc<Decoded>]) -> Result<Drawn> {
        let (src, dst, size) = (cx.src.clone(), cx.dst.clone(), (cx.w, cx.h));
        warp(cx, &src, size, &dst, size, &self.m, self.border, self.cval);
        Ok(Drawn::Dst)
    }
}
