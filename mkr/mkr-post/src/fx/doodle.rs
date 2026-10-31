//! doodle: pen strokes over the picture (mkmmd.picturefx.draw_strokes; PictureFx._doodle). Python (mkmmd.gpufx.doodle)
//! works out each mark's strokes and draws their masks with OpenCV's antialiased lines in the box they cover (a few
//! hundred px; written for this frame); the rim, the ink and the strength's mix are laid here (shaders/fx_doodle.wgsl).

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::figure;
use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_doodle.wgsl");

/// One mark: the box its strokes cover (x0, y0, x1, y1), its masks (rim, ink as v / 255; bw x bh pairs of float32,
/// removed once read) and its ink's colour.
#[derive(Debug, Clone, Deserialize)]
pub struct Mark {
    #[serde(rename = "box")]
    pub area: [u32; 4],
    pub masks: String,
    pub color: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Doodle {
    pub marks: Vec<Mark>,
    pub strength: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    x0: u32,
    y0: u32,
    bw: u32,
    bh: u32,
    pad0: u32,
    pad1: u32,
    colour: [f32; 4],
}

impl Mark {
    fn input(&self) -> Input {
        let [x0, y0, x1, y1] = self.area;
        Input::F32 { path: self.masks.clone(), count: 2 * ((x1 - x0) * (y1 - y0)) as usize }
    }
}

impl Effect for Doodle {
    fn inputs(&self) -> Vec<Input> {
        self.marks.iter().map(Mark::input).collect()
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        figure::copy(cx, &src, &dst, w, h);
        let k = cx.kernel("fx_doodle", SRC, "doodle");
        for (m, file) in self.marks.iter().zip(files) {
            let [x0, y0, x1, y1] = m.area;
            let masks = cx.upload(&m.input(), file);
            let c = m.color;
            let u = cx.uniform(&Params {
                w, h, x0, y0, bw: x1 - x0, bh: y1 - y0, pad0: 0, pad1: 0, colour: [c[0], c[1], c[2], 0.0],
            });
            cx.pass(&k, &[(0, &u), (2, &dst), (3, &masks)], crate::gpu::tiles(x1 - x0, y1 - y0));
        }
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
