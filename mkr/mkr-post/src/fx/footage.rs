//! footage: the figure as a sticker over a video clip (mkmmd.framefx.FrameFx._footage; mkmmd.gpufx.footage works out
//! the clip's frame, its fit and the figure's map). The clip's frame comes decoded and fitted by Python (OpenCV, as the
//! CPU path reads it: Input::Rgb); a cover fills the picture, a contained one is laid on its back (a colour, or the
//! picture as it reaches the op); the figure is pasted over it with its edge and shadow (fx/figure.rs), worked out over
//! the box they reach (the figure's support from its layer's Cover), and the whole mixed in by the strength.

use std::sync::Arc;

use anyhow::Result;
use serde::Deserialize;

use super::figure::{self, Cutout, Edge, Shadow};
use super::plane::Area;
use super::{Ctx, Decoded, Drawn, Effect, Input};

/// The clip's frame as mkmmd.footage.fitted leaves it: rgb24 bytes of `size` (w, h) in the file at `path`, removed once
/// read unless `keep` (a frame the run's outputs share: mkmmd.gpufx.footage.Kept, removed by mk post).
#[derive(Debug, Clone, Deserialize)]
pub struct Clip {
    pub path: String,
    pub size: [u32; 2],
    #[serde(default)]
    pub keep: bool,
}

impl Clip {
    fn input(&self) -> Input {
        Input::Rgb { path: self.path.clone(), size: self.size, keep: self.keep }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Footage {
    /// The clip's frame fitted: the whole picture (cover) or `size` of it at `at` (contain).
    pub clip: Clip,
    #[serde(default)]
    pub at: [u32; 2],
    /// Round a contained clip: a colour, or None for the picture as it reaches the op.
    #[serde(default)]
    pub back: Option<[f32; 3]>,
    pub figure: Cutout,
    /// The figure in a flat colour; None: as its picture shows it.
    #[serde(default)]
    pub fill: Option<[f32; 3]>,
    #[serde(default)]
    pub edge: Option<Edge>,
    #[serde(default)]
    pub shadow: Option<Shadow>,
    pub strength: f64,
}

impl Effect for Footage {
    fn inputs(&self) -> Vec<Input> {
        let mut v = vec![self.clip.input()];
        v.extend(self.figure.inputs());
        v
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let (w, h) = (cx.w, cx.h);
        let (src, dst) = (cx.src.clone(), cx.dst.clone());
        cx.mark("footage start");
        let clip = cx.upload(&self.clip.input(), &files[0]);
        let [vw, vh] = self.clip.size;
        if (vw, vh) == (w, h) {
            figure::unpack(cx, &clip, &dst, w, h);
        } else {
            match self.back {
                Some(c) => figure::fill(cx, &dst, w, h, c),
                None => figure::copy(cx, &src, &dst, w, h),
            }
            figure::blit(cx, &clip, (vw, vh), &dst, (w, h), self.at);
        }
        cx.mark("footage clip");
        // the figure over the box it, its ring and its shadow reach; none of it: nothing laid
        let reach = figure::paste_reach(self.edge.as_ref(), self.shadow.as_ref());
        let at = match figure::support(cx, &self.figure, &files[1]) {
            Some(None) => None,
            Some(Some([x0, y0, x1, y1])) => Some(Area::of((w, h), [x0.saturating_sub(reach), y0.saturating_sub(reach),
                                                                   (x1 + reach).min(w), (y1 + reach).min(h)])),
            None => Some(Area::whole(w, h)),
        };
        if let Some(at) = at {
            let cov = cx.scratch("figure.cov", at.bytes());
            let rgb = cx.scratch("figure.rgb", at.bytes4());
            figure::cut_box(cx, &self.figure, &files[1..], &at, &cov, self.fill.is_none().then_some(&rgb));
            figure::paste_figure(cx, &dst, &at, &cov, &rgb, self.fill, self.edge.as_ref(), self.shadow.as_ref());
        }
        if self.strength < 1.0 {
            figure::mix(cx, &src, &dst, w, h, self.strength);
        }
        Ok(Drawn::Dst)
    }
}
