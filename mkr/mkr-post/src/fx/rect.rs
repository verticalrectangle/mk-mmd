//! rect: an effect limited to a `rect` (mkmmd.framefx.FrameFx._apply), on any kind mkr-post draws. Python brackets the
//! entry's ops with a `begin` (the picture as it reaches the entry is kept) and an `end` that keeps what the ops drew
//! inside the rect and puts the kept picture back outside it, with the rect's ramps (mkmmd.picturefx.rect_ramps over the
//! whole frame: h row ramps then w column ramps, written once for the entry and kept: shaders/fx_rect.wgsl).

use std::sync::Arc;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

use super::{Ctx, Decoded, Drawn, Effect, Input};

const SRC: &str = include_str!("../shaders/fx_rect.wgsl");

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Rect {
    /// Keep the picture as it is before the entry's ops draw.
    Begin,
    /// Keep the ops' drawing inside the rect: the file of its ramps and how many float32 it holds (h + w).
    End { ramps: String, count: usize },
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

impl Effect for Rect {
    fn inputs(&self) -> Vec<Input> {
        match self {
            Rect::Begin => Vec::new(),
            Rect::End { ramps, count } => vec![Input::Kept { path: ramps.clone(), count: *count }],
        }
    }

    fn draw(&self, cx: &mut Ctx, files: &[Arc<Decoded>]) -> Result<Drawn> {
        let bytes = cx.w as u64 * cx.h as u64 * 16;
        let saved = cx.scratch("rect.before", bytes);
        match self {
            Rect::Begin => {
                cx.enc.copy_buffer_to_buffer(cx.src, 0, &saved, 0, bytes);
                Ok(Drawn::Src)
            }
            Rect::End { .. } => {
                let (src, dst) = (cx.src.clone(), cx.dst.clone());
                let table = cx.upload(&self.inputs()[0], &files[0]);
                let k = cx.kernel("fx_rect", SRC, "rect");
                let u = cx.uniform(&Params { w: cx.w, h: cx.h, pad0: 0, pad1: 0 });
                cx.pass(&k, &[(0, &u), (1, &src), (2, &dst), (3, &saved), (4, &table)], cx.tiles());
                Ok(Drawn::Dst)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Op;

    #[test]
    fn the_two_phases_parse_as_python_writes_them() {
        let begin: Op = serde_json::from_str(r#"{"op": "rect", "phase": "begin"}"#).unwrap();
        assert!(matches!(begin, Op::Rect(super::Rect::Begin)));
        assert!(begin.effect().inputs().is_empty());
        let end: Op = serde_json::from_str(r#"{"op": "rect", "phase": "end", "ramps": "r.f32", "count": 1500}"#).unwrap();
        let ins = end.effect().inputs();
        assert_eq!(ins.len(), 1);
        assert!(matches!(&ins[0], super::Input::Kept { path, count } if path == "r.f32" && *count == 1500));
        assert!(serde_json::from_str::<Op>(r#"{"op": "rect", "phase": "middle"}"#).is_err());
    }
}
