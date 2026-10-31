//! mk post on the GPU (docs/design.md: Post). `mkr-post serve --setup FILE` finishes one output's frames: per frame a
//! JSON job on stdin (job::Job: the cut's frame or a picture a Python worker made, the frame effects still to draw, the
//! screen type, the keep mask; or a frame Python finished, handed on as it is), per frame a record on stdout in the
//! jobs' order: the clip index (u32 LE), the darkest luma of the graded picture (f32 LE) and the w * h * 3 bytes the
//! encoder takes; with `--stats FD` the bytes alone on stdout (the encoder's own pipe) and the index and luma on FD.
//! mkmmd/postgpu.py drives it; every kernel follows the numpy / OpenCV code of mkmmd.post, mkmmd.framefx and
//! mkmmd.core.screentype.

pub mod engine;
pub mod fx;
pub mod gpu;
pub mod job;
pub mod pngio;
