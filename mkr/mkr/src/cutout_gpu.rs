//! The cut-out paper look finished on the GPU (cutout.rs says what and how; shaders/cutout.wgsl draws it): a vector
//! job's passes in, the frame's 8-bit RGBA out. The lit mask is opened and closed over the passes, the scissors' cut
//! warps the box round the figure, then the figure's shadow and rim and every piece far to near (its mask grown, its
//! colours carried out over the margin step by step as cutout._spread carries them, lifted, its shadow boxed and laid,
//! the piece on its paper), and the box is boxed down to the frame with the type laid over it.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::cutout::{Finish, Noise, PAD, Palette, Passes, Plan, Round, SCALE, fibres, unit};
use crate::gpu::Gpu;

const WGSL: &str = include_str!("shaders/cutout.wgsl");
const BINDINGS: u32 = 26;
const UNIFORM: u32 = 0;
/// The bindings the shader writes.
const WRITTEN: [u32; 10] = [10, 16, 17, 18, 19, 20, 21, 22, 23, 24];
const WRITTEN_OUT: u32 = 25;
/// _spread's order of a step: from the left, the right, above, below.
const SPREAD: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameGpu {
    bg: [f32; 4],
    rim: [f32; 4],
    shc: [f32; 4],
    size: [u32; 4],
    bbox: [u32; 4],
    misc: [u32; 4],
    misc2: [u32; 4],
    fl: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Op {
    win: [u32; 4],
    a: [i32; 4],
    b: [i32; 4],
    f: [f32; 4],
}

struct Pipes {
    lit_init: wgpu::ComputePipeline,
    lit_morph: wgpu::ComputePipeline,
    paper_sheet: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
    pic_init: wgpu::ComputePipeline,
    mask_init: wgpu::ComputePipeline,
    dil_square: wgpu::ComputePipeline,
    dil_cross: wgpu::ComputePipeline,
    spread: wgpu::ComputePipeline,
    cast: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    figure: wgpu::ComputePipeline,
    piece: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

/// The background's sheet, kept while the passes' size and the seed stay.
struct Sheet {
    key: (usize, usize, u64),
    grid: wgpu::Buffer,
    axes: wgpu::Buffer,
    descs: Vec<[u32; 4]>,
}

/// Scratch buffers, grown to the largest frame seen.
struct Scratch {
    pass: u64,
    cap: u64,
    tcap: u64,
    frame: u64,
    lit: wgpu::Buffer,
    wsrc: wgpu::Buffer,
    pw: wgpu::Buffer,
    texf: wgpu::Buffer,
    pic: wgpu::Buffer,
    mask: wgpu::Buffer,
    src: wgpu::Buffer,
    stamp: wgpu::Buffer,
    sbuf: wgpu::Buffer,
    out8: wgpu::Buffer,
    read: wgpu::Buffer,
}

pub struct CutoutKit {
    bgl: wgpu::BindGroupLayout,
    pipes: Pipes,
    frame: wgpu::Buffer,
    units: wgpu::Buffer,
    sheet: Option<Sheet>,
    scratch: Option<Scratch>,
    steps: BTreeMap<u64, Vec<u32>>,
}

fn layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    let ty = if binding == UNIFORM {
        wgpu::BufferBindingType::Uniform
    } else {
        wgpu::BufferBindingType::Storage { read_only: !(WRITTEN.contains(&binding) || binding == WRITTEN_OUT) }
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn buffer(device: &wgpu::Device, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

/// A storage buffer holding `bytes` (padded to whole words, at least 16 bytes).
fn upload(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let mut data = bytes.to_vec();
    data.resize(data.len().max(16).next_multiple_of(4), 0);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &data, usage: wgpu::BufferUsages::STORAGE })
}

/// Noise fields packed for the shader: their grids one after another, then each one's row and column tables (cell,
/// weight bits), and per field (grid offset, grid columns, row table, column table).
fn pack(fields: &[&Noise]) -> (Vec<f32>, Vec<[u32; 2]>, Vec<[u32; 4]>) {
    let (mut grid, mut axes, mut descs) = (Vec::new(), Vec::new(), Vec::new());
    for n in fields {
        let g = grid.len() as u32;
        grid.extend_from_slice(&n.grid);
        let ys = axes.len() as u32;
        axes.extend(n.ys.iter().map(|&(i, w)| [i, w.to_bits()]));
        let xs = axes.len() as u32;
        axes.extend(n.xs.iter().map(|&(i, w)| [i, w.to_bits()]));
        descs.push([g, n.cols as u32, ys, xs]);
    }
    (grid, axes, descs)
}

fn colour(c: [f64; 3]) -> [f32; 4] {
    [c[0] as f32, c[1] as f32, c[2] as f32, 1.0]
}

/// Workgroups of 16 x 16 over a window.
fn dispatch(pass: &mut wgpu::ComputePass, pipe: &wgpu::ComputePipeline, op: &Op) {
    let (w, h) = (op.win[2].saturating_sub(op.win[0]), op.win[3].saturating_sub(op.win[1]));
    if w == 0 || h == 0 {
        return;
    }
    pass.set_pipeline(pipe);
    pass.set_immediates(0, bytemuck::bytes_of(op));
    pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
}

fn op(win: [usize; 4]) -> Op {
    Op { win: win.map(|v| v as u32), ..Default::default() }
}

/// Grow the mask in half `cur` by `round` inside `win`; the half that holds the result.
fn dilate(pass: &mut wgpu::ComputePass, pipes: &Pipes, win: [usize; 4], round: Round, mut cur: i32) -> i32 {
    if round.square > 0 {
        for axis in 0..2 {
            dispatch(pass, &pipes.dil_square, &Op { a: [round.square as i32, axis, cur, 0], ..op(win) });
            cur = 1 - cur;
        }
    }
    for _ in 0..round.steps {
        dispatch(pass, &pipes.dil_cross, &Op { a: [0, 0, cur, 0], ..op(win) });
        cur = 1 - cur;
    }
    cur
}

/// Cast the shadow of the mask in half `cur` lifted by `lift` into the blur buffer's first half, boxed `r` px.
fn cast(pass: &mut wgpu::ComputePass, pipes: &Pipes, win: [usize; 4], cur: i32, shift: [i32; 2], lift: [i32; 2], r: u32) {
    dispatch(pass, &pipes.cast, &Op { a: [shift[0], shift[1], lift[0], lift[1]], b: [cur, 0, 0, 0], ..op(win) });
    if r > 0 {
        for k in 0..4 {
            // crayon.blur's order: down the columns, then along the rows, twice
            dispatch(pass, &pipes.blur, &Op { b: [r as i32, 1 - k % 2, k % 2, 0], ..op(win) });
        }
    }
}

impl CutoutKit {
    pub fn new(gpu: &Gpu) -> Result<CutoutKit> {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("cutout"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..BINDINGS).map(layout_entry).collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cutout"), entries: &entries });
        let p = |entry: &str| crate::render::compute_pipe(device, &module, &bgl, entry, std::mem::size_of::<Op>() as u32, gpu.cache.as_ref());
        let pipes = Pipes {
            lit_init: p("lit_init"),
            lit_morph: p("lit_morph"),
            paper_sheet: p("paper_sheet"),
            warp: p("warp"),
            pic_init: p("pic_init"),
            mask_init: p("mask_init"),
            dil_square: p("dil_square"),
            dil_cross: p("dil_cross"),
            spread: p("spread"),
            cast: p("cast_mask"),
            blur: p("blur"),
            figure: p("figure"),
            piece: p("piece"),
            finish: p("finish"),
        };
        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutout frame"),
            size: std::mem::size_of::<FrameGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let units: Vec<f32> = (0..256).map(unit).collect();
        let units = upload(device, "cutout units", bytemuck::cast_slice(&units));
        Ok(CutoutKit { bgl, pipes, frame, units, sheet: None, scratch: None, steps: BTreeMap::new() })
    }

    fn scratch(&mut self, device: &wgpu::Device, pass: u64, cap: u64, tcap: u64, frame: u64) -> &Scratch {
        let fits = self.scratch.as_ref().is_some_and(|s| s.pass >= pass && s.cap >= cap && s.tcap >= tcap && s.frame >= frame);
        if !fits {
            let old = self.scratch.take();
            let (pass, cap, tcap, frame) = match old {
                Some(s) => (s.pass.max(pass), s.cap.max(cap), s.tcap.max(tcap), s.frame.max(frame)),
                None => (pass, cap, tcap, frame),
            };
            let none = wgpu::BufferUsages::empty();
            self.scratch = Some(Scratch {
                pass,
                cap,
                tcap,
                frame,
                lit: buffer(device, "cutout lit", 8 * pass, none),
                wsrc: buffer(device, "cutout warp", 4 * cap, none),
                pw: buffer(device, "cutout pieces", 4 * cap, none),
                texf: buffer(device, "cutout paper", 4 * tcap, none),
                pic: buffer(device, "cutout picture", 16 * cap, none),
                mask: buffer(device, "cutout masks", 8 * cap, none),
                src: buffer(device, "cutout spread", 4 * cap, none),
                stamp: buffer(device, "cutout stamps", 4 * cap, none),
                sbuf: buffer(device, "cutout shadows", 8 * cap, none),
                out8: buffer(device, "cutout frame out", 4 * frame, wgpu::BufferUsages::COPY_SRC),
                read: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cutout readback"),
                    size: (4 * frame).max(16),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            });
        }
        self.scratch.as_ref().expect("scratch")
    }

    /// The finished frame of a job (`fin`) from its passes: (frame width, height, RGBA8 rows, top first).
    pub fn finish(&mut self, gpu: &Gpu, p: &Passes, fin: &Finish) -> Result<(u32, u32, Vec<u8>)> {
        let device = &gpu.device;
        if !p.w.is_multiple_of(SCALE) || !p.h.is_multiple_of(SCALE) {
            bail!("vector passes of {} x {} are not twice a frame", p.w, p.h);
        }
        let pal = Palette::new(fin, &mut self.steps)?;
        let plan = Plan::new(p, fin, &pal.ids)?;
        if pal.tones.iter().any(|t| t[0] == crate::cutout::KIND_DRAWN || t[1] != 0) && p.tex.is_none() {
            bail!("the look draws tones from the textures and the passes have no tex");
        }
        let (w, h) = (p.w, p.h);
        let (fw, fh) = (w / SCALE, h / SCALE);
        let key = (w, h, plan.sheet_seed);
        if self.sheet.as_ref().is_none_or(|s| s.key != key) {
            let f = fibres(h, w, plan.px, plan.sheet_seed);
            let (grid, axes, descs) = pack(&f.iter().collect::<Vec<_>>());
            self.sheet = Some(Sheet {
                key,
                grid: upload(device, "cutout sheet", bytemuck::cast_slice(&grid)),
                axes: upload(device, "cutout sheet axes", bytemuck::cast_slice(&axes)),
                descs,
            });
        }
        let [bx, by, bw, bh] = plan.bbox.unwrap_or([0; 4]);
        let (tw, th) = (bw + 2 * PAD, bh + 2 * PAD);
        let fields: Vec<&Noise> = plan.rough.iter().chain(&plan.tex).collect();
        let (grid, axes, mut descs) = pack(&fields);
        let sheet = self.sheet.as_ref().expect("sheet");
        let mut all = sheet.descs.clone();
        all.append(&mut descs);
        all.resize(12, [0; 4]);
        let ngroups = pal.ids.iter().map(|g| g[0]).max().unwrap_or(0) as usize + 1;
        let bboxes: Vec<u32> = (0..ngroups).flat_map(|_| [u32::MAX, u32::MAX, 0, 0]).collect();
        let s = &fin.cutout;
        let fr = FrameGpu {
            bg: colour(fin.background),
            rim: s.rim.as_ref().map_or([0.0; 4], |r| colour(r.color)),
            shc: s.shadow.as_ref().map_or([0.0; 4], |sh| colour(sh.color)),
            size: [w as u32, h as u32, fw as u32, fh as u32],
            bbox: [bx as u32, by as u32, bw as u32, bh as u32],
            misc: [tw as u32, p.tex.is_some() as u32, p.typ.is_some() as u32, fin.shade_level],
            misc2: [pal.ids.len() as u32, (bw * bh).max(1) as u32, (w * h) as u32, 0],
            fl: [(s.rough * plan.px) as f32, s.grain as f32, 0.0, 0.0],
        };
        gpu.queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&fr));
        let up = |label: &str, bytes: &[u8]| upload(device, label, bytes);
        let idb = up("cutout ids", p.id);
        let shadeb = up("cutout shade", p.shade);
        let texb = up("cutout tex", p.tex.unwrap_or(&[]));
        let typeb = up("cutout type", p.typ.unwrap_or(&[]));
        let idinfo = up("cutout id info", bytemuck::cast_slice(&pal.ids));
        let tones = up("cutout tones", bytemuck::cast_slice(&pal.tones));
        let colors = up("cutout colours", bytemuck::cast_slice(&pal.colors));
        let steps = up("cutout steps", bytemuck::cast_slice(&pal.steps));
        let fgrid = up("cutout grids", bytemuck::cast_slice(&grid));
        let faxes = up("cutout axes", bytemuck::cast_slice(&axes));
        let noises = up("cutout noises", bytemuck::cast_slice(&all));
        let bbox_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cutout piece boxes"),
            contents: bytemuck::cast_slice(&bboxes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        self.scratch(device, (w * h) as u64, (bw * bh) as u64, (tw * th) as u64, (fw * fh) as u64);
        let sc = self.scratch.as_ref().expect("scratch");
        let sheet = self.sheet.as_ref().expect("sheet");
        let bufs: [&wgpu::Buffer; BINDINGS as usize] = [
            &self.frame, &idb, &shadeb, &texb, &typeb, &idinfo, &tones, &colors, &steps, &self.units, &sc.lit,
            &sheet.grid, &sheet.axes, &fgrid, &faxes, &noises, &sc.wsrc, &sc.pw, &sc.texf, &sc.pic, &sc.mask, &sc.src,
            &sc.stamp, &sc.sbuf, &bbox_buf, &sc.out8,
        ];
        let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cutout"), layout: &self.bgl, entries: &entries });
        let pipes = &self.pipes;

        // the lit mask, the pieces' sheet, the cut, the background under the figure; then where each piece lies
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cutout cut") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cutout cut"), timestamp_writes: None });
            pass.set_bind_group(0, &bg, &[]);
            let full = [0, 0, w, h];
            dispatch(&mut pass, &pipes.lit_init, &op(full));
            for (k, mode) in [1, 0, 0, 1].into_iter().enumerate() {
                dispatch(&mut pass, &pipes.lit_morph, &Op { a: [mode, (k % 2) as i32, 0, 0], ..op(full) });
            }
            if plan.bbox.is_some() {
                dispatch(&mut pass, &pipes.paper_sheet, &op([0, 0, tw, th]));
                dispatch(&mut pass, &pipes.warp, &op([0, 0, bw, bh]));
                dispatch(&mut pass, &pipes.pic_init, &op([0, 0, bw, bh]));
            }
        }
        let boxes = if plan.bbox.is_some() {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cutout piece boxes read"),
                size: (bboxes.len() * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&bbox_buf, 0, &staging, 0, (bboxes.len() * 4) as u64);
            let v = map_read(gpu, enc, &staging, (bboxes.len() * 4) as u64)?;
            bytemuck::cast_slice::<u8, u32>(&v).to_vec()
        } else {
            gpu.queue.submit(Some(enc.finish()));
            Vec::new()
        };

        // the figure's shadow and rim, the pieces far to near, the frame
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cutout lay") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cutout lay"), timestamp_writes: None });
            pass.set_bind_group(0, &bg, &[]);
            if plan.bbox.is_some() {
                let whole = [0, 0, bw, bh];
                if s.rim.is_some() || s.shadow.is_some() {
                    dispatch(&mut pass, &pipes.mask_init, &op(whole));
                    let cur = plan.figure.map_or(0, |r| dilate(&mut pass, pipes, whole, r, 0));
                    if let Some(c) = plan.figure_cast {
                        cast(&mut pass, pipes, whole, cur, [c.dx, c.dy], [0, 0], c.r);
                    }
                    let alpha = plan.figure_cast.map_or(0.0, |c| c.alpha);
                    dispatch(&mut pass, &pipes.figure, &Op {
                        a: [plan.rim_paper[0] as i32, plan.rim_paper[1] as i32, 0, 0],
                        b: [cur, plan.figure_cast.is_some() as i32, 0, s.rim.is_some() as i32],
                        f: [alpha, 0.0, 0.0, 0.0],
                        ..op(whole)
                    });
                }
                for pc in &plan.pieces {
                    let g = pc.group as usize;
                    let b = &boxes[4 * g..4 * g + 4];
                    if b[0] > b[2] {
                        continue;
                    }
                    let m = plan.margin;
                    let win = [(b[0] as usize).saturating_sub(m), (b[1] as usize).saturating_sub(m),
                               (b[2] as usize).saturating_add(m + 1).min(bw), (b[3] as usize).saturating_add(m + 1).min(bh)];
                    dispatch(&mut pass, &pipes.mask_init, &Op { a: [g as i32, 0, 0, 0], ..op(win) });
                    let mut cur = 0;
                    if let Some(r) = plan.grow {
                        cur = dilate(&mut pass, pipes, win, r, 0);
                        for step in 0..plan.spread_steps {
                            for (k, (dx, dy)) in SPREAD.into_iter().enumerate() {
                                let stamp = (1 + 4 * step as usize + k) as i32;
                                dispatch(&mut pass, &pipes.spread, &Op { a: [dx, dy, stamp, 0], b: [cur, 0, 0, 0], ..op(win) });
                            }
                        }
                    }
                    if let Some(c) = plan.piece_cast {
                        cast(&mut pass, pipes, win, cur, [c.dx, c.dy], pc.lift, c.r);
                    }
                    dispatch(&mut pass, &pipes.piece, &Op {
                        a: [pc.lift[0], pc.lift[1], pc.paper[0] as i32, pc.paper[1] as i32],
                        b: [cur, plan.piece_cast.is_some() as i32, 0, 0],
                        f: [plan.piece_cast.map_or(0.0, |c| c.alpha), 0.0, 0.0, 0.0],
                        ..op(win)
                    });
                }
            }
            dispatch(&mut pass, &pipes.finish, &op([0, 0, fw, fh]));
        }
        let bytes = (4 * fw * fh) as u64;
        enc.copy_buffer_to_buffer(&sc.out8, 0, &sc.read, 0, bytes);
        let out = map_read(gpu, enc, &sc.read, bytes)?;
        Ok((fw as u32, fh as u32, out))
    }
}

/// Submit `enc` and read `bytes` of `buf` back.
fn map_read(gpu: &Gpu, enc: wgpu::CommandEncoder, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
    let index = gpu.queue.submit(Some(enc.finish()));
    let slice = buf.slice(..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).map_err(|e| anyhow!("GPU wait: {e}"))?;
    rx.recv().context("readback")?.map_err(|e| anyhow!("readback: {e}"))?;
    let out = slice.get_mapped_range().map_err(|e| anyhow!("readback: {e:?}"))?.to_vec();
    buf.unmap();
    Ok(out)
}
