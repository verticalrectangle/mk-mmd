//! The device, its compute kernels and the buffers they work on.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// The shared prelude of every kernel (shaders/common.wgsl).
const COMMON: &str = include_str!("shaders/common.wgsl");

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    /// Whether GPU timestamps can be written between passes (MKR_POST_PROFILE).
    pub timestamps: bool,
    /// Whether kernels may use 64-bit floats (a film's grain spread is worked out in them on the GPU).
    pub f64: bool,
}

impl Gpu {
    /// The most capable adapter (a discrete GPU first), with its own limits for buffer sizes (a 4K picture of vec4
    /// floats is past WebGPU's defaults), once this process holds the GPU lock (one job at a time: mkr_gpulock).
    pub fn new() -> Result<Gpu> {
        mkr_gpulock::hold().context("taking the GPU lock")?;
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // the indirect dispatches are written by mkr-post's own kernels (a pass gated on what the GPU found), always
        // within the limits: no validating pass before each (unless the environment asks for it)
        if std::env::var_os("WGPU_VALIDATION_INDIRECT_CALL").is_none() {
            desc.flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
        }
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;
        let have = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: have.max_storage_buffer_binding_size,
            max_buffer_size: have.max_buffer_size,
            max_storage_buffers_per_shader_stage: have.max_storage_buffers_per_shader_stage.min(16),
            // a pixel sort sorts a whole line of a frame in a workgroup's memory (fx/pixelsort.rs)
            max_compute_workgroup_storage_size: have.max_compute_workgroup_storage_size,
            ..wgpu::Limits::default()
        };
        let stamps = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timestamps = adapter.features().contains(stamps);
        let f64 = adapter.features().contains(wgpu::Features::SHADER_F64);
        let mut features = if timestamps { stamps } else { wgpu::Features::empty() };
        if f64 {
            features |= wgpu::Features::SHADER_F64;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mkr-post"),
            required_features: features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("no GPU device")?;
        Ok(Gpu { device, queue, adapter: adapter.get_info(), timestamps, f64 })
    }

    /// A compute pipeline for entry point `entry` of the kernel source `src` (the prelude prepended), its bind group
    /// layout the one the shader declares.
    pub fn kernel(&self, name: &str, src: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{src}").into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// A storage buffer of `bytes`, written by the queue and copied from.
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A storage buffer holding `data`.
    pub fn storage_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// A uniform buffer holding `data` (one pass's parameters).
    pub fn uniform(&self, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// A buffer of `n` indirect dispatches (three u32 each: x, y, z workgroups) a kernel writes for passes after it.
    pub fn dispatches(&self, label: &str, n: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (12 * n).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// One dispatch of `pipeline` over `groups` with the buffers `bind` (binding number, buffer) in its group 0.
    pub fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind: &[(u32, &wgpu::Buffer)],
        groups: (u32, u32),
    ) {
        self.dispatch(encoder, pipeline, bind, Dispatch::Groups(groups.0, groups.1));
    }

    /// One dispatch of `pipeline` as `how` says with the buffers `bind` (binding number, buffer) in its group 0.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind: &[(u32, &wgpu::Buffer)],
        how: Dispatch,
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = bind
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &group, &[]);
        match how {
            Dispatch::Groups(x, y) => cpass.dispatch_workgroups(x, y, 1),
            Dispatch::Indirect(args, at) => cpass.dispatch_workgroups_indirect(args, 12 * at),
        }
    }
}

/// How many workgroups a pass runs: so many, or the `at`th dispatch (three u32: x, y, z) of a buffer a kernel before it
/// wrote (a pass that runs only when the GPU found it needed, its own count or none: Gpu::dispatches).
#[derive(Clone, Copy)]
pub enum Dispatch<'a> {
    Groups(u32, u32),
    Indirect(&'a wgpu::Buffer, u64),
}

/// Workgroups of 16 x 16 covering a w x h picture.
pub fn tiles(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(16), h.div_ceil(16))
}
