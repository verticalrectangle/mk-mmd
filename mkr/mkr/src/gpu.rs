//! The GPU: a headless Vulkan device on the high-performance adapter, with ray queries for shadows, binding arrays for
//! the scene's images, timestamps for the report and a pipeline cache kept on disk (`$XDG_CACHE_HOME/mkr`, else
//! `~/.cache/mkr`), so a warm start compiles nothing.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: String,
    /// wgpu 30 does not expose SHADER_EARLY_DEPTH_TEST on Metal; the g-buffer pass drops its force.
    pub metal: bool,
    pub timestamps: bool,
    pub timestamp_period: f32,
    /// The driver's pipeline cache, loaded from and saved to `cache_path`.
    pub cache: Option<wgpu::PipelineCache>,
    cache_path: Option<PathBuf>,
}

/// What mkr cannot run without.
pub fn required_features(metal: bool) -> wgpu::Features {
    wgpu::Features::EXPERIMENTAL_RAY_QUERY
        | wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
        | wgpu::Features::FLOAT32_FILTERABLE
        | wgpu::Features::IMMEDIATES
        | wgpu::Features::DUAL_SOURCE_BLENDING
        | wgpu::Features::PRIMITIVE_INDEX
        | wgpu::Features::INDIRECT_FIRST_INSTANCE
        | wgpu::Features::DEPTH32FLOAT_STENCIL8
        | wgpu::Features::FLOAT32_BLENDABLE
}

/// How many pipelines to compile at once: each compile can hold ~1 GB in the driver, so at most one per 1.5 GB of the
/// memory available now, 4 by default (`MKR_COMPILE_THREADS` overrides), never more than the cores.
pub fn compile_threads() -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    if let Some(n) = std::env::var("MKR_COMPILE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()) {
        return n.clamp(1, cores);
    }
    let avail_gb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1)).and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0 / 1024.0);
    let by_memory = avail_gb.map(|g| (g / 1.5).floor() as usize).unwrap_or(4);
    by_memory.clamp(1, 4.min(cores))
}

fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("mkr"))
}

impl Gpu {
    /// The GPU, once this process holds the GPU lock (one job at a time: mkr_gpulock).
    pub fn new() -> Result<Gpu> {
        mkr_gpulock::hold().context("taking the GPU lock")?;
        pollster::block_on(Self::open())
    }

    async fn open() -> Result<Gpu> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::from_env()
            .unwrap_or(if cfg!(target_os = "macos") { wgpu::Backends::METAL } else { wgpu::Backends::VULKAN });
        desc.flags = wgpu::InstanceFlags::empty().with_env();
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .context("no Vulkan adapter")?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("only a CPU Vulkan adapter ({}): mkr needs a GPU", info.name);
        }
        let have = adapter.features();
        let metal = info.backend == wgpu::Backend::Metal;
        let need = required_features(metal)
            | if metal { wgpu::Features::empty() } else { wgpu::Features::SHADER_EARLY_DEPTH_TEST };
        if !have.contains(need) {
            bail!("the adapter {} lacks {:?}", info.name, need - have);
        }
        let timestamps = have.contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let caching = have.contains(wgpu::Features::PIPELINE_CACHE);
        let features = need
            | if timestamps { wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS } else { wgpu::Features::empty() }
            | if caching { wgpu::Features::PIPELINE_CACHE } else { wgpu::Features::empty() };
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mkr"),
                required_features: features,
                required_limits: limits,
                // SAFETY: ray queries are experimental in wgpu; mkr only uses them as documented (opaque BLASes plus
                // candidate confirmation for alpha-tested geometry).
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("cannot open the GPU")?;
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("mkr: GPU error: {e}");
            std::process::exit(3);
        }));
        let timestamp_period = queue.get_timestamp_period();
        // the pipeline cache: keyed by adapter and driver, started afresh when missing or refused
        let cache_path = if caching {
            wgpu::util::pipeline_cache_key(&info).and_then(|key| cache_dir().map(|d| d.join(key)))
        } else {
            None
        };
        let cache = cache_path.as_ref().map(|path| {
            let data = std::fs::read(path).ok();
            // SAFETY: the data is what this adapter's driver gave back (keyed by adapter and driver) or nothing;
            // `fallback` makes the driver start empty when it refuses it.
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("mkr pipelines"),
                    data: data.as_deref(),
                    fallback: true,
                })
            }
        });
        Ok(Gpu {
            device,
            queue,
            adapter: format!("{} ({:?}, {})", info.name, info.backend, info.driver_info),
            metal: info.backend == wgpu::Backend::Metal,
            timestamps,
            timestamp_period,
            cache,
            cache_path,
        })
    }

    /// Keep the pipeline cache on disk (through a temporary file: a crashed write never leaves half a cache).
    pub fn save_cache(&self) -> Result<()> {
        let (Some(cache), Some(path)) = (&self.cache, &self.cache_path) else { return Ok(()) };
        let Some(data) = cache.get_data() else { return Ok(()) };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("cannot make {}", dir.display()))?;
        }
        // a name of its own: several mkr processes may save at once (the last rename wins, each file whole)
        let tmp = path.with_extension(format!("{}.part", std::process::id()));
        std::fs::write(&tmp, &data).with_context(|| format!("cannot write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))?;
        Ok(())
    }

    /// Wait for everything submitted so far.
    pub fn wait(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("GPU wait: {e}"))?;
        Ok(())
    }
}
