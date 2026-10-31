//! Named tunables (`mkr render --params FILE.json`) for hill climbing against Blender's frames. The defaults are the values
//! that match Blender best; a file names only the ones it changes, and an unknown name is an error.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    /// Radiance multipliers per light type on top of Blender's conventions (area: energy / (pi area), point and spot:
    /// energy / (4 pi² r²) over the sphere, sun: strength as irradiance).
    pub light_area: f32,
    pub light_point: f32,
    pub light_spot: f32,
    pub light_sun: f32,
    /// Multipliers of the diffuse (and translucent) and of the glossy light.
    pub diffuse: f32,
    pub glossy: f32,
    /// The film filter: sigma of the Gaussian as a fraction of the radius, the radius as a multiple of filter_size,
    /// and the jitter's extent in pixels.
    pub filter_sigma: f32,
    pub filter_scale: f32,
    pub jitter: f32,
    /// Ray-traced shadows (`shadow_mode: "rays"`) start this far off the surface along its geometric normal (metres,
    /// plus `shadow_offset_rel` times the distance to the camera).
    pub shadow_offset: f32,
    pub shadow_offset_rel: f32,
    /// Fraction of the light's size the map tracing's shadow rays aim over (`shadow_mode: "map"`; ray-traced shadows
    /// aim over the whole shape). The default, 1.04, is measured: a little wider matches EEVEE's frames of MMD figures,
    /// the lab's plain shapes keep their best near 1.
    pub shadow_spread: f32,
    /// Shadows at all (false: every light reaches every surface it faces; separates shadow error from light error).
    pub shadows: bool,
    /// How shadows are drawn: "map", EEVEE Next's default (a shadow map per light drawn once a frame, a ray per sample
    /// traced through it toward a random point of the light, its light leaks included), or "rays" (ray queries toward
    /// the light: converged soft shadows, as EEVEE's jittered shadows and Cycles draw them).
    pub shadow_mode: ShadowMode,
    /// Shadow maps (shadows.rs), powers of two up to 8192 texels across (EEVEE's finest level: 2048 is its level 2):
    /// a window of the pool (and a sun's map), a whole cube face (its size halves while the faces would take more
    /// than 1 GB), and how many windows the lights share over the receivers that need finer texels than the faces'.
    pub shadow_map_size: u32,
    pub shadow_face_size: u32,
    pub shadow_windows: u32,
    /// The level a receiver gets (shaders/shadow_lod.wgsl): floor(log2(footprint / finest texel) + shadow_lod_bias),
    /// at least floor(log2(resolution limit / finest texel) - shadow_limit_lod). A shadow ray's steps within
    /// shadow_lod_reach texels (of that level) of the receiver read that level, the farther ones the whole face's map
    /// (measured: 2, the face's coarser texels, matches EEVEE's frames better than reading the windows farther out).
    pub shadow_lod_bias: f32,
    pub shadow_limit_lod: f32,
    pub shadow_lod_reach: f32,
    /// The map tracing's hit rule (shaders/shadow_map.wgsl): "interim" or "v2" (ShadowTrace's fit to EEVEE's probes,
    /// with shadow_trace 0.8 and shadow_step_power 2.25).
    pub shadow_rule: ShadowRule,
    /// Steps along a shadow ray (0: the scene's eevee shadow_step_count).
    pub shadow_steps: u32,
    /// Shadow rays per sample and light, as a multiple of the scene's eevee shadow_ray_count (at least 1: EEVEE's own
    /// count), and the sequences their random numbers follow (shaders/light.wgsl shadow_draw): "r2" (R2 over the
    /// light's shape and over the filter disk, R1 for the steps' jitter, each rotated per pixel) or "sobol" (a point of
    /// Sobol's sequence in five dimensions per ray, shuffled and Owen-scrambled per pixel and light: less noise for as
    /// many rays). Fewer samples with more rays each keep the shadows' noise for less of the geometry's work
    /// (docs/design.md: mkr).
    pub shadow_rays: u32,
    pub shadow_sequence: ShadowSequence,
    /// Where the steps sit: step k of S at shadow_trace ((k + u) / S)^shadow_step_power of the way from the shading
    /// point toward the light (u one random number per ray).
    pub shadow_trace: f32,
    pub shadow_step_power: f32,
    /// The light's shadow_filter_radius F as EEVEE's filter acts on the tracing: the ray end jittered over a disk of
    /// F x shadow_filter texel radii, and lifted off the surface by F x shadow_filter_offset more texel radii (x the sine
    /// of the light's angle to the normal: the shadow of a tilted receiver moves toward the light as F grows).
    pub shadow_filter: f32,
    pub shadow_filter_offset: f32,
    /// The tracing's normal offset (texel radii x sin of the light's angle to the normal) and depth tolerance (texel
    /// radii).
    pub shadow_normal_bias: f32,
    pub shadow_depth_bias: f32,
    /// Lights fade out at the distance where they fall under the scene's light threshold (EEVEE's influence radius).
    pub light_influence: bool,
    /// Dither amplitude as a multiple of the scene's dither_intensity / 255, and a constant added before rounding (in
    /// 8-bit levels).
    pub dither: f32,
    pub dither_bias: f32,
    /// Texture filtering: anisotropy (1: off) and the mip level bias.
    pub anisotropy: u16,
    pub mip_bias: f32,
    /// Hashed (dithered) alpha: the hash grid's cell size in pixel footprints (0.25 leaves EEVEE's noise, a layer alone
    /// and layers overlapping alike).
    pub alpha_cell: f32,
    /// Area lights' horizon: the sphere approximation, as EEVEE (the polygon's vector form factor as a sphere clipped
    /// to the lobe's hemisphere: lab area, point, spot and MMD cases closer), or clipping the polygon to the surface's
    /// hemisphere (false), for the diffuse (and translucent) and for the glossy lobes.
    pub ltc_sphere_diffuse: bool,
    pub ltc_sphere_glossy: bool,
    /// The glossy lobes' model: "eevee", EEVEE Next's (measured: its LTC table, GGX fitted by L3 error with a clamped
    /// cosine normalised over the whole sphere, in the frame of the lobe's mean direction, whose part under the
    /// surface's horizon is lost: rough lobes seen obliquely lose up to 12 %, and their highlights are tighter on the
    /// side toward the viewer; and its fade of lights whose bounding sphere sinks under the lobe's horizon), or "mkr",
    /// mkr's own table, renormalised over the surface's hemisphere (Cycles' lobe within a few percent), no fade.
    pub ltc_glossy: LtcFit,
    /// Exposure added to the scene's (stops).
    pub exposure: f32,
    /// Depth of field (dof.rs): "lens" (every sample through its own point of the aperture, the focus plane kept,
    /// converging on a thin lens as Cycles draws it; on real frames closer to EEVEE's than the post process below);
    /// "eevee", the scene's EEVEE scheme (an emulation of its post process, shaders/dof.wgsl, when the
    /// scene's use_bokeh_jittered is off; with it on the samples jittered over most of the aperture and the post
    /// process over the rest, EEVEE's over-blur); or "off" (the pinhole: separates the depth of field's error from the
    /// rest, as `shadows: false` does the shadows').
    pub dof: DofMode,
    /// The post process: rings of samples gathered, the colour weight 1 / (luma + dof_color) and the CoC weight's
    /// sharpness (per pixel of CoC) of its half-resolution copy, the CoC radius (pixels) up to which a pixel blurs at
    /// full resolution (the slight focus) rather than in the layers, and the least area (pixels) a sample spreads its
    /// light over.
    pub dof_rings: u32,
    pub dof_color: f32,
    pub dof_coc: f32,
    pub dof_slight: f32,
    pub dof_floor: f32,
    /// The world's diffuse light (world.rs): its L1 spherical harmonics' band 0 and band 1 weights, band 1 as a
    /// fraction of the cosine lobe's (EEVEE's frames: 0.68-0.75 of it, band 0 a hair under 1).
    pub world_band0: f32,
    pub world_band1: f32,
    /// Glossy lobes read the world's probe at the mip whose texels are this many lobe widths (GGX alpha) across, and
    /// blend into its harmonics' light from roughness world_glossy_mix_from to world_glossy_mix_to.
    pub world_glossy_blur: f32,
    pub world_glossy_mix_from: f32,
    pub world_glossy_mix_to: f32,
    /// Motion blur (`--motion-blur`, motion_blur.rs): "eevee", as the scene's EEVEE Next draws it (motion_blur_steps 1:
    /// each sample drawn at the shutter's middle, smeared along the pixels' velocities by its post process); or
    /// "quality", mkr's own (each sample drawn at its own time in the shutter: true motion blur, as Cycles draws it).
    /// The post process's taps per pixel, velocity and half of the shutter (at least 1; EEVEE takes 8, 6 are as close
    /// over 64 samples, fewer samples take more).
    pub motion_blur: MotionBlurMode,
    pub mb_taps: u32,
    /// EEVEE's ray tracing (raytrace.rs): "screen", its screen-traced reflections where the scene turns ray tracing on
    /// (method SCREEN), or "off" (the probe alone, as without ray tracing: separates the tracing's error); the steps a
    /// ray takes through the depth buffer, a multiplier of the scene's trace thickness, how far off the surface a ray
    /// starts (m, per metre of view distance plus one), and what a ray that finds nothing reads: negative, the probe as
    /// its lobe sees it without tracing (along the mirror direction); else the probe along the ray itself, blurred as
    /// a lobe that many times as rough (measured: 0.5; the mirror direction leaves rough floors 2-3 levels too bright);
    /// and the share of the thickness a ray going away from the camera keeps past a surface it crossed since its last
    /// step (measured: 0.25; the whole thickness on top of the stretch finds 11-15 % more of the lab's rt_emit cards'
    /// light than EEVEE's undenoised frames, from rays passing under and behind them).
    pub raytrace: RayTraceMode,
    pub trace_steps: u32,
    pub trace_thickness: f32,
    pub trace_offset: f32,
    pub trace_miss: f32,
    pub trace_cross: f32,
    /// Fast GI with ray tracing (raytrace.rs): drawn as the scene sets it (false: never), EEVEE's horizon scan of the
    /// depth buffer: its depth samples each way (0: the scene's fast_gi_step_count), multipliers of the scene's
    /// fast_gi_thickness_near, fast_gi_thickness_far (angular) and fast_gi_bias, how far it reaches (m; 0: the scene's
    /// fast_gi_distance, across the screen when that is 0 too); the roughness from which glossy lobes are wholly fast
    /// GI's (from trace_max_roughness up they fade into it from their traced ray); and how much of the
    /// GLOBAL_ILLUMINATION method's bounce mkr draws, one bounce of each sample's light off the sides of surfaces
    /// facing the pixel, clamped as EEVEE clamps indirect light (1, all of it, the scene's method: the lab's rt_* and
    /// gi_* cases 6.00 -> 3.35 levels against the occlusion alone, jwin +0.02 / +0.09; 0: the occlusion alone).
    pub fast_gi: bool,
    pub gi_steps: u32,
    pub gi_thickness: f32,
    pub gi_reach: f32,
    pub gi_far: f32,
    pub gi_bias: f32,
    pub gi_rough: f32,
    pub gi_bounce: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DofMode {
    Eevee,
    Lens,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MotionBlurMode {
    Eevee,
    Quality,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RayTraceMode {
    Screen,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowMode {
    Map,
    Rays,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowRule {
    Interim,
    V2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowSequence {
    R2,
    Sobol,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LtcFit {
    Eevee,
    Mkr,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            light_area: 1.0,
            light_point: 1.0,
            light_spot: 1.0,
            light_sun: 1.0,
            diffuse: 1.0,
            glossy: 1.0,
            filter_sigma: 0.284,
            filter_scale: 1.0,
            jitter: 1.0,
            shadow_offset: 2e-4,
            shadow_offset_rel: 2e-4,
            shadow_spread: 1.04,
            shadows: true,
            shadow_mode: ShadowMode::Map,
            shadow_map_size: 2048,
            shadow_face_size: 1024,
            shadow_windows: 8,
            shadow_lod_bias: 0.44,
            shadow_limit_lod: 0.6,
            shadow_lod_reach: 2.0,
            shadow_rule: ShadowRule::Interim,
            shadow_steps: 0,
            shadow_rays: 1,
            shadow_sequence: ShadowSequence::R2,
            shadow_trace: 0.825,
            shadow_step_power: 2.0,
            shadow_filter: 1.0,
            shadow_filter_offset: 0.77,
            shadow_normal_bias: 1.2,
            shadow_depth_bias: 0.048,
            light_influence: true,
            dither: 1.0,
            dither_bias: 0.0,
            anisotropy: 2,
            mip_bias: 0.0,
            alpha_cell: 0.25,
            ltc_sphere_diffuse: true,
            ltc_sphere_glossy: true,
            ltc_glossy: LtcFit::Eevee,
            exposure: 0.0,
            dof: DofMode::Lens,
            dof_rings: 5,
            dof_color: 4.0,
            dof_coc: 4.0,
            dof_slight: 4.0,
            dof_floor: 1.0,
            world_band0: 0.991,
            world_band1: 0.675,
            world_glossy_blur: 4.0,
            world_glossy_mix_from: 0.7,
            world_glossy_mix_to: 0.9,
            motion_blur: MotionBlurMode::Eevee,
            mb_taps: 6,
            raytrace: RayTraceMode::Screen,
            trace_steps: 32,
            trace_thickness: 1.0,
            trace_offset: 2e-4,
            trace_miss: 0.5,
            trace_cross: 0.25,
            fast_gi: true,
            gi_steps: 0,
            gi_thickness: 1.0,
            gi_reach: 0.0,
            gi_far: 1.0,
            gi_bias: 1.0,
            gi_rough: 1.0,
            gi_bounce: 1.0,
        }
    }
}

impl Params {
    pub fn load(path: Option<&Path>) -> Result<Params> {
        let Some(path) = path else { return Ok(Params::default()) };
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read --params {}", path.display()))?;
        let p: Params = serde_json::from_str(&text).with_context(|| format!("--params {}", path.display()))?;
        p.check().with_context(|| format!("--params {}", path.display()))
    }

    /// The params, unless a count no render can take (shadow rays or motion blur taps under 1).
    fn check(self) -> Result<Params> {
        if self.shadow_rays == 0 || self.mb_taps == 0 {
            bail!("shadow_rays {}, mb_taps {}: each at least 1", self.shadow_rays, self.mb_taps);
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_names_only_what_it_changes() {
        let p: Params = serde_json::from_str(r#"{"light_area": 1.25}"#).unwrap();
        assert_eq!(p.light_area, 1.25);
        assert_eq!(p.filter_sigma, Params::default().filter_sigma);
        assert!(serde_json::from_str::<Params>(r#"{"light_areas": 1}"#).is_err(), "unknown names are errors");
    }

    #[test]
    fn rays_and_taps_are_at_least_one() {
        let parse = |s: &str| serde_json::from_str::<Params>(s).unwrap().check();
        assert!(parse(r#"{"shadow_rays": 2, "shadow_sequence": "sobol", "mb_taps": 8}"#).is_ok());
        assert!(parse(r#"{"shadow_rays": 0}"#).is_err());
        assert!(parse(r#"{"mb_taps": 0}"#).is_err());
    }
}
