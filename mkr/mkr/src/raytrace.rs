//! EEVEE Next's ray tracing (`scene.eevee.use_raytracing`, ray_tracing_method SCREEN) as mkr draws it
//! (shaders/raytrace.wgsl). Each glossy lobe up to the scene's `trace_max_roughness` sends a ray a sample from its GGX
//! visible normals through the sample's depth buffer, as thick as the scene's `screen_trace_thickness`; a ray that
//! finds a surface takes the light the previous sample left there (the first sample's from a lighting pass of its own
//! before it), one that finds none the world's probe as without tracing. The film's samples average the rays, as
//! EEVEE's denoiser averages its neighbours'. Fast GI (`fast_gi_method`), EEVEE's horizon scan: per sample
//! `fast_gi_ray_count` slices of directions round the view vector, each scanned both ways in `fast_gi_step_count` depth
//! samples, every sample hiding the directions from its front to its back (`fast_gi_thickness_near` behind it, at least
//! `fast_gi_thickness_far` of angle); what stays open lets the world's light through to diffuse closures and (the
//! GLOBAL_ILLUMINATION method, `gi_bounce` of it) what hides it lights them. Glossy lobes rougher than the trace's
//! limit fade from their ray into fast GI.

use serde_json::Value;

use crate::camera::Cam;
use crate::params::{Params, RayTraceMode};

/// The scene's screen tracing (bake.json `lit.eevee`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTrace {
    /// Lobes up to this roughness trace rays; rougher ones fade into fast GI.
    pub max_roughness: f32,
    /// How far behind a surface of the depth buffer a ray may pass and still hit it (m along the view axis).
    pub thickness: f32,
    /// EEVEE's clamp of the light a ray brings (`clamp_surface_indirect`, 0: none): a ray's light whose largest
    /// channel passes it is scaled down to it (lab `rt_clamp`).
    pub clamp: f32,
    /// Fast GI: 0 none, 1 occlusion of the world's light (AMBIENT_OCCLUSION_ONLY), 2 occlusion and bounce
    /// (GLOBAL_ILLUMINATION); slices per sample, depth samples each way, thickness (m), how far it reaches (m, 0: across
    /// the screen), angular thickness (rad), how far the normal leans toward the camera.
    pub gi_method: u32,
    pub gi_rays: u32,
    pub gi_steps: u32,
    pub gi_thickness: f32,
    pub gi_reach: f32,
    pub gi_far: f32,
    pub gi_bias: f32,
}

impl RayTrace {
    /// The scene's tracing: None when it traces nothing (ray tracing off, another method, or `raytrace: "off"`).
    pub fn of(eevee: &Value, p: &Params) -> Option<RayTrace> {
        if p.raytrace == RayTraceMode::Off || !screen_traced(eevee) {
            return None;
        }
        let opts = eevee.get("ray_tracing_options");
        let f = |k: &str, d: f64| opts.and_then(|o| o.get(k)).and_then(Value::as_f64).unwrap_or(d) as f32;
        let e = |k: &str, d: f64| eevee.get(k).and_then(Value::as_f64).unwrap_or(d);
        let gi_method = match eevee.get("fast_gi_method").and_then(Value::as_str) {
            _ if !p.fast_gi => 0,
            Some("AMBIENT_OCCLUSION_ONLY") => 1,
            _ => 2,
        };
        // EEVEE traces fast_gi_ray_count rays per pixel of its fast GI picture, fast_gi_resolution screen pixels across
        // (a value string: "1", "2", ...): per screen pixel that many over its square, at least one slice a sample
        let res = eevee.get("fast_gi_resolution").and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()));
        let res = res.unwrap_or(2.0).max(1.0);
        Some(RayTrace {
            max_roughness: f("trace_max_roughness", 0.5),
            thickness: f("screen_trace_thickness", 0.2),
            clamp: e("clamp_surface_indirect", 10.0).max(0.0) as f32,
            gi_method,
            gi_rays: (e("fast_gi_ray_count", 2.0) / (res * res)).round().max(1.0) as u32,
            gi_steps: if p.gi_steps > 0 { p.gi_steps } else { e("fast_gi_step_count", 8.0).max(1.0) as u32 },
            gi_thickness: e("fast_gi_thickness_near", 0.25) as f32,
            gi_reach: if p.gi_reach > 0.0 { p.gi_reach } else { e("fast_gi_distance", 0.0).max(0.0) as f32 },
            gi_far: e("fast_gi_thickness_far", std::f64::consts::FRAC_PI_4) as f32,
            gi_bias: e("fast_gi_bias", 0.05) as f32,
        })
    }

    /// The frame record's tracing fields (`rt`, `rt2`, `rt3`, `rt4`, and `sampling.z`: the clamp) for camera `cam`.
    pub fn record(&self, cam: &Cam, p: &Params) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], f32) {
        let method = match self.gi_method {
            0 => 0.0,
            1 => 1.0,
            _ => 1.0 + p.gi_bounce,
        };
        ([1.0, self.max_roughness, self.thickness * p.trace_thickness, p.trace_steps.max(1) as f32],
         [p.trace_offset, p.trace_miss, cam.near, cam.far],
         [method, self.gi_rays as f32, self.gi_steps as f32, self.gi_thickness * p.gi_thickness],
         [self.gi_reach, self.gi_far * p.gi_far, self.gi_bias * p.gi_bias, p.gi_rough],
         self.clamp)
    }
}

/// Whether the scene traces EEVEE's rays through the screen (ray tracing on, method SCREEN).
pub fn screen_traced(eevee: &Value) -> bool {
    eevee.get("use_raytracing").and_then(Value::as_bool).unwrap_or(false)
        && eevee.get("ray_tracing_method").and_then(Value::as_str).unwrap_or("SCREEN") == "SCREEN"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fast_gi_follows_the_scene_a_slice_per_pixel_of_its_picture() {
        let mut eevee = json!({"use_raytracing": true, "ray_tracing_method": "SCREEN",
                               "fast_gi_method": "GLOBAL_ILLUMINATION", "fast_gi_resolution": "2",
                               "fast_gi_ray_count": 2, "fast_gi_step_count": 8,
                               "ray_tracing_options": {"trace_max_roughness": 0.5}});
        let p = Params::default();
        let rt = RayTrace::of(&eevee, &p).unwrap();
        assert_eq!((rt.gi_method, rt.gi_rays, rt.gi_steps), (2, 1, 8), "2 rays a pixel of a half-size picture");
        assert_eq!(rt.clamp, 10.0, "Blender's clamp_surface_indirect when the bake has none");
        eevee["clamp_surface_indirect"] = json!(0.0);
        assert_eq!(RayTrace::of(&eevee, &p).unwrap().clamp, 0.0);
        eevee["fast_gi_resolution"] = json!("1");
        assert_eq!(RayTrace::of(&eevee, &p).unwrap().gi_rays, 2);
        eevee["fast_gi_method"] = json!("AMBIENT_OCCLUSION_ONLY");
        assert_eq!(RayTrace::of(&eevee, &p).unwrap().gi_method, 1);
        assert_eq!(RayTrace::of(&eevee, &Params { fast_gi: false, ..Params::default() }).unwrap().gi_method, 0);
        eevee["use_raytracing"] = json!(false);
        assert!(RayTrace::of(&eevee, &p).is_none());
    }
}
