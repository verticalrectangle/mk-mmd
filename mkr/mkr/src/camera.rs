//! A bake camera on a frame as Blender frames it (mkmmd/site/web/js/scene/shotcam.js, mkmmd/render/plan.py): the view
//! takes glTF's world to Blender's camera axes (it looks down its -Z with +Y up), the projection is the lens on the sensor
//! fitted to the picture (AUTO: the larger side), with the shift in fractions of that side. Depth is reversed ([0, 1],
//! 1 at the near plane) for precision; the picture's row 0 is its top. With depth of field each sample looks from its
//! own point of the aperture through the focus plane (`view_projection`): a thin lens, what the samples converge to.

use anyhow::{Result, bail};
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::bake::{Bake, CameraRec, LIT_CAMERA_FLOATS, LitCamera};

/// glTF's axes to Blender's: glTF (X, Y, Z) is Blender (X, -Z, Y).
pub const TO_BLENDER: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 1.0, 0.0),
    Vec4::new(0.0, -1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fit {
    Auto,
    Horizontal,
    Vertical,
}

/// A camera's depth of field on a frame (Blender's `dof`: the focus distance in metres along the view axis, the f-stop,
/// the aperture's blades, their rotation in radians and the anamorphic ratio).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dof {
    pub focus_distance: f32,
    pub fstop: f32,
    pub blades: f32,
    pub rotation: f32,
    pub ratio: f32,
}

/// Point `i` of `n` spread evenly over the unit disk: a sunflower, radius sqrt((i + 1/2) / n), each point turned by the
/// golden angle from the last.
pub fn sunflower(i: u32, n: u32) -> Vec2 {
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
    let r = ((i as f64 + 0.5) / n.max(1) as f64).sqrt();
    let a = (i as f64 * GOLDEN_ANGLE).rem_euclid(std::f64::consts::TAU);
    Vec2::new((r * a.cos()) as f32, (r * a.sin()) as f32)
}

impl Dof {
    /// The aperture's radius in metres through a lens of `lens` mm: half the focal length over the f-stop.
    pub fn radius(&self, lens: f32) -> f32 {
        lens * 1e-3 / (2.0 * self.fstop.max(1e-5))
    }

    /// Point `i` of the `n` points of the unit aperture (x right, y up on the camera): the sunflower's, moved
    /// area-evenly onto the polygon of `blades` corners inscribed in the disk when there are three or more (a point
    /// keeps its share of the way from the centre and across its corner-to-corner sector: a triangle's area grows
    /// evenly with both), the polygon turned by `rotation`; then narrowed across by `ratio`.
    pub fn aperture(&self, i: u32, n: u32) -> Vec2 {
        let p = sunflower(i, n);
        let blades = self.blades.round();
        let p = if blades >= 3.0 {
            let sector = std::f32::consts::TAU / blades;
            let s = (p.to_angle() - self.rotation).rem_euclid(std::f32::consts::TAU) / sector;
            let first = self.rotation + s.floor() * sector;
            p.length() * Vec2::from_angle(first).lerp(Vec2::from_angle(first + sector), s.fract())
        } else {
            p
        };
        Vec2::new(p.x / self.ratio.max(1e-3), p.y)
    }
}

/// A camera on one (possibly fractional) frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    /// Camera to world, glTF axes (the bake's 3x4).
    pub world: Mat4,
    pub lens: f32,
    pub shift: Vec2,
    pub near: f32,
    pub far: f32,
    pub sensor: f32,
    pub sensor_height: f32,
    pub fit: Fit,
    /// Orthographic scale when the camera is orthographic.
    pub ortho: Option<f32>,
    pub dof: Option<Dof>,
}

pub fn affine(c: &[f32]) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(c[0], c[1], c[2], 0.0),
        Vec4::new(c[3], c[4], c[5], 0.0),
        Vec4::new(c[6], c[7], c[8], 0.0),
        Vec4::new(c[9], c[10], c[11], 1.0),
    )
}

impl Cam {
    /// The camera `rec` on bake frame `k` (`lit`: its v5 lens record, for orthographic cameras and depth of field).
    pub fn at(bake: &Bake, rec: &CameraRec, lit: Option<&LitCamera>, k: usize) -> Result<Cam> {
        let c = bake.f(rec.offset + 15 * k, 15)?;
        let clip = bake.f(rec.clip_at + 2 * k, 2)?;
        let fit = match rec.fit.as_str() {
            "HORIZONTAL" => Fit::Horizontal,
            "VERTICAL" => Fit::Vertical,
            _ => Fit::Auto,
        };
        let mut cam = Cam {
            world: affine(&c[..12]),
            lens: c[12],
            shift: Vec2::new(c[13], c[14]),
            near: clip[0],
            far: clip[1],
            sensor: rec.sensor as f32,
            sensor_height: if rec.sensor_height > 0.0 { rec.sensor_height as f32 } else { rec.sensor as f32 },
            fit,
            ortho: None,
            dof: None,
        };
        if let Some(l) = lit {
            let d = bake.f(l.offset + LIT_CAMERA_FLOATS * k, LIT_CAMERA_FLOATS)?;
            match l.kind.as_str() {
                "PERSP" => {}
                "ORTHO" => cam.ortho = Some(d[6]),
                other => bail!("camera {}: a {other} camera (Blender draws it)", rec.name),
            }
            if d[0] > 0.5 && d[2] > 0.0 {
                cam.dof = Some(Dof { focus_distance: d[1], fstop: d[2], blades: d[3], rotation: d[4], ratio: d[5] });
            }
        }
        if !(cam.near > 0.0 || cam.ortho.is_some()) || cam.far <= cam.near {
            bail!("camera {} on bake frame {k}: clip range {}..{}", rec.name, cam.near, cam.far);
        }
        Ok(cam)
    }

    /// World (glTF) to Blender camera axes.
    pub fn view(&self) -> Mat4 {
        TO_BLENDER * self.world.inverse()
    }

    /// The camera position in the world (glTF axes).
    pub fn position(&self) -> Vec3 {
        self.world.w_axis.truncate()
    }

    /// The way the camera looks (glTF world): Blender's camera -Z, which is its local glTF -Y.
    pub fn forward(&self) -> Vec3 {
        -self.world.y_axis.truncate().normalize()
    }

    /// (pixels of the sensor-fitted side, sensor size in mm along it) for a W x H picture.
    fn fitted(&self, w: f32, h: f32) -> (f32, f32) {
        match self.fit {
            Fit::Vertical => (h, self.sensor_height),
            Fit::Horizontal => (w, self.sensor),
            Fit::Auto => (w.max(h), self.sensor),
        }
    }

    /// A pixel's footprint on a W x H picture: metres per pixel at one metre of view depth (perspective) or anywhere
    /// (orthographic).
    pub fn pixel_footprint(&self, w: f32, h: f32) -> f32 {
        let (s, sensor) = self.fitted(w, h);
        match self.ortho {
            None => sensor / (self.lens * s),
            Some(scale) => scale / s,
        }
    }

    /// A point `d` metres along the view axis blurs over a circle of this times (1 / focus distance - 1 / d) pixels of
    /// a W x H picture (its radius; positive beyond the focus plane): the aperture's radius in pixels one metre away.
    /// Zero without depth of field and through an orthographic camera.
    pub fn coc_scale(&self, w: f32, h: f32) -> f32 {
        match self.dof {
            Some(d) if self.ortho.is_none() => d.radius(self.lens) / self.pixel_footprint(w, h),
            _ => 0.0,
        }
    }

    /// Blender's projection for a W x H picture, its pixels moved by `jitter` (in pixels: the sample is taken at pixel
    /// centre + jitter, x right, y down), into wgpu's clip space with reversed depth.
    pub fn projection(&self, w: f32, h: f32, jitter: Vec2) -> Mat4 {
        let (s, sensor) = self.fitted(w, h);
        let (n, f) = (self.near, self.far);
        let (sx, sy) = (2.0 * s * self.shift.x / w, 2.0 * s * self.shift.y / h);
        let p = match self.ortho {
            None => {
                let k = self.lens / sensor;
                Mat4::from_cols(
                    Vec4::new(2.0 * s * k / w, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, 2.0 * s * k / h, 0.0, 0.0),
                    Vec4::new(sx, sy, n / (f - n), -1.0),
                    Vec4::new(0.0, 0.0, n * f / (f - n), 0.0),
                )
            }
            Some(scale) => Mat4::from_cols(
                Vec4::new(2.0 * s / (w * scale), 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 * s / (h * scale), 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / (f - n), 0.0),
                Vec4::new(-sx, -sy, f / (f - n), 1.0),
            ),
        };
        let t = Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(-2.0 * jitter.x / w, 2.0 * jitter.y / h, 0.0, 1.0));
        t * p
    }

    /// Where on the aperture sample `i` of `n` looks from, in metres on the camera's x (right) and y (up) axes: zero
    /// for a single sample, without depth of field and for an orthographic camera (whose depth of field is Blender's:
    /// mkr info leaves those frames to it).
    pub fn lens_offset(&self, i: u32, n: u32) -> Vec2 {
        match self.dof {
            Some(d) if self.ortho.is_none() && n > 1 => d.aperture(i, n) * d.radius(self.lens),
            _ => Vec2::ZERO,
        }
    }

    /// The aperture's half extents across and up in metres: every `lens_offset` lies inside (zero where they all are).
    pub fn aperture_extent(&self) -> Vec2 {
        match self.dof {
            Some(d) if self.ortho.is_none() => {
                let r = d.radius(self.lens);
                Vec2::new(r / d.ratio.max(1e-3), r)
            }
            _ => Vec2::ZERO,
        }
    }

    /// A sample's view and projection: `projection(w, h, jitter)` seen from `lens` on the aperture (metres on the
    /// camera's x and y axes, `lens_offset`), the camera moved there and its frame sheared so that the focus plane
    /// keeps its place on the picture. A point d metres along the view axis then moves by the offset times the pixels
    /// a metre spans one metre away times (1 / focus - 1 / d): its circle of confusion, the far side's the aperture's
    /// shape and the near side's turned half round. Without depth of field or offset: projection x view.
    pub fn view_projection(&self, w: f32, h: f32, jitter: Vec2, lens: Vec2) -> Mat4 {
        let mut p = self.projection(w, h, jitter);
        let Some(d) = self.dof.filter(|_| lens != Vec2::ZERO) else {
            return p * self.view();
        };
        let s = d.focus_distance.max(1e-4);
        p.z_axis.x -= p.x_axis.x * lens.x / s;
        p.z_axis.y -= p.y_axis.y * lens.y / s;
        p * Mat4::from_translation(-lens.extend(0.0)) * self.view()
    }

    /// The camera between `a` (t = 0) and `b` (t = 1): matrices, lens, clip, focus distance and f-stop linearly, the
    /// rotation kept orthonormal.
    pub fn lerp(a: &Cam, b: &Cam, t: f32) -> Cam {
        if t == 0.0 {
            return *a;
        }
        let m = |x: Vec4, y: Vec4| x + (y - x) * t;
        let x = m(a.world.x_axis, b.world.x_axis).truncate();
        let z = m(a.world.z_axis, b.world.z_axis).truncate();
        // Gram-Schmidt keeps the interpolated frame a rotation; each axis keeps its interpolated scale (the chord of
        // two unit axes is shorter than either).
        let s = |x: Vec4, y: Vec4| x.truncate().length() + (y.truncate().length() - x.truncate().length()) * t;
        let (sx, sy, sz) =
            (s(a.world.x_axis, b.world.x_axis), s(a.world.y_axis, b.world.y_axis), s(a.world.z_axis, b.world.z_axis));
        let xn = x.normalize();
        let zn = (z - xn * z.dot(xn)).normalize();
        let yn = zn.cross(xn);
        let world = Mat4::from_cols(
            (xn * sx).extend(0.0),
            (yn * sy).extend(0.0),
            (zn * sz).extend(0.0),
            m(a.world.w_axis, b.world.w_axis),
        );
        let l = |x: f32, y: f32| x + (y - x) * t;
        Cam {
            world,
            lens: l(a.lens, b.lens),
            shift: a.shift + (b.shift - a.shift) * t,
            near: l(a.near, b.near),
            far: l(a.far, b.far),
            ortho: match (a.ortho, b.ortho) {
                (Some(x), Some(y)) => Some(l(x, y)),
                (o, _) => o,
            },
            dof: match (a.dof, b.dof) {
                (Some(x), Some(y)) => {
                    Some(Dof { focus_distance: l(x.focus_distance, y.focus_distance), fstop: l(x.fstop, y.fstop), ..x })
                }
                (o, _) => o,
            },
            ..*a
        }
    }

    /// Where a world point lands in a W x H picture (pixels from the top left), or None behind the camera.
    pub fn pixel(&self, w: f32, h: f32, p: Vec3) -> Option<Vec2> {
        let clip = self.projection(w, h, Vec2::ZERO) * self.view() * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new((ndc.x + 1.0) * 0.5 * w, (1.0 - ndc.y) * 0.5 * h))
    }
}

/// Blender's render size for an output at `percent` (it rounds down).
pub fn render_size(size: [u32; 2], percent: u32) -> [u32; 2] {
    [size[0] * percent / 100, size[1] * percent / 100]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(world: Mat4) -> Cam {
        Cam {
            world,
            lens: 50.0,
            shift: Vec2::ZERO,
            near: 0.1,
            far: 100.0,
            sensor: 36.0,
            sensor_height: 24.0,
            fit: Fit::Auto,
            ortho: None,
            dof: None,
        }
    }

    /// Blender camera (0.3, 0.2, -2) seen by a camera at the origin with no rotation (it looks down Blender's -Z):
    /// x = W/2 + S f x / -z, with S the larger side, f = lens / sensor; glTF's point is (0.3, -2, -0.2).
    #[test]
    fn projects_like_blender() {
        let c = cam(Mat4::IDENTITY);
        let p = c.pixel(1920.0, 1080.0, Vec3::new(0.3, -2.0, -0.2)).unwrap();
        let f = 50.0 / 36.0;
        assert!((p.x - (960.0 + 1920.0 * f * 0.15)).abs() < 1e-3, "{p}");
        assert!((p.y - (540.0 - 1920.0 * f * 0.1)).abs() < 1e-3, "{p}");
        // a portrait picture fits the sensor to its height
        let p = c.pixel(1080.0, 1920.0, Vec3::new(0.3, -2.0, -0.2)).unwrap();
        assert!((p.x - (540.0 + 1920.0 * f * 0.15)).abs() < 1e-3, "{p}");
        assert!(c.pixel(1920.0, 1080.0, Vec3::new(0.0, 2.0, 0.0)).is_none(), "behind");
    }

    /// The shift moves the frame by its fraction of the larger side: shift_x 0.1 moves the picture 192 px left.
    #[test]
    fn shift_moves_the_frame() {
        let mut c = cam(Mat4::IDENTITY);
        c.shift = Vec2::new(0.1, -0.05);
        let p = c.pixel(1920.0, 1080.0, Vec3::new(0.0, -2.0, 0.0)).unwrap();
        assert!((p.x - (960.0 - 192.0)).abs() < 1e-3, "{p}");
        assert!((p.y - (540.0 - 96.0)).abs() < 1e-3, "{p}");
    }

    /// The jittered projection draws the pixel centre + jitter at the pixel centre.
    #[test]
    fn jitter_samples_the_offset_position() {
        let c = cam(Mat4::IDENTITY);
        let (w, h) = (64.0, 32.0);
        let j = Vec2::new(0.3, -0.2);
        // a point that lands on pixel (10.5 + 0.3, 7.5 - 0.2) without jitter lands on (10.5, 7.5) with it
        let target = Vec2::new(10.8, 7.3);
        let ndc = Vec2::new(target.x / w * 2.0 - 1.0, 1.0 - target.y / h * 2.0);
        let (s, k) = (w, 50.0 / 36.0);
        let z = -3.0;
        let x = ndc.x * -z / (2.0 * s * k / w);
        let y = ndc.y * -z / (2.0 * s * k / h);
        let world = Vec3::new(x, z, -y); // Blender camera axes (x, y, z) to glTF (x, z, -y)
        let clip = c.projection(w, h, j) * c.view() * world.extend(1.0);
        let n = clip.truncate() / clip.w;
        let px = Vec2::new((n.x + 1.0) * 0.5 * w, (1.0 - n.y) * 0.5 * h);
        assert!((px - Vec2::new(10.5, 7.5)).length() < 1e-3, "{px}");
    }

    /// Reversed depth: 1 at the near plane, 0 at the far one.
    #[test]
    fn depth_is_reversed() {
        let c = cam(Mat4::IDENTITY);
        let d = |z: f32| {
            let clip = c.projection(100.0, 100.0, Vec2::ZERO) * c.view() * Vec3::new(0.0, z, 0.0).extend(1.0);
            clip.z / clip.w
        };
        assert!((d(-0.1) - 1.0).abs() < 1e-5);
        assert!(d(-100.0).abs() < 1e-5);
        assert!(d(-1.0) > d(-2.0));
    }

    /// An orthographic camera spans `ortho_scale` metres across the fitted side.
    #[test]
    fn orthographic_scale() {
        let mut c = cam(Mat4::IDENTITY);
        c.ortho = Some(4.0);
        let p = c.pixel(1920.0, 1080.0, Vec3::new(1.0, -5.0, -0.5)).unwrap();
        assert!((p.x - (960.0 + 480.0)).abs() < 1e-3, "{p}");
        assert!((p.y - (540.0 - 240.0)).abs() < 1e-3, "{p}");
    }

    /// The camera's forward is the view's -Z (Blender camera axes), for any rotation.
    #[test]
    fn forward_is_the_view_axis() {
        let c = cam(Mat4::from_rotation_y(0.4) * Mat4::from_rotation_x(-0.7) * Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
        let d = c.view().transform_vector3(c.forward());
        assert!((d - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5, "{d}");
    }

    /// A pixel's footprint: two points one footprint x depth apart across the view land one pixel apart; orthographic,
    /// one footprint apart.
    #[test]
    fn a_pixel_spans_its_footprint() {
        let mut c = cam(Mat4::IDENTITY);
        let (w, h) = (1920.0, 1080.0);
        let f = c.pixel_footprint(w, h);
        let depth = 2.5;
        let a = c.pixel(w, h, Vec3::new(0.0, -depth, 0.0)).unwrap();
        let b = c.pixel(w, h, Vec3::new(f * depth, -depth, 0.0)).unwrap();
        assert!(((b - a).length() - 1.0).abs() < 1e-3, "{a} {b}");
        c.ortho = Some(4.0);
        let f = c.pixel_footprint(w, h);
        let a = c.pixel(w, h, Vec3::new(0.0, -5.0, 0.0)).unwrap();
        let b = c.pixel(w, h, Vec3::new(f, -5.0, 0.0)).unwrap();
        assert!(((b - a).length() - 1.0).abs() < 1e-3, "{a} {b}");
    }

    #[test]
    fn lerp_keeps_a_rotation() {
        let a = cam(Mat4::IDENTITY);
        let b = cam(Mat4::from_rotation_y(0.4) * Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
        let m = Cam::lerp(&a, &b, 0.5);
        let r = glam::Mat3::from_mat4(m.world);
        assert!((r.determinant() - 1.0).abs() < 1e-5);
        assert!((m.world.x_axis.truncate().dot(m.world.z_axis.truncate())).abs() < 1e-6);
        assert_eq!(Cam::lerp(&a, &b, 0.0), a);
    }

    fn dof(blades: f32, rotation: f32, ratio: f32) -> Dof {
        Dof { focus_distance: 2.0, fstop: 2.0, blades, rotation, ratio }
    }

    /// Seen from any point of the aperture the focus plane keeps the pinhole's picture; a point beyond it moves with the
    /// lens offset by S lens / sensor x (1 / focus - 1 / d) pixels per metre of offset (camera x right and y up, the
    /// picture's y down), a point before it the other way; depth stays the pinhole's. Without depth of field the
    /// offset changes nothing.
    #[test]
    fn the_lens_keeps_the_focus_plane_and_blurs_the_rest() {
        let mut c = cam(Mat4::from_rotation_y(0.3) * Mat4::from_translation(Vec3::new(0.5, 0.2, -1.0)));
        c.dof = Some(dof(0.0, 0.0, 1.0));
        let (w, h) = (1920.0, 1080.0);
        // Blender camera axes (x, y, -d) are the camera's own glTF (x, -d, -y)
        let at = |x: f32, y: f32, d: f32| c.world.transform_point3(Vec3::new(x, -d, -y));
        let px = |vp: Mat4, p: Vec3| {
            let q = vp * p.extend(1.0);
            Vec2::new((q.x / q.w + 1.0) * 0.5 * w, (1.0 - q.y / q.w) * 0.5 * h)
        };
        let pin = c.view_projection(w, h, Vec2::ZERO, Vec2::ZERO);
        assert_eq!(pin, c.projection(w, h, Vec2::ZERO) * c.view());
        let lens = Vec2::new(0.006, -0.004);
        let vp = c.view_projection(w, h, Vec2::ZERO, lens);
        let per_m = 1920.0 * 50.0 / 36.0;
        for (x, y, d) in [(0.1, -0.05, 2.0), (-0.3, 0.2, 2.0), (0.2, 0.1, 5.0), (-0.1, -0.2, 0.8)] {
            let p = at(x, y, d);
            let moved = px(vp, p) - px(pin, p);
            let want = Vec2::new(lens.x, -lens.y) * per_m * (1.0 / 2.0 - 1.0 / d);
            assert!((moved - want).length() < 5e-3, "depth {d}: moved {moved}, want {want}");
        }
        let p = at(0.1, 0.1, 3.0);
        let (a, b) = (vp * p.extend(1.0), pin * p.extend(1.0));
        assert!((a.z / a.w - b.z / b.w).abs() < 1e-6);
        let mut flat = c;
        flat.dof = None;
        assert_eq!(flat.view_projection(w, h, Vec2::ZERO, lens), pin);
    }

    /// The aperture's points: inside the unit disk, inside the polygon of the blades when there are three or more,
    /// centred and evenly spread (the mean squared radius a disk's 1/2, a polygon's (2 + cos(2 pi / blades)) / 6),
    /// narrowed across by the ratio. A sample's lens offset is its point times the radius: zero without depth of field,
    /// through an orthographic camera or for a single sample.
    #[test]
    fn the_aperture_is_covered_evenly() {
        let n = 64;
        for (blades, rotation) in [(0.0, 0.0), (5.0, 0.0), (6.0, 0.3), (8.0, -1.0)] {
            let d = dof(blades, rotation, 1.0);
            let pts: Vec<Vec2> = (0..n).map(|i| d.aperture(i, n)).collect();
            let mean = pts.iter().sum::<Vec2>() / n as f32;
            assert!(mean.length() < 0.03, "{blades} blades: mean {mean}");
            let r2 = pts.iter().map(|p| p.length_squared()).sum::<f32>() / n as f32;
            let want = if blades >= 3.0 { (2.0 + (std::f32::consts::TAU / blades).cos()) / 6.0 } else { 0.5 };
            assert!((r2 - want).abs() < 0.01, "{blades} blades: mean squared radius {r2}, want {want}");
            for p in &pts {
                assert!(p.length() <= 1.0 + 1e-5, "{p}");
                if blades >= 3.0 {
                    let sector = std::f32::consts::TAU / blades;
                    for k in 0..blades as i32 {
                        let mid = Vec2::from_angle(rotation + (k as f32 + 0.5) * sector);
                        assert!(p.dot(mid) <= (sector * 0.5).cos() + 1e-5, "{blades} blades: {p} outside edge {k}");
                    }
                }
            }
        }
        let wide = dof(0.0, 0.0, 2.0);
        let across = (0..n).map(|i| wide.aperture(i, n).x.abs()).fold(0.0, f32::max);
        assert!(across <= 0.5 + 1e-5 && across > 0.45, "{across}");

        let mut c = cam(Mat4::IDENTITY);
        assert_eq!((c.lens_offset(3, n), c.aperture_extent()), (Vec2::ZERO, Vec2::ZERO));
        c.dof = Some(dof(0.0, 0.0, 1.0));
        let r = 50e-3 / (2.0 * 2.0);
        assert!((c.lens_offset(3, n) - dof(0.0, 0.0, 1.0).aperture(3, n) * r).length() < 1e-7);
        assert!((c.aperture_extent() - Vec2::splat(r)).length() < 1e-7);
        assert_eq!(c.lens_offset(0, 1), Vec2::ZERO);
        c.ortho = Some(4.0);
        assert_eq!((c.lens_offset(3, n), c.aperture_extent()), (Vec2::ZERO, Vec2::ZERO));
    }

    /// Between two frames the focus distance and the f-stop move linearly, the blades stay the first frame's.
    #[test]
    fn lerp_moves_the_focus() {
        let mut a = cam(Mat4::IDENTITY);
        let mut b = a;
        a.dof = Some(dof(6.0, 0.2, 1.0));
        b.dof = Some(Dof { focus_distance: 3.0, fstop: 4.0, ..dof(0.0, 0.0, 1.0) });
        let d = Cam::lerp(&a, &b, 0.25).dof.unwrap();
        assert!((d.focus_distance - 2.25).abs() < 1e-6 && (d.fstop - 2.5).abs() < 1e-6 && d.blades == 6.0);
    }

    #[test]
    fn render_size_rounds_down() {
        assert_eq!(render_size([1920, 1080], 50), [960, 540]);
        assert_eq!(render_size([1080, 1920], 33), [356, 633]);
    }
}
