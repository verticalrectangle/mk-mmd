// A bake's shot cameras as Blender frames them (docs/design.md: The site, Rendering): the view (glTF's axes back to
// Blender's camera axes) and the projection of a lens on a sensor fitted to the picture, with its shift.
// mkmmd/render/plan.py projects ring centres the same way.
import { invert, mul } from "../viewer/math.js";
import { mat4of } from "./draw.js";

// glTF's axes back to Blender's camera axes (it looks down its -Z with +Y up): the bake's matrices are A M A^T
export const UNAXES = new Float32Array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]);

// Blender's projection of a camera with `lens` (mm) for a W x H picture: the sensor fitted to the larger side (`fit`
// AUTO: its width), to W (HORIZONTAL: its width) or to H (VERTICAL: its height), the shift in fractions of that side,
// clipped at `near` and `far`. `cam` is the bake's camera record (sensor, sensor_height, fit).
export function cameraProjection(lens, cam, sx, sy, W, H, near, far) {
  const S = cam.fit === "VERTICAL" ? H : cam.fit === "HORIZONTAL" ? W : Math.max(W, H);
  const f = lens / (cam.fit === "VERTICAL" ? cam.sensor_height || cam.sensor : cam.sensor);
  return new Float32Array([2 * S * f / W, 0, 0, 0, 0, 2 * S * f / H, 0, 0, 2 * S * sx / W, 2 * S * sy / H,
    -(far + near) / (far - near), -1, 0, 0, -2 * far * near / (far - near), 0]);
}

// The view matrix of the camera whose 15 floats (3x4 world matrix, lens, shift x, shift y) start at `o` in the blob.
export const cameraView = (blob, o) => mul(UNAXES, invert(mat4of(blob, o)));
