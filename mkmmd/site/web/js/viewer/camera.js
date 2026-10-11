// An orbit camera around a target (+Y up): drag turns it, right-drag or shift-drag pans, the wheel zooms; presets
// (front, back, side, other side, top, bottom) and orthographic as Blender's numpad does (1, 3, 7, Ctrl for the
// opposite, 5 for ortho).
import { lookAt, mul, perspective, ortho, norm, cross, sub } from "./math.js";

export const PRESETS = { front: [0, 0], back: [Math.PI, 0], side: [Math.PI / 2, 0], "other side": [-Math.PI / 2, 0],
  top: [0, Math.PI / 2 - 0.0001], bottom: [0, -Math.PI / 2 + 0.0001] };

export class Orbit {
  constructor() {
    this.target = [0, 1, 0];
    this.dist = 3;
    this.yaw = 0;
    this.pitch = 0.06;
    this.fov = 0.6;
    this.ortho = false;
    this.radius = 1;
    this.view = "front";
  }

  eye() {
    const cp = Math.cos(this.pitch);
    return [this.target[0] + this.dist * Math.sin(this.yaw) * cp, this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * Math.cos(this.yaw) * cp];
  }

  frame(aspect) {
    const eye = this.eye(), V = lookAt(eye, this.target, [0, 1, 0]);
    const near = Math.max(this.dist - this.radius * 4, this.dist * 0.01, 0.001), far = this.dist + this.radius * 6;
    const P = this.ortho ? ortho(this.dist * Math.tan(this.fov / 2), aspect, near, far) : perspective(this.fov, aspect, near, far);
    const fwd = norm(sub(this.target, eye)), right = norm(cross(fwd, [0, 1, 0])), up = cross(right, fwd);
    const key = norm([right[0] * -0.45 + up[0] * 0.75 - fwd[0] * 0.55, right[1] * -0.45 + up[1] * 0.75 - fwd[1] * 0.55, right[2] * -0.45 + up[2] * 0.75 - fwd[2] * 0.55]);
    const fill = norm([right[0] * 0.7 - up[0] * 0.2 - fwd[0] * 0.4, right[1] * 0.7 - up[1] * 0.2 - fwd[1] * 0.4, right[2] * 0.7 - up[2] * 0.2 - fwd[2] * 0.4]);
    return { VP: mul(P, V), eye, key, fill, right, up };
  }

  orbit(dx, dy) {
    this.yaw -= dx * 0.008;
    this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.008));
    this.view = null;
  }

  pan(dx, dy, height) {
    const s = 2 * this.dist * Math.tan(this.fov / 2) / Math.max(1, height);
    const { right, up } = this.frame(1);
    this.target = [0, 1, 2].map((k) => this.target[k] - right[k] * dx * s + up[k] * dy * s);
  }

  zoom(k) { this.dist = Math.max(this.radius * 0.03, Math.min(this.radius * 60, this.dist * k)); }

  fit(bounds) {
    this.target = [0, 1, 2].map((k) => (bounds.min[k] + bounds.max[k]) / 2);
    this.radius = Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]) / 2 || 1;
    this.dist = this.radius / Math.sin(this.fov / 2) * 1.02;
  }

  preset(name) {
    [this.yaw, this.pitch] = PRESETS[name];
    this.view = name;
  }

  // Zoom in on a world point (a double-click): it becomes the target, closer.
  focus(p) {
    this.target = p;
    this.dist = Math.max(this.radius * 0.08, this.dist * 0.45);
  }
}
