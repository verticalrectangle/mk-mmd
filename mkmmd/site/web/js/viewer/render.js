// The viewer's WebGL2 renderer: a model's parts in one of the views (shaded, grey, lines, silhouette, x-ray, parts), a
// clip plane (the cut), thick overlay lines (sections, measures) and picking (the world point under a pixel, from a
// depth pass packed into RGBA8). It draws when asked: nothing runs while the view is still.
import { edges } from "./glb.js";
import { invert, project } from "./math.js";

export const MODES = { shaded: 0, grey: 1, silhouette: 2, xray: 3, parts: 4, pick: 5, lines: 1 };

const MESH_VS = `#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNor; layout(location=2) in vec2 aUV;
uniform mat4 uViewProj; uniform vec3 uOffset;
out vec3 vPos; out vec3 vNor; out vec2 vUV;
void main() { vPos = aPos + uOffset; vNor = aNor; vUV = aUV; gl_Position = uViewProj * vec4(vPos, 1.0); }`;

const MESH_FS = `#version 300 es
precision highp float;
in vec3 vPos; in vec3 vNor; in vec2 vUV;
uniform int uMode; uniform vec4 uColor; uniform bool uHasTex; uniform sampler2D uTex; uniform float uCutoff;
uniform vec3 uEye; uniform vec3 uKey; uniform vec3 uFill; uniform vec4 uClip; uniform bool uClipOn;
uniform vec3 uFlat; uniform bool uBackRed;
out vec4 o;
vec4 pack(float v) {
  vec4 e = fract(v * vec4(1.0, 255.0, 65025.0, 16581375.0));
  return e - e.yzww * vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
}
void main() {
  if (uClipOn && dot(uClip.xyz, vPos) + uClip.w < 0.0) discard;
  vec4 base = uColor;
  if (uHasTex && uMode == 0) base *= texture(uTex, vUV);
  if (uCutoff > 0.0 && base.a < uCutoff) discard;
  if (uMode == 5) { o = pack(gl_FragCoord.z); return; }
  if (uMode == 2) { o = vec4(uFlat, 1.0); return; }
  vec3 n = normalize(vNor);
  bool back = !gl_FrontFacing;
  if (back) n = -n;
  vec3 v = normalize(uEye - vPos);
  float facing = max(dot(n, v), 0.0);
  if (uMode == 3) { float a = 0.05 + 0.5 * pow(1.0 - facing, 2.5); o = vec4(uFlat, a); return; }
  float k = max(dot(n, uKey), 0.0), f = max(dot(n, uFill), 0.0), hemi = 0.5 + 0.5 * n.y;
  vec3 c; float l;
  if (uMode == 0) { c = base.rgb; l = 0.50 + 0.38 * k + 0.10 * f + 0.08 * hemi; }
  else { c = uMode == 4 ? uColor.rgb : vec3(0.80, 0.80, 0.82); l = 0.26 + 0.56 * k + 0.14 * f + 0.14 * hemi + 0.12 * pow(1.0 - facing, 3.0); }
  vec3 col = c * l;
  if (uBackRed && back) col = mix(col, vec3(0.92, 0.26, 0.26), 0.7);
  o = vec4(col, uMode == 0 ? base.a : 1.0);
}`;

const LINE_VS = `#version 300 es
layout(location=0) in vec3 aPos; uniform mat4 uViewProj; uniform vec3 uOffset; out vec3 vPos;
void main() { vPos = aPos + uOffset; gl_Position = uViewProj * vec4(vPos, 1.0); gl_Position.z -= 0.0004 * gl_Position.w; }`;
const LINE_FS = `#version 300 es
precision highp float; in vec3 vPos; uniform vec4 uColor; uniform vec4 uClip; uniform bool uClipOn; out vec4 o;
void main() { if (uClipOn && dot(uClip.xyz, vPos) + uClip.w < 0.0) discard; o = uColor; }`;

// Thick lines: a quad per segment (instanced), widened in screen pixels.
const THICK_VS = `#version 300 es
layout(location=0) in vec2 aCorner; layout(location=1) in vec3 aA; layout(location=2) in vec3 aB;
uniform mat4 uViewProj; uniform vec2 uViewport; uniform float uWidth;
void main() {
  vec4 a = uViewProj * vec4(aA, 1.0), b = uViewProj * vec4(aB, 1.0);
  vec2 sa = a.xy / a.w * uViewport, sb = b.xy / b.w * uViewport;
  vec2 d = sb - sa; d = length(d) > 1e-6 ? normalize(d) : vec2(1.0, 0.0);
  vec4 p = mix(a, b, aCorner.x);
  p.xy += vec2(-d.y, d.x) * aCorner.y * uWidth / uViewport * p.w;
  gl_Position = p;
}`;
const THICK_FS = `#version 300 es
precision highp float; uniform vec4 uColor; out vec4 o; void main() { o = uColor; }`;

function program(gl, vs, fs) {
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
    const name = gl.getActiveUniform(p, i).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}

// A colour per part, far apart: golden-angle hues.
export function partColor(i) {
  const hue = (i * 137.508) % 360, s = 0.55, l = 0.62;
  const k = (n) => (n + hue / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4), 1];
}

export class Renderer {
  constructor(canvas, onChange) {
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    if (!gl) throw new Error("this browser has no WebGL2");
    this.gl = gl;
    this.canvas = canvas;
    this.onChange = onChange || (() => {});
    this.mesh = program(gl, MESH_VS, MESH_FS);
    this.line = program(gl, LINE_VS, LINE_FS);
    this.thick = program(gl, THICK_VS, THICK_FS);
    this.cornerBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    this.segBuf = gl.createBuffer();
    this.thickVao = gl.createVertexArray();
    gl.bindVertexArray(this.thickVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.segBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribDivisor(2, 1);
    gl.bindVertexArray(null);
    this.pickFbo = null;
  }

  // GPU buffers for a parsed model; its textures arrive when decoded (onChange is called then).
  upload(model) {
    const gl = this.gl;
    const geos = model.geos.map((g) => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const attr = (loc, data, size) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        return b;
      };
      const pos = attr(0, g.positions, 3);
      attr(1, g.normals, 3);
      if (g.uvs) attr(2, g.uvs, 2); else gl.vertexAttrib2f(2, 0, 0);
      const ebo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, g.indices, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { vao, pos, src: g, lines: null };
    });
    const textures = new Map();
    const parts = model.parts.map((p, i) => {
      const part = { ...p, index: i, tex: null };
      const img = p.material.image;
      if (img) {
        if (!textures.has(img)) {
          const holder = { tex: null };
          textures.set(img, holder);
          createImageBitmap(img).then((bmp) => {
            const t = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, t);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
            gl.generateMipmap(gl.TEXTURE_2D);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
            holder.tex = t;
            this.onChange();
          }).catch(() => {});
        }
        part.holder = textures.get(img);
      }
      return part;
    });
    return { geos, parts, model };
  }

  // The line VAOs of a geometry (every edge, open edges, flipped faces' edges), made the first time Lines asks.
  lines(geo) {
    if (geo.lines) return geo.lines;
    const gl = this.gl, e = edges(geo.src), out = {};
    for (const [k, idx] of Object.entries(e)) {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, geo.pos);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      const ebo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      out[k] = { vao, count: idx.length };
    }
    geo.lines = out;
    return out;
  }

  size() {
    const c = this.canvas, dpr = devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return [w, h];
  }

  // frame: {VP, eye, key, fill, mode, items: [{h (an upload), offset, hidden (Set of part indices), solo, ghost}],
  //         clip: [nx, ny, nz, w] or null, overlays: [{segments: [ax, ay, az, bx, by, bz, ...], color, width}],
  //         bg, flat, edge, open, flipped, ghost}
  draw(frame) {
    const gl = this.gl, [w, h] = this.size();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(frame.bg[0], frame.bg[1], frame.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    for (const it of frame.items) this.drawItem(frame, it, it.ghost ? "ghost" : frame.mode);
    this.drawOverlays(frame, w, h);
  }

  drawItem(frame, it, mode) {
    const gl = this.gl, m = this.mesh, u = m.u;
    gl.useProgram(m.p);
    gl.uniformMatrix4fv(u.uViewProj, false, frame.VP);
    gl.uniform3fv(u.uOffset, it.offset || [0, 0, 0]);
    gl.uniform3fv(u.uEye, frame.eye);
    gl.uniform3fv(u.uKey, frame.key);
    gl.uniform3fv(u.uFill, frame.fill);
    const clip = it.clip === undefined ? frame.clip : it.clip;
    gl.uniform1i(u.uClipOn, clip ? 1 : 0);
    if (clip) gl.uniform4fv(u.uClip, clip);
    gl.uniform1i(u.uBackRed, mode === "lines" ? 1 : 0);
    const xray = mode === "xray" || mode === "ghost";
    gl.uniform1i(u.uMode, xray ? 3 : MODES[mode]);
    gl.uniform3fv(u.uFlat, mode === "ghost" ? frame.ghost : frame.flat);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    if (xray) { gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.disable(gl.CULL_FACE); }
    if (mode === "lines") { gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(1, 1); }
    const shown = it.h.parts.filter((p) => !(it.hidden && it.hidden.has(p.index)) && (it.solo == null || it.solo === p.index));
    for (const pass of xray ? [null] : [false, true]) {
      if (pass === true) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
      for (const p of shown) {
        if (pass !== null && (p.material.blend && mode === "shaded") !== pass) continue;
        if (!xray) { if (p.material.double || mode === "lines") gl.disable(gl.CULL_FACE); else gl.enable(gl.CULL_FACE); }
        const color = mode === "parts" ? partColor(p.index) : p.material.color;
        gl.uniform4fv(u.uColor, color);
        const tex = p.holder && p.holder.tex;
        gl.uniform1i(u.uHasTex, tex ? 1 : 0);
        gl.uniform1f(u.uCutoff, mode === "shaded" ? p.material.cutoff : 0);
        if (tex) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(u.uTex, 0); }
        gl.bindVertexArray(it.h.geos[p.geo].vao);
        gl.drawElements(gl.TRIANGLES, p.count, gl.UNSIGNED_INT, p.start * 4);
      }
      if (pass === true) { gl.disable(gl.BLEND); gl.depthMask(true); }
    }
    if (xray) { gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); }
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.disable(gl.CULL_FACE);
    if (mode === "lines") this.drawEdges(frame, it);
    gl.bindVertexArray(null);
  }

  drawEdges(frame, it) {
    const gl = this.gl, l = this.line;
    gl.useProgram(l.p);
    gl.uniformMatrix4fv(l.u.uViewProj, false, frame.VP);
    gl.uniform3fv(l.u.uOffset, it.offset || [0, 0, 0]);
    const clip = it.clip === undefined ? frame.clip : it.clip;
    gl.uniform1i(l.u.uClipOn, clip ? 1 : 0);
    if (clip) gl.uniform4fv(l.u.uClip, clip);
    const geos = new Set(it.h.parts.filter((p) => !(it.hidden && it.hidden.has(p.index)) && (it.solo == null || it.solo === p.index)).map((p) => p.geo));
    for (const [kind, color] of [["all", frame.edge], ["open", frame.open], ["flipped", frame.flipped]]) {
      gl.uniform4fv(l.u.uColor, color);
      for (const gi of geos) {
        const L = this.lines(it.h.geos[gi])[kind];
        if (!L.count) continue;
        gl.bindVertexArray(L.vao);
        gl.drawElements(gl.LINES, L.count, gl.UNSIGNED_INT, 0);
      }
    }
  }

  drawOverlays(frame, w, h) {
    const gl = this.gl, t = this.thick;
    if (!frame.overlays || !frame.overlays.length) return;
    gl.useProgram(t.p);
    gl.uniformMatrix4fv(t.u.uViewProj, false, frame.VP);
    gl.uniform2f(t.u.uViewport, w / 2, h / 2);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.thickVao);
    for (const o of frame.overlays) {
      if (!o.segments.length) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.segBuf);
      gl.bufferData(gl.ARRAY_BUFFER, o.segments instanceof Float32Array ? o.segments : new Float32Array(o.segments), gl.DYNAMIC_DRAW);
      gl.uniform4fv(t.u.uColor, o.color);
      gl.uniform1f(t.u.uWidth, o.width * (devicePixelRatio || 1) / 2);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, o.segments.length / 6);
    }
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);
  }

  // The world point drawn at CSS pixel (x, y), or null where nothing is.
  pick(frame, x, y) {
    const gl = this.gl, [w, h] = this.size(), dpr = devicePixelRatio || 1;
    const px = Math.round(x * dpr), py = Math.round(h - y * dpr);
    if (!this.pickFbo || this.pickFbo.w !== w || this.pickFbo.h !== h) {
      const fbo = gl.createFramebuffer(), color = gl.createRenderbuffer(), depth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, color);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w, h);
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, color);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      this.pickFbo = { fbo, w, h };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickFbo.fbo);
    gl.viewport(0, 0, w, h);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(px, py, 1, 1);
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    for (const it of frame.items) if (!it.ghost) this.drawItem(frame, it, "pick");
    const out = new Uint8Array(4);
    gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const d = out[0] / 255 + out[1] / 65025 + out[2] / 16581375 + out[3] / 4228250625;
    if (d >= 0.99999 || (out[0] === 255 && out[1] === 255 && out[2] === 255 && out[3] === 255)) return null;
    const inv = invert(frame.VP);
    if (!inv) return null;
    const p = project(inv, [(px + 0.5) / w * 2 - 1, (py + 0.5) / h * 2 - 1, d * 2 - 1]);
    return [p[0], p[1], p[2]];
  }
}
