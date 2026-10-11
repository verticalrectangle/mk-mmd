// The scene viewer's GPU half (docs/design.md: The site): a bake's meshes (build/<name>.bake, mkmmd/blender/build/bake.py)
// as vertex buffers, placed by their node matrix on each frame and skinned in the vertex shader from a texture of their
// joints' matrices; morph targets are added on the CPU when their weights change. Shaded draws the base colour and
// texture under a key and a fill light, Grey the shapes alone. Blended materials come after the opaque ones, in order.
import { openGLB } from "../viewer/glb.js";

const VS = `#version 300 es
precision highp float;
in vec3 aPos; in vec3 aNor; in vec2 aUv; in vec4 aJoint; in vec4 aWeight;
uniform mat4 uViewProj, uNode;
uniform bool uSkinned;
uniform highp sampler2D uJoints;
out vec3 vNor; out vec2 vUv;
mat4 joint(float j) {
  int x = int(j + 0.5) * 3;
  return transpose(mat4(texelFetch(uJoints, ivec2(x, 0), 0), texelFetch(uJoints, ivec2(x + 1, 0), 0),
                        texelFetch(uJoints, ivec2(x + 2, 0), 0), vec4(0.0, 0.0, 0.0, 1.0)));
}
void main() {
  vec4 p = uNode * vec4(aPos, 1.0);
  vec3 n = mat3(uNode) * aNor;
  if (uSkinned) {
    mat4 m = aWeight.x * joint(aJoint.x) + aWeight.y * joint(aJoint.y) + aWeight.z * joint(aJoint.z) + aWeight.w * joint(aJoint.w);
    p = m * p;
    n = mat3(m) * n;
  }
  vNor = n; vUv = aUv;
  gl_Position = uViewProj * p;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vNor; in vec2 vUv;
uniform vec4 uColor;
uniform bool uTex;
uniform sampler2D uImage;
uniform float uCutoff;
uniform int uMode;
uniform vec3 uKey, uFill;
out vec4 frag;
void main() {
  vec4 c = uColor;
  if (uTex) c *= texture(uImage, vUv);
  if (c.a < uCutoff) discard;
  vec3 n = normalize(vNor);
  if (!gl_FrontFacing) n = -n;
  float l = 0.5 + 0.4 * max(dot(n, uKey), 0.0) + 0.15 * max(dot(n, uFill), 0.0);
  frag = uMode == 1 ? vec4(vec3(0.8) * l, 1.0) : vec4(c.rgb * l, c.a);
}`;

function program(gl) {
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
  ["aPos", "aNor", "aUv", "aJoint", "aWeight"].forEach((name, i) => gl.bindAttribLocation(p, i, name));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (const n of ["uViewProj", "uNode", "uSkinned", "uJoints", "uColor", "uTex", "uImage", "uCutoff", "uMode", "uKey", "uFill"]) u[n] = gl.getUniformLocation(p, n);
  return { p, u };
}

// A bake's 3x4 matrix (12 floats, column by column) as a WebGL mat4.
export const mat4of = (b, o) => new Float32Array([b[o], b[o + 1], b[o + 2], 0, b[o + 3], b[o + 4], b[o + 5], 0, b[o + 6], b[o + 7], b[o + 8], 0,
  b[o + 9], b[o + 10], b[o + 11], 1]);

export class SceneGL {
  // `glb` the scene.glb bytes, `bake` bake.json, `blob` bake.bin as a Float32Array.
  constructor(gl, glb, bake, blob) {
    this.gl = gl;
    this.bake = bake;
    this.blob = blob;
    this.prog = program(gl);
    this.frame = -1;
    const { doc, read, deltas, materials } = openGLB(glb);
    const byName = new Map(doc.nodes.map((n, i) => [n.name, i]));
    const arms = new Map(bake.armatures.map((a) => [a.name, a]));
    const shown = new Map(bake.shown.map((s) => [s.name, s]));
    const morphs = new Map(bake.morphs.map((m) => [m.name, m]));
    this.textures = new Map();
    this.nodes = [];
    this.bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (const entry of bake.nodes) {
      const ni = byName.get(entry.name);
      if (ni === undefined || doc.nodes[ni].mesh === undefined) continue;
      const node = doc.nodes[ni], skin = node.skin !== undefined ? doc.skins[node.skin] : null;
      const N = { entry, shown: shown.get(entry.name) || null, morph: morphs.get(entry.name) || null, items: [], skin: null };
      if (skin && entry.armature && arms.has(entry.armature)) {
        const arm = arms.get(entry.armature), at = new Map(arm.bones.map((b, i) => [b, i]));
        const joints = Int32Array.from(skin.joints, (j) => at.get(doc.nodes[j].name) ?? -1);
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, joints.length * 3, 1, 0, gl.RGBA, gl.FLOAT, null);
        for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
        N.skin = { arm, joints, tex, rows: new Float32Array(joints.length * 12) };
      }
      const M = mat4of(blob, entry.offset);
      for (const prim of doc.meshes[node.mesh].primitives) {
        if ((prim.mode ?? 4) !== 4) continue;
        const A = prim.attributes, pos = read(A.POSITION), count = pos.length / 3;
        const acc = doc.accessors[A.POSITION];
        if (acc.min && acc.max) {
          for (const c of [acc.min, acc.max]) {
            const w = [0, 1, 2].map((r) => M[r] * c[0] + M[4 + r] * c[1] + M[8 + r] * c[2] + M[12 + r]);
            for (let r = 0; r < 3; r++) { this.bounds.min[r] = Math.min(this.bounds.min[r], w[r]); this.bounds.max[r] = Math.max(this.bounds.max[r], w[r]); }
          }
        }
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const attr = (loc, data, size) => {
          const b = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, b);
          gl.bufferData(gl.ARRAY_BUFFER, data, loc === 0 && N.morph ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
          return b;
        };
        const posBuf = attr(0, pos, 3);
        attr(1, A.NORMAL !== undefined ? read(A.NORMAL) : new Float32Array(count * 3).fill(0).map((_, i) => (i % 3 === 1 ? 1 : 0)), 3);
        attr(2, A.TEXCOORD_0 !== undefined ? read(A.TEXCOORD_0) : new Float32Array(count * 2), 2);
        if (N.skin && A.JOINTS_0 !== undefined) { attr(3, read(A.JOINTS_0), 4); attr(4, read(A.WEIGHTS_0), 4); }
        else { gl.disableVertexAttribArray(3); gl.disableVertexAttribArray(4); gl.vertexAttrib4f(3, 0, 0, 0, 0); gl.vertexAttrib4f(4, 1, 0, 0, 0); }
        const idx = prim.indices !== undefined ? read(prim.indices, true) : Uint32Array.from({ length: count }, (_, k) => k);
        const ib = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
        gl.bindVertexArray(null);
        const mat = materials[prim.material] || { color: [0.8, 0.8, 0.8, 1], image: null, blend: false, cutoff: 0 };
        N.items.push({ vao, count: idx.length, mat, posBuf, base: N.morph ? pos : null,
          targets: N.morph && prim.targets ? prim.targets.map((t) => deltas(t.POSITION)) : null,
          skinned: !!(N.skin && A.JOINTS_0 !== undefined) });
        if (mat.image) this.texture(mat.image);
      }
      this.nodes.push(N);
    }
    this.ready = Promise.all([...this.textures.values()].map((holder) => holder.loaded));   // every texture uploaded
  }

  texture(blob) {
    if (this.textures.has(blob)) return;
    const holder = { tex: null, loaded: null };
    this.textures.set(blob, holder);
    holder.loaded = createImageBitmap(blob).then((bmp) => {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      holder.tex = t;
      this.frame = -1;
      if (this.onChange) this.onChange();
    }).catch(() => {});                              // a texture that cannot be read leaves its parts untextured
  }

  // Bring every mesh to bake frame `k` (0 .. frames - 1): node matrices, visibility, joints, morphs.
  setFrame(k) {
    if (k === this.frame) return;
    this.frame = k;
    const gl = this.gl, b = this.blob;
    for (const N of this.nodes) {
      const e = N.entry;
      N.M = mat4of(b, e.offset + 12 * (e.frames > 1 ? k : 0));
      N.visible = !N.shown || b[N.shown.offset + k] > 0.5;
      if (!N.visible) continue;
      if (N.skin) {
        const { arm, joints, rows, tex } = N.skin, base = arm.offset + arm.bones.length * 12 * k;
        for (let j = 0; j < joints.length; j++) {
          const o = base + 12 * Math.max(joints[j], 0), r = j * 12;
          for (let row = 0; row < 3; row++) {
            rows[r + row * 4] = b[o + row]; rows[r + row * 4 + 1] = b[o + 3 + row]; rows[r + row * 4 + 2] = b[o + 6 + row]; rows[r + row * 4 + 3] = b[o + 9 + row];
          }
        }
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, joints.length * 3, 1, gl.RGBA, gl.FLOAT, rows);
      }
      if (N.morph) {
        const w = b.subarray(N.morph.offset + N.morph.count * k, N.morph.offset + N.morph.count * (k + 1));
        const key = Array.from(w, (x) => x.toFixed(4)).join(",");
        if (key === N.morphKey) continue;
        N.morphKey = key;
        for (const it of N.items) {
          if (!it.targets) continue;
          const pos = it.base.slice();
          it.targets.forEach((t, ti) => {
            const wt = w[ti];
            if (!wt) return;
            for (let j = 0; j < t.index.length; j++) {
              const v = t.index[j] * 3;
              pos[v] += wt * t.delta[j * 3]; pos[v + 1] += wt * t.delta[j * 3 + 1]; pos[v + 2] += wt * t.delta[j * 3 + 2];
            }
          });
          gl.bindBuffer(gl.ARRAY_BUFFER, it.posBuf);
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
        }
      }
    }
  }

  // Draw the frame set by setFrame: `VP` the view-projection, `mode` "shaded" or "grey", `key` / `fill` light directions.
  draw(VP, mode, key, fill) {
    const gl = this.gl, { p, u } = this.prog;
    gl.useProgram(p);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.uniformMatrix4fv(u.uViewProj, false, VP);
    gl.uniform1i(u.uMode, mode === "grey" ? 1 : 0);
    gl.uniform3fv(u.uKey, key);
    gl.uniform3fv(u.uFill, fill);
    gl.uniform1i(u.uJoints, 1);
    gl.uniform1i(u.uImage, 0);
    for (const pass of [false, true]) {
      if (pass && mode === "grey") break;
      if (pass) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); } else gl.disable(gl.BLEND);
      for (const N of this.nodes) {
        if (!N.visible) continue;
        gl.uniformMatrix4fv(u.uNode, false, N.M);
        if (N.skin) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, N.skin.tex); }
        for (const it of N.items) {
          if (mode !== "grey" && it.mat.blend !== pass) continue;
          const holder = it.mat.image ? this.textures.get(it.mat.image) : null;
          gl.uniform1i(u.uSkinned, it.skinned ? 1 : 0);
          gl.uniform4fv(u.uColor, it.mat.color);
          gl.uniform1i(u.uTex, holder && holder.tex ? 1 : 0);
          if (holder && holder.tex) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, holder.tex); }
          gl.uniform1f(u.uCutoff, mode === "grey" ? 0.5 : (it.mat.cutoff || (it.mat.blend ? 0.02 : 0)));
          gl.bindVertexArray(it.vao);
          gl.drawElements(gl.TRIANGLES, it.count, gl.UNSIGNED_INT, 0);
        }
      }
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }
}
