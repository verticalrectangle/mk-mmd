// mk's own renderer, the GPU half (docs/design.md: Rendering): the passes of the vector look (mkmmd.blender.styles)
// drawn from a bake as Blender's Workbench draws them, for mkmmd/render to finish with the same compose_vector:
//   id     the material colour pass: each material's id (two bytes) where it is drawn, no antialiasing (a material
//          under half opaque only hides what is behind it; of two surfaces at one depth the one drawn first stays), and
//          in another target the planar depth of the same draw (metres, the float's four bytes)
//   shade  the studio light, fixed in the world, on white, times Workbench's shadow factor; nothing inside the shadow
//          volumes the look's light casts (stencil, extruded from every mesh's outline as Workbench extrudes them for
//          a mesh that is not closed: every face casts, an edge between two faces counts twice). Into the id target's
//          third byte, straight, 8 bits like Blender's PNG
//   tex    the texture colour pass, flat: each material's texture (cut where its alpha is under 0.1) or its viewport
//          colour, in display values
// The shade and texture passes are antialiased as Workbench's are: 16 jittered samples (BLI_jitter_init), each spread
// over its 3 x 3 neighbours with a 2-pixel Blackman-Harris filter and summed in log2 space (half floats), then divided
// out and unpremultiplied. Every mesh is skinned once per frame into buffers in the world (transform feedback) that the
// passes draw and the shadow volumes read through a texture; the meshes baked frame by frame are in the world already.
import { program, SceneGL } from "../scene/draw.js";
import { cameraProjection, cameraView } from "../scene/shotcam.js";
import { mul } from "../viewer/math.js";

const TEX_W = 2048;                         // the width of a mesh's position texture (a vertex per texel)
const OPAQUE_TEX = 254.5 / 255;             // Workbench draws a material transparent when its alpha is not 255 of 255
const TAA_SAMPLES = 16;                     // mkmmd.blender.styles renders the shade and texture passes at render_aa "16"
const HEAD = "#version 300 es\nprecision highp float; precision highp int;\n";

const JOINT = `
mat4 joint(float j) {
  int x = int(j + 0.5) * 3;
  return transpose(mat4(texelFetch(uJoints, ivec2(x, 0), 0), texelFetch(uJoints, ivec2(x + 1, 0), 0),
                        texelFetch(uJoints, ivec2(x + 2, 0), 0), vec4(0.0, 0.0, 0.0, 1.0)));
}`;

const SKIN_VS = `${HEAD}
in vec3 aPos; in vec3 aNor; in vec4 aJoint; in vec4 aWeight;
uniform mat4 uNode;
uniform bool uSkinned;
uniform highp sampler2D uJoints;
out vec3 tPos; out vec3 tNor;
${JOINT}
void main() {
  mat4 m = uNode;
  if (uSkinned) m = (aWeight.x * joint(aJoint.x) + aWeight.y * joint(aJoint.y) + aWeight.z * joint(aJoint.z) + aWeight.w * joint(aJoint.w)) * uNode;
  tPos = (m * vec4(aPos, 1.0)).xyz;
  tNor = transpose(inverse(mat3(m))) * aNor;
  gl_Position = vec4(0.0);
}`;

const NONE_FS = `${HEAD}
out vec4 o;
void main() { o = vec4(0.0); }`;

const DRAW_VS = `${HEAD}
in vec3 aPos; in vec3 aNor; in vec2 aUv;
uniform mat4 uVP;
uniform vec4 uDepth;                        // planar depth: dot(xyz, p) + w
out vec3 vNor; out vec2 vUv; out float vDepth;
void main() {
  vNor = aNor; vUv = aUv;
  vDepth = dot(uDepth.xyz, aPos) + uDepth.w;
  gl_Position = uVP * vec4(aPos, 1.0);
}`;

const ID_FS = `${HEAD}
in float vDepth;
uniform int uId;
layout(location = 0) out vec4 oId;
layout(location = 1) out vec4 oDepth;
void main() {
  oId = vec4(float(uId & 255), float(uId >> 8), 0.0, 255.0) / 255.0;
  uint b = floatBitsToUint(vDepth);
  oDepth = vec4(float(b & 255u), float((b >> 8) & 255u), float((b >> 16) & 255u), float(b >> 24)) / 255.0;
}`;

// The texture colour, linear (sRGB images decoded by their texture format), with the material's alpha.
const TEX_FS = `${HEAD}
in vec2 vUv;
uniform bool uTex, uPremul, uClip;
uniform sampler2D uImage;
uniform float uCutoff;
uniform vec4 uColor;
out vec4 o;
void main() {
  vec3 c = uColor.rgb;
  if (uTex) {
    vec4 t = texture(uImage, vUv);
    if (uClip && (any(lessThan(vUv, vec2(0.0))) || any(greaterThan(vUv, vec2(1.0))))) t = vec4(0.0);
    if (uPremul && t.a > 0.0 && t.a < 1.0) t.rgb /= t.a;
    if (t.a < uCutoff) discard;
    c = t.rgb;
  }
  o = vec4(c, uColor.a);
}`;

const PRE_FS = `${HEAD}
in vec3 vNor;
out vec4 o;
void main() {
  vec3 n = normalize(vNor);
  o = vec4(gl_FrontFacing ? n : -n, 1.0);
}`;

const VOLUME = `
uniform highp sampler2D uP;
uniform mat4 uVP;
uniform vec3 uL;                            // the light's direction of travel
uniform vec4 uDepth;                        // planar depth: dot(xyz, p) + w
uniform float uFar;
vec3 P(float i) { int k = int(i + 0.5); return texelFetch(uP, ivec2(k % ${TEX_W}, k / ${TEX_W}), 0).xyz; }
vec3 back(vec3 p) {                         // extruded toward the far plane (Workbench's extrude_offset)
  float d = 1e5, lf = dot(uDepth.xyz, uL);
  if (lf > 0.0) d = (uFar - dot(uDepth.xyz, p) - uDepth.w) / lf - 1e-3;
  return p + uL * d;
}`;

// An edge: its two faces' far vertices and its ends, as Workbench orders them (v0 v1 v2 v3: the edge is v1 v2); a quad
// along the light where one face faces it and the other does not (six vertices: a strip's two triangles).
const EDGE_VS = `${HEAD}
in vec4 aPos;
${VOLUME}
void main() {
  vec3 v0 = P(aPos.x), v1 = P(aPos.y), v2 = P(aPos.z), v3 = P(aPos.w);
  vec3 n1 = cross(v2 - v1, v0 - v1), n2 = cross(v3 - v1, v2 - v1);
  bvec2 degen = lessThan(vec2(dot(n1, n1), dot(n2, n2)), vec2(4e-17));
  bvec2 bf = greaterThan(vec2(dot(n1, uL), dot(n2, uL)), vec2(0.0));
  bf.x = degen.x ? !bf.y : bf.x;
  bf.y = degen.y ? !bf.x : bf.y;
  if (all(degen) || bf.x == bf.y) { gl_Position = vec4(0.0); return; }
  vec3 a = bf.x ? v1 : v2, b = bf.x ? v2 : v1;
  int s = gl_VertexID < 3 ? gl_VertexID : gl_VertexID == 3 ? 2 : gl_VertexID == 4 ? 1 : 3;
  vec3 w = s == 0 ? a : s == 1 ? b : s == 2 ? back(a) : back(b);
  gl_Position = uVP * vec4(w, 1.0);
}`;

// A face: its front cap where it is and its back cap at the far end, both turned to face the light.
const CAP_VS = `${HEAD}
in vec3 aPos;
${VOLUME}
void main() {
  vec3 v0 = P(aPos.x), v1 = P(aPos.y), v2 = P(aPos.z);
  bool inv = dot(cross(v2 - v1, v0 - v1), uL) > 0.0;
  bool front = gl_VertexID < 3;
  int t = gl_VertexID % 3;
  ivec2 idx = inv ? ivec2(2, 1) : ivec2(1, 2);
  int vi = t == 0 ? 0 : t == 1 ? (front ? idx.x : idx.y) : (front ? idx.y : idx.x);
  vec3 p = vi == 0 ? v0 : vi == 1 ? v1 : v2;
  gl_Position = uVP * vec4(front ? p : back(p), 1.0);
  gl_Position.z += 0.00005;
}`;

const SCREEN_VS = `${HEAD}
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Workbench's studio light without specular (get_world_lighting) and its shadow factor (get_shadow), red channel; one
// TAA sample, premultiplied (where nothing is drawn it stays 0).
const LIGHT_FS = `${HEAD}
uniform highp sampler2D uNor;
uniform vec3 uLd[4];
uniform vec2 uLc[4];                        // red diffuse, wrap
uniform float uAmbient, uShift, uFocus, uMul, uAdd;
uniform vec3 uShadow;
uniform bool uInShadow;
out vec4 o;
float fast_rcp(float v) { return intBitsToFloat(0x7eef370b - floatBitsToInt(v)); }
float wrapped(float nl, float w) { float w1 = w + 1.0; return clamp((nl + w) * fast_rcp(w1 * w1), 0.0, 1.0); }
void main() {
  vec4 n4 = texelFetch(uNor, ivec2(gl_FragCoord.xy), 0);
  if (n4.a < 0.5) discard;
  vec3 n = normalize(n4.xyz);
  float d = uAmbient;
  for (int i = 0; i < 4; i++) d += wrapped(dot(uLd[i], n), uLc[i].y) * uLc[i].x;
  float mixv = smoothstep(uShift, uFocus, -dot(n, uShadow)) * (uInShadow ? 0.0 : uMul);
  o = vec4(d * (mixv + uAdd), 0.0, 0.0, 1.0);
}`;

// One TAA sample into the accumulation: its 3 x 3 neighbours weighted, in log2 space (workbench_effect_taa_frag).
const ACC_FS = `${HEAD}
uniform highp sampler2D uSample;
uniform float uW[9];
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), top = textureSize(uSample, 0) - 1;
  vec4 acc = vec4(0.0);
  int i = 0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++, i++) {
      vec4 c = clamp(texelFetch(uSample, clamp(p + ivec2(x, y), ivec2(0), top), 0), vec4(0.0), vec4(1e10));
      acc += log2(c + 0.5) * uW[i];
    }
  }
  o = acc;
}`;

// The accumulation divided out (workbench_smaa resolve without SMAA), unpremultiplied: the shade into the blue byte, or
// the texture colour through Blender's Standard view (sRGB).
const DONE_FS = `${HEAD}
uniform highp sampler2D uAcc;
uniform float uTotal;
uniform bool uColour;
out vec4 o;
vec3 srgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
void main() {
  vec4 v = exp2(texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0) / uTotal) - 0.5;
  if (v.a > 0.999) v.a = 1.0;
  vec3 c = v.a > 0.0 ? v.rgb / v.a : vec3(0.0);
  o = uColour ? vec4(srgb(c), 1.0) : vec4(0.0, 0.0, c.r, 0.0);
}`;

// Blender's (x, y, z) in glTF's axes.
const GL = (v) => [v[0], v[2], -v[1]];

// Workbench's TAA samples: BLI_jitter_init(n) (its RNG and relaxation, float32 as in C) moved by TaaSamples so the
// sample nearest the middle is at 0, in its [-1, 1] units; with each one's 3 x 3 Blackman-Harris weights.
export function taaSamples(num = TAA_SAMPLES) {
  const f = Math.fround;
  let state = (BigInt(31415926 + num) << 16n) | 0x330En;
  const rand = () => { state = (0x5DEECE66Dn * state + 0xBn) & 0xFFFFFFFFFFFFn; return Number(state >> 17n) / 0x80000000; };
  const sq = f(Math.sqrt(num)), rad1 = f(1 / sq), rad2 = f(1 / num), rad3 = f(sq / num);
  let jit = [];
  let x = 0;
  for (let i = 0; i < num; i++) {
    const a = f(x + f(rad1 * f(0.5 - rand())));
    const b = f(f(i / num) + f(rad1 * f(0.5 - rand())));
    jit.push([a, b]);
    x = f(x + rad3);
    x = f(x - Math.floor(x));
  }
  const near = (vx, vy, r, d) => {
    if (Math.abs(vx) < r && Math.abs(vy) < r) {
      let len = f(Math.sqrt(f(f(vx * vx) + f(vy * vy))));
      if (len > 0 && len < r) { len = f(len / r); d[0] = f(d[0] + f(vx / len)); d[1] = f(d[1] + f(vy / len)); }
    }
  };
  const jitterate1 = (r) => {
    const out = jit.map((p, i) => {
      const d = [0, 0], px = p[0], py = p[1];
      for (let j = num - 1; j >= 0; j--) {
        if (i === j) continue;
        let vx = f(f(jit[j][0] - px) - 1), vy = f(f(jit[j][1] - py) - 1);
        for (let k = 3; k > 0; k--) {
          near(vx, vy, r, d); vx = f(vx + 1);
          near(vx, vy, r, d); vx = f(vx + 1);
          near(vx, vy, r, d); vx = f(vx - 2); vy = f(vy + 1);
        }
      }
      let nx = f(px - f(d[0] / 18)), ny = f(py - f(d[1] / 18));
      return [f(nx - Math.floor(nx)), f(ny - Math.floor(ny))];
    });
    jit = out;
  };
  const jitterate2 = (r) => {
    jit = jit.map((p, i) => {
      let dx = 0, dy = 0;
      for (let j = num - 1; j >= 0; j--) {
        if (i === j) continue;
        let vx = f(f(jit[j][0] - p[0]) - 1), vy = f(f(jit[j][1] - p[1]) - 1);
        for (let k = 0; k < 3; k++) { if (Math.abs(vx) < r) dx = f(dx + f(vx * r)); vx = f(vx + 1); }
        for (let k = 0; k < 3; k++) { if (Math.abs(vy) < r) dy = f(dy + f(vy * r)); vy = f(vy + 1); }
      }
      const nx = f(p[0] - f(dx / 2)), ny = f(p[1] - f(dy / 2));
      return [f(nx - Math.floor(nx)), f(ny - Math.floor(ny))];
    });
  };
  for (let i = 0; i < 24; i++) { jitterate1(rad1); jitterate1(rad1); jitterate2(rad2); }
  jit = jit.map(([a, b]) => [f(a - 0.5), f(b - 0.5)]);
  let ci = 0, cd = 1;
  jit.forEach(([a, b], i) => { const d = f(f(a * a) + f(b * b)); if (d < cd) { cd = d; ci = i; } });
  const c = jit[ci];
  const bh = (r, width) => {
    if (r > width * 0.5) return 0;
    const t = 2 * Math.PI * Math.min(Math.max(r / width + 0.5, 0), 1);
    return 0.35875 - 0.48829 * Math.cos(t) + 0.14128 * Math.cos(2 * t) - 0.01168 * Math.cos(3 * t);
  };
  return jit.map(([a, b]) => {
    const o = [f(f(f(a - c[0]) + 0.5) % 1) * 2 - 1, f(f(f(b - c[1]) + 0.5) % 1) * 2 - 1];
    const w = [];
    for (let gx = -1; gx <= 1; gx++) for (let gy = -1; gy <= 1; gy++) w.push(bh(Math.hypot(gx - o[0], gy - o[1]), 2));
    return { offset: o, weights: w, sum: w.reduce((s, v) => s + v, 0) };
  });
}

// The edges and faces of a mesh's triangles (vertex indices from `base`) for its shadow volumes: an edge between two
// faces twice, an open edge once (its own face's far vertex stands in for the missing one).
function volumes(tris, base) {
  const edges = new Map(), out = [], twice = [];
  let nv = 0;
  for (let i = 0; i < tris.length; i++) nv = Math.max(nv, tris[i] + 1);
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t + e], b = tris[t + (e + 1) % 3], c = tris[t + (e + 2) % 3];
      if (a === b) continue;
      const key = a < b ? a * nv + b : b * nv + a, rec = edges.get(key);
      if (!rec) edges.set(key, [a, b, c, -1, 1]);
      else { if (rec[4] === 1) rec[3] = c; rec[4]++; }
    }
  }
  for (const [a, b, c, d, n] of edges.values()) {
    if (n === 2) { out.push(c + base, a + base, b + base, d + base); twice.push(c + base, a + base, b + base, d + base); }
    else out.push(c + base, a + base, b + base, c + base);
  }
  const caps = new Float32Array(tris.length);
  for (let i = 0; i < tris.length; i++) caps[i] = tris[i] + base;
  return { edges: Float32Array.from(out.concat(twice)), caps };
}

export class PassGL extends SceneGL {
  constructor(gl, glb, bake, blob) {
    super(gl, glb, bake, blob);
    this.depthClamp = gl.getExtension("EXT_depth_clamp");
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("this browser's WebGL cannot draw into float targets");
    this.p = {
      skin: program(gl, SKIN_VS, NONE_FS, ["uNode", "uSkinned", "uJoints"], ["tPos", "tNor"]),
      id: program(gl, DRAW_VS, ID_FS, ["uVP", "uDepth", "uId"]),
      tex: program(gl, DRAW_VS, TEX_FS, ["uVP", "uDepth", "uTex", "uPremul", "uClip", "uImage", "uCutoff", "uColor"]),
      pre: program(gl, DRAW_VS, PRE_FS, ["uVP", "uDepth"]),
      edge: program(gl, EDGE_VS, NONE_FS, ["uP", "uVP", "uL", "uDepth", "uFar"]),
      cap: program(gl, CAP_VS, NONE_FS, ["uP", "uVP", "uL", "uDepth", "uFar"]),
      light: program(gl, SCREEN_VS, LIGHT_FS, ["uNor", "uLd", "uLc", "uAmbient", "uShift", "uFocus", "uMul", "uAdd", "uShadow", "uInShadow"]),
      acc: program(gl, SCREEN_VS, ACC_FS, ["uSample", "uW"]),
      done: program(gl, SCREEN_VS, DONE_FS, ["uAcc", "uTotal", "uColour"]),
    };
    this.taa = taaSamples();
    this.tf = gl.createTransformFeedback();
    this.mats = bake.materials || {};
    this.casts = new Map((bake.objects || []).map((o) => [o.name, o.shadow !== false]));
    this.samplers = new Map();
    this.cams = new Map(bake.cameras.map((c) => [c.name, c]));
    this.W = 0; this.H = 0;
    for (const N of this.nodes) {
      if (N.flex) continue;
      let base = 0;
      const vols = [];
      for (const it of N.items) {
        it.at = base;                                        // its first vertex in the mesh's world buffers
        vols.push(volumes(it.index, base));
        base += it.vertices;
      }
      N.texH = Math.max(1, Math.ceil(base / TEX_W));
      N.wpos = this.buffer(TEX_W * N.texH * 12);
      N.wnor = this.buffer(Math.max(1, base) * 12);
      for (const it of N.items) {
        it.dvao = gl.createVertexArray();
        gl.bindVertexArray(it.dvao);
        gl.bindBuffer(gl.ARRAY_BUFFER, N.wpos);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, it.at * 12);
        gl.bindBuffer(gl.ARRAY_BUFFER, N.wnor);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, it.at * 12);
        gl.bindBuffer(gl.ARRAY_BUFFER, it.uvBuf);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, it.ib);
        gl.bindVertexArray(null);
      }
      const edges = new Float32Array(vols.reduce((s, v) => s + v.edges.length, 0));
      const caps = new Float32Array(vols.reduce((s, v) => s + v.caps.length, 0));
      let e = 0, c = 0;
      for (const v of vols) { edges.set(v.edges, e); e += v.edges.length; caps.set(v.caps, c); c += v.caps.length; }
      this.setVolumes(N, edges, caps);
      N.posTex = this.posTexture(N.texH);
    }
  }

  buffer(bytes) {
    const gl = this.gl, b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return b;
  }

  posTexture(rows) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, TEX_W, rows, 0, gl.RGB, gl.FLOAT, null);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    return t;
  }

  // The shadow volumes' instances of a mesh: edges (four vertex indices each) and faces (three).
  setVolumes(N, edges, caps) {
    const gl = this.gl, vao = (data, size) => {
      const v = gl.createVertexArray(), b = gl.createBuffer();
      gl.bindVertexArray(v);
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(0, 1);
      gl.bindVertexArray(null);
      return { v, b };
    };
    for (const old of [N.edgeVao, N.capVao]) if (old) { gl.deleteVertexArray(old.v); gl.deleteBuffer(old.b); }
    N.edgeVao = vao(edges, 4); N.nEdges = edges.length / 4;
    N.capVao = vao(caps, 3); N.nCaps = caps.length / 3;
  }

  // Images as Workbench samples them: colour-managed sRGB decoded to linear (or the raw values), alpha straight.
  texture(blob) {
    if (this.textures.has(blob)) return;
    const holder = { tex: null, srgb: null, loaded: null };
    this.textures.set(blob, holder);
    holder.loaded = createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" }).then((bmp) => {
      const gl = this.gl;
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
      for (const [key, fmt] of [["srgb", gl.SRGB8_ALPHA8], ["tex", gl.RGBA8]]) {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, fmt, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
        holder[key] = t;
      }
    });
  }

  // Workbench's sampler for an image node: linear or closest (no mipmaps), the node's extension.
  sampler(closest, extension) {
    const key = `${closest}/${extension}`;
    if (!this.samplers.has(key)) {
      const gl = this.gl, s = gl.createSampler(), f = closest ? gl.NEAREST : gl.LINEAR;
      const wrap = extension === "REPEAT" ? gl.REPEAT : extension === "MIRROR" ? gl.MIRRORED_REPEAT : gl.CLAMP_TO_EDGE;
      gl.samplerParameteri(s, gl.TEXTURE_MIN_FILTER, f);
      gl.samplerParameteri(s, gl.TEXTURE_MAG_FILTER, f);
      gl.samplerParameteri(s, gl.TEXTURE_WRAP_S, wrap);
      gl.samplerParameteri(s, gl.TEXTURE_WRAP_T, wrap);
      this.samplers.set(key, s);
    }
    return this.samplers.get(key);
  }

  // The targets at W x H: id (ids, shade) and depth bytes with their depth buffer; the normals with the depth and
  // stencil the shadow volumes count in; a TAA sample, its accumulation; the texture pass with its depth buffer.
  size(W, H) {
    if (W === this.W && H === this.H) return;
    const gl = this.gl;
    for (const t of this.texs || []) gl.deleteTexture(t);
    for (const r of this.rbs || []) gl.deleteRenderbuffer(r);
    for (const f of this.fbs || []) gl.deleteFramebuffer(f);
    const tex = (fmt) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, W, H);
      for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
      return t;
    };
    const rb = (fmt) => { const r = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, r); gl.renderbufferStorage(gl.RENDERBUFFER, fmt, W, H); return r; };
    const fb = (colors, depth, attach) => {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      colors.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
      if (depth) gl.framebufferRenderbuffer(gl.FRAMEBUFFER, attach, gl.RENDERBUFFER, depth);
      gl.drawBuffers(colors.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("a render target cannot be made");
      return f;
    };
    this.idTex = tex(gl.RGBA8); this.depthTex = tex(gl.RGBA8); this.texTex = tex(gl.RGBA8);
    this.norTex = tex(gl.RGBA16F); this.sampleTex = tex(gl.RGBA16F); this.accTex = tex(gl.RGBA16F);
    this.idDepth = rb(gl.DEPTH_COMPONENT24); this.texDepth = rb(gl.DEPTH_COMPONENT24); this.shadeDS = rb(gl.DEPTH24_STENCIL8);
    this.fbId = fb([this.idTex, this.depthTex], this.idDepth, gl.DEPTH_ATTACHMENT);
    this.fbPre = fb([this.norTex], this.shadeDS, gl.DEPTH_STENCIL_ATTACHMENT);
    this.fbLight = fb([this.sampleTex], this.shadeDS, gl.DEPTH_STENCIL_ATTACHMENT);
    this.fbTexSample = fb([this.sampleTex], this.texDepth, gl.DEPTH_ATTACHMENT);
    this.fbAcc = fb([this.accTex], null);
    this.fbShade = fb([this.idTex], null);
    this.fbTex = fb([this.texTex], null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.texs = [this.idTex, this.depthTex, this.texTex, this.norTex, this.sampleTex, this.accTex];
    this.rbs = [this.idDepth, this.texDepth, this.shadeDS];
    this.fbs = [this.fbId, this.fbPre, this.fbLight, this.fbTexSample, this.fbAcc, this.fbShade, this.fbTex];
    this.W = W; this.H = H;
  }

  // The world-space buffers of the drawn meshes on the frame set: skinned by transform feedback, and the position
  // textures of those that cast shadows (a mesh baked frame by frame fills its own from the bake).
  skin(drawn, casts) {
    const gl = this.gl, { p, u } = this.p.skin;
    gl.useProgram(p);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.uniform1i(u.uJoints, 1);
    for (const N of drawn) {
      if (N.flex) continue;
      gl.uniformMatrix4fv(u.uNode, false, N.M);
      if (N.skin) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, N.skin.tex); }
      for (const it of N.items) {
        gl.uniform1i(u.uSkinned, it.skinned ? 1 : 0);
        gl.bindVertexArray(it.vao);
        gl.bindBufferRange(gl.TRANSFORM_FEEDBACK_BUFFER, 0, N.wpos, it.at * 12, it.vertices * 12);
        gl.bindBufferRange(gl.TRANSFORM_FEEDBACK_BUFFER, 1, N.wnor, it.at * 12, it.vertices * 12);
        gl.beginTransformFeedback(gl.POINTS);
        gl.drawArrays(gl.POINTS, 0, it.vertices);
        gl.endTransformFeedback();
      }
    }
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindVertexArray(null);
    for (const N of casts) {
      if (N.flex) { this.flexVolumes(N); continue; }
      gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, N.wpos);
      gl.bindTexture(gl.TEXTURE_2D, N.posTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, N.texH, gl.RGB, gl.FLOAT, 0);
      gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
    }
  }

  // A frame-by-frame mesh's position texture and shadow volumes for the geometry it shows (made when it changes).
  flexVolumes(N) {
    if (N.volGeom === N.geom) return;
    const gl = this.gl, [vo, nv] = N.flex.geoms[N.geom], rows = Math.max(1, Math.ceil(nv / TEX_W));
    const pos = new Float32Array(TEX_W * rows * 3), v = this.blob;
    for (let i = 0; i < nv; i++) { pos[i * 3] = v[vo + i * 6]; pos[i * 3 + 1] = v[vo + i * 6 + 1]; pos[i * 3 + 2] = v[vo + i * 6 + 2]; }
    if (N.texH !== rows) { if (N.posTex) gl.deleteTexture(N.posTex); N.posTex = this.posTexture(rows); N.texH = rows; }
    gl.bindTexture(gl.TEXTURE_2D, N.posTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, rows, gl.RGB, gl.FLOAT, pos);
    if (N.volTris !== N.flex.geoms[N.geom][2]) {
      const vol = volumes(N.tris, 0);
      this.setVolumes(N, vol.edges, vol.caps);
      N.volTris = N.flex.geoms[N.geom][2];
    }
    N.volGeom = N.geom;
  }

  // Draw the items of the drawn meshes that `want(material)` takes, with `bind(item, material)` setting each one's
  // uniforms.
  each(drawn, want, bind) {
    const gl = this.gl;
    for (const N of drawn) {
      for (const it of N.items) {
        const m = this.mats[it.mat.name] || null;
        if (!want(m)) continue;
        bind(it, m);
        gl.bindVertexArray(N.flex ? N.vao : it.dvao);
        gl.drawElements(gl.TRIANGLES, it.count, gl.UNSIGNED_INT, it.first * 4);
      }
    }
    gl.bindVertexArray(null);
  }

  // A TAA sample's projection: Workbench's window_translate_m4 of the sample's offset (in its units over the target),
  // which moves the picture the way the offset points (measured against Blender's passes: the other way blurs them
  // apart).
  jitter(P, V, o) {
    const PV = mul(P, V), J = Float32Array.from(P);
    J[8] -= P[0] * o[0] / this.W / Math.hypot(PV[0], PV[4], PV[8]);
    J[9] -= P[5] * o[1] / this.H / Math.hypot(PV[1], PV[5], PV[9]);
    return mul(J, V);
  }

  // Draw one frame's passes: `job` {k (bake frame), camera, size [W, H] of the passes}, `look` {draw: Set of object
  // names, ids: {material: id}, light: [x, y, z] (Blender's light_direction), tex: bool}. Returns the bytes for
  // mkmmd/render: id and shade (4 per pixel), depth (4), and the texture pass (4) when the look draws from textures,
  // bottom row first.
  render(job, look) {
    const gl = this.gl, [W, H] = job.size;
    this.setFrame(job.k);
    this.size(W, H);
    const cam = this.cams.get(job.camera);
    if (!cam) throw new Error(`the bake has no camera ${job.camera}`);
    const o = cam.offset + 15 * job.k, b = this.blob, c = cam.clip_at + 2 * job.k;
    const V = cameraView(b, o), near = b[c], far = b[c + 1];             // the clip range of that frame (the clip stage)
    const P = cameraProjection(b[o + 12], cam, b[o + 13], b[o + 14], W, H, near, far), VP = mul(P, V);
    const depth = [-V[2], -V[6], -V[10], -V[14]];
    const drawn = this.nodes.filter((N) => N.visible && look.draw.has(N.name));
    const casts = drawn.filter((N) => this.casts.get(N.name) !== false);
    this.skin(drawn, casts);
    const cull = !!(this.bake.workbench || {}).cull;                    // the scene's shading culls back faces
    const faces = (on) => (on && cull ? gl.enable(gl.CULL_FACE) : gl.disable(gl.CULL_FACE));
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);                                               // the bytes are data: ids, depth, shade
    gl.disable(gl.STENCIL_TEST);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);                                               // a tie keeps the material drawn first
    gl.depthMask(true);
    faces(true);

    // id and depth: opaque materials first, then those under half opaque into the depth alone
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbId);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    const { p: pid, u: uid } = this.p.id;
    gl.useProgram(pid);
    gl.uniformMatrix4fv(uid.uVP, false, VP);
    gl.uniform4fv(uid.uDepth, depth);
    const halfOpaque = (m) => !m || m.alpha >= 0.5;
    this.each(drawn, halfOpaque, (it) => gl.uniform1i(uid.uId, look.ids[it.mat.name] || 0));
    gl.drawBuffers([gl.NONE, gl.COLOR_ATTACHMENT1]);
    this.each(drawn, (m) => !halfOpaque(m), () => {});
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

    // shade, sample by sample: normals and depth, the shadow volumes into the stencil, the light where it is 0
    const L = GL([-look.light[0], look.light[2], -look.light[1]]);    // Workbench turns light_direction: (-x, z, -y)
    this.accumulate(() => {
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    });
    let total = 0;
    for (const s of this.taa) {
      const VPs = this.jitter(P, V, s.offset);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbPre);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      gl.clearDepth(1);
      gl.clearStencil(0);
      gl.clear(gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
      const { p: ppre, u: upre } = this.p.pre;
      gl.useProgram(ppre);
      gl.uniformMatrix4fv(upre.uVP, false, VPs);
      gl.uniform4fv(upre.uDepth, depth);
      faces(true);
      this.each(drawn, () => true, () => {});
      faces(false);
      this.volumes(casts, VPs, L, depth, far);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbLight);
      gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      this.light(L);
      this.accumulate(() => this.addSample(s));
      total += s.sum;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbShade);
    gl.colorMask(false, false, true, false);
    this.finish(total, false);
    gl.colorMask(true, true, true, true);

    // textures, sample by sample
    if (look.tex) {
      this.accumulate(() => gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]));
      const { p: ptex, u: utex } = this.p.tex;
      const bind = (it, m) => {
        const holder = m && m.texture === "glb" && it.mat.image ? this.textures.get(it.mat.image) : null;
        const t = holder && (m.srgb === false ? holder.tex : holder.srgb);
        gl.uniform1i(utex.uTex, t ? 1 : 0);
        gl.uniform4fv(utex.uColor, [...(m ? m.color : [0.8, 0.8, 0.8]), m ? m.alpha : 1]);
        if (t) {
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, t);
          gl.bindSampler(0, this.sampler(!!m.closest, m.extension || "REPEAT"));
          gl.uniform1i(utex.uPremul, m.premul ? 1 : 0);
          gl.uniform1i(utex.uClip, m.extension === "CLIP" ? 1 : 0);
          gl.uniform1f(utex.uCutoff, m.cutoff ? 0.1 : -1);
        } else gl.uniform1f(utex.uCutoff, -1);
      };
      const opaque = (m) => !m || m.alpha >= OPAQUE_TEX;
      let ttotal = 0;
      for (const s of this.taa) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbTexSample);
        gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        gl.useProgram(ptex);
        gl.uniformMatrix4fv(utex.uVP, false, this.jitter(P, V, s.offset));
        gl.uniform4fv(utex.uDepth, depth);
        gl.uniform1i(utex.uImage, 0);
        faces(true);
        this.each(drawn, opaque, bind);
        gl.enable(gl.BLEND);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        this.each(drawn, (m) => !opaque(m), bind);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.bindSampler(0, null);
        faces(false);
        this.accumulate(() => this.addSample(s));
        ttotal += s.sum;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbTex);
      this.finish(ttotal, true);
    }

    // read back, bottom row first, into one buffer kept for every frame of this size (each frame's bytes are sent
    // before the next is drawn)
    const n = W * H * 4;
    if (!this.out || this.out.length !== n * 3) this.out = new Uint8Array(n * 3);
    const out = this.out.subarray(0, n * (look.tex ? 3 : 2));
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbId);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, out.subarray(0, n));
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, out.subarray(n, 2 * n));
    if (look.tex) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbTex);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, out.subarray(2 * n, 3 * n));
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }

  // Run `draw` into the TAA accumulation (additive, depth and stencil off).
  accumulate(draw) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbAcc);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    draw();
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
  }

  addSample(s) {
    const gl = this.gl, { p, u } = this.p.acc;
    gl.useProgram(p);
    gl.uniform1i(u.uSample, 4);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.sampleTex);
    gl.uniform1fv(u.uW, s.weights);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // The accumulation divided out into the bound target: the texture colour, or the shade into its blue byte.
  finish(total, colour) {
    const gl = this.gl, { p, u } = this.p.done;
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(p);
    gl.uniform1i(u.uAcc, 4);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.accTex);
    gl.uniform1f(u.uTotal, total);
    gl.uniform1i(u.uColour, colour ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);
  }

  // The stencil: every casting mesh's shadow volume counted where it lies behind what is drawn (depth fail, with caps)
  // when the light runs away from the camera, else where it lies in front (depth pass).
  volumes(casters, VP, L, depth, far) {
    const gl = this.gl;
    const fail = depth[0] * L[0] + depth[1] * L[1] + depth[2] * L[2] > 0;
    gl.colorMask(false, false, false, false);
    gl.depthMask(false);
    gl.enable(gl.STENCIL_TEST);
    gl.stencilFunc(gl.ALWAYS, 0, 0xff);
    if (fail) {
      gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.INCR_WRAP, gl.KEEP);
      gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.DECR_WRAP, gl.KEEP);
    } else {
      gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.KEEP, gl.INCR_WRAP);
      gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.KEEP, gl.DECR_WRAP);
    }
    if (this.depthClamp) gl.enable(this.depthClamp.DEPTH_CLAMP_EXT);
    for (const key of fail ? ["edge", "cap"] : ["edge"]) {
      const { p, u } = this.p[key];
      gl.useProgram(p);
      gl.uniformMatrix4fv(u.uVP, false, VP);
      gl.uniform3fv(u.uL, L);
      gl.uniform4fv(u.uDepth, depth);
      gl.uniform1f(u.uFar, far);
      gl.uniform1i(u.uP, 2);
      gl.activeTexture(gl.TEXTURE2);
      for (const N of casters) {
        const [vao, count] = key === "edge" ? [N.edgeVao, N.nEdges] : [N.capVao, N.nCaps];
        if (!count) continue;
        gl.bindTexture(gl.TEXTURE_2D, N.posTex);
        gl.bindVertexArray(vao.v);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
      }
    }
    if (this.depthClamp) gl.disable(this.depthClamp.DEPTH_CLAMP_EXT);
    gl.bindVertexArray(null);
    gl.disable(gl.STENCIL_TEST);
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
  }

  // One sample's shade into the bound target: lit where the stencil is 0, the shadow's floor elsewhere.
  light(L) {
    const gl = this.gl, { p, u } = this.p.light, wb = this.bake.workbench || {};
    const lights = (wb.lights || []).slice(0, 4);
    const r = wb.rotate_z || 0, cr = Math.cos(r), sr = Math.sin(r);
    const ld = [], lc = [];
    for (let i = 0; i < 4; i++) {
      const sl = lights[i];
      if (!sl || !sl.use) { ld.push(1, 0, 0); lc.push(0, 0); continue; }
      const [x, y, z] = sl.direction, d = [x, -z, y];                    // the studio light's axes in Blender's world
      ld.push(...GL([d[0] * cr + d[1] * sr, -d[0] * sr + d[1] * cr, d[2]]));
      lc.push(sl.diffuse[0], sl.wrap);
    }
    gl.useProgram(p);
    gl.disable(gl.DEPTH_TEST);
    gl.uniform1i(u.uNor, 3);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.norTex);
    gl.uniform3fv(u.uLd, ld);
    gl.uniform2fv(u.uLc, lc);
    gl.uniform1f(u.uAmbient, (wb.ambient || [1])[0]);
    const shift = 0.1, focus = 1e-4;                                     // as mkmmd.blender.styles sets them
    gl.uniform1f(u.uShift, shift);
    gl.uniform1f(u.uFocus, 1 - focus * (1 - shift));
    gl.uniform1f(u.uMul, 1);
    gl.uniform1f(u.uAdd, 0);
    gl.uniform3fv(u.uShadow, L);
    gl.enable(gl.STENCIL_TEST);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    for (const shadow of [false, true]) {
      gl.stencilFunc(shadow ? gl.NOTEQUAL : gl.EQUAL, 0, 0xff);
      gl.uniform1i(u.uInShadow, shadow ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.STENCIL_TEST);
    gl.enable(gl.DEPTH_TEST);
  }
}
