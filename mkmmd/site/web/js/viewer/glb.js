// A .glb (binary glTF 2.0) as what the viewer draws: geometries in world space (node transforms applied, +Y up) and
// parts (a triangle range of a geometry with its material). Primitives that share vertex accessors under one node
// share one geometry (mk model glb writes one vertex buffer and a primitive per material); triangles only.
import { mul, fromTRS, normalMatrix, det3, IDENTITY } from "./math.js";

const COMP = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const SIZE = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const MAX = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };
const GET = { 5120: "getInt8", 5121: "getUint8", 5122: "getInt16", 5123: "getUint16", 5125: "getUint32", 5126: "getFloat32" };

export function parseGLB(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("not a .glb file");
  const jsonLen = dv.getUint32(12, true);
  const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
  const binAt = 20 + jsonLen + 8, binLen = binAt - 8 < buf.byteLength ? dv.getUint32(binAt - 8, true) : 0;

  const read = (i, asIndex) => {
    const a = doc.accessors[i], bv = doc.bufferViews[a.bufferView], T = COMP[a.componentType], k = SIZE[a.type];
    const es = T.BYTES_PER_ELEMENT, stride = bv.byteStride || es * k, start = binAt + (bv.byteOffset || 0) + (a.byteOffset || 0);
    const n = a.count * k, out = asIndex ? new Uint32Array(n) : new Float32Array(n);
    const scale = !asIndex && a.normalized && MAX[a.componentType] ? 1 / MAX[a.componentType] : 1;
    if (stride === es * k && start % es === 0) {
      const src = new T(buf, start, n);
      if (scale === 1) out.set(src); else for (let j = 0; j < n; j++) out[j] = src[j] * scale;
    } else {
      const get = GET[a.componentType];
      for (let v = 0; v < a.count; v++) for (let c = 0; c < k; c++) out[v * k + c] = dv[get](start + v * stride + c * es, true) * scale;
    }
    return out;
  };

  const images = (doc.images || []).map((im) => {
    if (im.bufferView === undefined) return null;
    const bv = doc.bufferViews[im.bufferView];
    return new Blob([new Uint8Array(buf, binAt + (bv.byteOffset || 0), bv.byteLength)], { type: im.mimeType || "image/png" });
  });
  const materials = (doc.materials || []).map((m, i) => {
    const pbr = m.pbrMetallicRoughness || {};
    const tex = pbr.baseColorTexture ? doc.textures[pbr.baseColorTexture.index] : null;
    return {
      name: m.name || `material ${i + 1}`, color: pbr.baseColorFactor || [1, 1, 1, 1],
      image: tex && tex.source !== undefined ? images[tex.source] : null,
      blend: m.alphaMode === "BLEND", cutoff: m.alphaMode === "MASK" ? (m.alphaCutoff ?? 0.5) : 0, double: !!m.doubleSided,
    };
  });

  const geos = [], parts = [], groups = new Map();
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const walk = (ni, parent) => {
    const node = doc.nodes[ni];
    const local = node.matrix ? new Float32Array(node.matrix) : fromTRS(node.translation, node.rotation, node.scale);
    const world = parent ? mul(parent, local) : local;
    if (node.mesh !== undefined) {
      const mesh = doc.meshes[node.mesh], flip = det3(world) < 0;
      mesh.primitives.forEach((prim, pi) => {
        if ((prim.mode ?? 4) !== 4) return;
        const A = prim.attributes, key = `${ni}:${A.POSITION}:${A.NORMAL}:${A.TEXCOORD_0}`;
        let g = groups.get(key);
        if (!g) {
          const P = read(A.POSITION), count = P.length / 3, nm = normalMatrix(world);
          const positions = new Float32Array(P.length);
          for (let v = 0; v < count; v++) {
            const x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
            positions[v * 3] = world[0] * x + world[4] * y + world[8] * z + world[12];
            positions[v * 3 + 1] = world[1] * x + world[5] * y + world[9] * z + world[13];
            positions[v * 3 + 2] = world[2] * x + world[6] * y + world[10] * z + world[14];
            for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], positions[v * 3 + c]); hi[c] = Math.max(hi[c], positions[v * 3 + c]); }
          }
          let normals = null;
          if (A.NORMAL !== undefined) {
            const N = read(A.NORMAL);
            normals = new Float32Array(N.length);
            for (let v = 0; v < count; v++) {
              const x = N[v * 3], y = N[v * 3 + 1], z = N[v * 3 + 2];
              const nx = nm[0] * x + nm[1] * y + nm[2] * z, ny = nm[3] * x + nm[4] * y + nm[5] * z, nz = nm[6] * x + nm[7] * y + nm[8] * z;
              const l = Math.hypot(nx, ny, nz) || 1;
              normals[v * 3] = nx / l; normals[v * 3 + 1] = ny / l; normals[v * 3 + 2] = nz / l;
            }
          }
          g = { positions, normals, uvs: A.TEXCOORD_0 !== undefined ? read(A.TEXCOORD_0) : null, lists: [], count };
          g.index = geos.length;
          geos.push(g);
          groups.set(key, g);
        }
        let idx = prim.indices !== undefined ? read(prim.indices, true) : Uint32Array.from({ length: g.count }, (_, k) => k);
        if (flip) for (let t = 0; t < idx.length; t += 3) { const s = idx[t + 1]; idx[t + 1] = idx[t + 2]; idx[t + 2] = s; }
        g.lists.push(idx);
        const mat = materials[prim.material] || { name: mesh.name || `part ${parts.length + 1}`, color: [0.8, 0.8, 0.8, 1], image: null, blend: false, cutoff: 0, double: false };
        parts.push({ geo: g.index, list: g.lists.length - 1, name: mat.name, material: mat, node: node.name || mesh.name || "", prim: pi });
      });
    }
    (node.children || []).forEach((c) => walk(c, world));
  };
  const scene = doc.scenes ? doc.scenes[doc.scene || 0] : { nodes: doc.nodes.map((_, i) => i) };
  scene.nodes.forEach((ni) => walk(ni, null));
  if (!parts.length) throw new Error("no triangles in it");

  for (const g of geos) {                                   // one index buffer per geometry; parts are ranges of it
    const total = g.lists.reduce((s, l) => s + l.length, 0);
    g.indices = new Uint32Array(total);
    g.starts = [];
    let at = 0;
    for (const l of g.lists) { g.starts.push(at); g.indices.set(l, at); at += l.length; }
    if (!g.normals) g.normals = vertexNormals(g.positions, g.indices);
    delete g.lists;
  }
  for (const p of parts) {
    const g = geos[p.geo];
    p.start = g.starts[p.list];
    p.count = (p.list + 1 < g.starts.length ? g.starts[p.list + 1] : g.indices.length) - p.start;
  }
  return { geos, parts, bounds: { min: lo, max: hi }, extras: (doc.nodes[scene.nodes[0]] || {}).extras || {},
    triangles: geos.reduce((s, g) => s + g.indices.length / 3, 0), vertices: geos.reduce((s, g) => s + g.count, 0) };
}

export function vertexNormals(P, I) {
  const N = new Float32Array(P.length);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const k of [a, b, c]) { N[k] += nx; N[k + 1] += ny; N[k + 2] += nz; }
  }
  for (let v = 0; v < N.length; v += 3) {
    const l = Math.hypot(N[v], N[v + 1], N[v + 2]) || 1;
    N[v] /= l; N[v + 1] /= l; N[v + 2] /= l;
  }
  return N;
}

// Vertices at the same place, as one id each: a model's vertices are split along UV and normal seams, which are not
// holes in its surface.
export function weld(P, count, step = 1e-5) {
  const ids = new Uint32Array(count), at = new Map();
  for (let v = 0; v < count; v++) {
    const key = `${Math.round(P[v * 3] / step)},${Math.round(P[v * 3 + 1] / step)},${Math.round(P[v * 3 + 2] / step)}`;
    let id = at.get(key);
    if (id === undefined) { id = v; at.set(key, v); }
    ids[v] = id;
  }
  return ids;
}

// The edges of a geometry's triangles, as index pairs: every edge once (`all`), the open ones (`open`: a single
// triangle has them, a hole or a sheet's border) and the ones whose two triangles run them the same way (`flipped`:
// one of the two faces the wrong way). Edges are told apart where the vertices are, not by vertex id.
export function edges(g) {
  const I = g.indices, n = g.count, W = weld(g.positions, n), seen = new Map();
  for (let t = 0; t < I.length; t += 3) for (let k = 0; k < 3; k++) {
    const a = W[I[t + k]], b = W[I[t + (k + 1) % 3]];
    if (a === b) continue;
    const key = a < b ? a * n + b : b * n + a, dir = a < b ? 1 : -1, e = seen.get(key);
    if (e === undefined) seen.set(key, dir);                    // its first triangle's direction
    else if (e === 1 || e === -1) seen.set(key, e === dir ? 3 : 2);   // 2: shared and consistent; 3: flipped
  }
  const all = new Uint32Array(seen.size * 2), open = [], flipped = [];
  let i = 0;
  for (const [key, e] of seen) {
    const a = Math.floor(key / n), b = key - a * n;
    all[i++] = a; all[i++] = b;
    if (e === 1 || e === -1) open.push(a, b);
    else if (e === 3) flipped.push(a, b);
  }
  return { all, open: Uint32Array.from(open), flipped: Uint32Array.from(flipped) };
}
