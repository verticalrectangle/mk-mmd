// The server's API as the exported page has it (`mk export web` writes this file as js/api.js): the routes it needs are
// JSON files under data/ (scene, music, draft_<output>), and bytes() undoes the export's packing: a gzip file (scene.glb)
// is gunzipped, an MKZ1 file (bake.bin) is rebuilt into the float32 array mk build wrote (mkmmd/site/export.py: pack).

export async function api(method, name, { query = {} } = {}) {
  if (method !== "GET") throw new Error("this page is a static export: it only reads");
  const file = name === "draft" ? `draft_${query.output}` : name;
  const res = await fetch(`data/${encodeURIComponent(file)}.json`);
  if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`);
  return res.json();
}

export const fileUrl = (path) => path;

export async function bytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  const buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, Math.min(4, buf.byteLength));
  if (head[0] === 0x1f && head[1] === 0x8b) return gunzip(buf);            // a server that sent it encoded already gave plain bytes
  if (String.fromCharCode(...head) === "MKZ1") return unpack(buf);
  return buf;
}

async function gunzip(buf) {
  if (typeof DecompressionStream === "undefined") throw new Error("this browser cannot unpack gzip (no DecompressionStream)");
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

// MKZ1: the magic, a uint32 header length, the header (JSON: `floats`, the array's length, and `streams`, each a block of
// `rows` x `cols` floats at float offset `at`), then the gzipped payload, each stream's bytes in turn. A "raw" stream is
// float32. A "q" stream is int32 steps (per column `step`), zigzagged and split into four byte planes; with `delta` a row
// is stored as its change from the row above, with `ref` as its change from the steps of stream `ref` (same shape).
async function unpack(buf) {
  const view = new DataView(buf), n = view.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const body = new Uint8Array(await gunzip(buf.slice(8 + n)));
  const out = new Float32Array(head.floats), kept = new Map(), uses = new Map();
  for (const s of head.streams) if (s.ref !== undefined && s.ref !== null) uses.set(s.ref, (uses.get(s.ref) || 0) + 1);
  let pos = 0;
  head.streams.forEach((s, i) => {
    const count = s.rows * s.cols;
    if (s.mode === "raw") {
      out.set(new Float32Array(body.slice(pos, pos + 4 * count).buffer), s.at);
      pos += 4 * count;
      return;
    }
    const p0 = pos, p1 = pos + count, p2 = pos + 2 * count, p3 = pos + 3 * count;
    pos += 4 * count;
    const q = new Int32Array(count), ref = s.ref === undefined || s.ref === null ? null : kept.get(s.ref);
    const steps = Array.isArray(s.step) ? s.step : new Array(s.cols).fill(s.step);
    for (let k = 0; k < count; k++) {
      const u = (body[p0 + k] | (body[p1 + k] << 8) | (body[p2 + k] << 16) | (body[p3 + k] << 24)) >>> 0;
      let v = (u >>> 1) ^ -(u & 1);
      if (ref) v += ref[k];
      else if (s.delta && k >= s.cols) v += q[k - s.cols];
      q[k] = v;
      out[s.at + k] = v * steps[k % s.cols];
    }
    if (ref !== null && uses.has(s.ref) && uses.set(s.ref, uses.get(s.ref) - 1).get(s.ref) === 0) kept.delete(s.ref);
    if (uses.has(i)) kept.set(i, q);
  });
  return out.buffer;
}
