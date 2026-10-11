// mk's own renderer, the page (docs/design.md: Rendering): headless Chromium opens it for `mk render` when [render]
// engine = "mk". It asks mk for its job (the bake's files, the frames, the looks), draws each frame's passes on the GPU
// (passes.js) and sends them back, one frame at a time (mk answers once it has room for the next), and ends by saying
// it is done, or why it stopped.
import { api, bytes, fileUrl } from "../api.js";
import { PassGL } from "./passes.js";

async function main() {
  const job = await api("GET", "render/job");
  const [glb, bin, doc] = await Promise.all([bytes(fileUrl(job.files.glb)), bytes(fileUrl(job.files.bin)),
    bytes(fileUrl(job.files.json))]);
  const bake = JSON.parse(new TextDecoder().decode(doc));
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: "high-performance" });
  if (!gl) throw new Error("this browser has no WebGL2");
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  const R = new PassGL(gl, glb, bake, new Float32Array(bin));
  await R.ready;
  const looks = job.looks.map((l) => ({ ...l, draw: new Set(l.draw) }));
  const t0 = performance.now();
  let gpu = 0;
  for (let i = 0; i < job.items.length; i++) {
    const item = job.items[i], t = performance.now();
    const data = R.render(item, looks[item.look]);
    const err = gl.getError();
    if (err) throw new Error(`WebGL error ${err} drawing ${item.output} frame ${item.frame}`);
    gpu += performance.now() - t;
    const res = await fetch(`api/render/frame?i=${i}`, { method: "POST", body: data });
    const answer = await res.json().catch(() => ({}));        // read to the end: an unread answer keeps the frame's bytes
    if (!res.ok) throw new Error(`mk did not take ${item.output} frame ${item.frame}: ${answer.error || res.status}`);
  }
  await api("POST", "render/done", { body: { frames: job.items.length, renderer,
    seconds: (performance.now() - t0) / 1000, gpu_seconds: gpu / 1000 } });
}

main().catch((e) => api("POST", "render/done", { body: { error: String((e && e.stack) || e) } }).catch(() => {}));
