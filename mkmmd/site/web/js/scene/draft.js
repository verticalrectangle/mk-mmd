// mk post's draft of an output, played frame by frame to the scene's clock with WebCodecs (docs/design.md: The site):
// the samples go to a VideoDecoder in decode order, from the keyframe before the frame asked for when the clock jumps,
// a few past it (B-frames hold frames back until later ones arrive); decoded frames are kept near the playhead only.
const AHEAD = 6;                      // samples fed past the one asked for
const KEEP = 12;                      // decoded frames kept behind and ahead of it

const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export class DraftPlayer {
  // `meta` the server's index ({codec, description, width, height, samples}), `data` the file's bytes.
  constructor(meta, data) {
    this.meta = meta;
    this.data = new Uint8Array(data);
    const s = meta.samples;
    this.byPts = s.map((_, i) => i).sort((a, b) => s[a][2] - s[b][2]);     // presentation index -> decode index
    this.pts = this.byPts.map((i) => s[i][2]);
    this.frames = new Map();                                                // timestamp -> VideoFrame
    this.onFrame = null;
    this.error = null;
    this.open(0);
  }

  open(at) {
    if (this.dec && this.dec.state !== "closed") this.dec.close();
    for (const f of this.frames.values()) f.close();
    this.frames.clear();
    this.dec = new VideoDecoder({
      output: (f) => { this.frames.set(f.timestamp, f); if (this.onFrame) this.onFrame(); },
      error: (e) => { this.error = e; },
    });
    this.dec.configure({ codec: this.meta.codec, description: b64(this.meta.description), codedWidth: this.meta.width,
      codedHeight: this.meta.height, optimizeForLatency: true });
    this.next = at;
    this.flushed = false;
  }

  // The decoded frame to show for clip frame `k` (the latest one at or before it), feeding the decoder as needed.
  frame(k) {
    if (this.error) return null;
    k = Math.max(0, Math.min(this.pts.length - 1, k));
    const target = this.pts[k], di = this.byPts[k], s = this.meta.samples;
    const held = [...this.frames.keys()];
    const behind = held.length && target < Math.min(...held) && !this.frames.has(target);
    if (behind || di + AHEAD < this.next - 2 * AHEAD - 1 || di > this.next + 60) {
      let kf = di;
      while (kf > 0 && !s[kf][3]) kf--;
      this.open(kf);
    }
    while (this.next < s.length && this.next <= di + AHEAD && this.dec.decodeQueueSize < 16) {
      const [off, size, ts, key] = s[this.next++];
      this.dec.decode(new EncodedVideoChunk({ type: key ? "key" : "delta", timestamp: ts, data: this.data.subarray(off, off + size) }));
    }
    if (this.next >= s.length && !this.flushed) { this.flushed = true; this.dec.flush().catch(() => {}); }
    for (const [ts, f] of this.frames) {
      const pi = this.pts.indexOf(ts);
      if (pi < k - KEEP || pi > k + KEEP) { f.close(); this.frames.delete(ts); }
    }
    let best = null;
    for (const [ts, f] of this.frames) if (ts <= target && (!best || ts > best.timestamp)) best = f;
    return best;
  }

  close() {
    if (this.dec && this.dec.state !== "closed") this.dec.close();
    for (const f of this.frames.values()) f.close();
    this.frames.clear();
  }
}
