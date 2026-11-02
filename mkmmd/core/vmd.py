"""VMD (MMD motion) files: reader and motion analysis (tempo, beat phase, the fit to a song's beats, energy, travel).
numpy only.

Layout (little endian): 30-byte signature, 20-byte model name (cp932), then sections, each a u32 count followed by
records: bone keys (15-byte name, u32 frame, 3 f32 position, 4 f32 quaternion xyzw, 64 bytes interpolation),
morph keys (15-byte name, u32 frame, f32 weight), camera keys (u32 frame, f32 distance, 3 f32 position,
3 f32 rotation, 24 bytes interpolation, u32 view angle, u8 orthographic), then light, self-shadow and IK sections.
MMD units: 1 unit = 8 cm; y is up; 30 frames per second."""
import struct

import numpy as np

MMD_FPS = 30.0
MMD_UNIT_M = 0.08
LIMBS = ("腕", "ひじ", "手首", "頭", "首", "足", "ひざ")


def _name(raw):
    raw = raw.split(b"\0")[0]
    for enc in ("cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


class Motion:
    """bones: {name: (frames u32[n], pos f32[n,3], quat f32[n,4] xyzw)} sorted by frame; morphs: {name: (frames,
    weights)}; camera: dict of arrays or None."""

    def __init__(self, model, bones, morphs, camera):
        self.model = model
        self.bones = bones
        self.morphs = morphs
        self.camera = camera

    @property
    def n_frames(self):
        last = 0
        for fr, *_ in self.bones.values():
            last = max(last, int(fr[-1]))
        for fr, _ in self.morphs.values():
            last = max(last, int(fr[-1]))
        if self.camera is not None and len(self.camera["frame"]):
            last = max(last, int(self.camera["frame"][-1]))
        return last + 1

    @property
    def duration(self):
        return self.n_frames / MMD_FPS

    def summary(self):
        return {"model": self.model, "frames": self.n_frames, "duration_s": round(self.duration, 3),
                "bone_tracks": len(self.bones), "bone_keys": int(sum(len(v[0]) for v in self.bones.values())),
                "morph_tracks": len(self.morphs), "morph_keys": int(sum(len(v[0]) for v in self.morphs.values())),
                "camera_keys": 0 if self.camera is None else int(len(self.camera["frame"]))}


def read(path):
    with open(path, "rb") as fh:
        data = fh.read()
    sig = data[:30].split(b"\0")[0]
    if not sig.startswith(b"Vocaloid Motion Data"):
        raise ValueError(f"{path}: not a VMD file")
    model = _name(data[30:50])
    off = 50
    bones, morphs, camera = {}, {}, None

    def count():
        nonlocal off
        if off + 4 > len(data):
            return 0
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        return n

    n = count()
    raw = {}
    for _ in range(n):
        name = _name(data[off:off + 15])
        rec = struct.unpack_from("<I7f", data, off + 15)
        raw.setdefault(name, []).append(rec)
        off += 111
    for name, recs in raw.items():
        recs.sort(key=lambda r: r[0])
        arr = np.array(recs, dtype=np.float64)
        bones[name] = (arr[:, 0].astype(np.int64), arr[:, 1:4].astype(np.float32), arr[:, 4:8].astype(np.float32))
    n = count()
    raw = {}
    for _ in range(n):
        name = _name(data[off:off + 15])
        fr, w = struct.unpack_from("<If", data, off + 15)
        raw.setdefault(name, []).append((fr, w))
        off += 23
    for name, recs in raw.items():
        recs.sort()
        arr = np.array(recs, dtype=np.float64)
        morphs[name] = (arr[:, 0].astype(np.int64), arr[:, 1].astype(np.float32))
    n = count()
    if n:
        recs = []
        for _ in range(n):
            fr, dist, px, py, pz, rx, ry, rz = struct.unpack_from("<I7f", data, off)
            fov, ortho = struct.unpack_from("<IB", data, off + 56)
            recs.append((fr, dist, px, py, pz, rx, ry, rz, fov, ortho))
            off += 61
        recs.sort()
        arr = np.array(recs, dtype=np.float64)
        camera = {"frame": arr[:, 0].astype(np.int64), "distance": arr[:, 1], "position": arr[:, 2:5],
                  "rotation": arr[:, 5:8], "fov": arr[:, 8], "ortho": arr[:, 9].astype(bool)}
    return Motion(model, bones, morphs, camera)


SIDES = str.maketrans({"左": "右", "右": "左"})


def mirror(data):
    """The bytes of a VMD (`data`) mirrored left for right, as MMD mirrors a motion: every bone key's name with 左 and
    右 traded (the other side's bone, the same length in cp932), its position's x negated and its rotation reflected
    through the body's middle plane, (x, y, z, w) -> (x, -y, -z, w) (VMD rotations are in the world's axes, so a
    twist bone's turn about its own axis becomes the same turn about the other side's). The facial keys and the
    sections after them are kept as they are."""
    if not data[:30].split(b"\0")[0].startswith(b"Vocaloid Motion Data"):
        raise ValueError("not a VMD file")
    out = bytearray(data)
    (n,) = struct.unpack_from("<I", data, 50)
    off = 54
    for _ in range(n):
        raw = data[off:off + 15].split(b"\0")[0]
        try:
            swapped = raw.decode("cp932").translate(SIDES).encode("cp932")
        except UnicodeError:
            swapped = raw
        if len(swapped) == len(raw):
            out[off:off + len(raw)] = swapped
        fr, px, py, pz, qx, qy, qz, qw = struct.unpack_from("<I7f", data, off + 15)
        struct.pack_into("<I7f", out, off + 15, fr, -px, py, pz, qx, -qy, -qz, qw)
        off += 111
    return bytes(out)


# ---------------------------------------------------------------- analysis
def sample_position(track, nf):
    fr, pos, _ = track
    return np.stack([np.interp(np.arange(nf), fr, pos[:, i]) for i in range(3)], 1)


def angular_speed(track, nf):
    """Per-frame rotation angle (rad) of a bone track, with linear quaternion interpolation between keys."""
    fr, _, q = track
    qs = np.stack([np.interp(np.arange(nf), fr, q[:, i]) for i in range(4)], 1)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True) + 1e-9
    d = np.abs(np.sum(qs[1:] * qs[:-1], axis=1)).clip(0, 1)
    return np.concatenate([[0.0], 2 * np.arccos(d)])


def envelope(motion):
    """Onset envelope: centre-bone landings (upward acceleration) + limb angular acceleration; and limb energy."""
    nf = motion.n_frames
    env = np.zeros(nf)
    for b in ("センター", "グルーブ"):
        tr = motion.bones.get(b)
        if tr is not None and len(tr[0]) > 4:
            vel = np.gradient(sample_position(tr, nf)[:, 1])
            acc = np.maximum(0, np.gradient(vel))
            env += acc / (acc.max() + 1e-9)
    energy = np.zeros(nf)
    for name, tr in motion.bones.items():
        if len(tr[0]) > 8 and any(s in name for s in LIMBS):
            energy += angular_speed(tr, nf)
    ea = np.maximum(0, np.gradient(np.convolve(energy, np.ones(3) / 3, "same")))
    env += 0.5 * ea / (ea.max() + 1e-9)
    return env, energy


def tempo_phase(motion, lo_bpm=70.0, hi_bpm=180.0, step=0.25, segment=None):
    """Comb-filter tempo over [lo_bpm, hi_bpm]: (period_frames, phase_frame, score, contrast). Contrast (best score
    over the median score) above ~1.4 means a clear beat."""
    env, _ = envelope(motion)
    if len(env) < 40:
        return None
    a, b = segment if segment else (min(100, len(env) // 10), len(env) - min(100, len(env) // 10))
    best = (0.0, 0.0, 0.0)
    scores = []
    for bpm in np.arange(lo_bpm, hi_bpm + 1e-9, step):
        period = MMD_FPS * 60.0 / bpm
        top = (0.0, 0.0)
        for ph in np.arange(0, period, 0.5):
            idx = np.round(np.arange(a + ph, b, period)).astype(int)
            if len(idx) < 4:
                continue
            s = env[idx].mean()
            if s > top[0]:
                top = (s, ph)
        scores.append(top[0])
        if top[0] > best[0]:
            best = (top[0], period, (a + top[1]) % period)
    if best[0] <= 0.0:                                  # flat envelope: camera-only or motionless
        return None
    contrast = best[0] / (np.median(scores) + 1e-9)
    return best[1], best[2], best[0], contrast


def beat_fit(motion, beats, fps, start_t, src_from, bpm=None, per_beat=1.0):
    """How to play `motion` so that its beats land on a song's: (scale, start, info). `beats` are the song's beats in
    clip seconds and `fps` the scene's frame rate. The motion's beat is `bpm` when given (the tempo of the song it was
    made for: tempo_phase searches 70-180 bpm and can settle on half or twice the beat, or on a pattern of the moves
    that is not the beat at all), else tempo_phase's; its phase is tempo_phase's at that period. `per_beat` motion
    beats play to each song beat (2: on the 8ths, 0.5: one every other beat): the strip's scale stretches per_beat
    motion beats over the song's median beat. The first motion beat at or after source frame `src_from` lands on the
    song's beats (each cut into round(per_beat) parts when that is 2 or more) nearest to where it would fall with
    `src_from` played at clip second `start_t`. `start` is the strip's first frame, in scene frames from clip 0.
    Raises ValueError when the motion has no beat or the song fewer than two beats."""
    tp = tempo_phase(motion) if bpm is None else tempo_phase(motion, lo_bpm=bpm, hi_bpm=bpm)
    if tp is None:
        raise ValueError("no beat in the motion")
    b = np.asarray(beats, float)
    if len(b) < 2:
        raise ValueError("no beats in the timeline")
    period_v, phase_v = tp[0], tp[1]
    song_s = float(np.median(np.diff(b)))
    scale = song_s * fps / per_beat / period_v
    n = max(1, round(per_beat))
    grid = b if n == 1 else np.append((b[:-1, None] + np.diff(b)[:, None] * (np.arange(n) / n)).ravel(), b[-1])
    src_beat = phase_v + np.ceil((src_from - phase_v) / period_v) * period_v   # first motion beat at or after `from`
    want = start_t * fps + (src_beat - src_from) * scale   # where it would land unsnapped (frames from clip 0)
    land = grid[int(np.argmin(np.abs(grid * fps - want)))] * fps
    info = {"motion_bpm": round(MMD_FPS * 60.0 / period_v, 2), "song_bpm": round(60.0 / song_s, 2),
            "scale": round(scale, 4)}
    if per_beat != 1:
        info["per_beat"] = per_beat
    return scale, land - (src_beat - src_from) * scale, info


def travel(motion):
    """How far the centre bone wanders (m): xy extent, path length and vertical bounce std."""
    tr = motion.bones.get("センター")
    if tr is None or len(tr[0]) < 2:
        return {"xy_extent_m": 0.0, "path_m": 0.0, "bounce_std_m": 0.0, "in_place": True}
    p = sample_position(tr, motion.n_frames) * MMD_UNIT_M
    xz = p[:, [0, 2]]
    extent = float(np.max(np.ptp(xz, axis=0)))
    path = float(np.sum(np.linalg.norm(np.diff(xz, axis=0), axis=1)))
    return {"xy_extent_m": round(extent, 3), "path_m": round(path, 3), "bounce_std_m": round(float(p[:, 1].std()), 4),
            "in_place": extent < 0.6}


def analyse(motion):
    out = motion.summary()
    tp = tempo_phase(motion)
    if tp:
        period, phase, score, contrast = tp
        out["tempo"] = {"bpm": round(MMD_FPS * 60.0 / period, 2), "period_frames": round(period, 3),
                        "phase_frame": round(phase, 2), "score": round(float(score), 4),
                        "contrast": round(float(contrast), 3), "clear": bool(contrast > 1.4)}
    _, energy = envelope(motion)
    out["energy"] = {"mean_rad_per_frame": round(float(energy.mean()), 4),
                     "p90_rad_per_frame": round(float(np.percentile(energy, 90)), 4)} if len(energy) else None
    out["travel"] = travel(motion)
    out["bones_used"] = sorted(motion.bones)
    out["morphs_used"] = sorted(motion.morphs)
    return out
