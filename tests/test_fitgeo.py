"""mkmmd.core.fitgeo on synthetic words: letters (and outline copies of them) moved rigidly, per letter, tilted in
perspective and squashed like jelly, written to a blob and decoded back as mkr decodes them. `make_fixture` writes the
small bake mkr's screen_geo.rs test decodes (mkr/mkr/tests/fixtures/screen_geo; `python tests/test_fitgeo.py` rewrites
it); its test here decodes that fixture with fitgeo's own decoder."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mkmmd.core import fitgeo as FG

FIXTURE = Path(__file__).resolve().parents[1] / "mkr" / "mkr" / "tests" / "fixtures" / "screen_geo"


# ================================================================================================= synthetic words
def _square():
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    return np.array(pts, float), np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])


def _ell():
    pts = [(0, 0), (1, 0), (1, 0.3), (0.3, 0.3), (0.3, 1), (0, 1)]
    return np.array(pts, float), np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]])


def _ring(n=8):
    a = np.arange(n) * 2 * math.pi / n
    pts = np.concatenate([np.c_[np.cos(a), np.sin(a)], 0.5 * np.c_[np.cos(a), np.sin(a)]]) * 0.5 + 0.5
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[i, j, n + i], [j, n + j, n + i]]
    return pts, np.array(tris)


SHAPES = [_square(), _ell(), _ring()]


class Word:
    """A synthetic word: letters (SHAPES) in a row in the plane x/z of glTF's axes at depth y, each with `copies`
    outline copies around it (slot 1, a little behind) and itself filled (slot 0), as a frame's drawn arrays."""

    def __init__(self, letters=(0, 1, 2, 0), copies=4, em=0.04, gap=0.012, x0=-0.1, z0=0.05):
        self.letters, self.copies, self.em = list(letters), copies, em
        self.pos, self.tris, self.slot, self.letter, self.copy, uv = [], [], [], [], [], []
        self.offsets = [(0.0, 0.0)] + [(0.003 * math.cos(2 * math.pi * c / copies), 0.003 * math.sin(2 * math.pi * c / copies))
                                       for c in range(copies)]
        nv = 0
        for slot, offs in ((0, self.offsets[:1]), (1, self.offsets[1:])):
            for li, s in enumerate(self.letters):
                pts, tris = SHAPES[s]
                for ci, (ox, oz) in enumerate(offs):
                    p = np.c_[x0 + li * (em + gap) + pts[:, 0] * em + ox, np.full(len(pts), 0.002 if slot == 0 else 0.0016),
                              z0 + pts[:, 1] * em + oz]
                    self.pos.append(p)
                    self.tris.append(tris + nv)
                    self.slot += [slot] * len(tris)
                    self.letter.append(np.full(len(pts), li))
                    self.copy.append(np.full(len(pts), ci + (0 if slot == 0 else 1)))
                    uv.append(pts)                                  # the shape's own: alike on every copy
                    nv += len(pts)
        self.pos = np.concatenate(self.pos)
        self.tris = np.concatenate(self.tris)
        self.letter = np.concatenate(self.letter)
        self.copy = np.concatenate(self.copy)
        self.slot = np.array(self.slot)
        self.slots = []
        for s in (0, 1):
            idx = np.flatnonzero(self.slot == s)
            self.slots.append([s, int(idx[0]), len(idx)])
        self.src = np.arange(len(self.pos))
        self.uv = np.concatenate(uv).astype(np.float32)

    def normals(self, P):
        """Each vertex's face normal (a flat letter's: its plane's, signed by the winding), glTF axes."""
        a, b, c = P[self.tris[:, 0]], P[self.tris[:, 1]], P[self.tris[:, 2]]
        n = np.cross(b - a, c - a)
        out = np.zeros_like(P)
        np.add.at(out, self.tris[:, 0], n)
        np.add.at(out, self.tris[:, 1], n)
        np.add.at(out, self.tris[:, 2], n)
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)

    def frame(self, P, alpha=None, uv=None):
        """(P, N, attrs) of a frame, positions P (v, 3)."""
        P = np.asarray(P, np.float64)
        if alpha is None:
            alpha = np.ones(len(self.letters))
        attrs = {"attr:mk_alpha": np.asarray(alpha, np.float32)[self.letter][:, None],
                 "uv:": self.uv if uv is None else uv}
        return P.astype(np.float32), self.normals(P), attrs

    def topo(self):
        return self.tris.astype(np.uint32), [list(s) for s in self.slots], self.src


def _affine(P, A, t, about=(0.0, 0.0, 0.0)):
    about = np.asarray(about)
    return (P - about) @ np.asarray(A, float).T + about + np.asarray(t, float)


def _turn(deg, sx=1.0, sz=1.0):
    """A turn about glTF's y (the screen plane's normal) after a squash in x and z."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.diag([sx, 1.0, sz])


def _tilted(P, yaw, pitch, dist, pivot):
    """text.py's `_tilted`: the word about its pivot turned in depth and seen in perspective from `dist`, its own depth
    kept (in the text's frame: x right, y up, z toward the viewer; here glTF x, -z, y)."""
    p = P - pivot
    x, y, z = p[:, 0], -p[:, 2], p[:, 1]
    ry, rx = math.radians(yaw), math.radians(-pitch)
    # Euler XYZ: rotate about x (pitch), then about y (yaw)
    y1, z1 = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
    x2, z2 = x * math.cos(ry) + z1 * math.sin(ry), -x * math.sin(ry) + z1 * math.cos(ry)
    k = dist / np.maximum(dist - z2, 0.1 * dist)
    out = np.c_[x2 * k, z, -(y1 * k)] + pivot
    return out


class Blob:
    def __init__(self):
        self.parts, self.size = [], 0

    def put(self, a, dt):
        a = np.ascontiguousarray(a, FG.DT[dt])
        pad = -self.size % 16
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        at = self.size
        self.parts.append(a.tobytes())
        self.size += a.nbytes
        return at

    def bytes(self):
        return np.frombuffer(b"".join(self.parts), np.uint8)


def _encode(word, frames, n=None):
    """Encode frames [(P, alpha, uv, None or the Word of a new topology)] of a word, None hidden: (Word encoder, lit,
    record, what each frame became)."""
    blob = Blob()
    w = FG.Word(n or len(frames), blob.put)
    kinds, cur = [], word
    for k, f in enumerate(frames):
        if f is None:
            kinds.append(None)
            continue
        cur = f[3] or cur
        topo = cur.topo() if (k == 0 or f[3] is not None) else None
        kinds.append(w.add(k, *cur.frame(*f[:3]), topo=topo))
    return w, blob.bytes(), w.record(0), kinds


def _check(word, frames, lit, rec, tol=FG.TOL):
    """Every frame decodes to its input: positions within tol, normals, attributes and triangles. Answers the worst
    position error."""
    worst, cur = 0.0, word
    for k, f in enumerate(frames):
        out = FG.decode(lit, rec, k)
        if f is None:
            assert out is None
            continue
        cur = f[3] or cur
        P, N, attrs = cur.frame(*f[:3])
        g = rec["geoms"][rec["frames"][k]]
        L = FG.Layout(*cur.topo())
        assert g["vertices"] == len(L.src) and len(out["position"]) == len(L.src)
        err = np.linalg.norm(out["position"].astype(float) - P[L.src].astype(float), axis=1).max()
        worst = max(worst, err)
        assert err <= tol, (k, err)
        cos = np.einsum("vi,vi->v", out["normal"].astype(float), N[L.src].astype(float))
        assert cos.min() > math.cos(FG.NORMAL_TOL), (k, cos.min())
        for key, a in attrs.items():
            assert np.array_equal(out["attrs"][key], np.asarray(a, np.float32).reshape(len(P), -1)[L.src]), (k, key)
        # the triangles are the input's, slot by slot, through the materialised vertices
        got = {(s, tuple(sorted(t))) for s, a, c in out["slots"] for t in L.src[out["tris"][a:a + c].astype(int)]}
        want = {(s, tuple(sorted(t))) for s, a, c in cur.slots for t in cur.tris[a:a + c]}
        assert got == want
    return worst


# ================================================================================================= tests
def test_uv_that_differs_between_copies_keeps_their_prototypes_apart():
    w = Word(letters=(0, 0), copies=2)
    seg = FG.Segment(*w.frame(w.pos), *w.topo())
    assert len(seg.pv0) - 1 == 1                                            # the shape's own uv: one prototype
    placed = (w.pos[:, [0, 2]] * 3.0).astype(np.float32)                    # uv from where each copy stands
    seg = FG.Segment(*w.frame(w.pos, uv=placed), *w.topo())
    assert len(seg.pv0) - 1 == seg.layout.count


def test_a_rigid_word_is_one_map_a_frame():
    w = Word()
    frames = [(w.pos, None, None, None)]
    for t in range(1, 6):
        A = _turn(7.0 * t) * (1.0 + 0.05 * t)
        frames.append((_affine(w.pos, A, (0.01 * t, 0.0, -0.02 * t), about=(0.0, 0.002, 0.07)), None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds == ["geom"] + ["word"] * 5
    assert enc.stats["geoms"] == 1 and enc.stats["refit"] == 0
    assert _check(w, frames, lit, rec) < 1e-6


def test_letters_moving_apart_are_a_map_per_island():
    w = Word()
    frames = [(w.pos, None, None, None)]
    rng = np.random.default_rng(1)
    for t in range(1, 5):
        P = w.pos.copy()
        for li in range(len(w.letters)):
            sel = w.letter == li
            c = P[sel].mean(0)
            A = _turn(rng.uniform(-20, 20), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2))
            P[sel] = _affine(P[sel], A, rng.uniform(-0.01, 0.01, 3) * (1, 0, 1), about=c)
        frames.append((P, None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds == ["geom"] + ["islands"] * 4
    assert enc.stats["projective"] == 0
    assert _check(w, frames, lit, rec) < 1e-6


def test_a_word_tilted_in_perspective_is_projective_and_its_depth_kept():
    w = Word()
    pivot = w.pos.mean(0)
    frames = [(w.pos, None, None, None)]
    for t in range(1, 5):
        frames.append((_tilted(w.pos, 12.0 * t, -6.0 * t, 8 * w.em, pivot), None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds == ["geom"] + ["word"] * 4
    assert enc.stats["projective"] == 4
    assert _check(w, frames, lit, rec) < 2e-6
    # its letters tilted a little on their own: a bent map per island holds that within TOL
    frames = [(w.pos, None, None, None)]
    for t in range(1, 4):
        P = w.pos.copy()
        for li in range(len(w.letters)):
            sel = w.letter == li
            P[sel] = _tilted(P[sel], 15.0 * t * (1 if li % 2 else -1), 5.0 * t, 8 * w.em, P[sel].mean(0))
        frames.append((P, None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds[0] == "geom" and set(kinds[1:]) <= {"bent", "islands"} and enc.stats["refit"] == 0
    _check(w, frames, lit, rec)
    # tilted hard and seen from close by: a projective map per island
    frames = [(w.pos, None, None, None)]
    for t in range(1, 3):
        P = w.pos.copy()
        for li in range(len(w.letters)):
            sel = w.letter == li
            P[sel] = _tilted(P[sel], 35.0 * t * (1 if li % 2 else -1), 10.0 * t, 1.5 * w.em, P[sel].mean(0))
        frames.append((P, None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds[0] == "geom" and set(kinds[1:]) <= {"bent", "islands"} and enc.stats["refit"] == 0, kinds
    assert enc.stats["projective"] > 0
    _check(w, frames, lit, rec)


def _bend(P, R, pivot):
    """textring.py's bend: a point x along the word and y up from its middle goes to the angle x / R round a circle of
    radius R + y about a centre R below the pivot (glTF: x right, -z up)."""
    x, y = P[:, 0] - pivot[0], -(P[:, 2] - pivot[2])
    a, r = x / R, R + y
    return np.c_[pivot[0] + r * np.sin(a), P[:, 1], pivot[2] - (r * np.cos(a) - R)]


def test_letters_turning_before_the_word_bends_round_a_ring_are_bent_maps():
    w = Word()
    pivot = w.pos.mean(0)
    frames = []
    for t in range(5):
        P = w.pos.copy()
        for li in range(len(w.letters)):
            sel = w.letter == li
            P[sel] = _affine(P[sel], _turn(3.0 * t * (1 if li % 2 else -1)), (0.0, 0.0, 0.001 * t), P[sel].mean(0))
        frames.append((_bend(P, 0.3, pivot), None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds == ["geom"] + ["bent"] * 4, kinds
    assert _check(w, frames, lit, rec) <= FG.TOL


def test_jelly_is_no_map_and_starts_a_geometry_a_frame():
    w = Word()
    frames = [(w.pos, None, None, None)]
    for t in range(1, 4):
        P = w.pos.copy()
        P[:, 0] += 0.002 * np.sin(40.0 * P[:, 2] + t)                     # the strokes wave across each letter
        frames.append((P, None, None, None))
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds == ["geom"] * 4
    assert enc.stats["refit"] == 3
    assert _check(w, frames, lit, rec) <= FG.TOL                            # copies may still share a shape within TOL


def test_island_attributes_ride_the_poses_and_vertex_ones_must_hold():
    w = Word()
    frames = [(w.pos, None, None, None)]
    alpha = np.ones(len(w.letters))
    for t in range(1, 4):
        alpha = alpha * 0.8
        A = _turn(3.0 * t)
        frames.append((_affine(w.pos, A, (0.0, 0.0, 0.0)), alpha.copy(), None, None))
    frames.append((frames[-1][0], frames[-1][1], w.uv + 0.5, None))         # the uv changed: a new geometry
    enc, lit, rec, kinds = _encode(w, frames)
    assert kinds == ["geom", "word", "word", "word", "geom"]
    assert _check(w, frames, lit, rec) < 1e-6


def test_the_typewriter_starts_a_geometry_and_hidden_frames_decode_to_nothing():
    w3 = Word(letters=(0, 1, 2))
    w4 = Word(letters=(0, 1, 2, 0))
    frames = [(w3.pos, None, None, w3), None, (w4.pos, None, None, w4), (_affine(w4.pos, _turn(4.0), (0, 0, 0)), None, None, None)]
    enc, lit, rec, kinds = _encode(w4, frames)
    assert kinds == ["geom", None, "geom", "word"]
    assert rec["frames"] == [0, -1, 1, 1] and rec["poses"][1] == -1
    assert rec["geoms"][0]["vertices"] < rec["geoms"][1]["vertices"]
    _check(w3, frames, lit, rec)


def test_a_block_turning_in_3d_turns_its_normals():
    """A cube (flat-shaded: its corners split per face, joined by Blender's vertices) turning about a slanted axis."""
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 0.05
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    pos, tris, src = [], [], []
    for f in faces:
        base = len(pos)
        pos += [c[i] for i in f]
        src += f
        tris += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    pos, tris, src = np.array(pos), np.array(tris), np.array(src)

    def normals(P):
        a, b, cc = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        n = np.cross(b - a, cc - a)
        out = np.zeros_like(P)
        for j in range(3):
            np.add.at(out, tris[:, j], n)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    blob = Blob()
    enc = FG.Word(4, blob.put)
    frames = []
    axis = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
    for t in range(4):
        th = 0.4 * t
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        P = pos @ R.T + (0.1, 0.0, 0.1)
        frames.append(P)
        enc.add(t, P.astype(np.float32), normals(P).astype(np.float32), {},
                topo=(tris.astype(np.uint32), [[0, 0, len(tris)]], src) if t == 0 else None)
    assert enc.stats["geoms"] == 1 and enc.stats["word"] == 3
    lit, rec = blob.bytes(), enc.record(0)
    L = FG.Layout(tris, [[0, 0, len(tris)]], src)
    assert L.count == 1
    for t, P in enumerate(frames):
        out = FG.decode(lit, rec, t)
        assert np.abs(out["position"] - P[L.src]).max() < 1e-6
        assert np.einsum("vi,vi->v", out["normal"], normals(P)[L.src]).min() > 1 - 1e-6


def test_record_offsets_move_with_the_part():
    w = Word()
    frames = [(w.pos, None, None, None), (_affine(w.pos, _turn(3.0), (0, 0, 0)), None, None, None)]
    enc, lit, rec, kinds = _encode(w, frames)
    moved = enc.record(4096)
    pad = np.concatenate([np.zeros(4096, np.uint8), lit])
    for k in range(2):
        a, b = FG.decode(lit, rec, k), FG.decode(pad, moved, k)
        assert np.array_equal(a["position"], b["position"])


def test_a_word_done_drops_its_fitting_state_and_keeps_what_decodes():
    """A clip's screen type is hundreds of words, each fitted over its own window: once its window is over, the bake
    lets go of what fitting it took (its segment's maps and arrays, megabytes a word), keeping the records."""
    w = Word()
    frames = [(w.pos, None, None, None), (_affine(w.pos, _turn(3.0), (0, 0, 0)), None, None, None)]
    enc, lit, rec, kinds = _encode(w, frames)
    assert enc.seg is not None
    enc.done()
    assert enc.seg is None and enc.last is None and enc.record(0) == rec
    _check(w, frames, lit, enc.record(0))


# ================================================================================================= mkr's fixture
def make_fixture(folder):
    """The small bake mkr's screen_geo.rs test decodes: one screen word (lit.screen.meshes[0]) over 9 frames (a base,
    a rigid turn, letters moving apart, a perspective tilt, an alpha fade, the word bent round a ring, a hidden frame,
    the typewriter's next letter, a turn of it), and raw.json: each frame's input (positions, normals, island and
    vertex attributes) in the materialised order of the geometry that holds it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    w3, w4 = Word(letters=(0, 1, 2), copies=3), Word(letters=(0, 1, 2, 0), copies=3)
    pivot = w3.pos.mean(0)
    P1 = _affine(w3.pos, _turn(9.0) * 1.1, (0.01, 0.0, -0.005), about=pivot)
    P2 = P1.copy()
    for li in range(3):
        sel = w3.letter == li
        P2[sel] = _affine(P2[sel], _turn(6.0 * (li - 1), 1.0, 1.0 + 0.05 * li), (0.0, 0.0, 0.002 * li), P2[sel].mean(0))
    P3 = _tilted(P2, 14.0, -8.0, 8 * w3.em, pivot)
    frames = [(w3.pos, None, None, w3), (P1, None, None, None), (P2, None, None, None), (P3, None, None, None),
              (P3, [0.9, 0.6, 0.3], None, None), (_bend(P2, 0.6, pivot), [0.9, 0.6, 0.3], None, None), None,
              (w4.pos, None, None, w4), (_affine(w4.pos, _turn(-5.0), (0.0, 0.0, 0.01)), [1.0, 1.0, 1.0, 0.5], None, None)]
    enc, lit, rec, kinds = _encode(w3, frames)
    assert kinds == ["geom", "word", "islands", "word", "word", "bent", None, "geom", "word"], kinds
    rec_mesh = {"name": "fixture_word", "kind": "by_frame", "materials": ["fill", "outline"],
                "visible": {"camera": True, "shadow": True, "holdout": False}, "aspect": "16x9",
                "object": {"color": [1, 1, 1, 1], "pass_index": 0}, "instance": None, "shown": None, "matrix": None,
                "why": "screen type", "by_frame": rec, "show": [1, 1 + len(frames)], "knockout": False}
    lut = {"file": "screen_lut.bin", "n": 65, "lo": -12.5, "hi": 12.5}
    doc = {"version": 5, "fps": 30, "frame0": 1, "frames": len(frames),
           "outputs": [{"name": "16x9", "size": [1920, 1080]}],
           "lit": {"file": "lit.bin", "view": {"lut": lut}, "screen": {"meshes": [rec_mesh], "left_out": [], "lut": lut}}}
    (folder / "bake.json").write_text(json.dumps(doc, indent=1))
    (folder / "lit.bin").write_bytes(lit.tobytes())
    (folder / "bake.bin").write_bytes(b"")
    raw = []
    for k, f in enumerate(frames):
        if f is None:
            raw.append(None)
            continue
        wd = f[3] or (w3 if k < 7 else w4)
        P, N, attrs = wd.frame(*f[:3])
        L = FG.Layout(*wd.topo())
        raw.append({"position": P[L.src].astype(float).ravel().tolist(), "normal": N[L.src].astype(float).ravel().tolist(),
                    "attrs": {key: np.asarray(a, np.float32).reshape(len(P), -1)[L.src].astype(float).ravel().tolist()
                              for key, a in attrs.items()},
                    "triangles": len(wd.tris)})
    (folder / "raw.json").write_text(json.dumps({"tol": FG.TOL, "frames": raw}))
    return rec


def test_mkr_fixture_decodes_to_its_raw_frames():
    """The committed fixture (mkr's screen_geo.rs test decodes it too) decodes here to the frames it was made of."""
    doc = json.loads((FIXTURE / "bake.json").read_text())
    lit = np.fromfile(FIXTURE / "lit.bin", np.uint8)
    raw = json.loads((FIXTURE / "raw.json").read_text())
    rec = doc["lit"]["screen"]["meshes"][0]["by_frame"]
    for k, want in enumerate(raw["frames"]):
        out = FG.decode(lit, rec, k)
        if want is None:
            assert out is None
            continue
        assert np.abs(out["position"].ravel() - np.array(want["position"])).max() <= raw["tol"]
        assert np.einsum("vi,vi->v", out["normal"], np.array(want["normal"]).reshape(-1, 3)).min() > 0.999
        for key, a in want["attrs"].items():
            assert np.array_equal(out["attrs"][key].ravel(), np.array(a, np.float32)), key
        assert len(out["tris"]) == want["triangles"]


if __name__ == "__main__":                       # python tests/test_fitgeo.py: rewrite mkr's fixture
    make_fixture(sys.argv[1] if len(sys.argv) > 1 else FIXTURE)
    print("fixture in", FIXTURE)
