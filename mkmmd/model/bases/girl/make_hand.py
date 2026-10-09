"""Rebuild hand.npz, the girl base's hand mesh, from its CC0 source: the left glove of B-chan by AmarilloArts (CC0 1.0,
https://amarilloarts.itch.io/blender-chan; the file is the OpenGameArt copy, edited only to remove a logo:
https://opengameart.org/content/b-chan).

    uv run python -m mkmmd.model.bases.girl.make_hand [--source FILE.blend] [--out FILE.npz]

1. The .blend is downloaded into <assets>/sources/bchan/ when it is not there (a direct link, no login).
2. Blender (mk's, from the config) exports the gloves at rest with export_glove.py: the cage (the mirror applied, solidify
   and subdivision off), the rig's vertex-group weights and its deform bones.
3. Here (numpy): the left glove without its flared cuff ring, the rig's groups summed onto the body's semantic bones
   (GROUP), the joints from the rig's bones, and the fingers re-posed from the rig's spread rest to a relaxed one: FAN
   (each finger's direction about the palm normal from the hand's axis, + towards the little finger), CURL (at its three
   joints) and THUMB_IN (the thumb turned towards the index). The result has body_hand_mesh's format, with `license` and
   `source` recorded in it.
docs/model_base.md tells how this hand was made and what to watch when making a part from another artist's mesh."""
import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from .... import config as CFG
from ...parts import body_hand_mesh as BHM

URL = "https://opengameart.org/sites/default/files/bchan_edited_0.blend"
SOURCE = "https://opengameart.org/content/b-chan"
LICENSE = "CC0-1.0"
LEVELS = 2
GROUP = {"hand.l": "wrist", "c_index1_base.l": "wrist", "c_middle1_base.l": "wrist", "c_ring1_base.l": "wrist",
         "c_pinky1_base.l": "wrist", "forearm_twist.l": "forearm", "forearm_stretch.l": "forearm",
         "thumb1.l": "thumb0", "c_thumb2.l": "thumb1", "c_thumb3.l": "thumb2"}
RIG_FINGER = {"index": "index", "middle": "middle", "ring": "ring", "little": "pinky"}     # Auto-Rig Pro's finger names
for _f, _a in RIG_FINGER.items():
    GROUP[f"{_a}1.l"], GROUP[f"c_{_a}2.l"], GROUP[f"c_{_a}3.l"] = _f + "1", _f + "2", _f + "3"
FAN = {"index": 1.5, "middle": 1.0, "ring": 0.0, "little": -1.5}
CURL = {"index": (4, 4, 5), "middle": (5, 6, 6), "ring": (6, 8, 7), "little": (8, 11, 8)}
THUMB_IN = 14.0


def default_source():
    return Path(CFG.load()["assets"]) / "sources" / "bchan" / "bchan_edited.blend"


def fetch(dest):
    """Download the source .blend to `dest` unless it is there; it must be a Blender file (plain or zstd)."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(URL, headers={"User-Agent": "mk-mmd"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    head = tmp.read_bytes()[:7]
    if not (head.startswith(b"BLENDER") or head.startswith(b"\x28\xb5\x2f\xfd")):
        tmp.unlink()
        raise RuntimeError(f"{URL} did not give a .blend file")
    tmp.rename(dest)
    return dest


def export(source, raw):
    """Blender writes the gloves of `source` to `raw` (export_glove.py)."""
    blender = CFG.load()["blender"]
    script = Path(__file__).with_name("export_glove.py")
    r = subprocess.run([blender, "-b", str(source), "--python", str(script), "--", str(raw)], capture_output=True,
                       text=True)
    if r.returncode != 0 or not Path(raw).is_file():
        raise RuntimeError(f"Blender could not export the gloves of {source}:\n{(r.stdout + r.stderr)[-2000:]}")


def _unit(v):
    return v / np.linalg.norm(v)


def _rot(axis, deg):
    a = _unit(np.asarray(axis, float))
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _about(R, p):
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = p - R @ p
    return X


def prepare(raw, fan=None, thumb_in=None):
    """The hand asset (a dict of arrays, body_hand_mesh's format) from the exported gloves `raw`."""
    z = np.load(raw, allow_pickle=False)
    V = z["verts"]
    faces = [[int(v) for v in f] for f in np.split(z["face_flat"], np.cumsum(z["face_sizes"])[:-1])]
    rig = {str(n): (h, t) for n, h, t in zip(z["bone_names"], z["bone_heads"], z["bone_tails"])}
    names = [str(n) for n in z["weight_names"]]
    W = z["weights"]
    # the left glove (+x), without the flared cuff: its open boundary ring and the faces on it
    faces = [f for f in faces if all(V[v, 0] > 0 for v in f)]
    cnt = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            cnt[(min(a, b), max(a, b))] += 1
    cuff = {v for e, c in cnt.items() if c == 1 for v in e}
    faces = [f for f in faces if not any(v in cuff for v in f)]
    used = np.unique(np.concatenate([np.array(f) for f in faces]))
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    faces = [[int(remap[v]) for v in f] for f in faces]
    V = V[used].copy()
    bones = list(BHM.BONES)
    Wb = np.zeros((len(used), len(bones)))
    for k, g in enumerate(names):
        if g in GROUP:
            Wb[:, bones.index(GROUP[g])] += W[used, k]
    Wb /= np.maximum(Wb.sum(1, keepdims=True), 1e-12)
    # joints from the rig
    J = {"wrist": rig["hand.l"][0], "thumb0": rig["thumb1.l"][0], "thumb1": rig["c_thumb2.l"][0],
         "thumb2": rig["c_thumb3.l"][0], "thumb_tip": rig["c_thumb3.l"][1]}
    for f, a in RIG_FINGER.items():
        J[f + "1"], J[f + "2"], J[f + "3"] = rig[f"{a}1.l"][0], rig[f"c_{a}2.l"][0], rig[f"c_{a}3.l"][0]
        J[f + "_tip"] = rig[f"c_{a}3.l"][1]
    a_ = _unit(J["middle1"] - J["wrist"])
    r_ = J["little1"] - J["index1"]
    r_ = _unit(r_ - (r_ @ a_) * a_)
    n_ = -np.cross(a_, r_)                                   # out of the palm (a left hand)

    def fan_of(f):
        d = J[f + "2"] - J[f + "1"]
        d = d - (d @ n_) * n_
        return np.degrees(np.arctan2(d @ r_, d @ a_))
    fan = FAN if fan is None else fan
    M = {b: np.eye(4) for b in bones}
    J2 = dict(J)
    for f in RIG_FINGER:
        j1, j2, j3, tip = J[f + "1"], J[f + "2"], J[f + "3"], J[f + "_tip"]
        turn = -(fan[f] - fan_of(f))                         # _rot(n, +x) turns towards the thumb: hence the sign
        M1 = _about(_rot(n_, turn) @ _rot(np.cross(_unit(j2 - j1), n_), CURL[f][0]), j1)
        p2, p3 = (M1 @ np.r_[j2, 1])[:3], (M1 @ np.r_[j3, 1])[:3]
        M2 = _about(_rot(np.cross(_unit(p3 - p2), n_), CURL[f][1]), p2) @ M1
        p3b, pt = (M2 @ np.r_[j3, 1])[:3], (M2 @ np.r_[tip, 1])[:3]
        M3 = _about(_rot(np.cross(_unit(pt - p3b), n_), CURL[f][2]), p3b) @ M2
        M[f + "1"], M[f + "2"], M[f + "3"] = M1, M2, M3
        J2[f + "2"], J2[f + "3"], J2[f + "_tip"] = p2, p3b, (M3 @ np.r_[tip, 1])[:3]
    Mt = _about(_rot(n_, -(THUMB_IN if thumb_in is None else thumb_in)), J["thumb0"])
    for b in ("thumb0", "thumb1", "thumb2"):
        M[b] = Mt
    for nm in ("thumb1", "thumb2", "thumb_tip"):
        J2[nm] = (Mt @ np.r_[J[nm], 1])[:3]
    Vh = np.c_[V, np.ones(len(V))]
    V2 = np.zeros_like(V)
    for k, b in enumerate(bones):
        V2 += Wb[:, k:k + 1] * (Vh @ M[b].T)[:, :3]
    jn = list(J2)
    return dict(verts=V2, face_flat=np.concatenate([np.array(f) for f in faces]),
                face_sizes=np.array([len(f) for f in faces]), bones=np.array(bones), weights=Wb,
                joint_names=np.array(jn), joints=np.array([J2[n] for n in jn]), levels=LEVELS,
                license=np.array(LICENSE), source=np.array(SOURCE))


def make(out, source=None):
    """Write the hand asset to `out` from `source` (default: default_source(), downloaded when missing); returns a summary.
    Quiet: body_hand_mesh calls this when the asset is missing."""
    source = fetch(source or default_source())
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "gloves.npz"
        export(source, raw)
        asset = prepare(raw)
    np.savez(out, **asset)
    return {"wrote": str(out), "cage_vertices": len(asset["verts"]), "faces": len(asset["face_sizes"]),
            "joints": len(asset["joint_names"]), "source": str(source)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mkmmd.model.bases.girl.make_hand", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, help="the B-chan .blend (default: <assets>/sources/bchan/bchan_edited.blend, "
                                                "downloaded when missing)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("hand.npz"), help="the hand mesh to write")
    args = ap.parse_args(argv)
    print(json.dumps(make(args.out, args.source)))


if __name__ == "__main__":
    main()
