"""Rebuild body.npz, the girl base's body mesh, from its CC0 source: the stylized female body of Blender Studio's Human Base
Meshes v1.4.1 ("by Blender Studio and community contributions", CC0 1.0, https://www.blender.org/download/demo-files/).

    uv run python -m mkmmd.model.bases.girl.make_body [--source FILE.blend] [--out FILE.npz]

1. The bundle (a 49 MB zip, a direct link) is downloaded into <assets>/sources/blender_human_base_meshes/ when its .blend
   is not there, and the .blend is taken out of it.
2. Blender (mk's, from the config) exports the body with export_body.py: its vertices, faces and UV map.
3. Here (numpy, scipy): the body centred on x = 0; the head cut off with a level plane just under the jaw (NECK_CUT) and
   anything the cut leaves loose inside the neck dropped; the UV tiles packed into the top half of the skin atlas (PACK);
   the donor's own joints (JOINTS, measured on its mesh); every vertex's share of the torso, each arm and each leg
   (harmonic fields between seed regions, so a share follows the surface: the side of the chest under the arm stays
   torso); the edge loops round each forearm near the wrist (the body is cut at one of them for the hand) and every
   vertex's mirror twin. The result has body_donor's format, with `license` and `source` recorded in it.
docs/model_base.md tells how this body was made and how it is fitted to a character."""
import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .... import config as CFG

URL = "https://www.blender.org/download/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip"
SOURCE = "https://www.blender.org/download/demo-files/ (Human Base Meshes v1.4.1, GEO-body_female_stylized)"
LICENSE = "CC0-1.0"
MEMBER = "human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend"

# The donor's own joints, left side, metres (after centring on x = 0), measured on its mesh:
#   shoulder  the centre of the ball under the deltoid: on the upper arm's axis, 3 cm above the top of the armpit
#   elbow     the centre of the elbow's section, between the crease (front) and the point of the elbow (back)
#   wrist     just past the forearm's narrowest section, where the hand starts to widen
#   hip       6 cm above the crotch, 57 % of the hips' half width out, at the depth of the thigh's centre line
#   knee      the narrowest section between the thigh and the calf
#   ankle     the shin's axis carried down to the height of the ankle bones
JOINTS = {"shoulder": (0.130, 0.012, 1.218), "elbow": (0.237, 0.021, 1.056), "wrist": (0.326, -0.014, 0.896),
          "hip": (0.090, -0.026, 0.850), "knee": (0.0806, -0.0215, 0.450), "ankle": (0.068, -0.002, 0.075)}
NECK_CUT = 1.32                  # just under the jaw at the front; the cut ring is where the head part's neck begins
WAIST_Z, BUST_Z = 1.02, 1.14     # the torso's narrowest section, the bust's deepest
SNAP = 0.0015                    # vertices this close to the cut move onto it (no sliver faces)
# UDIM tile -> its place (u0, v0, u1, v1) in the skin atlas; the hands' tiles of the atlas are its bottom half
PACK = {1: (0.0, 0.5, 0.5, 1.0),        # torso, arms, legs
        4: (0.5, 0.75, 0.75, 1.0),      # feet
        2: (0.5, 0.5, 0.75, 0.75),      # hands (cut off when a character is built: the hand mesh takes their place)
        0: (0.75, 0.75, 1.0, 1.0)}      # the head's tile: only the top of the neck is left of it
PLAIN_UV = (0.875, 0.625)               # a texel no mark reaches (the bridge to the hand takes it)


def default_source():
    return Path(CFG.load()["assets"]) / "sources" / "blender_human_base_meshes" / MEMBER


def fetch(dest):
    """The bundle's .blend at `dest`, downloaded and taken out of the zip unless it is there."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "bundle.zip"
        req = urllib.request.Request(URL, headers={"User-Agent": "mk-mmd"})
        with urllib.request.urlopen(req, timeout=600) as r, open(zp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        with zipfile.ZipFile(zp) as z, z.open(MEMBER) as src, open(dest.with_suffix(".part"), "wb") as out:
            shutil.copyfileobj(src, out)
    head = dest.with_suffix(".part").read_bytes()[:7]
    if not (head.startswith(b"BLENDER") or head.startswith(b"\x28\xb5\x2f\xfd")):
        dest.with_suffix(".part").unlink()
        raise RuntimeError(f"{URL} did not hold a .blend file at {MEMBER}")
    dest.with_suffix(".part").rename(dest)
    return dest


def export(source, raw):
    """Blender writes the body of `source` to `raw` (export_body.py)."""
    blender = CFG.load()["blender"]
    script = Path(__file__).with_name("export_body.py")
    r = subprocess.run([blender, "-b", str(source), "--python", str(script), "--", str(raw)], capture_output=True,
                       text=True)
    if r.returncode != 0 or not Path(raw).is_file():
        raise RuntimeError(f"Blender could not export the body of {source}:\n{(r.stdout + r.stderr)[-2000:]}")


# ---------------------------------------------------------------- geometry
def pack_uv(uv):
    """Corner UVs (k, 2) in the UDIM tiles -> the skin atlas (PACK); a tile's island keeps its aspect."""
    tile = np.floor(uv[:, 0]).astype(int)
    out = np.empty_like(uv)
    for t in np.unique(tile):
        if int(t) not in PACK:
            raise ValueError(f"the body's UVs use tile {int(t)}, which has no place in the atlas")
        u0, v0, u1, v1 = PACK[int(t)]
        m = tile == t
        out[m, 0] = u0 + (u1 - u0) * (uv[m, 0] - t)
        out[m, 1] = v0 + (v1 - v0) * uv[m, 1]
    return out


def clip_below(V, faces, uvs, z):
    """The polygon mesh below the level plane at height z: vertices within SNAP of it move onto it, faces across it are
    cut there (new vertices on their edges, UVs interpolated per corner). Returns (V, faces, uvs)."""
    V = V.copy()
    d = V[:, 2] - z
    near = np.abs(d) < SNAP
    V[near, 2] = z
    d[near] = 0.0
    out_f, out_uv, cut = [], [], {}
    pts = [V]
    n = len(V)

    def on_edge(a, b):
        nonlocal n
        key = (min(a, b), max(a, b))
        if key not in cut:
            t = d[a] / (d[a] - d[b])
            pts.append((V[a] + t * (V[b] - V[a]))[None])
            cut[key] = n
            n += 1
        return cut[key]

    for f, fu in zip(faces, uvs):
        df = d[f]
        if (df <= 0).all():
            out_f.append(list(f))
            out_uv.append(fu)
            continue
        if (df >= 0).all():
            continue
        nf, nu = [], []
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if d[a] <= 0:
                nf.append(a)
                nu.append(fu[i])
            if d[a] * d[b] < 0:
                t = d[a] / (d[a] - d[b])
                nf.append(on_edge(a, b))
                nu.append(fu[i] + t * (fu[(i + 1) % len(f)] - fu[i]))
        if len(nf) >= 3:
            out_f.append(nf)
            out_uv.append(np.array(nu))
    return np.concatenate(pts, 0), out_f, out_uv


def largest_part(V, faces, uvs):
    """The faces of the largest edge-connected piece, the vertices renumbered (what a cut leaves loose is dropped)."""
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            e = (min(f[i], f[(i + 1) % len(f)]), max(f[i], f[(i + 1) % len(f)]))
            if e in owner:
                parent[find(fi)] = find(owner[e])
            else:
                owner[e] = fi
    roots = np.array([find(i) for i in range(len(faces))])
    keep = roots == np.bincount(roots).argmax()
    faces = [f for f, k in zip(faces, keep) if k]
    uvs = [u for u, k in zip(uvs, keep) if k]
    used = np.unique(np.concatenate([np.array(f) for f in faces]))
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used], [[int(remap[v]) for v in f] for f in faces], uvs


def mirror_twins(V, tol=1e-5):
    """Index of every vertex's mirror image in x = 0 (the body is symmetric: every vertex has one)."""
    key = lambda P: [tuple(k) for k in np.round(P / tol).astype(np.int64)]
    at = {k: i for i, k in enumerate(key(V))}
    twins = np.array([at.get(k, -1) for k in key(V * np.array([-1.0, 1.0, 1.0]))])
    if (twins < 0).any():
        raise ValueError(f"{int((twins < 0).sum())} vertices of the body have no mirror twin")
    return twins


def triangles(faces):
    return np.array([(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)], int)


def cotan_laplacian(V, T):
    """The cotangent Laplacian (n, n) of the triangle mesh (negative weights clamped), negative semidefinite."""
    import scipy.sparse as sp
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, o = T[:, k], T[:, (k + 1) % 3], T[:, (k + 2) % 3]
        a, b = V[i] - V[o], V[j] - V[o]
        cot = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-12)
        w = np.maximum(0.5 * cot, 0.0)
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(V),) * 2).tocsr()
    return L - sp.diags(np.asarray(L.sum(1)).ravel())


def harmonic(L, fixed, values):
    """The harmonic function on the mesh that takes `values` at the vertices `fixed`."""
    import scipy.sparse.linalg as spla
    free = np.ones(L.shape[0], bool)
    free[fixed] = False
    x = np.zeros(L.shape[0])
    x[fixed] = values
    x[free] = spla.spsolve(L[free][:, free].tocsc(), -L[free][:, ~free] @ x[~free])
    return x


def _segment(P, a, b):
    """(parameter along a-b, distance to the segment) of points P."""
    ab = b - a
    t = ((P - a) @ ab) / float(ab @ ab)
    return t, np.linalg.norm(P - (a + np.clip(t, 0.0, 1.0)[:, None] * ab), axis=1)


def seeds(V):
    """Vertices that surely belong to one region: an arm from 60 % down the upper arm (within the arm's flesh) to the
    fingertips; a leg below the crotch; the torso along its middle above the hips and on its flanks below the armpits.
    Nothing near an arm is a torso or leg seed, so the regions meet over the shoulder and the hip in a long, smooth
    hand-over."""
    out, zone = {}, np.zeros(len(V), bool)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        M = np.array([sx, 1.0, 1.0])
        sh, el, wr = (np.array(JOINTS[k]) * M for k in ("shoulder", "elbow", "wrist"))
        tip = wr + 0.20 * (wr - el) / np.linalg.norm(wr - el)
        t_up, d_up = _segment(V, sh, el)
        t_fo, d_fo = _segment(V, el, tip)
        on = V[:, 0] * sx > 0.05
        out[f"arm.{side}"] = on & (((t_up > 0.60) & (d_up < 0.05)) | ((t_fo >= 0.0) & (d_fo < 0.06)))
        zone |= on & (((t_up > -0.10) & (d_up < 0.06)) | ((t_fo >= 0.0) & (d_fo < 0.07)))
    x, z = np.abs(V[:, 0]), V[:, 2]
    for side, sx in (("L", 1.0), ("R", -1.0)):
        out[f"leg.{side}"] = (V[:, 0] * sx > 0.0) & (x < 0.2) & (z < 0.72) & ~zone
    out["torso"] = ~zone & (((x < 0.06) & (z > 0.86)) | ((z > 0.93) & (z < 1.12)))
    return out


REGIONS = ("torso", "arm.L", "arm.R", "leg.L", "leg.R")


def memberships(V, faces):
    """(n, len(REGIONS)) every vertex's share of each region, rows summing to one: per limb the harmonic function that is
    1 on its seeds and 0 on every other region's; the torso takes the rest."""
    S = seeds(V)
    L = cotan_laplacian(V, triangles(faces))
    out = np.zeros((len(V), len(REGIONS)))
    for k, name in enumerate(REGIONS[1:], 1):
        pos = S[name]
        neg = np.zeros(len(V), bool)
        for other in REGIONS:
            if other != name:
                neg |= S[other]
        neg &= ~pos
        fixed = np.flatnonzero(pos | neg)
        out[:, k] = np.clip(harmonic(L, fixed, pos[fixed].astype(float)), 0.0, 1.0)
    out[:, 1:] /= np.maximum(out[:, 1:].sum(1, keepdims=True), 1.0)
    out[:, 0] = 1.0 - out[:, 1:].sum(1)
    return out


def ring_loops(V, faces, centre, axis, along=(-0.065, 0.0), tol=0.25):
    """Closed edge loops round `axis` whose middle lies within `along` (m, along the axis) of `centre`: each loop as its
    vertices in order, the loops sorted along the axis. A loop is walked across the quads of an edge ring: from an edge
    running along the axis, through each quad to the opposite edge, until it comes back."""
    edge_faces, nbr = defaultdict(list), defaultdict(set)
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edge_faces[(min(a, b), max(a, b))].append(fi)
            nbr[a].add(b)
            nbr[b].add(a)

    def walk(a, b):
        loop, cur, prev = [a], (a, b), None
        for _ in range(200):
            fs = [f for f in edge_faces[(min(cur), max(cur))] if f != prev]
            if not fs or len(faces[fs[0]]) != 4:
                return None
            f = faces[fs[0]]
            i, j = f.index(cur[0]), f.index(cur[1])
            na = f[(i - 1) % 4] if f[(i + 1) % 4] == cur[1] else f[(i + 1) % 4]
            nb = f[(j - 1) % 4] if f[(j + 1) % 4] == cur[0] else f[(j + 1) % 4]
            cur, prev = (na, nb), fs[0]
            if {na, nb} == {a, b}:
                return loop
            loop.append(na)
        return None

    axis = axis / np.linalg.norm(axis)
    s = (V - centre) @ axis
    radial = np.linalg.norm((V - centre) - np.outer(s, axis), axis=1)
    cand = np.flatnonzero((s > along[0] - 0.01) & (s < along[1] + 0.01) & (radial < 0.06))
    found = {}
    for a in cand:
        for b in nbr[int(a)]:
            e = V[b] - V[a]
            if abs(e @ axis) < (1.0 - tol) * np.linalg.norm(e):
                continue
            loop = walk(int(a), int(b))
            if loop is None or not (along[0] <= float(s[loop].mean()) <= along[1]):
                continue
            found.setdefault(frozenset(loop), loop)
    loops = sorted(found.values(), key=lambda lp: float(s[lp].mean()))
    if not loops:
        raise ValueError("no edge loops round the forearm near the wrist")
    sizes = {len(lp) for lp in loops}
    if len(sizes) != 1:
        raise ValueError(f"the forearm's loops near the wrist differ in size: {sorted(sizes)}")
    return np.array(loops, int)


def prepare(raw):
    """The body asset (a dict of arrays, body_donor's format) from the exported body `raw`."""
    z = np.load(raw, allow_pickle=False)
    V = np.asarray(z["verts"], float).copy()
    sizes = np.asarray(z["face_sizes"], int)
    flat = np.asarray(z["face_flat"], int)
    uv = pack_uv(np.asarray(z["uv"], float))
    cuts = np.cumsum(sizes)[:-1]
    faces = [list(map(int, f)) for f in np.split(flat, cuts)]
    uvs = np.split(uv, cuts)
    V[:, 0] -= 0.5 * (V[:, 0].min() + V[:, 0].max())
    V, faces, uvs = largest_part(*clip_below(V, faces, uvs, NECK_CUT))
    twin = mirror_twins(V)
    member = memberships(V, faces)
    wr, el = np.array(JOINTS["wrist"]), np.array(JOINTS["elbow"])
    loops = ring_loops(V, faces, wr, wr - el)
    foot = V[(V[:, 0] > 0.0) & (V[:, 2] < 0.06) & (np.abs(V[:, 0]) < 0.2)]
    tip = foot[np.argmin(foot[:, 1])]
    heel = foot[np.argmax(foot[:, 1])]
    joints = dict(JOINTS)
    joints["toe_end"] = (float(tip[0]), float(tip[1]), 0.0)          # the front of the toes, at the floor
    joints["heel"] = tuple(float(x) for x in heel)                   # the back of the heel
    names = list(joints)
    f32, i32 = np.float32, np.int32
    return dict(verts=V.astype(f32), face_flat=np.concatenate([np.array(f) for f in faces]).astype(i32),
                face_sizes=np.array([len(f) for f in faces], i32), uv=np.concatenate(uvs, 0).astype(f32),
                joint_names=np.array(names), joints=np.array([joints[n] for n in names], float),
                neck_z=np.array(NECK_CUT), waist_z=np.array(WAIST_Z), bust_z=np.array(BUST_Z),
                regions=np.array(REGIONS), membership=member.astype(f32), wrist_loops=loops.astype(i32),
                mirror=twin.astype(i32), plain_uv=np.array(PLAIN_UV), license=np.array(LICENSE), source=np.array(SOURCE))


def make(out, source=None):
    """Write the body asset to `out` from `source` (default: default_source(), downloaded when missing); returns a summary.
    Quiet: body_donor calls this when the asset is missing."""
    source = fetch(source or default_source())
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "body.npz"
        export(source, raw)
        asset = prepare(raw)
    np.savez_compressed(out, **asset)
    return {"wrote": str(out), "vertices": len(asset["verts"]), "faces": len(asset["face_sizes"]),
            "wrist_loops": len(asset["wrist_loops"]), "source": str(source)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mkmmd.model.bases.girl.make_body", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, help="the Human Base Meshes .blend (default: <assets>/sources/"
                                                f"blender_human_base_meshes/{MEMBER}, downloaded when missing)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("body.npz"), help="the body mesh to write")
    args = ap.parse_args(argv)
    print(json.dumps(make(args.out, args.source)))


if __name__ == "__main__":
    main()
