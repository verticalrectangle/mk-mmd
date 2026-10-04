"""Face morphs as deformation fields on the skin (and the parts that ride on it).

Every morph is first evaluated on the SKIN vertices (the face mesh, head-local metres); lashes, lip lines and brow decals
are bound to the skin and follow it (head_decal), eyeball layers and the mouth interior are moved by their own rules.
The result is one (n, 3) offset array per mesh and morph, in head-local metres (the caller scales them to model space)."""
import numpy as np

from . import head_brow as HB
from . import head_eye as EY
from . import head_mouth as MO


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------------- the morph book
class MorphBook:
    """name -> {mesh name: (n, 3) offsets} for the meshes of the part; sizes: {mesh name: vertex count}."""

    def __init__(self, sizes):
        self.sizes = dict(sizes)
        self.data = {}

    def get(self, name, mesh):
        d = self.data.setdefault(name, {})
        if mesh not in d:
            d[mesh] = np.zeros((self.sizes[mesh], 3))
        return d[mesh]

    def add(self, name, mesh, delta, rows=None):
        a = self.get(name, mesh)
        if rows is None:
            a += delta
        else:
            a[rows] += delta

    def names(self):
        return list(self.data)


def piece_rows(acc, piece):
    a, b = acc.pieces[piece]
    return np.arange(a, b)


class Morpher:
    """Builds every face morph from the geometry: skin offsets from fields, decals riding on the skin, eyeball layers,
    brow strips. `H` is a simple namespace with: V (skin vertices), surf (SkinSurface), decals {piece: Decal}, lines /
    eyes (MeshAcc), lids / sheets / strips (head_lid), eyes_info {side: eye dict}, brows {side: centre line}, bind_strip, brow_cfg."""

    def __init__(self, H):
        self.H = H
        self.book = MorphBook({"face": len(H.V), "lines": len(H.lines.V), "eyes": len(H.eyes.V),
                               "mouth": len(H.mouth_mesh.V), "lids": len(H.lids_mesh.V)})

    # ---- skin offsets -> decals
    def skin(self, name, d):
        """Store skin offsets `d` (n_skin, 3) and move every decal with them."""
        H = self.H
        self.book.add(name, "face", d)
        Vm = H.V + d
        from .head_decal import vertex_normals
        Nm = vertex_normals(Vm, H.surf.T)
        for piece, dec in H.decals.items():
            if piece.startswith("brow"):
                continue                                   # brows are moved by their own morphs (the skin there is rigid)
            delta = dec.positions(Vm, Nm) - dec.positions()
            if np.abs(delta).max() > 0:
                self.book.add(name, "lines", delta, piece_rows(H.lines, piece))

    # ---- eyes
    def eye_layers(self, name, k_iris=1.0, k_pupil=1.0, k_hl=1.0, dv=0.0, back=0.0, sides=("L", "R")):
        """Scale the iris, pupil and highlight discs about the iris centre (highlights about their own centre) and shift
        them down by dv metres on the eyeball."""
        H = self.H
        for sd in sides:
            e = H.eyes_info[sd]
            bar = e["barrel"]
            cx, cz = e["centre"]
            for piece, k, gap in (("iris", k_iris, EY.LAYER_GAP), ("pupil", k_pupil, 2 * EY.LAYER_GAP),
                                  ("highlight1", k_hl, 3 * EY.LAYER_GAP), ("highlight2", k_hl, 3 * EY.LAYER_GAP)):
                rows = piece_rows(H.eyes, f"{piece}.{sd}")
                P = np.asarray(H.eyes.V)[rows]
                if piece.startswith("highlight"):
                    c = P[:, [0, 2]].mean(0)
                else:
                    c = np.array([cx, cz])
                x = c[0] + k * (P[:, 0] - c[0])
                z = c[1] + k * (P[:, 2] - c[1]) - dv
                d = bar.layer(x, z, gap) - P
                d[:, 1] += back
                self.book.add(name, "eyes", d, rows)

    # ---- brows
    def brows(self, name, **params):
        H = self.H
        for sd, sg in (("L", 1.0), ("R", -1.0)):
            cl, w = HB.centreline(H.brow_cfg, **params)
            lo, hi = HB.edges(cl, w)
            lo, hi = lo * [sg, 1.0], hi * [sg, 1.0]
            dec, _ = H.bind_strip(H.surf, lo, hi, HB.BROW["height"], tip=True)
            piece = f"brow.{sd}"
            delta = dec.positions() - H.decals[piece].positions()
            self.book.add(name, "lines", delta, piece_rows(H.lines, piece))


# name tables: semantic key -> (PMX name, panel, English name)
NAMES = {
    "blink": ("まばたき", "eye", "blink"), "smile_eyes": ("笑い", "eye", "smile"), "wink_l": ("ウィンク", "eye", "wink"),
    "wink_r": ("ウィンク右", "eye", "wink right"), "wink2_l": ("ウィンク２", "eye", "wink 2"),
    "wink2_r": ("ｳｨﾝｸ２右", "eye", "wink 2 right"), "hau": ("はぅ", "eye", "hau"), "calm": ("なごみ", "eye", "calm"),
    "surprised": ("びっくり", "eye", "surprised"), "jito": ("じと目", "eye", "half-lidded"),
    "pupils_small": ("瞳小", "eye", "small pupils"), "star_eyes": ("星目", "eye", "star eyes"),
    "brow_up": ("上", "brow", "brow up"), "brow_down": ("下", "brow", "brow down"), "cheerful": ("にこり", "brow", "cheerful"),
    "serious": ("真面目", "brow", "serious"), "troubled": ("困る", "brow", "troubled"), "angry": ("怒り", "brow", "angry"),
    "sad": ("悲しみ", "brow", "sad"),
    "a": ("あ", "mouth", "a"), "i": ("い", "mouth", "i"), "u": ("う", "mouth", "u"), "e": ("え", "mouth", "e"),
    "o": ("お", "mouth", "o"), "omega": ("ω", "mouth", "omega"), "mouth_smile": ("口角上げ", "mouth", "mouth corners up"),
    "mouth_down": ("口角下げ", "mouth", "mouth corners down"), "grin": ("にやり", "mouth", "grin"),
    "blush": ("照れ", "other", "blush"),
}


def build_eye_and_brow_morphs(M):
    """Fill the book with the eye and brow morphs (the mouth and blush morphs are added by their own builders). The lids
    close with sheets and strips (head_lid): the skin does not move."""
    from . import head_lid as HL
    n = lambda k: NAMES[k][0]
    HL.lid_morphs(M, M.H)
    M.eye_layers(n("pupils_small"), k_iris=0.86, k_pupil=0.52, k_hl=0.95)
    M.eye_layers(n("surprised"), k_iris=0.93, k_pupil=0.68, k_hl=1.05)
    M.eye_layers(n("hau"), k_iris=0.92, k_pupil=0.9, k_hl=1.12, dv=0.0010)
    for key, params in HB.MORPHS.items():
        M.brows(n(key), **params)


def build_mouth_morphs(M):
    """Mouth shape morphs: lip field on the skin; the interior, teeth and tongue follow the lips."""
    H = M.H
    m = H.mouth
    V = H.V
    ids3 = m["ids3"]
    side = np.zeros(len(V))
    side[ids3[m["upper"]]] = 1.0
    side[ids3[m["lower"]]] = -1.0
    n = m["n"]
    A_rows = piece_rows(H.mouth_mesh, "cavity")
    follow, cols = m["cav_follow"], m["cav_cols"]
    teeth_rows = piece_rows(H.mouth_mesh, "teeth")
    tongue_rows = piece_rows(H.mouth_mesh, "tongue")
    ids_up = np.concatenate([[m["left"]], m["upper"][::-1], [m["right"]]])
    xs_up = V[ids3[ids_up]][:, 0]
    order = np.argsort(xs_up)
    for key, params in MO.SHAPES.items():
        d = MO.lip_field(V, side, m["cfg"], **params)
        M.skin(NAMES[key][0], d)
        dl = d[ids3]                                      # lip margin offsets in ring-3 column order
        name = NAMES[key][0]
        M.book.add(name, "mouth", dl[cols] * follow[:, None], A_rows)
        du = dl[ids_up][order]
        tx = H.teeth_x
        t_off = np.stack([np.interp(tx, xs_up[order], du[:, k]) for k in range(3)], -1)
        td = np.zeros((len(teeth_rows), 3))
        td[0::2] = t_off
        td[1::2] = t_off
        M.book.add(name, "mouth", td, teeth_rows)
        low = dl[m["lower"]].mean(0) * 0.5
        low[0] = 0.0
        M.book.add(name, "mouth", np.tile(low, (len(tongue_rows), 1)), tongue_rows)
