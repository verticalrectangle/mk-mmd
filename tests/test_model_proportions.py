"""mkmmd.model.proportions: whole-figure changes of the [proportions] tables (leg_extra)."""
import copy

import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import proportions as PR
from mkmmd.model import spec as SP


def test_leg_extra_lengthens_the_legs_and_what_sits_on_the_body_rides_up():
    """[proportions] leg_extra: from the hip joint up every height rises by it, the knee by its share of the stretch, the
    feet stay; the torso rises whole (its lowest cuts are under the hip joint: stretched, its rings would land elsewhere
    and change its girths); what the parts place on the body keeps its place there (the eye line, the sash at the waist,
    the collar at the neck, the hairline under the crown); the loaded spec is not changed."""
    spec = SP.load("base:girl", ["proportions.leg_extra=0.03"])
    was = copy.deepcopy(dict(spec["proportions"]))
    w = SP.watch(spec)
    PR.apply(w)
    now, L0 = w["proportions"], was["landmarks"]
    L = now["landmarks"]
    hip, ankle = L0["leg.L"][2], L0["ankle.L"][2]
    assert L["head_tip"][2] - L0["head_tip"][2] == pytest.approx(0.03)
    assert L["knee.L"][2] - L0["knee.L"][2] == pytest.approx(0.03 * (L0["knee.L"][2] - ankle) / (hip - ankle))
    assert L["ankle.L"] == L0["ankle.L"] and L["toe_end.L"] == L0["toe_end.L"]
    assert np.allclose(np.subtract(now["sections"]["torso"]["z"], was["sections"]["torso"]["z"]), 0.03)

    def waist(P):
        return P["sections"]["torso"]["z"][int(np.argmin(P["sections"]["torso"]["width"]))]
    for p in (now, was):
        p["pairs"] = (p["face"]["eye_z"] - p["landmarks"]["eye.L"][2], p["outfit_guides"]["sash_z"] - waist(p),
                      p["outfit_guides"]["collar_top_z"] - p["sections"]["neck"]["z"],
                      p["hair_guides"]["hairline_centre_z"] - p["head"]["crown_skull"],
                      p["face"]["mouth_z"] - p["head"]["chin_point"][1])
    assert now["pairs"] == pytest.approx(was["pairs"])
    assert spec["proportions"]["landmarks"]["head_tip"] == L0["head_tip"]


@pytest.mark.parametrize("source, outfit_tol", [("procedural", 1e-6), ("mesh", 1e-3)])
def test_built_with_leg_extra_what_hangs_above_the_hips_is_the_same_only_higher(tmp_path, source, outfit_tol):
    """Built, the body from 10 cm above the hip joints (the thighs reach into the pelvis below that) and the dress above
    its sash (the skirt hangs from the waist to below the knees, longer with the legs) are the same 3 cm higher, point for
    point: the torso's tuned heights rise with it, the girl's mesh body keeps its girths (her size leaves leg_extra out)
    and the dress keeps its size (S leaves leg_extra out). On the mesh body the dress may differ by a hair where its
    points sit in a valley of the skin on the mirror plane (the collar's front): which side the push off the skin takes is
    rounding (micrometres, a few tenths of a millimetre at the bow's tails). Meshes are compared as point sets (welding
    may order them anew)."""
    from scipy.spatial import cKDTree
    built = {}
    for extra in (0.0, 0.03):
        spec = SP.load("base:girl", [f"proportions.leg_extra={extra}", f'body.source="{source}"'])
        built[extra] = {p.name: p for p in BD.run(spec, only="outfit", tex_dir=tmp_path / str(extra))}
    cut = {"body": float(built[0.0]["body"].info["landmarks"]["leg.L"][2]) + 0.10,
           "outfit": float(spec["proportions"]["outfit_guides"]["sash_z"]) + 0.03}     # the loaded spec: the base's sash
    tol = {"body": 1e-6, "outfit": outfit_tol}
    compared = 0
    for part in ("body", "outfit"):
        for a, b in zip(built[0.0][part].meshes, built[0.03][part].meshes):
            pa, pb = a.verts[a.verts[:, 2] > cut[part]] + [0.0, 0.0, 0.03], b.verts[b.verts[:, 2] > cut[part] + 0.03]
            assert len(pa) == len(pb), (part, a.name)
            if len(pa):
                assert max(cKDTree(pb).query(pa)[0].max(), cKDTree(pa).query(pb)[0].max()) < tol[part], (part, a.name)
                compared += 1
    assert compared >= 4                      # the body, the bodice and sleeves, the frills, the bows
