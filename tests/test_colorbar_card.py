"""The colour-bar card's layout (mkmmd.blender.library.props.colorbar_card_layout, no bpy): a closed rounded card with
rolled edges in its frame, the surface panel matching the printed face, the pictures flipped right for Blender (the
texture's top row on the face's top edge, not mirrored; the back read from behind), the options and the palette."""
import numpy as np
import pytest

from mkmmd.blender.library.props import colorbar_card_layout as LAY
from mkmmd.core import colorbars as CB
from mkmmd.core import palette as PAL

DAWN = PAL.PALETTES["rose-pine-dawn"]


def _verts(m, mat):
    ids = np.unique(np.concatenate([m.Q[m.Qm == LAY.MATS.index(mat)].ravel(), m.T[m.Tm == LAY.MATS.index(mat)].ravel()]))
    return ids.astype(int)


def test_the_card_is_a_closed_rounded_plate_facing_minus_y():
    (w, h), t, style = LAY.options({})
    assert (w, h, t, style) == (0.60, 0.45, 0.012, "both")
    m = LAY.mesh((w, h), t)
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert hi[[0, 2]] - lo[[0, 2]] == pytest.approx([w, h], abs=1e-6) and (lo + hi)[[0, 2]] == pytest.approx(0.0, abs=1e-6)
    assert lo[1] == pytest.approx(-0.5 * t - LAY.DOME, abs=1e-6) and hi[1] == pytest.approx(0.5 * t, abs=1e-6)
    face, back = _verts(m, "face"), _verts(m, "back")
    assert m.V[face, 1].max() < -0.25 * t and m.V[back, 1].min() > 0.25 * t     # the print runs onto the roll's start
    n = m.vertex_normals()
    assert (n[face, 1] < -0.45).all() and (n[back, 1] > 0.45).all()
    edge = _verts(m, "edge")
    assert 0.5 * t < np.ptp(m.V[edge, 1]) <= t + 1e-9                        # the cream rim between face and back


def test_the_surface_panel_matches_the_printed_face():
    (w, h), t, style = LAY.options({"size": [0.8, 0.6], "thick": 0.02})
    m = LAY.mesh((w, h), t)
    c = LAY.card("bars", (w, h), t, style)
    (s,) = c["use"]["surface"]
    assert s["name"] == "face" and s["normal"] == [0.0, -1.0, 0.0] and s["up"] == [0.0, 0.0, 1.0]
    assert s["size"] == [0.8, 0.6] and s["center"] == pytest.approx([0.0, -0.5 * t, 0.0])
    face = _verts(m, "face")
    u, v = m.UV[face, 0], m.UV[face, 1]
    x = s["center"][0] + (u - 0.5) * s["size"][0]
    z = s["center"][2] + (v - 0.5) * s["size"][1]
    assert x == pytest.approx(m.V[face, 0], abs=1e-9) and z == pytest.approx(m.V[face, 2], abs=1e-9)   # uv = the panel
    assert c["use"]["look"] == [{"name": "centre", "point": s["center"]}]
    assert c["size"] == [0.8, 0.02, 0.6] and c["origin"] == "center" and c["front"] == "-Y"
    assert c["objects"] == ["bars_card"] and c["style"] == "both"


@pytest.mark.parametrize("style", CB.STYLES)
def test_the_texture_s_top_row_lands_on_the_face_s_top_edge_unmirrored(style):
    size, t = (0.6, 0.45), 0.012
    m = LAY.mesh(size, t)
    face, back = LAY.pictures(size, DAWN, style)
    H, W = face.shape[:2]
    buf = LAY.pixels(face).reshape(H, W, 4)                     # Blender's rows run up: row 0 is the bottom
    assert buf[..., 3].min() == 1.0

    def sample(img_buf, uv):
        col = np.clip(np.floor(uv[:, 0] * W).astype(int), 0, W - 1)
        row = np.clip(np.floor(uv[:, 1] * H).astype(int), 0, H - 1)
        return img_buf[row, col, :3]

    ids = _verts(m, "face")
    top = ids[m.V[ids, 2] > 0.5 * size[1] - 0.01]               # the face's top edge
    assert len(top) and np.allclose(sample(buf, m.UV[top]), face[0, np.clip((m.UV[top, 0] * W).astype(int), 0, W - 1)])
    bottom = ids[m.V[ids, 2] < -0.5 * size[1] + 0.01]
    assert np.allclose(sample(buf, m.UV[bottom]), face[-1, np.clip((m.UV[bottom, 0] * W).astype(int), 0, W - 1)])
    left = ids[np.argmin(m.V[ids, 0])]                          # the viewer's left from the front is -x: column 0
    assert m.UV[left, 0] < 0.02
    bids = _verts(m, "back")
    behind_left = bids[np.argmax(m.V[bids, 0])]                 # seen from behind, +x is the viewer's left
    assert m.UV[behind_left, 0] < 0.02 and np.allclose(back.shape, face.shape)


def test_options_palette_and_roles_are_checked():
    for bad in ({"size": [0.6]}, {"size": "big"}, {"size": [0.01, 0.4]}, {"thick": 0.5}, {"thick": "x"}):
        with pytest.raises(ValueError, match="colorbar_card"):
            LAY.options(bad)
    with pytest.raises(ValueError, match="colour bars style"):                 # EraLooks' error for a bad style
        LAY.pictures((0.6, 0.45), None, LAY.options({"style": "plaid"})[2])
    assert LAY.palette({}) is None and LAY.palette({"style": "dawn"}) is None
    assert LAY.palette(DAWN) == DAWN
    moon = PAL.PALETTES["rose-pine-moon"]
    assert LAY.palette(dict(moon, love="gold"))["love"] == moon["gold"]       # a slot naming another slot
    assert LAY.palette({"love": "#123456"})["text"] == DAWN["text"]           # Dawn's for the slots not given
    with pytest.raises(ValueError, match="palette slot love"):
        LAY.palette({"love": "#12345"})
    edge = LAY.Pal(DAWN).lin("edge")
    assert min(edge) > 0.5                                                    # cream paper in Dawn
    assert min(LAY.Pal(PAL.PALETTES["rose-pine-moon"]).lin("edge")) > 0.4     # and in a dark palette
