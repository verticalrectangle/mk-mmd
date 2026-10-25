"""The grade's floor (docs/design.md: Post): without a `floor` key the darkest colour allowed is the palette's darkest slot,
so a light palette (Rosé Pine Dawn, whose `base` is its lightest colour) keeps its blacks dark instead of washing every
frame toward its paper white. Synthetic frames, no Blender."""
import numpy as np

from mkmmd import post as P
from mkmmd.core import palette as PL

FLAT = {"saturation": 1.0, "contrast": 1.0, "vignette": 0.0, "grain": {"amount": 0.0}}


def luma(rgb):
    return float(np.asarray(rgb, np.float32) @ P.Y_W)


def test_darkest_slot_of_each_palette():
    assert PL.darkest(PL.get("rose-pine-moon")) == PL.get("rose-pine-moon")["base"]
    assert PL.darkest(PL.get("rose-pine")) == PL.get("rose-pine")["base"]
    dawn = PL.get("rose-pine-dawn")
    assert PL.darkest(dawn) == dawn["text"]
    assert luma(PL.srgb(dawn["text"])) < luma(PL.srgb(dawn["base"]))


def test_default_floor_keeps_a_light_palette_dark():
    dawn = PL.get("rose-pine-dawn")
    black = np.zeros((8, 8, 3), np.float32)
    out = P.Grade(FLAT, dawn, (8, 8))(black)
    floor = luma(PL.srgb(dawn["text"]))
    assert abs(float((out @ P.Y_W).mean()) - floor) < 0.02          # black lands on the darkest slot ...
    assert float((out @ P.Y_W).max()) < luma(PL.srgb(dawn["base"])) - 0.4   # ... nowhere near the paper white


def test_explicit_floor_still_wins():
    dawn = PL.get("rose-pine-dawn")
    black = np.zeros((4, 4, 3), np.float32)
    lifted = P.Grade({**FLAT, "floor": "base"}, dawn, (4, 4))(black)
    assert float((lifted @ P.Y_W).min()) > 0.9                    # asked for: lifted to the paper
    off = P.Grade({**FLAT, "floor": False}, dawn, (4, 4))(black)
    assert float(off.max()) == 0.0                                 # no floor: black stays black
