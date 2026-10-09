"""The model bases shipped with mk (mkmmd/model/bases): `mk model new` starts characters from them, so each must build
from its own files alone, every part valid and consistent with the others. bpy-free."""
import pytest

from mkmmd.model import build as BD
from mkmmd.model import spec as SP

SMALL = "outfit.texture_sizes={dress = 64, frill = 64, satin = 64, leather = 64}"


@pytest.mark.parametrize("name", SP.bases())
def test_base_builds(name, tmp_path):
    spec = SP.load(f"base:{name}", [SMALL])
    parts = BD.run(spec, tex_dir=tmp_path)
    assert [p.name for p in parts] == spec["model"]["parts"]
    assert BD.check_refs(parts, strict=True) == []


def test_rin_keeps_her_eye_drawing_wherever_the_girl_draws_her_own():
    """The girl base draws softer eyes; every lash value it sets, the Rin example puts back to the head part's own
    drawing (Rin's), so changing the girl's eyes never changes Rin's."""
    from mkmmd.model.parts import head_eye as EY
    own = {"lash": EY.UPPER_LASH, "lashwing": EY.LASH_WING, "lashfork": EY.LASH_FORK, "crease": EY.CREASE,
           "lashlow": EY.LOWER_LASH}
    girl, rin = SP.load("base:girl")["head"], SP.load("base:rin")["head"]
    changed = [(t, k) for t in own for k in (girl.get(t) or {})]
    assert changed                                                   # the girl has eyes of her own
    for t, k in changed:
        assert rin[t][k] == own[t][k], (t, k)
