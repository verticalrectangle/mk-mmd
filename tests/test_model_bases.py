"""The model bases shipped with mk (mkmmd/model/bases): `mk model new` starts characters from them, so each must build
from its own files alone, every part valid and consistent with the others. bpy-free."""
import pytest

from mkmmd.model import build as BD
from mkmmd.model import spec as SP

SMALL = "outfit.texture_sizes={dress = 64, frill = 64, satin = 64, leather = 64}"
# what a base keeps for features it switches off (a character may switch them on): nothing reads these in its own build
OFF = {"girl": {"colors.ears", "colors.black.ribbon_shade", "colors.black.ribbon_sheen", "colors.accent"},
       "rin": {"colors.accent"}}


@pytest.mark.parametrize("name", SP.bases())
def test_base_builds(name, tmp_path):
    """Each base builds, every part valid and consistent; and every key in its files is read by a part, but the ones of
    the features it switches off: a dead key in a base would be copied into characters and changed to no effect."""
    spec = SP.load(f"base:{name}", [SMALL])
    ctx = BD.BuildCtx(spec, tmp_path, seed=SP.model_cfg(spec)["seed"])
    parts = BD.run(spec, ctx=ctx)
    assert [p.name for p in parts] == spec["model"]["parts"]
    assert BD.check_refs(parts, strict=True) == []
    assert {SP.dotted(p) for p, _ in SP.unread(ctx.spec, bases=True)} <= OFF.get(name, set())


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
