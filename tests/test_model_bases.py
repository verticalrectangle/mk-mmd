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
