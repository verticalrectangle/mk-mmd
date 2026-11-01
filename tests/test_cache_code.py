"""mkmmd.cache.code: a cache key's fingerprint of the code that makes a result changes with that code and nothing else
(the bake's screen-word cache, mkmmd.blender.build.bake_lit._screen_code, keys on it)."""
import importlib
import itertools
import sys
import textwrap

import pytest

from mkmmd import cache as CA

_n = itertools.count()

FIT = '''
"""The fitter."""
TOL = 1e-5


def fit(x):
    return round(x / TOL) * TOL


def unused():
    return 1
'''

BAKE = '''
"""The bake."""
from . import fit as FG
from .util import clamp

SCALE = 2


def _read(x):
    # read a frame
    return clamp(x) * SCALE


class Word:
    """A word."""

    def add(self, x):
        return FG.fit(self._prep(_read(x)))

    def _prep(self, x):
        return x + 1


class Lit:
    def drive(self, w, x):
        return self._show(w.add(x))

    def _show(self, v):
        return v

    def lights(self):
        return 0


def elsewhere():
    return 42
'''

UTIL = '''
def clamp(x):
    return max(0.0, min(1.0, x))


def other(x):
    return x
'''


def fingerprint(tmp_path, bake=BAKE, fit=FIT, util=UTIL):
    """code() of the bake module's screen-word parts, in a fresh copy of the package `codepkg` with these sources."""
    where = tmp_path / f"v{next(_n)}"
    root = where / "codepkg"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    for mod, src in (("bake", bake), ("fit", fit), ("util", util)):
        (root / f"{mod}.py").write_text(textwrap.dedent(src))
    for m in [m for m in sys.modules if m == "codepkg" or m.startswith("codepkg.")]:
        del sys.modules[m]
    sys.path.insert(0, str(where))
    try:
        return CA.code(importlib.import_module("codepkg.bake"), "Word", "Lit.drive")
    finally:
        sys.path.remove(str(where))


def test_comments_docstrings_layout_and_unreached_code_keep_the_fingerprint(tmp_path):
    base = fingerprint(tmp_path)
    assert fingerprint(tmp_path, bake=BAKE.replace("# read a frame", "# reads one frame, world space")) == base
    assert fingerprint(tmp_path, bake=BAKE.replace('"""A word."""', '"""A word of screen type."""')) == base
    assert fingerprint(tmp_path, bake=BAKE.replace("return x + 1", "return (x\n                + 1)")) == base
    assert fingerprint(tmp_path, bake=BAKE.replace("return 42", "return 43")) == base           # elsewhere()
    assert fingerprint(tmp_path, bake=BAKE.replace("return 0", "return 7")) == base             # Lit.lights
    assert fingerprint(tmp_path, fit=FIT.replace("return 1", "return 2")) == base               # fit.unused
    assert fingerprint(tmp_path, util=UTIL.replace("return x\n", "return -x\n")) == base         # util.other


@pytest.mark.parametrize("where, old, new", [
    ("bake", "return x + 1", "return x + 2"),                       # a method of the class
    ("bake", "* SCALE", "* SCALE * 1.0"),                           # a function it calls
    ("bake", "SCALE = 2", "SCALE = 3"),                             # a constant that function reads
    ("bake", "return v\n", "return -v\n"),                          # a method the root method calls on self
    ("fit", "TOL = 1e-5", "TOL = 1e-6"),                            # a module used through its import alias
    ("fit", "round(x / TOL)", "int(x / TOL)"),
    ("util", "max(0.0,", "max(-1.0,"),                              # a function imported by name
])
def test_an_edit_to_code_the_parts_reach_changes_it(tmp_path, where, old, new):
    src = {"bake": BAKE, "fit": FIT, "util": UTIL}
    assert old in src[where]
    edited = dict(src, **{where: src[where].replace(old, new)})
    assert fingerprint(tmp_path, **edited) != fingerprint(tmp_path)


def test_a_name_the_module_lacks_is_an_error(tmp_path):
    name = f"codepkg{next(_n)}"
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("")
    (tmp_path / name / "m.py").write_text("def f():\n    return 1\n")
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(f"{name}.m")
        with pytest.raises(KeyError):
            CA.code(mod, "g")
    finally:
        sys.path.remove(str(tmp_path))
