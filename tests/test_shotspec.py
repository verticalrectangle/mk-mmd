"""The [[shot]] table (mkmmd/core/shotspec.py) and the wear override tables (mkmmd/core/wear.py `table`), bpy-free: unknown keys
and outputs are errors that list what there is, the per-output tables merge key by key, and a worn prop's `strap` and `cable`
overrides merge over the card's."""
import pytest

from mkmmd.core import shotspec as SP
from mkmmd.core import shotstyle as SS
from mkmmd.core import wear as WR

OUTS = ["9x16", "16x9"]
SHOT = {"name": "hood", "from": 0.0, "to": 4.0, "at": [0, -1.6, 1.3], "look": "cast:rin.head", "lens": 35,
        "frame": {"subject": ["cast:rin.head"], "fill": 0.45}, "dof": {"focus": "cast:rin.head", "fstop": 2.8},
        "keys": [{"t": 0.0}, {"t": 4.0, "at": [0, -1.1, 1.25], "lens": 50}], "style": "silhouette", "colors": {"accent": "gold"},
        "aspect": {"9x16": {"frame": {"fill": 0.6}, "lens": 28}}}


def test_a_shot_with_every_key_it_takes_passes():
    SP.check(SHOT, OUTS)
    SP.check({"name": "p", "plate": True}, OUTS)
    SP.check({"name": "p", "plate": True, "from": 1.0, "to": 2.0}, OUTS)
    SP.check({"name": "r", "from": 0, "to": 1, "reflection": {"object": "glass"}, "aspect": {"16x9": {"reflection": False}}}, OUTS)


@pytest.mark.parametrize("patch, frag", [
    ({"lenss": 40}, "shot 'hood': unknown key 'lenss' (known: name, from, to, plate, aspect, mount, at, look"),
    ({"frame": {"subject": [[0, 0, 1]], "fil": 0.5}}, "shot 'hood' frame: unknown key 'fil' (known: subject, fill, solve)"),
    ({"dof": {"focus": "cast:rin", "fstopp": 2}}, "shot 'hood' dof: unknown key 'fstopp' (known: focus, fstop)"),
    ({"keys": [{"t": 0}, {"t": 1, "lenz": 2}]}, "shot 'hood' keys[1]: unknown key 'lenz' (known: t, at, look, lens, shift)"),
    ({"aspect": {"1x1": {"lens": 40}}}, "[shot.aspect.1x1] is for an output the project does not have (outputs: 9x16, 16x9)"),
    ({"aspect": {"9x16": {"from": 1.0}}}, "shot 'hood' aspect.9x16: unknown key 'from'"),
    ({"aspect": {"9x16": {"frame": {"fil": 1}}}}, "shot 'hood' aspect.9x16 frame: unknown key 'fil'"),
])
def test_unknown_keys_and_outputs_are_errors_that_list_what_there_is(patch, frag):
    with pytest.raises(SP.ShotError) as e:
        SP.check(dict(SHOT, **patch), OUTS)
    assert frag in str(e.value)


def test_a_plate_with_from_and_no_to_says_so():
    with pytest.raises(SP.ShotError, match="a plate takes `from` and `to` together .* only `from`"):
        SP.check({"name": "p", "plate": True, "from": 1.0}, OUTS)
    with pytest.raises(SP.ShotError, match="only `to`"):
        SP.check({"name": "p", "plate": True, "to": 1.0}, OUTS)


def test_an_output_changes_frame_without_losing_the_subject():
    sp = SP.merged(SHOT, SHOT["aspect"]["9x16"])
    assert sp["frame"] == {"subject": ["cast:rin.head"], "fill": 0.6} and sp["lens"] == 28
    assert SHOT["frame"]["fill"] == 0.45                                            # the shot itself is untouched
    assert SP.merged(SHOT, None) == SHOT
    sp = SP.merged(SHOT, {"dof": {"fstop": 4.0}, "colors": {"background": "base"}})
    assert sp["dof"] == {"focus": "cast:rin.head", "fstop": 4.0} and sp["colors"] == {"accent": "gold", "background": "base"}


def test_targets_and_lists_are_replaced_whole_not_merged():
    shot = {"name": "s", "at": {"path": "road:road", "s": 640, "offset": 7, "z": 1.2}, "keys": [{"t": 0}, {"t": 1}]}
    sp = SP.merged(shot, {"at": {"prop": "car", "point": [0, 0, 1]}, "keys": [{"t": 2}]})
    assert sp["at"] == {"prop": "car", "point": [0, 0, 1]} and sp["keys"] == [{"t": 2}]    # no path / prop hybrid


def test_reflection_false_in_an_aspect_switches_the_look_off_for_that_output_only():
    shot = {"name": "r", "reflection": {"object": "glass", "strength": 0.4}, "aspect": {"16x9": {"reflection": False},
                                                                                      "9x16": {"reflection": {"dim": 0.2}}}}
    off, kept = SP.merged(shot, shot["aspect"]["16x9"]), SP.merged(shot, shot["aspect"]["9x16"])
    assert SS.normalize(off, {}) == {}
    refl = SS.normalize(kept, {})["reflection"]
    assert refl["object"] == "glass" and refl["strength"] == 0.4 and refl["dim"] == 0.2     # merged, not replaced


# ---------------------------------------------------------------- wear overrides
ENTRY = {"name": "stand", "strap": {"top": "strap_top", "bottom": "strap_bottom", "width": 0.05},
         "cable": {"object": "g_cable", "anchor": "jack", "radius": 0.0032}, "at": [0, 0, 0], "neck_deg": 30.0}


def test_wear_overrides_may_carry_strap_and_cable_tables():
    over = {"neck_deg": 20, "cable": {"trail": [1, 0], "reach": 0.3}, "strap": {"width": 0.03}}
    assert WR.params(ENTRY, over)["neck_deg"] == 20                                  # the tables no longer trip `unknown key`
    assert WR.table(ENTRY, over, "cable") == {"object": "g_cable", "anchor": "jack", "radius": 0.0032, "trail": [1, 0],
                                              "reach": 0.3}
    assert WR.table(ENTRY, over, "strap") == {"top": "strap_top", "bottom": "strap_bottom", "width": 0.03}
    assert WR.table(ENTRY, {}, "cable") == ENTRY["cable"] and WR.table({"name": "x"}, {}, "strap") is None
    with pytest.raises(WR.WearError, match="unknown key 'bogus'"):
        WR.params(ENTRY, {"bogus": 1})


def test_wear_override_tables_are_checked_and_need_the_card_to_have_them():
    with pytest.raises(WR.WearError, match=r"wear cable: unknown key 'trial' \(known: object, anchor"):
        WR.table(ENTRY, {"cable": {"trial": [1, 0]}}, "cable")
    with pytest.raises(WR.WearError, match="`strap` is a table of top, bottom"):
        WR.table(ENTRY, {"strap": 0.03}, "strap")
    with pytest.raises(WR.WearError, match="entry 'plain' has no `cable` to change"):
        WR.table({"name": "plain"}, {"cable": {"reach": 0.2}}, "cable")
