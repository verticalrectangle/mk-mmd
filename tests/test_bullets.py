"""Lyric bullets (`lyrics.bullets`, mkmmd.core.wordtype.expand with `marks`): words that fly past a target, each on its
own free panel at the target's point on the frame it lands, shown from `lead` before to `after` after, with their flight
path. The Blender half (the scene read on those frames, the object keyed along the path) is smoke-tested in a scene."""
import pytest
from test_wordtype import FPS, no_text_in, timeline

from mkmmd.core import wordtype as LY


def _entry(lyr=None, **bullets):
    return {"name": "m", "fit": 0.8, "lyrics": {"timeline": "t.json", "line": 1, "style": "pop", **(lyr or {}),
                                                "bullets": {"at": "cast:rin.center", **bullets}}}


def _fly(entry, tl, asked=None):
    def marks(at, times):
        if asked is not None:
            asked.append((at, list(times)))
        return [(float(k), 0.0, 1.0) for k in range(len(times))]       # word k's target stands at x = k
    return LY.expand(entry, tl, fps=FPS, marks=marks)


@pytest.mark.parametrize("bullets, lyr, frag", [
    ({"offsets": [[0, 0]]}, {}, "offsets"), ({"at": None}, {}, "needs `at`"), ({"lead": -1}, {}, "lead"),
    ({"box": [0, 1]}, {}, "box"), ({"speed": 3}, {}, "unknown"), ({}, {"zones": {}}, "zones"),
    ({}, {"leave": "drip"}, "drop `leave`")])
def test_bad_bullets_say_what_is_wrong(bullets, lyr, frag):
    with pytest.raises(LY.LyricsError, match=frag):
        _fly(_entry(lyr, **bullets), timeline())


@pytest.mark.parametrize("key", ["on", "at", "box", "screen"])
def test_the_words_place_themselves_so_the_entry_gives_no_placement(key):
    with pytest.raises(LY.LyricsError, match=f"drop \\['{key}'\\]"):
        _fly({**_entry(), key: [0, 0, 0]}, timeline())
    with pytest.raises(LY.LyricsError, match="need the scene"):
        LY.expand(_entry(), timeline(), fps=FPS)


def test_each_word_flies_past_its_mark_on_the_frame_it_lands():
    tl, asked = timeline(), []
    specs, summ = _fly(_entry(offsets=[[0.0, 0.0, -0.7], [0.0, 0.0, 0.5]]), tl, asked)     # feet, head, feet, head
    lands = [LY.landing_frame(w["start"], FPS) / FPS for w in tl["lines"][0]["words"]]
    assert asked == [("cast:rin.center", lands)]                     # the scene is read once, on the landing frames
    for k, (sub, t) in enumerate(zip(specs, lands)):
        z = 1.0 - 0.7 if k % 2 == 0 else 1.0 + 0.5
        assert sub["at"] == [float(k), 0.0, z] and sub["box"] == [1.0, 0.3] and sub["facing"] == [0.0, -1.0, 0.0]
        assert sub["kinetic"]["show"] == pytest.approx([t - 0.6, t + 0.6]) and "scale" in sub["kinetic"]  # pop still pops
        (t0, p0), (t1, p1), (t2, p2) = sub["_path"]
        assert (t0, t1, t2) == pytest.approx((t - 0.6, t, t + 0.6)) and p1 == sub["at"]
        assert p0 == [k + 6.0, 0.0, z] and p2 == [k - 6.0, 0.0, z]                  # in from ahead, out behind
    assert summ["bullets"] == 4 and summ["cut_frame"] == LY.cut_frame(lands[-1] + 0.6, FPS)    # the last one has flown
    assert no_text_in([{k: v for k, v in s.items() if k != "text"} for s in specs] + [summ], tl)
