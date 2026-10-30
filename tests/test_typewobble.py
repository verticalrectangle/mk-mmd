"""Wobbling letters (mkmmd/core/typewobble.py and `lyrics.wobble` in mkmmd/core/wordtype.py): a wave that travels along the
word while it is sung, as big as the voice is loud, bounded by its amount, sway and tilt, still when the voice is silent,
the same every build. Synthetic timelines of filler words."""
import json
import math
import random

import pytest

from mkmmd.core import typefx as FX
from mkmmd.core import typewobble as TW
from mkmmd.core import wordtype as L

FPS = 30.0


def timeline(vdb=-8.0, frames=300):
    words = [{"text": "aaaa", "start": 1.0, "end": 2.2, "voiced_end": 2.2},       # held
             {"text": "bb", "start": 2.4, "end": 2.5, "voiced_end": 2.5},         # short
             {"text": "cccccc", "start": 3.0, "end": 4.5, "voiced_end": 4.5}]     # held
    vocal = [vdb(i) if callable(vdb) else vdb for i in range(frames)]
    return {"lines": [{"words": words, "start": 1.0, "end": 4.5}], "vocal_db": vocal}


def entry(**lyr):
    return {"name": "w", "screen": True, "lyrics": {"timeline": "t.json", "line": 1, "style": "none", "to": 6.0,
                                                    "wobble": {}, **lyr}}


def run(tl, **lyr):
    return L.expand(entry(**lyr), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=None)


def test_letters_rest_when_the_voice_is_silent():
    for follow in (1.0, 0.0):
        o = TW.options({"follow": follow})
        assert TW.keys(1.0, 2.2, 30, 200, FPS, lambda t: 0.0, o) is None           # a silent voice: nothing to key
        assert all(TW.amplitude(t / 10, 1.0, 2.2, 0.0, o) == 0.0 for t in range(0, 40))
    specs, summ = run(timeline(vdb=-60.0))                                          # far under the -34 dB floor
    assert summ["wobble"] == 0 and not any("wobble" in s["kinetic"] for s in specs)
    loud, summ = run(timeline(vdb=-8.0))
    assert summ["wobble"] == 3


def test_the_wobble_never_exceeds_its_amount_sway_and_tilt():
    o = TW.options({"amount": 0.09, "sway": 0.4, "tilt": 8.0})
    rng = random.Random(7)
    for _ in range(2000):
        i, phase = rng.randrange(0, 12), rng.uniform(-3, 3)
        amp = TW.amplitude(rng.uniform(0, 4), 1.0, 2.2, rng.uniform(0, 1.6), o)    # a level past 1 is clipped
        assert 0.0 <= amp <= 1.0
        c = TW.letter(i, phase, amp, o)
        assert abs(c["dy"]) <= 0.09 * amp + 1e-12 and abs(c["dx"]) <= 0.4 * 0.09 * amp + 1e-12
        assert abs(c["rot"]) <= 8.0 * amp + 1e-12
    specs, _ = run(timeline(vdb=lambda i: -2.0 + 3.0 * math.sin(i / 3.0)))       # louder than the loud end at times
    for s in specs:
        w = FX.kinetic_spec(s)["wobble"]
        assert all(0.0 <= v <= 1.0 for _, v in w["amp"])


def test_the_amplitude_follows_the_voice_while_the_note_is_held_and_dies_after_it():
    o = TW.options({"follow": 1.0, "attack": 0.1, "release": 0.3})
    assert TW.amplitude(0.99, 1.0, 2.0, 0.8, o) == 0.0                              # before the note
    assert TW.amplitude(1.5, 1.0, 2.0, 0.8, o) == pytest.approx(0.8)                 # held: the level itself
    assert TW.amplitude(1.5, 1.0, 2.0, 0.4, o) == pytest.approx(0.4)
    assert 0.0 < TW.amplitude(2.15, 1.0, 2.0, 0.8, o) < 0.8                          # dying away after it
    assert TW.amplitude(2.31, 1.0, 2.0, 0.8, o) == 0.0
    flat = TW.options({"follow": 0.0})
    assert TW.amplitude(1.5, 1.0, 2.0, 0.4, flat) == pytest.approx(1.0)              # full whenever the voice sounds
    swell = timeline(vdb=lambda i: -30.0 if i < 45 else -6.0)                        # quiet, then loud, in one note
    (held, *_), _ = run(swell)
    keys = held["kinetic"]["wobble"]["amp"]                                          # linear keys, straight runs dropped

    def at(f):
        t = f / FPS
        for (ta, va), (tb, vb) in zip(keys, keys[1:]):
            if ta <= t <= tb:
                return va + (vb - va) * (t - ta) / (tb - ta)
        return keys[-1][1]
    quiet = max(at(f) for f in range(34, 41))
    loud = min(at(f) for f in range(52, 63))
    assert loud > 3 * quiet > 0


def test_the_wave_travels_along_the_word_from_its_first_letter_to_its_last():
    o = TW.options({"wave": 4.0, "rate": 2.0})
    for phase in (0.0, 0.13, 0.61):
        for i in range(5):                                   # letter i + 1 does what letter i did a quarter cycle before
            a, b = TW.letter(i, phase, 0.7, o), TW.letter(i + 1, phase + 0.25, 0.7, o)
            assert (a["dx"], a["dy"], a["rot"]) == pytest.approx((b["dx"], b["dy"], b["rot"]))
    (held, *_), _ = run(timeline(), wobble={"rate": 2.0})
    (t0, p0), (t1, p1) = held["kinetic"]["wobble"]["phase"]
    assert (p1 - p0) / (t1 - t0) == pytest.approx(2.0)                                # cycles a second


def test_wobble_is_the_same_every_build_and_words_do_not_bob_in_step():
    a, _ = run(timeline())
    b, _ = run(timeline())
    assert json.dumps([s["kinetic"] for s in a]) == json.dumps([s["kinetic"] for s in b])
    starts = {s["kinetic"]["wobble"]["phase"][0][1] for s in a}
    assert len(starts) == 3


def test_min_hold_keeps_short_notes_still_and_a_punch_word_can_wobble_on_its_own():
    specs, summ = run(timeline(), wobble={"min_hold": 0.5})
    assert [("wobble" in s["kinetic"]) for s in specs] == [True, False, True]
    specs, _ = run(timeline(), wobble=False, punch={"words": [2], "wobble": {"amount": 0.15}})
    assert [("wobble" in s["kinetic"]) for s in specs] == [False, True, False]
    assert specs[1]["kinetic"]["wobble"]["amount"] == 0.15


def test_a_wobbling_word_starts_still_and_ends_still():
    (held, *_), _ = run(timeline())
    amp = held["kinetic"]["wobble"]["amp"]
    assert amp[0][1] == pytest.approx(0.0, abs=1e-9) and amp[-1][1] == pytest.approx(0.0, abs=1e-9)
    on = held["kinetic"]["show"][0]
    assert amp[0][0] == pytest.approx(on)


def test_wobble_errors_name_the_key():
    tl = timeline()
    for bad, frag in (({"shake": 1}, r"wobble keys \['shake'\] are unknown"), ({"rate": 0}, "rate"),
                      ({"follow": 2}, "follow"), ({"amount": -1}, "amount"), ({"wave": "x"}, "wave")):
        with pytest.raises(L.LyricsError, match=frag):
            run(tl, wobble=bad)
    with pytest.raises(L.LyricsError, match="punch wobble"):
        run(tl, punch={"top": 1, "wobble": {"rate": -1}})
    for bad in ({"amp": [[0, 1]]}, {"amp": [[0, 1]], "phase": [[0, 0]], "wave": 0}, {"amp": [], "phase": [[0, 0]]}):
        with pytest.raises(Exception, match="kinetic wobble"):
            FX.kinetic_spec({"kinetic": {"wobble": bad}})
