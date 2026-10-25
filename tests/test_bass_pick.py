"""Picking single strings along a bass line (mkmmd.core.strum, `rhythm = "notes"`): the strokes are the notes, each centred on
its string and alternating, and the pick tip meets that string at the strike and clears the others; pure maths on a
synthetic four-string card."""
import numpy as np
import pytest

from mkmmd.core import fretting as FR
from mkmmd.core import strum as SM
from test_bass_notes import TUNING, bass_entry

BEAT = 0.5561
TL = {"beats": [round(i * BEAT, 4) for i in range(60)], "downbeats": [round(i * BEAT, 4) for i in range(0, 60, 4)]}
STRUM = {"name": "strum", "type": "strum", "center": [0.0, -0.0185, 0.085], "along": [0, 0, 1], "across": [1, 0, 0],
         "normal": [0, -1, 0]}
SPEC = {"rhythm": "notes", "hand": "R", "prop": "bass"}
RIFF = [(24.455, 45), (24.638, 47), (24.807, 47), (25.518, 48), (25.721, 48), (25.882, 47), (26.076, 47), (26.229, 47),
        (26.406, 45), (27.206, 46), (27.744, 47), (28.494, 48), (28.687, 48)]


def picked(notes):
    """[(t, c, w, string)] of a line of (t, midi) placed on the synthetic bass, and the placed notes."""
    placed = FR.place(notes, TUNING, FR.fret_z(bass_entry()))
    return SM.pick_notes(placed, bass_entry(), STRUM), placed


def on_strings(pairs):
    """[(t, c, w, string)] of notes given as (t, string) on the synthetic bass (whatever their frets)."""
    return SM.pick_notes([{"t": t, "string": s} for t, s in pairs], bass_entry(), STRUM)


# ------------------------------------------------------------------------------------------------ which strokes
def test_notes_is_a_rhythm_with_its_own_defaults():
    assert SM.parse_rhythm("notes") == ("notes", None) and SM.picking(SPEC) and not SM.picking({"rhythm": "beats:8"})
    p = SM.params(SPEC)
    assert p["span"] == SM.PICK["span"] < SM.DEFAULTS["span"] and p["up_scale"] == 1.0 and p["beat_tol"] == SM.DEFAULTS["beat_tol"]
    assert SM.params(dict(SPEC, span=0.03))["span"] == 0.03                     # the spec's own number wins
    assert SM.params({"rhythm": "beats:8"}) == SM.DEFAULTS
    with pytest.raises(SM.StrumError, match="none was given"):
        SM.plan(SPEC, TL)


def test_pick_notes_put_each_note_on_its_string_at_the_strum_centre():
    e = bass_entry()
    notes = on_strings([(1.0, 4), (2.0, 3), (3.0, 1)])
    assert [n[3] for n in notes] == [4, 3, 1]
    for (t, c, w, s), t0 in zip(notes, (1.0, 2.0, 3.0)):
        x = FR.string_point(e, s, 0.085)[0]
        assert t == t0 and c == pytest.approx(x) and w == pytest.approx(FR.string_entry(e, s)["radius"] + SM.CONTACT)
    assert notes[0][1] < notes[1][1] < 0.0 < notes[2][1]                       # the low E on the low side (u < 0)
    line, placed = picked([(1.0, 45)])
    assert line[0][3] == placed[0]["string"]                                    # the fretting hand's string, picked


def test_picked_strokes_alternate_down_first_and_start_down_again_after_a_rest():
    notes, _ = picked(RIFF)
    st = SM.plan(dict(SPEC, **{"from": 24.42, "to": 28.84}), TL, notes=notes)
    assert [s.t for s in st] == pytest.approx([t for t, _m in RIFF])
    assert all(s.string == 1 and s.c == pytest.approx(notes[0][1]) for s in st)   # the riff stays on the G string
    rest = sum(SM.params(SPEC)[k] for k in ("lead", "attack", "follow", "approach", "retreat")) + 0.1
    for a, b in zip(st, st[1:]):
        assert b.dir == (1 if b.t - a.t > rest else -a.dir)                     # alternate picking, down after a rest
    assert [s.dir for s in st[:4]] == [1, -1, 1, -1]                            # 24.807 -> 25.518 is just short of a rest
    assert st[9].dir == 1                                                       # 26.406 -> 27.206 rests
    assert all(s.amp == pytest.approx(1.0) for s in st)                         # no accent asked: up and down alike


def test_the_window_min_gap_and_accents_apply_to_picked_notes():
    notes, _ = picked([(1.0, 45), (1.03, 45), (1.5, 47), (2.0, 48)])
    st = SM.plan(dict(SPEC, to=1.6), TL, notes=notes)
    assert [round(s.t, 3) for s in st] == [1.0, 1.5]                            # a ghost 30 ms after a note is that note
    acc = SM.plan(dict(SPEC, accent=[1.5]), TL, notes=notes)
    assert [s.accent for s in acc] == [False, True, False]
    assert acc[1].amp == pytest.approx(SM.PICK["accent_gain"])


# ------------------------------------------------------------------------------------------------ where the pick is
def run(notes, **kw):
    spec = dict(SPEC, **kw)
    st = SM.plan(spec, TL, notes=notes)
    ts = np.arange(min(n[0] for n in notes) - 1.0, max(n[0] for n in notes) + 1.0, 0.0005)
    return st, ts, SM.path(st, ts, 0.0265, **SM.params(spec))


def at(P, ts, t):
    i = int(np.argmin(np.abs(ts - t)))
    return P["u"][i], P["h"][i]


def test_the_pick_meets_the_note_string_at_the_strike_and_pushes_through_it():
    notes, _ = picked(RIFF)
    st, ts, P = run(notes)
    p = SM.params(SPEC)
    for s in st:
        u, h = at(P, ts, s.t)
        assert u == pytest.approx(s.c - s.dir * s.w, abs=2e-4)                  # the near side of its string
        assert h == pytest.approx(-p["depth"], abs=1e-4)                        # past the string's axis plane: it catches it
        u2, _h2 = at(P, ts, s.t + s.attack)
        assert u2 == pytest.approx(s.c + s.dir * s.w, abs=2e-4)                 # and out the other side
    assert np.abs(P["u"] - notes[0][1]).max() < 0.06                           # small strokes about the string ...
    assert np.abs(np.diff(P["u"])).max() / 0.0005 < 1.0                        # ... with no jumps (m/s)


def test_crossing_strings_the_pick_passes_over_the_others():
    """E, G, A, D, E: each stroke strikes its own string; whenever the tip is over another string it is clear of it."""
    notes = on_strings([(10.0, 4), (10.25, 1), (10.5, 3), (10.75, 2), (11.0, 4)])
    st, ts, P = run(notes)
    p = SM.params(SPEC)
    for s in st:
        assert at(P, ts, s.t)[0] == pytest.approx(s.c - s.dir * s.w, abs=2e-4)
    for string in (1, 2, 3, 4):
        c, w = next((n[1], n[2]) for n in notes if n[3] == string)
        near = np.abs(P["u"] - c) < w + 0.0005
        mine = np.isin(P["stroke"], [k for k, s in enumerate(st) if s.string == string])
        clear = P["h"][near & ~mine]
        assert len(clear) == 0 or clear.min() > 0.5 * p["lift"], string        # over a string it does not play
    assert np.abs(np.diff(P["u"])).max() / 0.0005 < 3.0


def test_the_hand_comes_in_from_rest_lifted_over_the_strings_and_goes_back():
    notes, _ = picked([(5.0, 45), (5.2, 47)])
    st, ts, P = run(notes)
    p = SM.params(SPEC)
    win = SM.window(st, **p)
    approach = (ts > win[0]) & (ts < st[0].t - st[0].lead)
    assert P["h"][approach].max() > p["lift"] + 0.5 * p["return_lift"]          # lifted on the way in
    assert (P["u"][ts < win[0] - 1e-9] == 0).all() and (P["u"][ts > win[1] + 1e-9] == 0).all()
    assert abs(at(P, ts, win[1])[0]) < 1e-3


def test_strums_keep_their_geometry():
    """A strum stroke (no string of its own) still crosses all the strings: centre 0, half-width sigma."""
    assert SM.geometry(SM.Stroke(1.0, 1), 0.02) == (0.0, 0.02)
    assert SM.geometry(SM.Stroke(1.0, 1, c=0.01, w=0.001, string=2), 0.02) == (0.01, 0.001)
    ends = SM.extremes([SM.Stroke(1.0, 1, c=-0.02, w=0.002), SM.Stroke(1.2, -1, c=0.02, w=0.002)], SM.PICK["span"], 0.03)
    assert ends[0][1] == ends[1][0] == pytest.approx(0.02 + 0.5 * SM.PICK["span"])   # the turn clears both strings
    assert SM.extremes([SM.Stroke(1.0, 1), SM.Stroke(1.2, -1)], 0.09, 0.0225)[0][1] == pytest.approx(0.045)
