"""Ring type (mkmmd/core/typering.py and `layout = {kind = "ring"}` in mkmmd/core/wordtype.py): words round circles about a
point, reading round them, spaced by their width, a ring per repeat of the phrase or each ring filled before the next,
the rings turning in turn. The words here are made-up filler; the geometry checks rasterise each word's footprint (its
padded ink box bent round its ring, as the text stage bends it) instead of trusting the layout's own sums."""
import json
import math

import numpy as np
import pytest

from mkmmd.core import typefx as FX
from mkmmd.core import typering as TR
from mkmmd.core import wordtype as L

FPS = 30.0


def opts(**kw):
    return TR.options(kw)


def boxes_of(widths, y0=-0.2, y1=0.75):
    return [(float(w), y0, y1) for w in widths]


def footprint(word, ring, box, pad, bend=True, face="out", t_turn=0.0, n=24):
    """Points (m, 2) round the outline of a word's padded box as it stands on its ring, the ring turned `t_turn` degrees
    clockwise: bent round the ring (the letters follow it) or straight on its tangent."""
    w, y0, y1 = box
    hw, hh = 0.5 * w + pad, 0.5 * (y1 - y0) + pad
    xs = np.concatenate([np.linspace(-hw, hw, n), np.full(n, hw), np.linspace(hw, -hw, n), np.full(n, -hw)])
    ys = np.concatenate([np.full(n, -hh), np.linspace(-hh, hh, n), np.full(n, hh), np.linspace(hh, -hh, n)])
    R = ring["radius"]
    s = 1.0 if face == "out" else -1.0
    phi = math.radians(word["angle"] + t_turn)
    if bend:                                         # x along the ring line (clockwise for out), y away from the centre
        a = phi + s * xs / R
        r = R + s * ys
    else:                                            # straight on the tangent
        tx, ty = math.cos(phi) * s, -math.sin(phi) * s
        px, py = R * math.sin(phi), R * math.cos(phi)
        nx, ny = math.sin(phi) * s, math.cos(phi) * s
        X = px + xs * tx + ys * nx
        Y = py + xs * ty + ys * ny
        return np.stack([X, Y], 1)
    return np.stack([r * np.sin(a), r * np.cos(a)], 1)


def inside(poly, pts):
    """Even-odd point in polygon for many points."""
    x, y = pts[:, 0], pts[:, 1]
    out = np.zeros(len(pts), bool)
    px, py = poly[:, 0], poly[:, 1]
    for i in range(len(poly)):
        x0, y0, x1, y1 = px[i - 1], py[i - 1], px[i], py[i]
        hit = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-300) + x0)
        out ^= hit
    return out


def overlaps(polys, grid=500):
    """Pairs of footprints that share a pixel of a fine grid over them (each polygon tested only inside its own box)."""
    allp = np.concatenate(polys)
    lo, hi = allp.min(0), allp.max(0)
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    cells = []
    for p in polys:
        near = np.flatnonzero(np.all((P >= p.min(0)) & (P <= p.max(0)), axis=1))
        cells.append(set(near[inside(p, P[near])].tolist()))
    return [(i, j) for i in range(len(polys)) for j in range(i + 1, len(polys)) if cells[i] & cells[j]]


# ------------------------------------------------------------------------------------------------------ placement
def test_word_centres_sit_on_their_ring_about_the_centre():
    o = opts(center=[0.2, -0.1], rings="fill", radius=0.3)
    res = TR.layout_ring(boxes_of([2.0, 1.2, 3.1, 0.8, 2.4, 1.6, 2.2, 1.0, 2.8]), [1] * 9, (2.0, 1.2), 0.9, None, 0.7, o)
    cx, cy = res["center"]
    assert (cx, cy) == pytest.approx((0.4, -0.12))                     # shares of the panel from its centre
    for wd in res["words"]:
        R = res["rings"][wd["ring"]]["radius"]
        x, y = wd["offset"]
        assert math.hypot(x - cx, y - cy) == pytest.approx(R, rel=1e-9)
        assert math.degrees(math.atan2(x - cx, y - cy)) % 360 == pytest.approx(wd["angle"] % 360, abs=1e-6)


def test_out_reads_clockwise_with_the_tops_out_and_in_reads_counter_clockwise_with_the_tops_in():
    for face, sign in (("out", 1.0), ("in", -1.0)):
        res = TR.layout_ring(boxes_of([1.5, 2.0, 1.0, 2.5]), [1] * 4, (1.0, 1.0), 0.9, None, 0.7, opts(face=face))
        ang = np.unwrap(np.radians([w["angle"] for w in res["words"]]))
        assert np.all(sign * np.diff(ang) > 0), face                     # word k + 1 follows word k round the ring
        for w in res["words"]:
            up = (-math.sin(math.radians(w["orient"])), math.cos(math.radians(w["orient"])))    # the word's own up
            x, y = w["offset"]
            radial = (x / math.hypot(x, y), y / math.hypot(x, y))
            assert sign * (up[0] * radial[0] + up[1] * radial[1]) == pytest.approx(1.0, abs=1e-9)
        even = TR.layout_ring(boxes_of([1.8] * 4), [1] * 4, (1.0, 1.0), 0.9, 0.02, 0.7, opts(face=face))["words"]
        a = np.radians([w["angle"] for w in even])                          # a line short of a full ring is centred on start
        assert math.degrees(math.atan2(np.sin(a).sum(), np.cos(a).sum())) % 360 == pytest.approx(
            0.0 if face == "out" else 180.0, abs=1e-6)


def test_long_words_take_more_of_the_ring_and_the_clear_gap_between_neighbours_is_even():
    widths = [1.0, 4.0, 1.0, 2.5]
    o = opts(gap=0.3, rings="line")
    res = TR.layout_ring(boxes_of(widths), [1] * 4, (1.0, 1.0), 0.9, None, 0.7, o)
    R = res["rings"][0]["radius"] / res["em"]                                # em
    ang = np.unwrap(np.radians([w["angle"] for w in res["words"]]))
    clear = [R * (ang[j + 1] - ang[j]) - 0.5 * (widths[j] + widths[j + 1]) for j in range(3)]
    assert clear == pytest.approx([clear[0]] * 3, rel=1e-9)                   # the same clear arc whatever the widths
    assert clear[0] >= 0.3
    assert ang[1] - ang[0] > ang[3] - ang[2]                                 # the long word pushes its neighbour further


def test_each_repeat_adds_a_ring_further_out_and_no_two_stickers_ever_touch():
    widths = [2.0, 1.0, 3.0, 1.5, 2.0, 1.0, 3.0, 1.5, 2.2, 0.8, 2.9, 1.7, 1.2]
    lines = [1] * 4 + [2] * 4 + [3] * 5
    pad = 0.14
    for bend in (True, False):
        o = opts(rings="line", bend=bend, turn=12.0)
        res = TR.layout_ring(boxes_of(widths), lines, (1.78, 1.0), 0.9, None, 0.7, o, pads=[pad] * 13)
        assert [w["ring"] for w in res["words"]] == [0] * 4 + [1] * 4 + [2] * 5
        radii = [r["radius"] for r in res["rings"]]
        assert radii == sorted(radii) and len(set(radii)) == 3
        assert [r["turn"] for r in res["rings"]] == [12.0, -12.0, 12.0]      # alternate directions
        em = res["em"]
        for turn_t in (0.0, 1.7, 9.3):                                       # rings turned against each other
            polys = [footprint(w, res["rings"][w["ring"]], (wd * em, -0.2 * em, 0.75 * em), pad * em, bend,
                               t_turn=res["rings"][w["ring"]]["turn"] * turn_t)
                     for w, wd in zip(res["words"], widths)]
            assert overlaps(polys, 420) == [], (bend, turn_t)


def test_fill_winds_round_each_ring_until_it_is_full_then_goes_on_round_the_next():
    widths = [2.0, 1.0, 3.0, 1.5] * 5
    o = opts(rings="fill", radius=0.3)
    res = TR.layout_ring(boxes_of(widths), [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], (1.0, 1.0),
                         0.9, None, 0.7, o, pads=[0.14] * 20)
    rings = [w["ring"] for w in res["words"]]
    assert rings == sorted(rings) and rings[-1] >= 2                          # in reading order, ring after ring
    assert all(r["full"] for r in res["rings"][:-1]) and not res["rings"][-1]["full"]
    em = res["em"]
    for i, r in enumerate(res["rings"][:-1]):                               # a full ring closes evenly ...
        ks = r["words"]
        ang = np.unwrap(np.radians([res["words"][k]["angle"] for k in ks] + [res["words"][ks[0]]["angle"]]))
        rho = r["radius"] / em
        gaps = [rho * (ang[j + 1] - ang[j] if j < len(ks) - 1 else ang[-1] - ang[-2] + 0) for j in range(len(ks) - 1)]
        clear = [g - 0.5 * (widths[ks[j]] + widths[ks[j + 1]]) for j, g in enumerate(gaps)]
        assert max(clear) - min(clear) < 1e-6
        nxt = res["rings"][i + 1]["words"][0]                               # ... and the next word would not have fitted
        used = sum(TR._extent(boxes_of([widths[k]])[0], 0.14, rho, True, 0.3)[0] for k in ks)
        assert used + TR._extent(boxes_of([widths[nxt]])[0], 0.14, rho, True, 0.3)[0] > 2 * math.pi
    polys = [footprint(w, res["rings"][w["ring"]], (wd * em, -0.2 * em, 0.75 * em), 0.14 * em)
             for w, wd in zip(res["words"], widths)]
    assert overlaps(polys, 420) == []


def test_the_rings_fit_the_room_round_the_centre_and_a_size_caps_the_type():
    widths = [2.0, 1.0, 3.0, 1.5] * 3
    lines = [1] * 4 + [2] * 4 + [3] * 4
    for center, fit in (([0.0, 0.0], 0.9), ([0.25, 0.1], 0.8)):
        o = opts(center=center, rings="line")
        res = TR.layout_ring(boxes_of(widths), lines, (1.6, 0.9), fit, None, 0.7, o, pads=[0.14] * 12)
        room = fit * min(0.8 - abs(center[0] * 1.6), 0.45 - abs(center[1] * 0.9))
        em = res["em"]
        outer = max(res["rings"][w["ring"]]["radius"] + (0.475 + 0.14) * em for w in res["words"])
        assert outer <= room * (1 + 1e-6) and outer > 0.97 * room              # as big as fits
    small = TR.layout_ring(boxes_of(widths), lines, (1.6, 0.9), 0.9, 0.01, 0.7, opts(rings="line"))
    assert small["em"] == pytest.approx(0.01 / 0.7)


def test_a_line_never_wraps_into_itself_on_its_ring():
    res = TR.layout_ring(boxes_of([3.0] * 9 + [1.0]), [1] * 9 + [2], (1.0, 1.0), 0.9, None, 0.7, opts(radius=0.2))
    rho = res["rings"][0]["radius"] / res["em"]
    assert sum(TR._extent(b, 0.0, rho, True, 0.3)[0] for b in boxes_of([3.0] * 9)) <= 2 * math.pi * (1 + 1e-9)


def test_ring_errors_carry_numbers_only():
    with pytest.raises(TR.RingError, match="leaves no room"):
        TR.layout_ring(boxes_of([1.0]), [1], (1.0, 1.0), 0.9, None, 0.7, opts(radius=1.2))
    with pytest.raises(TR.RingError, match="needs a panel"):
        TR.layout_ring(boxes_of([1.0]), [1], None, 0.9, None, 0.7, opts())
    for bad, frag in (({"rings": "spiral"}, "rings is"), ({"face": "up"}, "face is"), ({"radius": 0}, "radius"),
                      ({"spin": 3}, "unknown"), ({"turn": []}, "turn"), ({"center": [1]}, "center")):
        with pytest.raises(TR.RingError, match=frag):
            TR.options(bad)
    assert TR.is_target("cast:rin.head") and TR.is_target([0.0, 0.0, 1.4]) and not TR.is_target([0.1, 0.2])


# ------------------------------------------------------------------------------------------------------ the lyrics entry
def timeline():
    """Three lines of filler words, the same phrase sung three times: 1.0-2.6 s, 4.0-5.6 s, 8.0-9.6 s."""
    fill = ["aa", "bbbb", "cc", "dddddd"]
    lines = []
    for t in (1.0, 4.0, 8.0):
        ws = [{"text": f, "start": t + 0.4 * i, "end": t + 0.4 * i + 0.3, "voiced_end": t + 0.4 * i + 0.3}
              for i, f in enumerate(fill)]
        lines.append({"words": ws, "start": t, "end": t + 1.6})
    return {"lines": lines, "vocal_db": [-8.0] * 400}


def measure(s, font=None, tracking=None):
    return {"w": 0.55 * len(s), "y0": -0.2, "y1": 0.7, "cap": 0.7}


def ring_entry(**layout):
    return {"name": "r", "screen": True, "outline": {"color": "#ffffff", "width": 0.14, "alpha": 1.0},
            "lyrics": {"timeline": "t.json", "lines": [1, 3], "style": "slap", "to": 11.0,
                       "layout": {"kind": "ring", **layout}}}


def test_a_phrase_sung_three_times_accumulates_ring_by_ring_and_every_word_stays_to_the_cut():
    tl = timeline()
    specs, summ = L.expand(ring_entry(turn=10.0), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    assert summ["layout"] == "ring" and summ["words"] == 12
    assert [r["words"] for r in summ["rings"]] == [4, 4, 4]
    assert [r["first_frame"] for r in summ["rings"]] == [30, 120, 240]          # each ring appears with its repeat
    assert all(round(s["kinetic"]["show"][1] * FPS) == 330 for s in specs)      # nothing replaces anything: all up to the cut
    radii = [r["radius"] for r in summ["rings"]]
    assert radii == sorted(radii)
    blob = json.dumps(summ)
    assert not any(f in blob for f in ("aa", "bbbb", "dddddd"))


def test_a_ring_turns_rigidly_from_its_first_word_and_each_word_lands_on_its_turning_place():
    tl = timeline()
    specs, summ = L.expand(ring_entry(turn=[10.0, -4.0, 6.0]), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    for ring, (t_ref, turn) in enumerate(((1.0, 10.0), (4.0, -4.0), (8.0, 6.0))):
        words = [s for s in specs if s["lyric"][0] == ring + 1]
        for s in words:
            kin = FX.kinetic_spec(s)                                       # the text stage takes what the expansion made
            keys = kin["orbit"]["keys"]
            t_on = kin["show"][0]
            assert keys[0] == pytest.approx((t_on, -turn * (t_on - math.floor(t_ref * FPS + 0.4) / FPS)), abs=1e-9)
            if len(keys) > 1:                                                # linear: two keys are the whole turn
                assert len(keys) == 2 and (keys[1][1] - keys[0][1]) / (keys[1][0] - keys[0][0]) == pytest.approx(-turn)
            assert kin["bend"]["side"] == 1 and kin["bend"]["radius"] == pytest.approx(summ["rings"][ring]["radius"], abs=1e-5)
            assert kin["orbit"]["center"] == [0.0, 0.0]


def test_ring_words_face_their_ring_and_keep_the_entrys_random_tilt_on_top():
    tl = timeline()
    e = ring_entry()
    e["lyrics"]["style"] = "none"                                          # no arrival: the turn is all the rot there is
    plain, _ = L.expand(e, tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    e["lyrics"]["tilt"] = {"random": 6}
    tilted, _ = L.expand(e, tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    for a, b in zip(plain, tilted):
        (ra,), (rb,) = [v for _, v in a["kinetic"]["rot"]], [v for _, v in b["kinetic"]["rot"]]
        x, y = a["offset"]
        assert ra == pytest.approx(-math.degrees(math.atan2(x, y)), abs=1e-9)  # tops out: rot = -angle from the top
        assert 1e-6 < abs(rb - ra) <= 6.0 + 1e-9                               # the sticker's own tilt on top


def test_a_ring_round_a_point_of_the_scene_asks_the_build_to_track_it():
    tl = timeline()
    specs, summ = L.expand(ring_entry(center="cast:rin.head"), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    assert summ["ring_track"] == {"target": "cast:rin.head", "center": [0.0, 0.0]}
    assert all(s["kinetic"]["orbit"]["center"] == [0.0, 0.0] for s in specs)


def test_ring_layout_errors_name_the_entry_and_the_key():
    tl = timeline()
    with pytest.raises(L.LyricsError, match=r"text 'r': lyrics: layout 'ring' keys \['spin'\] are unknown"):
        L.expand(ring_entry(spin=2), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    with pytest.raises(L.LyricsError, match="leaves no room"):
        L.expand(ring_entry(radius=2.0), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=measure)
    with pytest.raises(L.LyricsError, match="measures the words"):
        L.expand(ring_entry(), tl, fps=FPS, frame0=91, panel=(1.7, 0.92), measure=None)
