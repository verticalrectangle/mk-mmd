"""Toy alphabet blocks (mkmmd/core/blocktype.py, `[[text]]` with `lyrics.style = "blocks"` or a label in `blocks`): one block
per letter or digit, landing on its word's onset after a ballistic fall from above the picture, bouncing, hopping or slammed
on a lane's hits, the rows building up from the bottom, a line tumbling away as the next comes down; the block's mesh (a
rounded cube with raised or printed glyphs, a rim and pictures) facing out. Synthetic timelines of made-up filler words; no
lyric text leaves the plan except as geometry."""
import json
import math
import re

import numpy as np
import pytest

from mkmmd.core import blocktype as BT

FPS = 30.0
FILL = ["abc", "de", "fghi", "jk", "lmn", "o'p"]
BAND = {"panel": (1.6, 0.4), "top": 0.8, "bottom": -0.2}      # a band whose centre is 0.3 below the picture's middle


def timeline():
    """Two lines of filler words: line 1 at 1.0-2.1 s, line 2 at 4.0-5.2 s (its last word has an apostrophe)."""
    spec = [[(1.0, 1.2), (1.25, 1.5), (1.8, 2.1)], [(4.0, 4.3), (4.35, 4.9), (5.0, 5.2)]]
    lines, k = [], 0
    for ws in spec:
        row = []
        for s, e in ws:
            row.append({"text": FILL[k], "start": s, "end": e, "voiced_end": e})
            k += 1
        lines.append({"words": row, "start": ws[0][0], "end": ws[-1][1]})
    return {"lines": lines}


def no_text_in(obj):
    """True when no filler word appears anywhere in the JSON of `obj` (as a word of its own)."""
    blob = json.dumps(obj, default=str)
    return not any(re.search(r"\b" + re.escape(w) + r"\b", blob) for w in FILL)


def entry(blocks=None, **lyr):
    base = {"timeline": "t.json", "style": "blocks", "to": 9.0}
    if "lines" not in lyr:
        base["line"] = 1
    base.update(lyr)
    return {"name": "bk", "screen": True, "lyrics": base, "blocks": blocks or {}}


def run(e, **kw):
    args = dict(BAND, fps=FPS, frame0=31, tl=timeline(), screen=True)
    args.update(kw)
    return BT.plan(e, **args)


def at(b, frame):
    """(loc, quat, scale) of block `b` on clip frame `frame`."""
    i = int(np.searchsorted(b.keys["frames"], frame))
    assert b.keys["frames"][i] == frame
    return b.keys["loc"][i], b.keys["quat"][i], b.keys["scale"][i]


def panel_y(p, loc):
    return float((p.matrix @ np.array([*loc, 1.0]))[1])


def by_tag(p):
    return {b.tag: b for b in p.blocks}


# ------------------------------------------------------------------------------------------------------ the blocks
def test_one_block_per_letter_or_digit_tagged_by_line_word_and_letter():
    p = run(entry(lines=[1, 2]))
    tags = [b.tag for b in p.blocks]
    assert tags[:5] == ["l1w1c1", "l1w1c2", "l1w1c3", "l1w2c1", "l1w2c2"]
    assert len(tags) == sum(len(BT.letters_of(w)) for w in FILL) == 16            # the apostrophe gets no block
    assert all(re.fullmatch(r"l\d+w\d+c\d+", t) for t in tags)
    assert [b.lyric for b in p.blocks if b.tag.startswith("l2w3")] == [(2, 3), (2, 3)]
    assert p.summary["blocks"] == 16 and p.summary["words"] == 6 and p.summary["lines"] == [1, 2]


def test_a_word_lands_on_its_onset_frame_its_letters_a_stagger_apart_squashed_on_the_impact():
    p = run(entry())
    b = by_tag(p)
    assert [b[f"l1w1c{j}"].land for j in (1, 2, 3)] == [30, 31, 32]            # floor(1.0 * 30 + 0.4), then 0.03 s apart
    assert b["l1w2c1"].land == math.floor(1.25 * FPS + 0.4) and b["l1w3c1"].land == math.floor(1.8 * FPS + 0.4)
    for blk in p.blocks:
        loc, q, s = at(blk, blk.land)
        assert s[1] == pytest.approx(1.0 - 0.12, abs=1e-9)                         # the whole squash: a fall from above
        assert loc[1] - 0.5 * s[1] == pytest.approx(blk.rest[1] - 0.5, abs=1e-9)    # its bottom on its floor
        assert q == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-9)                 # flat on the impact
        before, q0, _ = at(blk, blk.land - 1)
        assert before[1] > blk.rest[1] + 0.5 and abs(q0[0]) < 1.0                 # still falling, tilted


def test_the_fall_is_ballistic_from_just_above_the_picture():
    p = run(entry())
    g = BT.FALL["g"]
    for blk in p.blocks:
        t_land = blk.land / FPS
        t0 = t_land - math.sqrt(2.0 * blk.height / g)
        fr = blk.keys["frames"]
        fall = fr < blk.land
        t = fr[fall] / FPS
        want = blk.rest[1] + blk.height - 0.5 * g * (t - t0) ** 2
        assert blk.keys["loc"][fall, 1] == pytest.approx(want, abs=1e-9)
        assert blk.on == math.ceil(t0 * FPS - 1e-6)
        first = blk.keys["loc"][0]
        assert panel_y(p, first) - 0.5 * p.edge >= BAND["top"]                     # it drops in from out of the picture


def test_bounces_shrink_by_the_restitution_squared_and_come_to_rest():
    p = run(entry(blocks={"fall": {"bounce": 0.4}}))
    blk = p.blocks[0]
    fr, y = blk.keys["frames"], blk.keys["loc"][:, 1] - blk.rest[1]
    after = (fr > blk.land) & (fr <= blk.land + 8)
    peak = y[after].max()
    assert peak == pytest.approx(0.4 ** 2 * blk.height, rel=0.15)               # h1 = e^2 H
    rest = fr >= blk.land + 30
    assert np.abs(y[rest]).max() < 1e-6 and np.abs(blk.keys["scale"][rest] - 1.0).max() < 1e-3


def test_a_jump_on_a_lane_hops_every_standing_block_rippling_along_the_row():
    hits = {"kick": [{"t": 3.0, "db": -2.0}, {"t": 6.0, "db": -30.0}]}
    p = run(entry(blocks={"hits": {"lane": "kick", "move": "jump", "min_db": -10, "height": 0.8, "wave": 0.02}}),
            hits=hits)
    assert p.summary["hits"] == {"jump": 1} and p.summary["moves"] == {"jump": 9}
    left = min(p.blocks, key=lambda b: b.rest[0])
    right = max(p.blocks, key=lambda b: b.rest[0])
    hop = lambda b, f: at(b, f)[0][1] - b.rest[1]
    assert hop(left, 90) == pytest.approx(0.0, abs=1e-9) and hop(left, 92) > 0.3
    assert max(hop(left, f) for f in range(90, 100)) == pytest.approx(0.8, rel=0.12)
    assert hop(right, 92) < hop(left, 92)                                           # the ripple reaches it later
    assert all(m[0] == "jump" for b in p.blocks for m in b.moves)


def test_a_slam_lifts_the_pile_before_the_hit_and_presses_it_flat_on_the_hits_frame():
    p = run(entry(blocks={"layout": "stack", "hits": {"lane": "sub", "move": "slam", "squash": 0.3}}), hits={"sub": [3.0]})
    low = [b for b in p.blocks if b.rest[1] == 0.5][0]
    high = [b for b in p.blocks if b.rest[1] == 1.5][0]
    for b in (low, high):
        assert at(b, 89)[0][1] - b.rest[1] > 0.25                                  # lifted a moment before the hit
    (l0, _, s0), (l1, _, s1) = at(low, 90), at(high, 90)
    assert s0[1] == pytest.approx(0.7, abs=1e-9) and s1[1] == pytest.approx(0.7, abs=1e-9)
    assert l0[1] - 0.5 * s0[1] == pytest.approx(0.0, abs=1e-9)                     # slammed onto the floor on the frame
    assert l1[1] - 0.5 * s1[1] == pytest.approx(l0[1] + 0.5 * s0[1], abs=1e-9)    # the pile shortens: no gap opens


def test_a_hit_while_a_block_is_still_falling_or_bouncing_moves_nothing():
    p = run(entry(blocks={"hits": [{"lane": "kick", "move": "jump"}, {"lane": "sub", "move": "slam"}]}),
            hits={"kick": [0.95], "sub": [1.05]})
    assert all(not b.moves for b in p.blocks) and p.summary["moves"] == {}


def test_next_tumble_knocks_a_line_off_as_the_next_line_comes_down_in_its_place():
    p = run(entry(lines=[1, 2], blocks={"next": "tumble"}))
    one = [b for b in p.blocks if b.lyric[0] == 1]
    two = [b for b in p.blocks if b.lyric[0] == 2]
    first = min(two, key=lambda b: b.land)
    t_new = first.land / FPS - math.sqrt(2.0 * first.height / BT.FALL["g"])
    assert p.summary["tumbles"] == [{"t": round(t_new, 4), "blocks": len(one)}]
    assert max(b.off for b in one) < min(b.land for b in two) + 2 * FPS          # gone soon after
    for b in one:
        loc = b.keys["loc"][-1]
        side = np.sign(b.rest[0] - first.rest[0])
        if abs(b.rest[0] - first.rest[0]) > 0.3:
            assert np.sign(loc[0] - b.rest[0]) == side                              # away from where the new line lands
        assert panel_y(p, loc) < BAND["bottom"] + p.edge                           # out of the bottom of the picture
    lo = lambda bs: (min(b.rest[1] for b in bs), max(b.rest[1] for b in bs))
    assert lo(one)[0] == lo(two)[0] == 0.5                                          # both on the floor: the same place


def test_next_stack_builds_each_line_on_top_of_the_last():
    p = run(entry(lines=[1, 2]))
    one = [b.rest[1] for b in p.blocks if b.lyric[0] == 1]
    two = [b.rest[1] for b in p.blocks if b.lyric[0] == 2]
    assert min(two) > max(one) and min(one) == 0.5 and p.summary["tumbles"] == []


def test_a_crash_tumbles_every_block_out_of_the_picture_spinning():
    p = run(entry(blocks={"crash": 3.0, "tumble": {"gap": 0.0}}))
    assert p.summary["tumbles"] == [{"t": 3.0, "blocks": 9}]
    for b in p.blocks:
        assert b.off < 9.0 * FPS and b.off <= math.ceil((3.0 + BT.TUMBLE["life"]) * FPS)
        last = b.keys["loc"][-1]
        assert panel_y(p, last) + BT.REACH * p.edge >= BAND["bottom"]              # still in on its last frame...
        assert abs(b.keys["quat"][-1][0]) < 0.999                                   # ...turning
        q = b.keys["quat"]
        assert (np.einsum("ij,ij->i", q[1:], q[:-1]) >= 0).all()                    # one hemisphere: blends the short way


def test_in_the_scene_a_crashed_block_lands_on_the_floor_slides_and_lies_flat_on_a_face():
    e = {"name": "tw", "at": [0, 0, 0], "facing": [0, -1, 0], "box": [2.0, 2.0],
         "lyrics": {"timeline": "t.json", "line": 1, "style": "blocks", "to": 9.0},
         "blocks": {"layout": "stack", "crash": 3.0}}
    p = BT.plan(e, fps=FPS, frame0=31, tl=timeline(), panel=(2.0, 2.0))
    assert p.summary["tumbles"] == [{"t": 3.0, "blocks": 9}]
    for b in p.blocks:
        assert b.off == 270                                                         # it lies there until the cut
        after = b.keys["frames"] >= 90
        for loc, q in zip(b.keys["loc"][after], b.keys["quat"][after]):
            low = loc[1] - 0.5 * np.abs(BT._qmat(q)[1]).sum()
            assert low > -0.005                                                     # never through the floor
        R = BT._qmat(b.keys["quat"][-1])
        assert np.abs(R).round(6).tolist() in np.abs(BT.CUBE_TURNS).tolist()       # flat on one of its faces
        assert b.keys["loc"][-1][1] == pytest.approx(0.5, abs=1e-6)
        assert b.keys["loc"][-1] == pytest.approx(b.keys["loc"][-20], abs=1e-6)    # at rest


def test_the_blocks_fill_the_panel_s_fit_and_a_size_caps_them():
    p = run(entry(), panel=(1.6, 0.4))
    corners = np.array([[*(np.asarray(b.rest) + c), 1.0] for b in p.blocks for c in BT.CORNERS]) @ p.matrix.T
    w, h = np.ptp(corners[:, 0]), np.ptp(corners[:, 1])
    assert max(w / 1.6, h / 0.4) == pytest.approx(0.9, abs=1e-6) and p.summary["fits"]
    assert abs(corners[:, 1].min() + corners[:, 1].max()) < 1e-9                   # centred (valign middle)
    assert run({**entry(), "size": 0.01}).edge == pytest.approx(0.01)
    assert run({**entry(), "size": 5.0, "fit": 0.9}).edge == pytest.approx(p.edge)
    assert (corners[:, 2] >= -1e-9).all()                                           # nothing pokes behind the panel


def test_auto_rows_set_a_line_in_more_rows_on_a_tall_panel_than_on_a_wide_one():
    wide = run(entry(lines=[1, 2]), panel=(1.6, 0.4))
    tall = run(entry(lines=[1, 2]), panel=(0.4, 1.6))
    assert tall.summary["rows"][0] > wide.summary["rows"][0] >= 1
    assert run(entry(lines=[1, 2], blocks={"rows": 3})).summary["rows"] == [3, 3]


def test_a_label_reads_top_to_bottom_and_lands_bottom_row_first_left_to_right():
    e = {"name": "lb", "text": "AB\nCDE", "blocks": {"land": 1.0}}
    p = BT.plan(e, fps=FPS, panel=(1.0, 1.0), end=200)
    b = by_tag(p)
    assert sorted(b) == [f"c{i}" for i in range(1, 6)]
    assert b["c1"].rest[1] > b["c3"].rest[1] and b["c1"].rest[0] < b["c2"].rest[0]
    assert [b[t].land for t in ("c3", "c4", "c5", "c1", "c2")] == [30, 31, 32, 33, 34]
    assert all(x.lyric is None and x.off == 200 for x in p.blocks) and p.summary["label"]


# ------------------------------------------------------------------------------------------------------ errors and privacy
@pytest.mark.parametrize("change, frag", [
    ({"blocks": {"colour": "love"}}, "blocks: unknown keys ['colour']"),
    ({"color": "love"}, "['color'] do not apply to blocks"),
    ({"lyrics": {"timeline": "t", "line": 1, "style": "blocks", "layout": "flow"}}, "lyrics keys ['layout'] do not apply"),
    ({"lyrics": {"timeline": "t", "line": 1, "style": "pop"}}, "lyrics style is 'blocks'"),
    ({"blocks": {"hits": {"lane": "kick", "turn": 90}}}, "whole turns"),
    ({"blocks": {"hits": {"lane": "kick", "move": "stomp"}}}, "move is one of"),
    ({"blocks": {"fall": {"bounce": 1.5}}}, "blocks.fall.bounce"),
    ({"blocks": {"view": [90, 0]}}, "blocks.view pitch"),
    ({"blocks": {"land": 1.0}}, "blocks.land is a label's"),
])
def test_mistakes_are_errors_that_name_the_key(change, frag):
    e = entry()
    e.update(change)
    with pytest.raises(BT.BlockError, match=re.escape(frag)) as err:
        run(e)
    assert str(err.value).startswith("text 'bk': ") and no_text_in(str(err.value))


def test_errors_about_words_carry_numbers_never_letters():
    with pytest.raises(BT.BlockError) as err:
        run(entry(), draws=lambda c: c != "e")
    assert "letter 2 of word (1, 2)" in str(err.value) and no_text_in(str(err.value))
    with pytest.raises(BT.BlockError, match="reads the \\[audio\\] hits file"):
        run(entry(blocks={"hits": {"lane": "kick"}}))
    with pytest.raises(BT.BlockError, match="no lane 'kick'"):
        run(entry(blocks={"hits": {"lane": "kick"}}), hits={"snare": [1.0]})
    with pytest.raises(BT.BlockError, match="give `size`"):
        BT.plan({"name": "lb", "text": "AB"}, fps=FPS)


def test_no_lyric_text_in_the_summary_the_tags_or_any_repr():
    p = run(entry(lines=[1, 2], blocks={"next": "tumble"}), hits={"kick": [3.0]})
    assert no_text_in(p.summary) and no_text_in([b.tag for b in p.blocks])
    assert no_text_in([repr(b) for b in p.blocks] + [repr(b.mesh) for b in p.blocks] + [repr(u) for u in BT._lyric_units(
        entry(), timeline(), FPS, 0)[0]])


def test_is_blocks_needs_a_blocks_table_or_the_blocks_style():
    assert BT.is_blocks({"text": "AB", "blocks": True}) and BT.is_blocks({"lyrics": {"style": "blocks"}})
    assert not BT.is_blocks({"text": "AB"}) and not BT.is_blocks({"lyrics": {"style": "pop"}, "blocks": False})


def test_thinned_keys_give_back_every_frame_exactly():
    f = np.arange(100, dtype=float)
    v = np.where(f < 30, 1.0, np.where(f < 50, 1.0 + 0.1 * (f - 30), 3.0 + np.sin(f / 7.0)))
    keep = BT.thin(f, v)
    assert len(keep) < 70 and keep[0] == 0 and keep[-1] == 99
    assert np.interp(f, f[keep], v[keep]) == pytest.approx(v, abs=1e-6)


# ------------------------------------------------------------------------------------------------------ geometry
def test_the_rounded_cube_is_a_unit_block_with_flat_painted_squares_and_wooden_rounds():
    r = 0.12
    V, Q, N, which = BT.rounded_cube(r)
    assert np.abs(V).max() == pytest.approx(0.5) and np.linalg.norm(N, axis=1) == pytest.approx(1.0)
    assert (np.einsum("ij,ij->i", N, V) > 0).all()                                # every normal points out
    for fi, f in enumerate(BT.FACES):
        nrm = np.array(BT.FRAMES[f][0], float)
        quad = Q[which == fi]
        assert len(quad) == 1 and (V[quad[0]] @ nrm == pytest.approx(0.5))          # one flat square per face
        a = 0.5 - r
        assert np.abs(V[quad[0]]).max(axis=0) @ np.abs(nrm) == pytest.approx(0.5)
        assert sorted(np.abs(V[quad[0]]).max(axis=0).tolist()) == pytest.approx(sorted([a, a, 0.5]))
    corner = V[np.argmax(V.sum(axis=1))]
    assert np.linalg.norm(corner - (0.5 - r)) == pytest.approx(r, abs=1e-6)      # a sphere's octant at the corner


def square_glyph(c):
    return np.array([[0.0, 0.0], [0.6, 0.0], [0.6, 0.7], [0.0, 0.7]]), np.array([[0, 1, 2], [0, 2, 3]])


@pytest.mark.parametrize("letter", ["raised", "printed"])
def test_a_block_mesh_faces_out_its_glyphs_stand_on_their_faces_and_slots_match_its_colours(letter):
    o = BT.options(entry(blocks={"letter": letter}), screen=True)
    spec, extra = BT.face_plan("a", o)
    m = BT.block_mesh(spec, square_glyph, 0.7)
    at_ = 0
    for size, mat in zip(m.sizes, m.mats):
        P = m.V[m.loops[at_:at_ + size]]
        n = sum(np.cross(P[i], P[(i + 1) % size]) for i in range(size))
        if np.linalg.norm(n) > 1e-12:
            assert min(float(n / np.linalg.norm(n) @ x) for x in m.normals[at_:at_ + size]) > 0.9
        at_ += size
    front = m.V[:, 2].max()
    assert front == pytest.approx(0.5 + (o["relief"] if letter == "raised" else BT.PRINT_LIFT))
    assert m.extra == extra and m.mats.max() == BT.SLOT_EXTRA + len(extra) - 1
    assert set(np.unique(m.mats)) >= {BT.SLOT_WOOD, BT.SLOT_INK, *range(1, 7)}


def test_the_other_faces_show_other_letters_or_pictures_as_the_sides_say():
    spec, extra = BT.face_plan("a", BT.options(entry(), screen=True))
    f = dict(zip(BT.FACES, spec.faces))
    assert f["front"] == ("glyph", "a") and f["back"] == ("glyph", "A") and f["right"] == ("glyph", "b")
    assert f["bottom"][0] == "glyph" and f["bottom"][1].isdigit()
    assert f["top"][0] == f["left"][0] == "picture" and f["top"][1] in BT.PICTURES and extra
    plain, _ = BT.face_plan("a", BT.options(entry(blocks={"sides": "plain"}), screen=True))
    assert plain.faces[0] == ("glyph", "a") and plain.faces[1:] == (None,) * 5
    shy, _ = BT.face_plan("a", BT.options(entry(blocks={"sides": "letters"}), screen=True), draws=lambda c: c == "a")
    assert shy.faces[1:] == (None,) * 5                                            # a glyph the font lacks: a plain face
    o = BT.options(entry(blocks={"paint": ["love", "gold"], "ground": "base"}), screen=True)
    cols = BT._colors(1, spec, extra, o)
    assert cols[0] == o["wood"] and cols[1] == "gold" and cols[BT.SLOT_INK] == o["ink"]
    assert cols[1 + BT.FACES.index("top")] == "base"                                # a picture's face: the ground
