import struct

import numpy as np
import pytest

from mkmmd.core import families, vmd


def _vmd_bytes(model="model", bones=(), morphs=(), camera=()):
    out = b"Vocaloid Motion Data 0002".ljust(30, b"\0") + model.encode("cp932").ljust(20, b"\0")
    out += struct.pack("<I", len(bones))
    for name, frame, pos, quat in bones:
        out += name.encode("cp932").ljust(15, b"\0") + struct.pack("<I7f", frame, *pos, *quat) + bytes(64)
    out += struct.pack("<I", len(morphs))
    for name, frame, w in morphs:
        out += name.encode("cp932").ljust(15, b"\0") + struct.pack("<If", frame, w)
    out += struct.pack("<I", len(camera))
    for frame, dist, pos, rot, fov in camera:
        out += struct.pack("<I7f", frame, dist, *pos, *rot) + bytes(24) + struct.pack("<IB", fov, 0)
    return out


def test_vmd_reads_bones_morphs_camera_sorted(tmp_path):
    p = tmp_path / "m.vmd"
    p.write_bytes(_vmd_bytes(
        "テスト", bones=[("センター", 10, (0, 1, 0), (0, 0, 0, 1)), ("センター", 0, (0, 0, 0), (0, 0, 0, 1)),
                       ("右腕", 5, (0, 0, 0), (0, 0, 0.7071, 0.7071))],
        morphs=[("まばたき", 3, 1.0)], camera=[(0, -45.0, (0, 10, 0), (0, 0, 0), 30)]))
    m = vmd.read(p)
    assert m.model == "テスト"
    fr, pos, quat = m.bones["センター"]
    assert fr.tolist() == [0, 10] and pos[1].tolist() == [0, 1, 0]
    assert m.morphs["まばたき"][0].tolist() == [3]
    assert m.camera["fov"].tolist() == [30.0]
    assert m.n_frames == 11


def test_vmd_travel_in_place_and_moving(tmp_path):
    p = tmp_path / "walk.vmd"
    keys = [("センター", f, (f * 0.5, 0, 0), (0, 0, 0, 1)) for f in range(0, 61, 10)]
    p.write_bytes(_vmd_bytes(bones=keys))
    t = vmd.travel(vmd.read(p))
    assert abs(t["xy_extent_m"] - 30 * vmd.MMD_UNIT_M) < 1e-6 and not t["in_place"]


def test_vmd_tempo_finds_a_clear_beat(tmp_path):
    p = tmp_path / "bounce.vmd"
    period = 15                                        # 120 bpm at 30 fps
    keys = []
    for f in range(0, 600, 1):
        y = -abs(np.sin(np.pi * f / period)) * 2.0     # sharp landings every `period` frames
        keys.append(("センター", f, (0, y, 0), (0, 0, 0, 1)))
    p.write_bytes(_vmd_bytes(bones=keys))
    period_f, phase, score, contrast = vmd.tempo_phase(vmd.read(p), lo_bpm=80, hi_bpm=160)
    assert abs(period_f - period) < 0.5 and contrast > 1.2


def _bounce(tmp_path, bpm, frames=600):
    """A VMD whose centre lands sharply every beat of `bpm` (at 30 fps), from frame 0."""
    period = vmd.MMD_FPS * 60.0 / bpm
    keys = [("センター", f, (0, -abs(np.sin(np.pi * f / period)) * 2.0, 0), (0, 0, 0, 1)) for f in range(frames)]
    p = tmp_path / f"bounce{bpm}.vmd"
    p.write_bytes(_vmd_bytes(bones=keys))
    return vmd.read(p)


def test_beat_fit_lays_per_beat_motion_beats_on_each_song_beat(tmp_path):
    """`[[motion]] retime = "beats"` with `per_beat` (vmd.beat_fit): per_beat motion beats stretch over each song beat
    and the first motion beat lands on the song's beats cut into per_beat parts (2: its 8ths)."""
    mo = _bounce(tmp_path, 120)                                   # a beat every 15 frames
    beat = 0.8225                                                 # a song at 72.95 bpm
    beats = [beat * k for k in range(37)]
    period, src_beat = vmd.tempo_phase(mo)[:2]                    # ~15 frames; the first motion beat after frame 0
    assert period == pytest.approx(15.0, abs=0.05)
    for per_beat, lands in ((1, 1.0), (2, 1.5)):                  # unsnapped 1.4 beats in: beat 1, or the 8th after it
        expect = beat * 30.0 / per_beat / period
        start_t = 1.4 * beat - src_beat * expect / 30.0           # where source frame 0 plays, so the beat falls at 1.4
        scale, start, info = vmd.beat_fit(mo, beats, 30.0, start_t, 0.0, per_beat=per_beat)
        assert scale == pytest.approx(expect, rel=1e-6)
        assert (start + src_beat * scale) / 30.0 == pytest.approx(lands * beat, abs=1e-6)
        assert info["song_bpm"] == 72.95 and info["motion_bpm"] == round(1800.0 / period, 2)
        assert info.get("per_beat", 1) == per_beat


def test_beat_fit_takes_the_motion_s_own_tempo_when_given(tmp_path):
    """A motion faster than the comb filter's 70-180 bpm (a 204 bpm routine): the search settles elsewhere, `bpm`
    gives the beat it was made on."""
    mo = _bounce(tmp_path, 204)
    beats = [0.8225 * k for k in range(37)]
    found = vmd.beat_fit(mo, beats, 30.0, 0.0, 0.0)[2]["motion_bpm"]
    assert abs(found - 204.0) > 20.0
    scale, _, info = vmd.beat_fit(mo, beats, 30.0, 0.0, 0.0, bpm=204, per_beat=2)
    assert info["motion_bpm"] == 204.0
    assert scale == pytest.approx(0.8225 * 30.0 / 2 / (30.0 * 60.0 / 204), rel=1e-6)


def test_beat_fit_needs_two_song_beats(tmp_path):
    with pytest.raises(ValueError, match="no beats"):
        vmd.beat_fit(_bounce(tmp_path, 120), [1.0], 30.0, 0.0, 0.0)


def test_a_mirrored_vmd_plays_each_side_s_keys_on_the_other_reflected_through_the_middle(tmp_path):
    """`[[motion]] mirror = true` (vmd.mirror): 左 and 右 trade bones, positions flip their x and rotations reflect
    through the body's middle plane, as MMD mirrors a motion: twice is the motion itself, and the facial keys stay."""
    raise_l = (0.0, 0.1, 0.4226, 0.9001)                              # the left arm raised, a little turned
    src = _vmd_bytes(bones=[("左腕", 0, (0, 0, 0), raise_l), ("右足ＩＫ", 5, (0.5, 0.0, -1.0), (0, 0, 0, 1)),
                            ("センター", 3, (1.0, 2.0, 3.0), (0.2, 0.3, 0.1, 0.9274))],
                     morphs=[("ウィンク", 4, 1.0)])
    p = tmp_path / "m.vmd"
    p.write_bytes(vmd.mirror(src))
    m = vmd.read(p)
    assert set(m.bones) == {"右腕", "左足ＩＫ", "センター"}
    q = m.bones["右腕"][2][0]
    assert np.allclose(q, [raise_l[0], -raise_l[1], -raise_l[2], raise_l[3]], atol=1e-6)
    assert np.allclose(m.bones["左足ＩＫ"][1][0], [-0.5, 0.0, -1.0]) and np.allclose(m.bones["センター"][1][0], [-1, 2, 3])
    assert np.allclose(m.bones["センター"][2][0], [0.2, -0.3, -0.1, 0.9274], atol=1e-6)
    assert list(m.morphs) == ["ウィンク"]
    assert vmd.mirror(vmd.mirror(src)) == src


def test_families_classify_common_names():
    assert families.classify("後髪1-1.R") == "back_hair"
    assert families.classify("前髪_0_1.L") == "bangs"
    assert families.classify("横髪_2_1.R") == "side_hair"
    assert families.classify("耳_1_1.R") == "ears"
    assert families.classify("スカート_0_3") == "skirt"
    assert families.classify("尻尾1") == "tail"
    assert families.classify("ponytail_02") == "hair"
    assert families.classify("elbow_helper") == "other"
    assert families.classify("Hair_L_01") == "hair"
    assert families.classify("rear_skirt_1") == "skirt"


def test_families_normalise_and_avoid_false_hits():
    assert families.classify("ｽｶｰﾄ_0_0") == "skirt"            # half-width katakana
    assert families.classify("ﾈｸﾀｲ１") == "ribbon"
    assert families.classify("前SK_1") == "skirt" and families.classify("mask_1") == "other"
    assert families.classify("flequillo1") == "bangs" and families.classify("前髮_1") == "bangs"
    assert families.classify("M_chestbow") == "ribbon" and families.classify("胸リボン") == "ribbon"
    assert families.classify("胸_1") == "breasts"
    assert families.classify("earring_1") == "accessory"
    assert families.classify("袴_0_1") == "skirt"
    assert families.classify("サイド0.L") == "side_hair"
