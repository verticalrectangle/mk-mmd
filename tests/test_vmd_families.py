import struct

import numpy as np

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
