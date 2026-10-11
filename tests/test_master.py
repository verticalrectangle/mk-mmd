"""Mastering a clip's audio for small speakers (mkmmd.master, mk post's [audio.master]): on synthetic mixes whose answers
are known. A phone's speakers play little below about 250 Hz, so 808 booms at 100 Hz vanish there."""
import numpy as np
import pytest

from mkmmd import master as MS

SR = MS.SR
HITS = [1.0, 2.0, 3.0]


def _t(seconds):
    return np.arange(int(seconds * SR)) / SR


def _level(x, t0, t1):
    i0, i1 = int(t0 * SR), int(t1 * SR)
    return 20 * np.log10(np.sqrt((np.asarray(x)[i0:i1] ** 2).mean()))


def _mix():
    """Three 808 booms at 100 Hz on HITS, a 100 Hz bass note at 3.4-3.8 s that is no boom, an 8 kHz tone throughout (the
    rest of the mix): stereo."""
    t = _t(4.5)
    x = 0.05 * np.sin(2 * np.pi * 8000 * t)
    for h in HITS:
        d = np.clip(t - h, 0, None)
        x += np.where(t >= h, 0.6 * np.exp(-d / 0.15) * np.sin(2 * np.pi * 100 * d), 0.0)
    x += np.where((t > 3.4) & (t < 3.8), 0.6 * np.sin(2 * np.pi * 100 * t), 0.0)
    return np.stack([x, x], 1)


def test_the_meter_reads_the_standard_s_own_references():
    """ITU-R BS.1770: a 997 Hz sine at full scale in one channel reads -3.01 LUFS, in both 0; quieter by its amplitude. The
    K-weighting hears little of a 30 Hz rumble and about 4 dB more of 10 kHz than of 1 kHz, and a passage more than 10 LU
    under the rest does not drag the loudness down (the relative gate)."""
    t = _t(3.0)
    s = np.sin(2 * np.pi * 997 * t)
    assert MS.loudness(np.stack([0.1 * s, 0.1 * s], 1)) == pytest.approx(-20.0, abs=0.05)
    assert MS.loudness(np.stack([0.1 * s, 0 * s], 1)) == pytest.approx(-23.01, abs=0.05)
    assert MS.loudness(np.stack([0.5 * s, 0.5 * s], 1)) == pytest.approx(-6.02, abs=0.05)
    assert MS.loudness(np.zeros((SR, 2))) == -np.inf
    at = lambda hz: MS.loudness(np.stack([0.1 * np.sin(2 * np.pi * hz * t)] * 2, 1))   # noqa: E731
    assert -12.0 < at(30) - at(997) < -6.0 and 3.0 < at(10000) - at(997) < 5.0
    quiet = np.concatenate([0.1 * s, 0.01 * s])                        # -20 LUFS, then -40
    assert MS.loudness(np.stack([quiet, quiet], 1)) == pytest.approx(-20.0, abs=0.3)


def test_the_booms_get_overtones_a_phone_can_play_and_the_rest_of_the_mix_dips_under_them():
    """Booms at 100 Hz barely register through a phone's speakers; mastered, they stand out by several dB there, a bass
    note that is no boom gets nothing, and the rest of the mix (the 8 kHz tone) dips `duck` dB just after each boom and is
    back after `release`."""
    x = _mix()
    hits = {"808": [{"t": h, "db": 0.0} for h in HITS]}
    y, rep = MS.master(x, SR, hits, {"loudness": -14.0})
    times = [(h, 1.0) for h in HITS]
    before, after = MS.boom_contrast(x, SR, times), MS.boom_contrast(y, SR, times)
    assert before < 2.0 and after > before + 5.0 and rep["boom_phone_db"] == {"before": before, "after": after}
    px, py = MS.phone(x, SR).mean(1), MS.phone(y, SR).mean(1)
    tone_x, tone_y = MS.band(x, SR, lo=6000, hi=10000).mean(1), MS.band(y, SR, lo=6000, hi=10000).mean(1)
    gain = _level(tone_y, 0.3, 0.8) - _level(tone_x, 0.3, 0.8)          # the loudness's own change, away from any boom
    assert _level(py, 3.45, 3.75) - _level(px, 3.45, 3.75) == pytest.approx(gain, abs=1.0)    # the bass note: no overtones
    d = MS.DEFAULTS
    for h in HITS:
        dip = _level(tone_y, h + 0.02, h + 0.05) - _level(tone_y, h - 0.2, h - 0.1)
        back = _level(tone_y, h + 0.04 + d["release"] + 0.1, h + 0.04 + d["release"] + 0.3) - _level(tone_y, h - 0.2, h - 0.1)
        assert dip == pytest.approx(-d["duck"], abs=0.6) and back == pytest.approx(0.0, abs=0.3)


def test_nothing_passes_the_ceiling_the_rumble_goes_and_the_loudness_is_set():
    """A hot mix with a 15 Hz rumble under it: at a calm target the loudness lands where it was asked and the rumble is
    25 dB down against the rest; at a hot one the limiter turns the peaks down so that no sample passes the ceiling, and
    comes back where nothing is near it."""
    x = _mix() * 3.0
    x += 0.3 * np.sin(2 * np.pi * 15 * _t(4.5))[:, None]
    hits = {"808": HITS}
    calm, rep = MS.master(x, SR, hits, {"loudness": -14.0})
    assert MS.loudness(calm) == pytest.approx(-14.0, abs=0.05) and rep["loudness_out"] == pytest.approx(-14.0, abs=0.05)
    rumble = lambda s: _level(MS.band(s, SR, hi=20).mean(1), 0.5, 4.0) - _level(MS.band(s, SR, lo=6000).mean(1), 0.5, 4.0)  # noqa: E731
    assert rumble(calm) < rumble(x) - 25.0
    hot, rep = MS.master(x, SR, hits, {"loudness": -6.0, "ceiling": -1.0})
    assert np.abs(hot).max() <= 10 ** (-1.0 / 20) + 1e-9 and rep["peak_out_dbfs"] <= -1.0 and rep["limiter_db"] > 1.0
    lim, cut = MS.limit(x, SR, -1.0)
    quiet = np.abs(x).max(1) < 0.3                                   # where nothing comes near the ceiling ...
    assert cut > 0 and np.abs(lim[quiet]).sum() / np.abs(x[quiet]).sum() > 0.9    # ... the gain is (nearly) back to 1


@pytest.mark.parametrize("cfg, hits, fragment", [
    ({"lanes": "808"}, {"808": [1.0]}, "unknown key 'lanes'"),
    ({"lane": "kick"}, {"808": [1.0], "snare": [2.0]}, "no such lane \\(have 808, snare\\)"),
    ({"duck": "lots"}, {"808": [1.0]}, "duck = 'lots': a number"),
    ({"ceiling": 1.0}, {"808": [1.0]}, "ceiling at most 0 dBFS"),
    ("hard", {"808": [1.0]}, "a table of"),
])
def test_bad_settings_are_refused_naming_the_fault(cfg, hits, fragment):
    with pytest.raises(MS.MasterError, match=fragment):
        MS.master(np.zeros((SR, 2)), SR, hits, cfg)
