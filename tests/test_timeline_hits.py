"""Hit lanes (mkmmd.timeline.hits): drum bands told apart, 16th-note rolls kept apart, quiet leakage dropped, silences
with their length, note pitches, clip times; and `mk timeline hits` writing a hits file from a timeline's cached stems.
No audio files, no Demucs."""
import json

import numpy as np
import pytest
from test_timeline_onsets import SR, FakeSong, hat, kick, run_cli

from mkmmd.timeline import analyze as A
from mkmmd.timeline import hits as H

SIXTEENTH = 60.0 / 158.2 / 4                         # a 16th note at 158 bpm: 95 ms


def snare(x, t0, rng, amp=0.6):
    """Noise band-passed to 200-900 Hz with a 2 ms attack and a 60 ms decay: a snare's body, no lows, no highs."""
    n = int(0.3 * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(f < 200) | (f > 900)] = 0
    b = np.fft.irfft(spec, n)
    i = int(round(t0 * SR))
    env = np.exp(-np.arange(n) / SR / 0.06) * np.minimum(1.0, np.arange(n) / SR / 0.002)
    x[i:i + n] += amp * b / np.abs(b).max() * env


def kit(seconds=8.0, seed=4):
    """A drum stem: kicks on the beats, snares on 2 and 4, hats on the 8ths (offbeats), at 120 bpm from 0.5 s."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(seconds * SR))
    beats = np.arange(0.5, seconds - 0.6, 0.5)
    kicks, snares, hats = beats[::2], beats[1::2], beats[:-1] + 0.25
    for t in kicks:
        kick(x, t, amp=0.8, f=60.0)
    for t in snares:
        snare(x, t, rng)
    for t in hats:
        hat(x, t, rng, amp=0.4)
    return x, kicks, snares, hats


def times(lane):
    return np.array([h["t"] for h in lane])


def near(got, want, tol):
    """Every wanted time has a hit within tol s, and every hit is within tol of a wanted time."""
    got, want = np.asarray(got), np.asarray(want)
    return (len(got) == len(want) and np.abs(got[:, None] - want[None, :]).min(1).max() < tol
            and np.abs(got[:, None] - want[None, :]).min(0).max() < tol)


def test_each_drum_lane_finds_its_own_hits_and_not_the_others():
    x, kicks, snares, hats = kit()
    st = {"drums": np.stack([x, x]).astype(np.float32)}
    hits, meta = H.lanes(st, st["drums"], 0.0, len(x) / SR, gaps={}, notes={})
    assert list(hits) == ["kick", "snare", "hat"] and meta["snare"]["band"] == [180.0, 1000.0]
    assert near(times(hits["kick"]), kicks, 0.006) and near(times(hits["snare"]), snares, 0.006)
    assert near(times(hits["hat"]), hats, 0.004)
    alone = times(H.onset_lane(x, *H.ONSETS["snare"][1:]))            # the kicks' thump is in the snare's band too
    assert len(alone) > len(snares)


def test_a_16th_note_roll_stays_four_hits():
    rng = np.random.default_rng(7)
    x = np.zeros(4 * SR)
    roll = 1.0 + SIXTEENTH * np.arange(4)
    for t in roll:
        snare(x, t, rng)
    assert near(times(H.onset_lane(x, *H.ONSETS["snare"][1:])), roll, 0.008)


def test_levels_are_against_the_loudest_and_leakage_far_below_it_is_dropped():
    rng = np.random.default_rng(8)
    x = np.zeros(5 * SR)
    for t, amp in ((1.0, 0.8), (2.0, 0.8 * 10 ** (-6 / 20)), (3.0, 0.8 * 10 ** (-40 / 20))):
        snare(x, t, rng, amp)
    lane = H.onset_lane(x, *H.ONSETS["snare"][1:])
    assert [round(h["t"]) for h in lane] == [1, 2]                     # the -40 dB one is leakage
    assert lane[0]["db"] == 0.0 and -7.5 < lane[1]["db"] < -4.5


def test_times_are_clip_seconds_inside_the_clip():
    x, kicks, _, _ = kit()
    lane = H.onset_lane(x, *H.ONSETS["kick"][1:], pre=2.0, duration=3.0)
    want = kicks[(kicks >= 2.0) & (kicks <= 5.0)] - 2.0
    assert near(times(lane), want, 0.006) and lane[0]["db"] == 0.0


def test_a_silence_is_a_gap_with_its_length_and_a_short_one_is_not():
    rng = np.random.default_rng(9)
    x = 0.3 * rng.standard_normal(6 * SR)
    x[int(2.0 * SR):int(2.15 * SR)] = 0.0                              # the music stops for 150 ms
    x[int(4.0 * SR):int(4.04 * SR)] *= 0.001                           # a 40 ms dip
    gaps = H.gap_lane(x, 10.0, 0.08)
    assert len(gaps) == 1 and abs(gaps[0]["t"] - 2.0) < 0.025 and abs(gaps[0]["dur"] - 0.15) < 0.04
    assert gaps[0]["db"] < -40
    clipped = H.gap_lane(x, 10.0, 0.08, pre=2.05, duration=2.0)        # starts before the clip: cut to it
    assert len(clipped) == 1 and clipped[0]["t"] == 0.0 and clipped[0]["dur"] < 0.12
    with pytest.raises(H.HitsError):
        H.gap_lane(x, 0.0, 0.08)


@pytest.mark.parametrize("f0, midi", [(440.0, 69.0), (523.25, 72.0), (98.0, 43.0)])
def test_a_note_s_pitch_is_its_fundamental(f0, midi):
    t = np.arange(int(0.3 * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4, 5))   # harmonics louder than nothing at f0
    x = np.concatenate([np.zeros(SR // 2), tone * np.exp(-t / 0.2), np.zeros(SR // 2)])
    assert abs(H.pitch(x, 0.5) - midi) < 0.2
    lane = H.note_lane(x, *H.NOTES["notes"][1:])
    assert len(lane) == 1 and abs(lane[0]["t"] - 0.5) < 0.01 and abs(lane[0]["midi"] - midi) < 0.2


def test_lane_specs_parse_and_bad_ones_say_the_form():
    assert H.parse("clap=drums:1000-4000", "onsets") == ("clap", ("drums", 1000.0, 4000.0))
    assert H.parse("breath=vocals:12:0.15", "gaps") == ("breath", ("vocals", 12.0, 0.15))
    for text, kind, frag in (("clap=drums", "onsets", "NAME=STEM:LO-HI"), ("x=piano:1-2", "onsets", "piano"),
                             ("b=bass:12", "gaps", "DROP_DB")):
        with pytest.raises(H.HitsError, match=frag):
            H.parse(text, kind)
    with pytest.raises(H.HitsError, match="floor"):
        H.onset_lane(np.zeros(SR), 500.0, 100.0)


def test_mk_timeline_hits_writes_the_lanes_of_the_timeline_span(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    song = FakeSong(monkeypatch)                                       # kicks every 0.5 s at 80 Hz from 0.3 s
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    tl = A.analyse(str(audio), start=6.0, duration=8.0, fps=30, cache_dir=str(tmp_path), words=False, context=4.0,
                   log=lambda *a: None)
    path = tmp_path / "audio" / "timeline.json"
    path.parent.mkdir()
    path.write_text(json.dumps(tl), encoding="utf-8")
    code, out = run_cli(["timeline", "hits", "--timeline", str(path), "--context", "4"], capsys)
    assert code == 0, out
    hits = json.loads((tmp_path / "audio" / "hits.json").read_text(encoding="utf-8"))
    assert list(hits) == ["kick", "snare", "hat", "stop", "bass_out", "notes"]
    want = np.arange(0.3, 24 - 0.6, 0.5) - 6.0
    want = want[(want >= 0) & (want <= 8.0)]
    assert near(times(hits["kick"]), want, 0.006) and song.keys[-1] == song.keys[0]   # the cached window again
    assert out["lanes"]["kick"]["count"] == len(want) and out["lanes"]["kick"]["band"] == [35.0, 150.0]

    code, out = run_cli(["timeline", "hits", "--timeline", str(path), "--context", "4", "--lane", "kick=drums:60-100",
                         "--gap", "hush=mix:12:0.2", "--out", str(tmp_path / "h2.json")], capsys)
    assert code == 0 and out["lanes"]["kick"]["band"] == [60.0, 100.0] and out["lanes"]["hush"]["kind"] == "gaps"
    code, out = run_cli(["timeline", "hits", "--timeline", str(path), "--context", "3"], capsys)
    assert code == 2 and "--context" in out["error"]
    code, out = run_cli(["timeline", "hits", "--timeline", str(path), "--lane", "kick=drums"], capsys)
    assert code == 2 and "NAME=STEM:LO-HI" in out["error"]
