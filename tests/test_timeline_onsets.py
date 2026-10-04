"""Note onsets of the timeline: signal.band_onsets on synthetic plucks, strums, kicks, hats and silence;
analyze.stem_onsets / timeline_onsets / onset_stats; and `mk timeline onsets` merging into an existing timeline.
No audio files, no Demucs."""
import hashlib
import json

import numpy as np
import pytest

from mkmmd.cli import main as MAIN
from mkmmd.timeline import analyze as A
from mkmmd.timeline import signal as S
from mkmmd.timeline import words as W

SR = S.SR
EIGHTH = 0.236                                       # an 8th note at 127 bpm


def attack(n, seconds, smooth=False):
    r = np.minimum(1.0, np.arange(n) / SR / seconds)
    return 0.5 - 0.5 * np.cos(np.pi * r) if smooth else r


def noise_pluck(x, t0, rng, amp=0.3, tau=0.05, rise=0.001):
    i, n = int(round(t0 * SR)), int(0.4 * SR)
    x[i:i + n] += amp * np.exp(-np.arange(n) / SR / tau) * attack(n, rise) * rng.standard_normal(n)


def sine_pluck(x, t0, amp=0.3, f=2000.0, tau=0.12, rise=0.001):
    i, n = int(round(t0 * SR)), int(0.6 * SR)
    t = np.arange(n) / SR
    x[i:i + n] += amp * np.exp(-t / tau) * attack(n, rise) * np.sin(2 * np.pi * f * t)


def kick(x, t0, amp=0.9, f=100.0):
    i, n = int(round(t0 * SR)), int(0.5 * SR)
    t = np.arange(n) / SR
    x[i:i + n] += amp * np.exp(-t / 0.08) * attack(n, 0.005, smooth=True) * np.sin(2 * np.pi * f * t)


def hat(x, t0, rng, amp=0.5, fc=8500.0):
    """Noise high-passed at fc (only above 8 kHz), a 1 ms attack and a 12 ms decay."""
    i, n = int(round(t0 * SR)), int(0.06 * SR)
    spec = np.fft.rfft(rng.standard_normal(n + 8192))
    spec[np.fft.rfftfreq(n + 8192, 1 / SR) < fc] = 0
    h = np.fft.irfft(spec, n + 8192)[4096:4096 + n]
    x[i:i + n] += amp * h / np.abs(h).max() * np.exp(-np.arange(n) / SR / 0.012) * attack(n, 0.001)


def pluck_train(seconds=12.0, kicks=False, hats=False, seed=1):
    """Alternating noise bursts and decaying sines every 8th note (+ a loud 100 Hz kick on each and in each gap, a hat
    every 16th): (signal, pluck times)."""
    rng = np.random.default_rng(seed)
    times = 0.5 + EIGHTH * np.arange(int((seconds - 1.5) / EIGHTH))
    x = np.zeros(int(seconds * SR))
    for k, t0 in enumerate(times):
        noise_pluck(x, t0, rng) if k % 2 == 0 else sine_pluck(x, t0)
        if kicks:
            kick(x, t0 + 0.002)
            kick(x, t0 + EIGHTH / 2)
        if hats:
            hat(x, t0, rng)
            hat(x, t0 + EIGHTH / 2, rng)
    return x, times


def strums(spread, seconds=12.0, click=3.0, seed=5):
    """A strum every 8th note: six staggered notes (a tone with six harmonics and a 4 ms pick click) over `spread` s."""
    rng = np.random.default_rng(seed)
    m = int(0.004 * SR)
    pick = rng.standard_normal(m) * np.exp(-np.arange(m) / SR / 0.0012)
    starts = 0.5 + EIGHTH * np.arange(int((seconds - 1.5) / EIGHTH))
    x = np.zeros(int(seconds * SR))
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    for t0 in starts:
        for j, f0 in enumerate((330, 415, 523, 659, 784, 988)):
            i = int(round((t0 + spread * j / 5) * SR))
            x[i:i + n] += 0.1 * np.exp(-t / 0.12) * attack(n, 0.0005) * sum(np.sin(2 * np.pi * f0 * h * t + 0.7 * h) / h
                                                                          for h in range(1, 7))
            x[i:i + m] += 0.1 * click * pick
    return x, starts


def test_plucks_are_found_within_5_ms_and_kick_and_hat_do_not_trigger():
    x, times = pluck_train(kicks=True, hats=True)
    t, s = S.band_onsets(x, SR)
    assert len(t) == len(times) and np.abs(t - times).max() < 0.005
    assert np.all(np.diff(t) > 0) and np.all(s > 2.0)
    clean, _ = pluck_train()
    assert np.abs(S.band_onsets(clean, SR)[0] - times).max() < 0.005


def test_a_kick_and_a_hat_alone_are_no_onsets():
    x = np.zeros(int(8 * SR))
    rng = np.random.default_rng(0)
    for t0 in np.arange(0.5, 7.0, EIGHTH):
        kick(x, t0)
        hat(x, t0 + EIGHTH / 2, rng)
    assert len(S.band_onsets(x, SR)[0]) == 0


def test_silence_has_no_onsets_and_plucks_in_silence_are_exact():
    rng = np.random.default_rng(2)
    assert len(S.band_onsets(np.zeros(5 * SR), SR)[0]) == 0
    assert len(S.band_onsets(3e-6 * rng.standard_normal(5 * SR), SR)[0]) == 0         # -110 dBFS noise
    x = np.zeros(8 * SR)
    noise_pluck(x, 3.0, rng)
    sine_pluck(x, 5.0)
    t, _ = S.band_onsets(x, SR)
    assert len(t) == 2 and np.abs(t - [3.0, 5.0]).max() < 0.002
    assert len(S.band_onsets(x[:100], SR)[0]) == 0 and len(S.band_onsets(x[:700], SR)[0]) == 0   # under one window


@pytest.mark.parametrize("spread", [0.06, 0.075, 0.09])
def test_a_strum_spread_over_six_notes_is_one_onset_at_its_start(spread):
    x, starts = strums(spread)
    t, _ = S.band_onsets(x, SR)
    assert len(t) == len(starts)
    assert np.abs(t - starts).max() < 0.012                       # the first note, not the strongest or the last


def test_the_latency_is_measured_on_synthetic_plucks_and_removed():
    lat = S.onset_latency()
    assert -0.012 < lat < -0.002                                  # centred frames: the flux peaks early
    rng = np.random.default_rng(4)
    for rise in (0.0005, 0.002, 0.005, 0.01):
        x = np.zeros(4 * SR)
        starts = 0.5 + 0.7 * np.arange(5)
        for t0 in starts:
            noise_pluck(x, t0, rng, rise=rise)
        raw, _ = S.band_onsets(x, SR, latency=0.0)
        fixed, _ = S.band_onsets(x, SR)
        assert len(raw) == len(fixed) == 5
        assert abs(np.mean(raw - starts) - lat) < 0.003           # the raw peaks lead by what was measured ...
        assert np.abs(fixed - starts).max() < 0.003               # ... and the compensated times land on the starts
    wide = S.onset_latency(SR, 800.0, 6000.0, 2048)
    assert wide < lat - 0.005                                     # a longer window: earlier peaks, more to give back
    x, times = pluck_train()
    assert np.abs(S.band_onsets(x, SR, n=2048)[0] - times).max() < 0.005


def test_the_detector_does_not_depend_on_the_level():
    x, _ = pluck_train()
    t1, s1 = S.band_onsets(x, SR)
    t2, s2 = S.band_onsets(0.03 * x, SR)
    assert len(t1) == len(t2) and np.abs(t1 - t2).max() < 0.0005 and np.allclose(s1, s2, atol=0.05)


def test_onsets_closer_than_the_separation_merge_and_farther_ones_stay():
    rng = np.random.default_rng(6)
    close, far = np.zeros(3 * SR), np.zeros(3 * SR)
    for x, gap in ((close, 0.06), (far, 0.14)):
        noise_pluck(x, 1.0, rng)
        noise_pluck(x, 1.0 + gap, rng)
    assert len(S.band_onsets(close, SR)[0]) == 1 and len(S.band_onsets(far, SR)[0]) == 2
    assert abs(S.band_onsets(close, SR)[0][0] - 1.0) < 0.01       # the start of the pair


def test_the_band_selects_which_notes_are_onsets():
    x = np.zeros(6 * SR)
    for t0 in (1.0, 2.0, 3.0):
        kick(x, t0, amp=0.3, f=250.0)                             # a low note (the shape of a kick: a smooth attack)
    for t0 in (1.5, 2.5, 3.5):
        sine_pluck(x, t0, f=2000.0)
    t, _ = S.band_onsets(x, SR)                                   # 800-6000 Hz: the plucks only
    assert len(t) == 3 and np.abs(t - [1.5, 2.5, 3.5]).max() < 0.002
    t, s = S.band_onsets(x, SR, lo=100.0, hi=700.0)               # a low band: the low notes (and the plucks' splatter)
    assert len(t[s > 10]) == 3 and np.abs(t[s > 10] - [1.0, 2.0, 3.0]).max() < 0.012


# ---------------------------------------------------------------------------------------------------- the timeline
def test_onset_stats_describe_the_8th_note_grid():
    beats = list(np.arange(0.0, 8.5, 0.5))                         # 17 beats, 16 beat intervals: 32 8ths
    grid = A.eighths(beats)
    assert len(grid) == 33 and grid[1] == 0.25
    on = list(grid[:-1] + 0.012)                                   # every 8th, 12 ms late
    st = A.onset_stats(on, beats)
    assert st["count"] == 32 and st["per_beat"] == 2.0 and st["empty_8ths"] == 0 and st["off_8th"] == 0
    assert st["offset_ms"] == {"median": 12.0, "mad": 0.0, "p95": 12.0}
    gap = A.onset_stats(on[:10] + on[11:] + [3.125 + 0.012], beats)         # one 8th dropped, one onset between 8ths
    assert gap["empty_8ths"] == 1 and gap["off_8th"] == 1 and gap["offset_ms"]["median"] == 12.0
    assert A.onset_stats([1.0, 2.0], [])["count"] == 2 and "per_beat" not in A.onset_stats([1.0], [0.5])


def test_stem_onsets_are_clip_times_inside_the_clip():
    x = np.zeros(10 * SR)
    rng = np.random.default_rng(7)
    for t0 in (0.5, 1.6, 2.5, 4.0, 8.0):                          # window seconds; the clip is 2.0 .. 6.0
        noise_pluck(x, t0, rng)
    st = {"other": np.stack([x, x]).astype(np.float32), "bass": np.zeros((2, 10 * SR), np.float32)}
    on, meta = A.stem_onsets(st, 2.0, 4.0)
    assert list(on) == ["other"] and len(on["other"]) == 2
    assert np.abs(np.array(on["other"]) - [0.5, 2.0]).max() < 0.003
    m = meta["other"]
    assert m["band"] == [800.0, 6000.0] and m["latency_ms"] == round(S.onset_latency() * 1000, 2)
    assert len(m["strength"]) == 2 and min(m["strength"]) > 2.0
    assert A.stem_onsets(st, 2.0, 4.0, names=("bass",))[0] == {"bass": []}
    with pytest.raises(ValueError):
        A.stem_onsets(st, 2.0, 4.0, names=("vocals",))


class FakeSong:
    """A synthetic song standing in for ffmpeg and Demucs: S.load slices it the way ffmpeg would and records its
    arguments, the stems function returns the stems of the window it was handed and records the cache key."""

    def __init__(self, monkeypatch, seconds=24.0):
        rng = np.random.default_rng(3)
        n = int(seconds * SR)
        other, drums = np.zeros(n), np.zeros(n)
        for t0 in np.arange(0.3, seconds - 0.6, 0.25):            # 8th notes at 120 bpm
            noise_pluck(other, t0, rng)
        for k, t0 in enumerate(np.arange(0.3, seconds - 0.6, 0.5)):
            kick(drums, t0, amp=0.8 if k % 4 == 0 else 0.56, f=80.0)
        self.parts = {name: np.stack([x, x]).astype(np.float32) for name, x in
                      (("drums", drums), ("other", other), ("bass", 0 * other), ("vocals", 0 * other))}
        self.mix = sum(self.parts.values())
        self.loads, self.keys = [], []
        monkeypatch.setattr(S, "load", self.load)
        monkeypatch.setattr(W, "stems", self.stems)

    def load(self, path, start=0.0, duration=None, sr=SR, channels=2):
        a = int(round(start * SR))
        b = self.mix.shape[1] if duration is None else min(self.mix.shape[1], a + int(round(duration * SR)))
        self.loads.append((str(path), f"{start:.6f}", f"{duration:.6f}"))
        return self.mix[:, a:b].copy()

    def stems(self, mix, cache_dir, sr=SR):
        a = int(round(float(self.loads[-1][1]) * SR))
        self.keys.append(hashlib.sha256(mix.tobytes()).hexdigest()[:20])
        return {k: v[:, a:a + mix.shape[1]] for k, v in self.parts.items()}


def analysed(monkeypatch, tmp_path):
    song = FakeSong(monkeypatch)
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    tl = A.analyse(str(audio), start=6.0, duration=8.0, fps=30, cache_dir=str(tmp_path), words=False, context=4.0,
                   log=lambda *a: None)
    return song, tl


def test_analyse_stores_the_onsets_of_the_other_stem(monkeypatch, tmp_path):
    song, tl = analysed(monkeypatch, tmp_path)
    on = tl["onsets"]["other"]
    assert list(tl["onsets"]) == ["other"] and list(tl["onsets_meta"]) == ["other"] and "drums" not in tl["onsets"]
    assert on == sorted(on) and 0 <= on[0] and on[-1] <= 8.0 and len(on) == 32       # every 8th of the 8 s clip
    plucks = (np.arange(0.3, 24 - 0.6, 0.25) - 6.0)
    plucks = plucks[(plucks >= 0) & (plucks <= 8.0)]
    assert np.abs(np.array(on) - plucks).max() < 0.003
    s = A.summary(tl)["onsets"]["other"]
    assert s["count"] == 32 and s["per_beat"] == 2.0 and s["empty_8ths"] == 0 and abs(s["offset_ms"]["median"]) < 3


def test_the_onsets_helper_reproduces_the_window_analyse_read(monkeypatch, tmp_path):
    song, tl = analysed(monkeypatch, tmp_path)
    assert song.loads == [(tl["audio"]["file"], "2.000000", "16.000000")]        # start - pre, duration + pre + context
    on, meta = A.timeline_onsets(tl, str(tmp_path), context=4.0, log=lambda *a: None)
    assert song.loads[1] == song.loads[0] and song.keys[1] == song.keys[0]        # the same samples: the same cache key
    assert on == tl["onsets"] and meta == tl["onsets_meta"]
    with pytest.raises(ValueError, match="--context"):                            # another window: new stems
        A.timeline_onsets(tl, str(tmp_path), context=3.0, log=lambda *a: None)
    with pytest.raises(ValueError, match="audio"):
        A.timeline_onsets({"bpm": 120.0}, str(tmp_path), log=lambda *a: None)


def test_the_window_is_clamped_to_the_file_like_analyse(monkeypatch, tmp_path):
    song = FakeSong(monkeypatch, seconds=12.0)
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    tl = A.analyse(str(audio), start=2.0, duration=8.0, words=False, context=4.0, cache_dir=str(tmp_path),
                   log=lambda *a: None)                                              # 2 s before, only 2 s after
    assert tl["audio"]["context"] == [2.0, 2.0]
    on, _ = A.timeline_onsets(tl, str(tmp_path), context=4.0, log=lambda *a: None)
    assert on == tl["onsets"] and song.keys[1] == song.keys[0]


def run_cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def test_mk_timeline_onsets_merges_and_keeps_every_other_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    song, tl = analysed(monkeypatch, tmp_path)
    expected = {"onsets": tl.pop("onsets"), "onsets_meta": tl.pop("onsets_meta")}    # an older timeline: no onsets yet
    tl["lines"] = [{"start": 1.0, "end": 2.0, "words": [{"text": "zxqvword", "start": 1.0, "end": 2.0}]}]
    path = tmp_path / "audio" / "timeline.json"
    path.parent.mkdir()
    before = json.dumps(tl, ensure_ascii=False, indent=1)
    path.write_text(before, encoding="utf-8")

    code, out = run_cli(["timeline", "onsets", "--timeline", str(path), "--context", "4"], capsys)
    assert code == 0 and "zxqvword" not in json.dumps(out)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert {k: after[k] for k in expected} == expected
    rest = {k: v for k, v in after.items() if k not in expected}
    assert json.dumps(rest, ensure_ascii=False, indent=1) == before                  # the other keys: byte for byte
    assert list(after)[:-2] == list(tl)
    st = out["onsets"]["other"]
    assert st["count"] == 32 and st["per_beat"] == 2.0 and st["band"] == [800.0, 6000.0] and "offset_ms" in st

    again = path.read_text(encoding="utf-8")                                         # idempotent
    run_cli(["timeline", "onsets", "--timeline", str(path), "--context", "4"], capsys)
    assert path.read_text(encoding="utf-8") == again
    code, out = run_cli(["timeline", "onsets", "--timeline", str(path), "--context", "4", "--stem", "bass"], capsys)
    merged = json.loads(path.read_text(encoding="utf-8"))                            # per stem: `other` stays
    assert code == 0 and merged["onsets"]["other"] == expected["onsets"]["other"] and merged["onsets"]["bass"] == []
    assert list(merged["onsets_meta"]) == ["other", "bass"]


def test_mk_timeline_onsets_usage_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    code, out = run_cli(["timeline", "onsets"], capsys)
    assert code == 2 and "timeline" in out["error"]
    path = tmp_path / "tl.json"
    path.write_text(json.dumps({"bpm": 120.0}), encoding="utf-8")
    code, out = run_cli(["timeline", "onsets", "--timeline", str(path)], capsys)
    assert code == 2 and "audio" in out["error"]
    path.write_text(json.dumps({"audio": {"file": str(tmp_path / "gone.wav"), "start": 0.0, "duration": 1.0,
                                          "context": [0.0, 0.0]}}), encoding="utf-8")
    code, out = run_cli(["timeline", "onsets", "--timeline", str(path)], capsys)
    assert code == 2 and "not found" in out["error"]
    song, tl = analysed(monkeypatch, tmp_path)
    path.write_text(json.dumps(tl), encoding="utf-8")
    code, out = run_cli(["timeline", "onsets", "--timeline", str(path), "--context", "7"], capsys)
    assert code == 2 and "--context" in out["error"]
