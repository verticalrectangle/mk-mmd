"""A sung word begins where its sound begins, not where the aligner is surest of its first letter."""
import numpy as np

from mkmmd.timeline import signal as S
from mkmmd.timeline import words as W


def _voice(notes, seconds, sr=S.SR):
    """A sung line: harmonic tones [(start, end, f0, attack seconds)], silence between."""
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for a, b, f0, rise in notes:
        env = np.clip((t - a) / rise, 0, 1) * np.clip((b - t) / 0.03, 0, 1)
        x += env * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return (0.3 * x).astype(np.float32)


def test_a_word_the_aligner_puts_well_into_its_sound_starts_at_the_sounds_attack(monkeypatch):
    # two words: one after silence with a soft attack, one re-attacked after a 70 ms dip with no real breath; a third
    # sung straight on from the second (a pitch change, no dip) whose letter the aligner puts where it belongs
    voc = _voice([(0.50, 1.40, 220.0, 0.08), (1.47, 2.20, 247.0, 0.05), (2.20, 2.90, 294.0, 0.01)], 3.2)
    late = {(0, 0): (0.72, 0.74), (0, 1): (0.74, 0.90), (1, 0): (1.74, 1.76), (2, 0): (2.21, 2.23), (2, 1): (2.23, 2.5)}
    monkeypatch.setattr(W, "ctc_align", lambda v, sr, words, t0, t1: (late, None))
    lines, _, stats = W.word_timeline(voc, S.SR, [["aa", "b", "cc"]], 3.2)
    s = [w["start"] for w in lines[0]["words"]]
    assert abs(s[0] - 0.50) < 0.06 and abs(s[1] - 1.47) < 0.06, s        # 0.22 and 0.27 s before their letters
    assert abs(s[2] - 2.20) < 0.06, s                                    # no dip before it: it stays with its letter
    assert stats["attack"] >= 1 and [w["ctc_start"] for w in lines[0]["words"]] == [0.72, 1.74, 2.21]
