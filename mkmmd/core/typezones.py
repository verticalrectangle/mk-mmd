"""Lyric type per shot, the numbers (docs/design.md: Text, Lyrics: zones): which words of a line stand in which shot.

A line of lyrics that runs across a cut is set again in every shot it is on screen in: `lyrics = {zones = {<shot> = {...}}}`
gives each shot its own placement and look (a band at the bottom where the faces are high, a plain stencil over a silhouette),
and the words that were already up at the cut come with it, set in the new shot's zone. This module only decides who stands
where, in frames; `mkmmd/core/wordtype.py` builds the entries."""


def plan_blocks(lands, hides, shots, min_run=2):
    """The words of a line shot by shot.

    lands, hides   per word, the clip frame it lands on and the first frame it is gone (all the same without a drip)
    shots          [(name, first, end)] in cut order: a shot's frames are first .. end - 1
    min_run        a word has to be on screen for this many frames inside a shot to be set in it (one that lands the frame
                   before a cut is carried to the next shot instead of flashing up)

    Returns [{"shot", "words", "first", "end"}]: the shot, the indices of its words (in order, never empty; the words are
    always a run of the line since they land in order), and the frames the block runs, `first` .. `end` - 1, the shot's own
    clipped to when the words are up. A shot nobody stands in is left out; so is a word that never gets min_run frames."""
    out = []
    for name, a, b in shots:
        words = [k for k in range(len(lands)) if min(hides[k], b) - max(lands[k], a) >= min_run]
        if words:
            out.append({"shot": name, "words": words, "first": max(a, min(lands[k] for k in words)),
                        "end": min(b, max(hides[k] for k in words))})
    return out


def split_words(blocks, n):
    """The indices (0-based) of the words that no block holds: they never get on screen."""
    held = {k for b in blocks for k in b["words"]}
    return [k for k in range(n) if k not in held]
