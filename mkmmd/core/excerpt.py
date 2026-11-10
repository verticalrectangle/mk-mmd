"""Excerpts, bpy-free (docs/design.md: Post): `[[excerpt]]`, spans of the clip that `mk post` writes as videos of their
own beside each output, cut out of it as it is encoded (a loop for a feed, a teaser): nothing is finished or encoded
twice over, the encoder writes the span's frames to a second file.

    [[excerpt]]   name, from, to (clip seconds), outputs = [names] (default: every output), loop = false

An excerpt holds the frames a shot or an effect over the same span holds: round(from x fps) up to, not with,
round(to x fps), so one cut on a bar line starts on that bar's first frame and stops before the next bar's. Its sound
is the clip's over those frames, faded out over its last 60 ms as the clip's is; a `loop` is not faded: it runs back
into its start."""
import re

KEYS = ("name", "from", "to", "outputs", "loop")
FADE = 0.06                                   # seconds an excerpt's sound fades out over, as the clip's


class ExcerptError(ValueError):
    pass


def normalize(specs, duration, fps, outputs):
    """The `[[excerpt]]` entries checked against the clip (`duration` s at `fps`) and its `outputs` (names):
    [{name, first (frames from the clip's first), frames, outputs, loop}]; an ExcerptError says what is wrong."""
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ExcerptError("[[excerpt]] is a list of tables")
    outputs, out, last = list(outputs), [], int(round(duration * fps))
    for i, s in enumerate(specs):
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not s["name"]:
            raise ExcerptError(f"excerpt {i}: `name` is required (the video is <project>_<output>_<name>.mp4)")
        what = f"excerpt {s['name']!r}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", s["name"]):
            raise ExcerptError(f"{what}: a name is letters, digits, - and _ (it is part of a file's name)")
        if s["name"] in [e["name"] for e in out]:
            raise ExcerptError(f"{what}: two excerpts have this name")
        extra = sorted(set(s) - set(KEYS))
        if extra:
            raise ExcerptError(f"{what}: unknown keys {extra} (known: {', '.join(KEYS)})")
        for k in ("from", "to"):
            if isinstance(s.get(k), bool) or not isinstance(s.get(k), (int, float)):
                raise ExcerptError(f"{what}: `{k}` is required, in clip seconds")
        first, end = int(round(s["from"] * fps)), int(round(s["to"] * fps))
        if not 0 <= first < end <= last:
            raise ExcerptError(f"{what}: from {s['from']} to {s['to']} is no span inside the clip (0 to {duration} s)")
        names = s.get("outputs", outputs)
        names = [names] if isinstance(names, str) else names
        if not isinstance(names, list) or not names or any(n not in outputs for n in names):
            raise ExcerptError(f"{what}: outputs = {s.get('outputs')!r}: one or a list of {', '.join(outputs)}")
        if not isinstance(s.get("loop", False), bool):
            raise ExcerptError(f"{what}: loop is true or false, not {s['loop']!r}")
        out.append({"name": s["name"], "first": first, "frames": end - first, "outputs": list(names),
                    "loop": s.get("loop", False)})
    return out


def filters(e, fps):
    """(video filter, audio filter) that cut excerpt `e` out of the clip's stream in ffmpeg, both starting at 0: the
    frames by their number, the sound over the same span, faded out at its end unless it is a loop."""
    a, b = e["first"], e["first"] + e["frames"]
    sound = f"atrim=start={a / fps:.6f}:end={b / fps:.6f},asetpts=PTS-STARTPTS"
    if not e["loop"]:
        sound += f",afade=t=out:st={max(0.0, e['frames'] / fps - FADE):.4f}:d={FADE}"
    return f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS", sound
