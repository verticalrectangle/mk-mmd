"""Feet that stay where they land. A captured or retargeted motion slides a resting foot a few millimetres a frame (it
was made on a body with other legs, or the capture is noisy); `plant` finds the frames a foot rests on the floor and
holds it where it landed, then eases it back onto its own path as it lifts off (mkmmd.blender.build.motion: a motion's
FK legs followed with `legs = "fk"`). Positions are metres with z up, one row a frame. numpy only."""
import numpy as np

HEIGHT = 0.02     # m above where the foot stands flat: still resting
SPEED = 0.012     # m a frame across the floor (0.36 m/s at 30 fps): slower is resting, faster a step or a glide
LEAST = 4         # frames: a shorter touch is a step through
GAP = 2           # frames: a rest broken this briefly is one rest
EASE = 5          # frames back onto the foot's own path after it lifts off


def contacts(pos, floor_z, height=HEIGHT, speed=SPEED, least=LEAST, gap=GAP):
    """[(first, last)] frame spans (inclusive) where a foot at `pos` (n x 3: its ankle, or the IK bone on it) rests:
    less than `height` above `floor_z` (where it is when the foot stands flat on the floor) and moving slower than
    `speed` m a frame across it, for at least `least` frames, spans broken for `gap` frames or fewer joined."""
    p = np.asarray(pos, float)
    if len(p) < 2:
        return []
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    v = np.concatenate([[d[0]], (d[:-1] + d[1:]) / 2.0, [d[-1]]])       # central speed, m a frame
    down = (p[:, 2] - floor_z < height) & (v < speed)
    spans = []
    k, n = 0, len(down)
    while k < n:
        if not down[k]:
            k += 1
            continue
        a = k
        while k < n and down[k]:
            k += 1
        if spans and a - spans[-1][1] - 1 <= gap:
            spans[-1] = (spans[-1][0], k - 1)
        else:
            spans.append((a, k - 1))
    return [(a, b) for a, b in spans if b - a + 1 >= least]


def plant(pos, floor_z, height=HEIGHT, speed=SPEED, least=LEAST, gap=GAP, ease=EASE):
    """(positions, spans): `pos` (n x 3) with the foot held across the floor where it landed over each span of
    `contacts`, its height as the motion has it but never under `floor_z`, and eased back onto its own path (smoothstep)
    over the `ease` frames after each span, or fewer when it lands again sooner."""
    p = np.asarray(pos, float)
    out = p.copy()
    spans = contacts(p, floor_z, height, speed, least, gap)
    for k, (a, b) in enumerate(spans):
        anchor = p[a, :2]
        out[a:b + 1, :2] = anchor
        out[a:b + 1, 2] = np.maximum(p[a:b + 1, 2], floor_z)
        nxt = spans[k + 1][0] if k + 1 < len(spans) else len(p)
        m = min(ease, nxt - b - 1)
        for j in range(1, m + 1):
            u = j / (m + 1.0)
            w = u * u * (3.0 - 2.0 * u)
            out[b + j, :2] = anchor * (1.0 - w) + p[b + j, :2] * w
    return out, spans
