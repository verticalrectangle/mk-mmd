"""split on the GPU (mkr-post fx/split.rs): each band's shifts from FrameFx.split_at, the CPU path's own draw of its
randomness."""


def ops(ffx, fx, frame):
    bands = ffx.split_at(fx, frame)
    if bands is None:
        return []
    return [{"op": "split", "bands": [[a, b, list(s)] for a, b, s in bands]}]
