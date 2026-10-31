"""`mk post` over worker processes (docs/design.md: Post, mkmmd.postframes): the frames come out in order and are the
frames one process makes, whatever the split, with what runs on from frame to frame re-created where a chunk starts
(a missing frame repeating the last one rendered, the loop's wrap over frame 0 as finished, a film's held frame, a
flowing mosh's carried picture); heavy frames are planned long before their turn, within their room; mkr-post's jobs
go out in order from a thread of their own, and what stops it is what the caller hears; the encoder and the number of
processes come from [post] or the flags. Tiny synthetic frames, the encoder and mkr-post stood in for; no Blender, no
ffmpeg."""
import json
import queue
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import footage as FT
from mkmmd import postframes as PF
from mkmmd import postgpu as PG
from mkmmd.cli import main as MAIN
from mkmmd.cli import post as POST
from mkmmd.core import screentype as SR

W, H, FPS, F0 = 64, 36, 10, 1
END = F0 + 19                                     # a 2 s clip, frames 1 .. 20
GAP = 8                                           # not rendered (a film step starts on it): frame 7 stands in for it
FX = ('[[fx]]\nkind = "film"\nfrom = 0.3\nto = 1.3\nfps = 3\nseed = 4\n'
      '[[fx]]\nkind = "mosh"\nfrom = 0.5\nto = 1.4\nflow = { kind = "drift", amount = 300.0, steps = 2 }\n'
      '[[fx]]\nkind = "tint"\nfrom = 1.2\nto = 2.0\ncolor = "#ff8000"\n')
WRAP = '[[transition]]\nat = 2.0\nkind = "page"\nease = "out"\ndir = "left"\nangle = 0.0\nshadow = 0.0\n'


def project(root, post=""):
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = 2.0\n[[output]]\nname = "wide"\n'
        f'size = [{W}, {H}]\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 1.0\n[[shot]]\nname = "b"\nfrom = 1.0\nto = 2.0\n'
        + WRAP + FX + '[post]\ngrain = { amount = 0.01 }\n' + post, encoding="utf-8")


def rendered(root):
    """Every frame but GAP, each its own picture (a gradient moving with the frame), and type on some of them."""
    d = root / "renders" / "draft" / "wide"
    (d / SR.LAYER_DIR).mkdir(parents=True)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    for f in range(F0, END + 1):
        if f == GAP:
            continue
        a = np.stack([(xs * 4 + f * 9) % 256, (ys * 7 + f * 5) % 256, (xs + ys + f * 13) % 256], -1).astype(np.uint8)
        Image.fromarray(a).save(d / f"{f:05d}.png")
        if f % 3 == 0:
            t = np.zeros((H, W, 4), np.uint8)
            t[4:12, 8 + f:20 + f] = (240, 240, 250, 210)
            Image.fromarray(t, "RGBA").save(d / SR.layer_rel(f))


class Encoder:
    """Stands in for ffmpeg: keeps the frames mk post writes and how it was asked to encode."""

    def __init__(self):
        self.frames, self.stdin, self.kwargs = [], self, None

    def __call__(self, *a, **k):
        self.kwargs = dict(zip(("frames", "size", "fps", "out", "audio", "crf", "encoder", "preset"), a), **k)
        return self

    def write(self, data):
        self.frames.append(bytes(data))

    def close(self):
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None

    def kill(self):
        pass


def run_post(root, capsys, monkeypatch, *extra):
    enc = Encoder()
    monkeypatch.setattr(POST, "encode", enc)
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--no-audio", "--allow-gaps", "--project", str(root), *extra])
    out = capsys.readouterr().out
    rep = json.loads(out[out.index("{"):])
    return e.value.code, rep, enc


def test_workers_give_the_frames_one_process_gives_in_order(tmp_path, capsys, monkeypatch):
    project(tmp_path)
    rendered(tmp_path)
    code, one, solo = run_post(tmp_path, capsys, monkeypatch, "--jobs", "1")
    assert code == 0, one
    monkeypatch.setattr(PF, "CHUNK", 3)                      # many chunks, starting all over the stateful spans
    code, many, split = run_post(tmp_path, capsys, monkeypatch, "--jobs", "3")
    assert code == 0, many
    assert len(solo.frames) == len(split.frames) == END + 1 - F0
    assert len(set(solo.frames)) > 10                         # the frames differ: order shows
    assert split.frames == solo.frames                        # byte for byte, in order
    a, b = one["outputs"]["wide"], many["outputs"]["wide"]
    assert (a["jobs"], b["jobs"]) == (1, 3)
    assert {k: v for k, v in a.items() if k not in ("jobs", "seconds")} == \
        {k: v for k, v in b.items() if k not in ("jobs", "seconds")}


def test_the_encoder_and_the_processes_come_from_post_or_the_flags(tmp_path, capsys, monkeypatch):
    project(tmp_path, 'encoder = "x264"\nx264_preset = "veryfast"\njobs = 2\n')
    rendered(tmp_path)
    code, rep, enc = run_post(tmp_path, capsys, monkeypatch)
    assert code == 0, rep
    assert (enc.kwargs["encoder"], enc.kwargs["preset"]) == ("x264", "veryfast") and rep["outputs"]["wide"]["jobs"] == 2
    code, rep, enc = run_post(tmp_path, capsys, monkeypatch, "--x264-preset", "slow", "--jobs", "1")
    assert code == 0 and enc.kwargs["preset"] == "slow" and rep["outputs"]["wide"]["jobs"] == 1
    for bad, said in (('jobs = 0\n', "jobs"), ('encoder = "nvenc"\n', "encoder"), ('x264_preset = "turbo"\n', "x264_preset")):
        project(tmp_path, bad)
        code, rep, _ = run_post(tmp_path, capsys, monkeypatch)
        assert code == 2 and said in rep["error"], rep


class Pool:
    """Stands in for PF.Planners: each run planned on one of `n` threads (a heavy frame's run slowly), every submission
    and every frame handed on noted in `log` in the order they happen."""

    def __init__(self, n, heavy):
        self.ex, self.heavy, self.log = ThreadPoolExecutor(n), heavy, []

    def submit(self, out, ks):
        self.log.append(("submit", ks[0]))

        def plan():
            time.sleep(0.05 if ks[0] in self.heavy else 0.002)
            return [(k, "gpu", {"k": k, "ops": []}, (None, 0, 0, 0)) for k in ks], []
        return self.ex.submit(plan)

    def stop(self):
        self.ex.shutdown(cancel_futures=True)


def test_heavy_frames_are_planned_long_before_their_turn_within_their_room(monkeypatch):
    heavy = (30, 31, 32, 33)
    out = types.SimpleNamespace(frames=list(range(40)), safe=lambda k: True, heavy=lambda k: k in heavy, size=(4, 2),
                                notes=[], wait=lambda k: None, ready=lambda k: True)
    monkeypatch.setattr(PF, "EARLY_BYTES", 2 * 4 * 2 * 12)    # two heavy frames' pictures planned ahead at most
    pool = Pool(3, heavy)
    got = []
    for k, *_ in PF._planned(out, 3, 4, pool):
        pool.log.append(("out", k))
        got.append(k)
    assert got == out.frames                                  # in order, each once
    submitted = [k for what, k in pool.log if what == "submit"]
    assert submitted.index(30) < submitted.index(24) and submitted.index(31) < submitted.index(24)
    starts = [r[0] for r in PF.chunks(out, 4, 2, 1)]          # as _planned cuts them: two workers, room for one each
    sent, ahead = set(), set()                                # heavy runs submitted, their frame not yet handed on
    for what, k in pool.log:
        if what == "out":
            ahead.discard(k)
            continue
        sent.add(k)
        if k in heavy:
            ahead.add(k)
            if any(s < k and s not in sent for s in starts):  # before its turn: within the room
                assert len(ahead) <= 2, pool.log


def test_a_heavy_frame_is_planned_ahead_only_once_a_followed_render_has_its_files(monkeypatch):
    heavy = (30,)
    rendered_up_to = [0]
    out = types.SimpleNamespace(frames=list(range(40)), safe=lambda k: True, heavy=lambda k: k in heavy, size=(4, 2),
                                notes=[], wait=lambda k: rendered_up_to.__setitem__(0, max(rendered_up_to[0], k)),
                                ready=lambda k: k <= rendered_up_to[0])
    pool = Pool(3, heavy)
    assert [k for k, *_ in PF._planned(out, 3, 4, pool)] == out.frames
    submitted = [k for what, k in pool.log if what == "submit"]
    assert submitted == sorted(submitted)                     # never past what the render has


class Server:
    """Stands in for mkmmd.postgpu.Server with a sink: answers each job in the order it was sent (its clip index, a
    darkest luma of k / 100); `end` makes it stop after the jobs it has."""

    def __init__(self, folder):
        self.folder, self.sink, self.sent = folder, object(), []
        self.answers = queue.SimpleQueue()
        self.proc = types.SimpleNamespace(kill=lambda: None)

    def send(self, job):
        self.sent.append(job)
        self.answers.put((job["k"], job["k"] / 100.0, None))

    def receive(self):
        got = self.answers.get(timeout=10)
        if got is None:
            raise RuntimeError("mkr-post stopped")
        return got

    def end(self):
        self.answers.put(None)


def sent_frames(tmp_path, monkeypatch, fail_at=None):
    """finished() over ten frames planned here (every third finished here, the others mkr-post's, every other one with
    a clip frame to hand over first) against Server; what came out, what was sent, and the threads the clip frames
    were handed on."""
    def plan(k):
        if k % 3 == 0:
            return "cpu", bytes([k]) * 6, (0.25, 0, 0, 0)
        return "gpu", {"k": k, "ops": [{"op": "footage", "clip": {"name": "c", "k": k}}] if k % 2 else []}, \
            (None, 1, 0, 0)
    handed = []

    def hand(ffx, ops, folder, footage_of, kept=None):
        k = ops[0]["clip"]["k"]
        if k == fail_at:
            raise FT.FootageError(f"clip c: frame {k} could not be decoded")
        handed.append((k, threading.current_thread().name))
        return ops
    monkeypatch.setattr(PG, "hand", hand)
    out = types.SimpleNamespace(frames=list(range(10)), wait=lambda k: None, plan=plan, ffx=None, gpu=tmp_path,
                                args={"proj": types.SimpleNamespace(root=tmp_path)})
    server = Server(tmp_path)
    got = list(PF.finished(out, jobs=1, server=server))
    return got, server.sent, handed


def test_mkr_posts_jobs_go_out_in_order_from_a_thread_of_their_own(tmp_path, monkeypatch):
    got, sent, handed = sent_frames(tmp_path, monkeypatch)
    assert [k for k, _, _ in got] == list(range(10))
    assert [job["k"] for job in sent] == list(range(10))     # every frame through mkr-post, in order
    for k, data, stats in got:
        assert data is None and stats[0] == (0.25 if k % 3 == 0 else k / 100.0)
    for job in sent:
        if job["k"] % 3 == 0:                                 # finished here: its bytes passed on as they were
            assert Path(job["finished"]["path"]).read_bytes() == bytes([job["k"]]) * 6
    assert [k for k, _ in handed] == [1, 5, 7]               # the clip frames, in order, off the caller's thread
    assert {name for _, name in handed} == {"mkpost-send"}


def test_a_clip_that_fails_on_the_senders_thread_is_what_the_caller_hears(tmp_path, monkeypatch):
    with pytest.raises(FT.FootageError, match="frame 5"):
        sent_frames(tmp_path, monkeypatch, fail_at=5)
