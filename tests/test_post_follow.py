"""`mk post --follow` and the render manifest (mkmmd.core.progress, docs/design.md: Post, Following a render): a file
is taken once it is all there, each frame once every file the render promised for it and the frames before it is (or
asked whether it is, without waiting), the end of a render without a promised file is an error, post following a
render that writes its frames one by one gives the frames a post after it gives, in order, from one process or
several, and each output is finished while the render of the next has not begun. Tiny synthetic frames, the encoder
stood in for; no Blender, no ffmpeg."""
import io
import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from PIL import Image

import test_post_jobs as TJ
from mkmmd.core import progress as PR
from mkmmd.core import screentype as SR


def png_bytes(w=8, h=6, colour=(10, 200, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, "PNG")
    return buf.getvalue()


def test_a_file_is_finished_once_all_of_it_is_there(tmp_path):
    data = png_bytes()
    p = tmp_path / "a.png"
    assert not PR.finished(p)                                   # not there
    p.write_bytes(b"")
    assert not PR.finished(p)                                   # a claim
    p.write_bytes(data[:len(data) // 2])
    assert not PR.finished(p)                                   # half written
    p.write_bytes(data)
    assert PR.finished(p)
    j = tmp_path / "point.json"
    j.write_text('{"x": 0.5, "y"', encoding="utf-8")
    assert not PR.finished(j)
    j.write_text('{"x": 0.5, "y": 0.25, "m": 1.0}', encoding="utf-8")
    assert PR.finished(j)


def later(delay, fn, *args):
    t = threading.Thread(target=lambda: (time.sleep(delay), fn(*args)), daemon=True)
    t.start()
    return t


def test_a_frame_is_waited_for_until_every_file_promised_up_to_it_is_there(tmp_path):
    PR.write(tmp_path, {1: ["00001.png"], 2: ["00002.png", "screen/00002.png"], 3: ["00003.png"]}, [1, 2, 3])
    (tmp_path / "screen").mkdir()
    data = png_bytes()
    follow = PR.Follow(tmp_path, [1, 2, 3], poll=0.01)
    later(0.05, (tmp_path / "00001.png").write_bytes, data)
    t0 = time.time()
    follow.wait(1)
    assert PR.finished(tmp_path / "00001.png") and time.time() - t0 >= 0.04
    later(0.05, (tmp_path / "00002.png").write_bytes, data)
    later(0.15, (tmp_path / "screen" / "00002.png").write_bytes, data)
    follow.wait(2)                                              # its screen layer too, which comes last
    assert PR.finished(tmp_path / "screen" / "00002.png")
    assert not (tmp_path / "00003.png").exists()                # frame 3 not asked for yet
    assert follow.waited >= 0.15


def test_ready_says_whether_the_files_are_there_without_waiting(tmp_path):
    PR.write(tmp_path, {1: ["00001.png"], 2: ["00002.png", "screen/00002.png"]}, [1, 2])
    (tmp_path / "screen").mkdir()
    follow = PR.Follow(tmp_path, [1, 2], poll=0.05)
    t0 = time.time()
    assert not follow.ready(1) and time.time() - t0 < 0.04     # not there: no waiting
    (tmp_path / "00001.png").write_bytes(png_bytes())
    (tmp_path / "00002.png").write_bytes(png_bytes())
    assert follow.ready(1) and not follow.ready(2)              # frame 2's screen layer still to come
    PR.finish(tmp_path, "mkr: out of memory")
    assert not follow.ready(2)                                  # never coming: wait says so
    with pytest.raises(FileNotFoundError, match="out of memory"):
        follow.wait(2)


def test_a_render_that_stops_without_a_promised_file_is_an_error(tmp_path):
    PR.write(tmp_path, {1: ["00001.png"], 2: ["00002.png"]}, [1, 2])
    (tmp_path / "00001.png").write_bytes(png_bytes())
    follow = PR.Follow(tmp_path, [1, 2], poll=0.01)
    follow.wait(1)
    later(0.05, PR.finish, tmp_path, "mkr: out of memory")
    with pytest.raises(FileNotFoundError, match="out of memory"):
        follow.wait(2)


def test_a_render_that_died_is_over_and_one_that_does_not_say_its_files_is_waited_for_to_the_end(tmp_path):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    PR.write(tmp_path, {1: ["00001.png"]}, [1], pid=gone.pid)   # a render killed outright: never marked done
    with pytest.raises(FileNotFoundError, match="00001.png"):
        PR.Follow(tmp_path, [1], poll=0.01).wait(1)
    other = tmp_path / "other"
    other.mkdir()
    PR.write(other, None, [1, 2])                               # an engine that cannot say which files a frame gets
    follow = PR.Follow(other, [1, 2], poll=0.01)
    later(0.1, PR.finish, other)
    t0 = time.time()
    follow.wait(1)
    assert time.time() - t0 >= 0.09 and PR.read(other)["done"]


def test_without_a_manifest_the_folder_is_taken_as_it_is(tmp_path):
    follow = PR.Follow(tmp_path, [1, 2], poll=0.01, appear=0.05)
    assert follow.promised is None
    follow.wait(2)                                              # nothing to wait for


def render_slowly(root, gap, slow):
    """Write the frames TJ.rendered() makes one by one into a fresh folder under a manifest, each frame's screen layer
    after the frame, every third file half first: what a render does while post follows it."""
    src = root / "renders" / "draft" / "wide"
    done = root / "done"
    TJ.rendered(done.parent)                                    # the pictures, rendered into done/../renders: moved
    (root / "renders").rename(done)
    (src / SR.LAYER_DIR).mkdir(parents=True)
    frames = [f for f in range(TJ.F0, TJ.END + 1) if f != gap]
    files = {f: [f"{f:05d}.png"] + ([SR.layer_rel(f)] if f % 3 == 0 else []) for f in frames}
    PR.write(src, files, frames)

    def run():
        for i, f in enumerate(frames):
            for rel in files[f]:
                data = (done / "draft" / "wide" / rel).read_bytes()
                with open(src / rel, "wb") as fh:
                    if i % 3 == 0:
                        fh.write(data[:len(data) // 2])
                        fh.flush()
                        time.sleep(slow)
                    fh.write(data[len(data) // 2:] if i % 3 == 0 else data)
                time.sleep(slow)
        PR.finish(src)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


@pytest.mark.parametrize("jobs", ["1", "3"])
def test_post_following_a_render_gives_the_frames_a_post_after_it_gives(tmp_path, capsys, monkeypatch, jobs):
    TJ.project(tmp_path)
    if jobs != "1":
        monkeypatch.setattr(TJ.PF, "CHUNK", 3)                  # many chunks, each waited for
    writer = render_slowly(tmp_path, TJ.GAP, 0.08)
    code, rep, follow = TJ.run_post(tmp_path, capsys, monkeypatch, "--jobs", jobs, "--follow")
    writer.join()
    assert code == 0, rep
    assert rep["outputs"]["wide"]["waited_for_render"] > 0     # it did wait: the render was still writing
    code, again, after = TJ.run_post(tmp_path, capsys, monkeypatch, "--jobs", "1")
    assert code == 0, again
    assert len(follow.frames) == TJ.END + 1 - TJ.F0 and follow.frames == after.frames
    a = {k: v for k, v in rep["outputs"]["wide"].items() if k not in ("seconds", "waited_for_render", "jobs")}
    b = {k: v for k, v in again["outputs"]["wide"].items() if k not in ("seconds", "jobs")}
    assert a == b


def test_post_following_a_render_that_stops_short_says_which_file_never_came(tmp_path, capsys, monkeypatch):
    TJ.project(tmp_path)
    TJ.rendered(tmp_path)
    src = tmp_path / "renders" / "draft" / "wide"
    frames = [f for f in range(TJ.F0, TJ.END + 1) if f != TJ.GAP]
    (src / "00015.png").unlink()
    PR.write(src, {f: [f"{f:05d}.png"] for f in frames}, frames)
    PR.finish(src, "mkr: the GPU was lost")
    code, rep, _ = TJ.run_post(tmp_path, capsys, monkeypatch, "--jobs", "1", "--follow")
    assert code == 2 and "00015.png" in rep["error"] and "the GPU was lost" in rep["error"], rep
    assert json.loads((src / PR.MANIFEST).read_text())["done"]


class Turns:
    """Stands in for ffmpeg, one encoder an output, and says when each has its last frame (the event of its name)."""

    def __init__(self):
        self.closed, self.frames = {}, {}

    def __call__(self, frames, size, fps, out, *a, **k):
        name = Path(out).stem.removeprefix("t_").removesuffix("_draft")
        self.closed.setdefault(name, threading.Event())
        got = self.frames.setdefault(name, [])
        closed = self.closed[name]
        enc = types.SimpleNamespace(stdin=types.SimpleNamespace(write=lambda b: got.append(bytes(b)), close=closed.set),
                                    wait=lambda timeout=None: 0, poll=lambda: None, kill=lambda: None)
        return enc


def test_each_output_is_finished_while_the_render_of_the_next_has_not_begun(tmp_path, capsys, monkeypatch):
    TJ.project(tmp_path)
    toml = tmp_path / "mk.toml"
    toml.write_text(toml.read_text().replace("[[shot]]", f'[[output]]\nname = "late"\nsize = [{TJ.W}, {TJ.H}]\n'
                                             "[[shot]]", 1), encoding="utf-8")
    TJ.rendered(tmp_path)
    wide, late = (tmp_path / "renders" / "draft" / n for n in ("wide", "late"))
    frames = [f for f in range(TJ.F0, TJ.END + 1) if f != TJ.GAP]
    files = {f: [f"{f:05d}.png"] + ([SR.layer_rel(f)] if f % 3 == 0 else []) for f in frames}
    PR.write(wide, files, frames)
    PR.finish(wide)                                             # the first output's render is over
    (late / SR.LAYER_DIR).mkdir(parents=True)
    PR.write(late, files, frames)                               # the next one's has drawn nothing yet
    enc = Turns()
    enc.closed["wide"] = threading.Event()
    order = []

    def render_late():
        order.append("wide finished" if enc.closed["wide"].wait(20) else "late began first")
        for f in frames:
            for rel in files[f]:
                (late / rel).write_bytes((wide / rel).read_bytes())
        PR.finish(late)
    writer = threading.Thread(target=render_late, daemon=True)
    writer.start()
    monkeypatch.setattr(TJ.POST, "encode", enc)
    with pytest.raises(SystemExit) as e:
        TJ.MAIN.main(["post", "--preset", "draft", "--no-audio", "--allow-gaps", "--follow", "--jobs", "1", "--project",
                      str(tmp_path)])
    writer.join()
    out = capsys.readouterr().out
    rep = json.loads(out[out.index("{"):])
    assert e.value.code == 0, rep
    assert order == ["wide finished"]                           # wide went on its own, late followed after it
    assert enc.frames["wide"] == enc.frames["late"] and len(enc.frames["late"]) == TJ.END + 1 - TJ.F0


def test_frames_older_than_the_built_scene_are_stale_and_a_resumed_render_s_are_not(tmp_path):
    """A render skips the frames on disk so that a stopped one resumes; frames drawn before the scene was built again
    show the scene of the build before, and their layers with them."""
    import os

    from mkmmd.cli import render as RN
    d = tmp_path / "renders" / "draft" / "wide"
    (d / "plate").mkdir(parents=True)
    built = 1_000_000.0
    for rel, age in (("00101.png", -60), ("00102.png", -60), ("00103.png", 60), ("plate/00101.png", -60),
                     ("plate/00103.png", 60), ("render.json", -60), ("00250.png", -60)):
        (d / rel).write_bytes(b"x")
        os.utime(d / rel, (built + age, built + age))
    old = [str(p.relative_to(d)) for p in RN.stale_files(d, range(101, 104), built)]
    assert old == ["00101.png", "00102.png", "plate/00101.png"]          # not the newer ones, the manifest or frame 250
    assert RN.stale_files(tmp_path / "none", [101], built) == []
