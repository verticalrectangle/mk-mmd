"""The site's server (mkmmd.site): it answers only with its token and to its own address, serves files only from the
folders a page was opened for (with byte ranges, which audio and video seek with), a review goes through it the way
the page uses it (opened, answers checked and kept, sent), a project's music comes back in clip seconds (the clip's
audio cut from the song where [audio] starts, the hits lanes the project names, the shots and effects, and the words by
number, never their text), its baked scene is found beside [project] blend with the drafts mk post made, a draft's
samples point at its H.264 frames, and the project's checks keep their last results for the Checks tab (merged by name,
never overwritten by an ad-hoc run) with each worst moment in clip seconds."""
import base64
import http.client
import json
import math
import os
import shutil
import struct
import subprocess
import threading
import urllib.parse
import wave

import pytest

from mkmmd import review as RV
from mkmmd.cli import main as MAIN
from mkmmd.site import api as API
from mkmmd.site import draft as DR
from mkmmd.site import music as MU
from mkmmd.site import server as S


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    s = S.Site(routes=API.routes())
    httpd = S.serve(s)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield s, httpd.server_port
    httpd.shutdown()
    httpd.server_close()


def ask(site, method, rest, body=None, host=None, token=None, headers=None):
    s, port = site
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = None if body is None else json.dumps(body).encode()
    c.request(method, f"/{token or s.token}/{rest}", body=data,
              headers={"Host": host or f"127.0.0.1:{port}", **(headers or {})})
    r = c.getresponse()
    out = r.status, r.read(), dict(r.getheaders())
    c.close()
    return out


def file_rest(path):
    return "file?path=" + urllib.parse.quote(str(path))


def test_only_its_token_and_its_own_address_reach_it(site, tmp_path):
    assert ask(site, "GET", "api/ping")[0] == 200
    assert ask(site, "GET", "api/ping", token="guess")[0] == 404
    assert ask(site, "GET", "api/ping", host="evil.example:80")[0] == 403          # DNS rebinding
    assert ask(site, "GET", "app/../server.py")[0] == 404


def test_files_come_only_from_opened_folders_and_seek_by_range(site, tmp_path):
    inside, outside = tmp_path / "proj", tmp_path / "secret"
    inside.mkdir(), outside.mkdir()
    (inside / "clip.bin").write_bytes(bytes(range(100)))
    (outside / "key.txt").write_text("no")
    (inside / "link.txt").symlink_to(outside / "key.txt")
    assert ask(site, "GET", file_rest(inside / "clip.bin"))[0] == 403
    site[0].allow(inside)
    status, data, _ = ask(site, "GET", file_rest(inside / "clip.bin"))
    assert (status, data) == (200, bytes(range(100)))
    for path in (outside / "key.txt", inside / "link.txt", inside / ".." / "secret" / "key.txt"):
        assert ask(site, "GET", file_rest(path))[0] == 403, path
    status, data, h = ask(site, "GET", file_rest(inside / "clip.bin"), headers={"Range": "bytes=10-19"})
    assert (status, data, h["Content-Range"]) == (206, bytes(range(10, 20)), "bytes 10-19/100")
    status, data, _ = ask(site, "GET", file_rest(inside / "clip.bin"), headers={"Range": "bytes=-5"})
    assert (status, data) == (206, bytes(range(95, 100)))
    assert ask(site, "GET", file_rest(inside / "clip.bin"), headers={"Range": "bytes=200-"})[0] == 416


def test_a_review_goes_through_the_server_as_the_page_uses_it(site, tmp_path):
    f = tmp_path / "pick.review.toml"
    f.write_text('[review]\ntitle = "Pick"\n[[question]]\nid = "eyes"\nask = "Eyes?"\n'
                 '[[question.option]]\nid = "A"\nlabel = "Rin\'s"\n', encoding="utf-8")
    q = "?path=" + urllib.parse.quote(str(f))
    assert ask(site, "GET", "api/review" + q)[0] == 403                            # not opened yet
    status, data, _ = ask(site, "POST", "api/open", {"review": str(f)})
    assert status == 200 and json.loads(data)["page"].startswith("?review=")
    assert json.loads(ask(site, "GET", "api/review" + q)[1])["title"] == "Pick"
    status, data, _ = ask(site, "PUT", "api/answers" + q, {"answers": {"eyes": {"choice": "Z"}}})
    assert status == 422 and "'Z'" in json.loads(data)["error"]
    assert ask(site, "PUT", "api/answers" + q, {"answers": {"eyes": {"choice": "A", "notes": "yes"}}})[0] == 200
    status, data, _ = ask(site, "POST", "api/send" + q)
    sent = json.loads(data)
    assert status == 200 and sent["sent"] and "A: Rin's" in sent["message"]
    assert RV.show(f)["sent"] == sent["sent"] and RV.answers(f)["answers"][0]["notes"] == "yes"


def song(path, seconds=6.0, tone_from=1.0, rate=8000):
    """A stereo WAV: silence, then a loud 440 Hz tone from `tone_from` seconds on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(rate)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            v = int(16000 * math.sin(2 * math.pi * 440 * i / rate)) if i >= tone_from * rate else 0
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))


def music_project(root):
    """A 4 s clip at 30 fps from Blender frame 101, of a song whose clip starts at 1.0 s: two shots, a ring, a freeze, a
    lyric text on line 1, the project's hits file and a timeline whose words have text."""
    (root / "audio").mkdir(parents=True)
    song(root / "audio" / "song.wav")
    (root / "mk.toml").write_text("""
[project]
fps = 30
frame0 = 101
duration = 4.0
[audio]
file = "audio/song.wav"
start = 1.0
hits = "audio/hits.json"
lanes = ["kick", "snare"]
[[shot]]
name = "a"
from = 0.0
to = 2.0
[[shot]]
name = "b"
from = 2.0
to = 4.0
[[ring]]
at = 0.5
center = [0.0, 0.0, 1.0]
dur = 0.8
[[freeze]]
from = 1.0
to = 1.5
[[text]]
name = "sung"
lyrics = { line = 1 }
""", encoding="utf-8")
    (root / "audio" / "hits.json").write_text(json.dumps({
        "beats": [0.0, 0.5], "kick": [0.5, {"t": 1.5, "db": -3.0}], "snare": [1.0], "hats": [0.25]}), encoding="utf-8")
    (root / "audio" / "timeline.json").write_text(json.dumps({
        "fps": 30, "bpm": 120.0, "beats": [0.0, 0.5, 1.0], "downbeats": [0.0], "onsets": {"other": [0.75]},
        "lines": [{"start": 1.2, "end": 2.6, "words": [{"text": "SECRETWORD", "start": 1.2, "end": 1.9},
                                                     {"text": "OTHERWORD", "start": 2.0, "end": 2.6}]}]}), encoding="utf-8")


def test_the_music_comes_in_clip_seconds_with_words_by_number_only(tmp_path):
    root = tmp_path / "proj"
    music_project(root)
    m = MU.music(root, tmp_path / "cache")
    assert "SECRETWORD" not in json.dumps(m) and "OTHERWORD" not in json.dumps(m)
    assert m["words"] == [{"line": 1, "word": 1, "start": 1.2, "end": 1.9}, {"line": 1, "word": 2, "start": 2.0, "end": 2.6}]
    assert [(lane["name"], lane["source"]) for lane in m["lanes"]] == [("kick", "hits"), ("snare", "hits"), ("other", "onsets")]
    assert m["lanes"][0]["hits"] == [{"t": 0.5, "db": None}, {"t": 1.5, "db": -3.0}]
    assert m["shots"] == [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}]
    spans = {e["kind"]: (e["from"], e["to"]) for e in m["effects"]}
    assert spans == {"ring": (0.5, 1.3), "freeze": (1.0, 1.5), "text": (1.2, 2.6)}
    assert m["problems"] == []
    if shutil.which("ffmpeg"):
        with wave.open(m["audio"]["path"]) as w:                      # the clip: from 1.0 s of the song, 4 s long
            assert abs(w.getnframes() / w.getframerate() - 4.0) < 0.05 and m["audio"]["offset"] == 0.0
            first = struct.unpack(f"<{2 * 441}h", w.readframes(441))      # its first 10 ms are the tone, not silence
            assert max(abs(v) for v in first) > 8000


def test_music_is_served_for_the_projects_its_pages_were_opened_for(site, tmp_path):
    root = tmp_path / "proj"
    music_project(root)
    f = tmp_path / "talk" / "t.review.toml"
    f.parent.mkdir()
    f.write_text(f'[review]\ntitle = "T"\n[[question]]\nid = "q"\nask = "?"\nmusic = {{ project = "{root}", from = 1.0 }}\n',
                 encoding="utf-8")
    q = "api/music?path=" + urllib.parse.quote(str(root))
    assert ask(site, "GET", q)[0] == 403
    assert ask(site, "POST", "api/open", {"review": str(f)})[0] == 200
    status, data, _ = ask(site, "GET", q)
    m = json.loads(data)
    assert status == 200 and [s["name"] for s in m["shots"]] == ["a", "b"]
    status, wav, _ = ask(site, "GET", m["audio"]["url"])
    assert status == 200 and wav[:4] == b"RIFF"


def test_the_scene_is_the_bake_beside_the_blend_with_the_drafts_mk_post_made(site, tmp_path):
    root = tmp_path / "proj"
    (root / "build").mkdir(parents=True)
    (root / "mk.toml").write_text('[project]\nname = "p"\nfps = 30\nframe0 = 1\nduration = 2.0\nblend = "build/p.blend"\n'
                                  '[[output]]\nname = "9x16"\nsize = [540, 960]\n[[output]]\nname = "1x1"\nsize = [540, 540]\n',
                                  encoding="utf-8")
    (root / "build" / "p.blend").write_bytes(b"blend")
    q = "api/scene?path=" + urllib.parse.quote(str(root))
    assert ask(site, "GET", q)[0] == 403                                           # the page was not opened for it
    assert ask(site, "POST", "api/open", {"project": str(root)})[0] == 200
    status, data, _ = ask(site, "GET", q)
    assert status == 404 and "mk build" in json.loads(data)["error"]
    bake = root / "build" / "p.bake"
    bake.mkdir()
    for name in ("bake.json", "bake.bin", "scene.glb"):
        (bake / name).write_bytes(b"{}")
    (root / "out").mkdir()
    (root / "out" / "p_9x16_draft.mp4").write_bytes(b"")
    os.utime(root / "build" / "p.blend", (1_000_000, 1_000_000))
    got = json.loads(ask(site, "GET", q)[1])
    assert (got["stale"], got["drafts"]) == (False, ["9x16"])
    assert ask(site, "GET", got["json"])[1] == b"{}"
    os.utime(root / "build" / "p.blend")                                         # saved again after the bake
    assert json.loads(ask(site, "GET", q)[1])["stale"] is True


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_a_drafts_index_sets_up_its_decoder_and_points_at_each_frame_in_decode_order(tmp_path):
    mp4 = tmp_path / "d.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "12",
                    "-c:v", "libx264", "-bf", "2", "-g", "6", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    d = DR.index(mp4, tmp_path / "cache")
    record = base64.b64decode(d["description"])
    assert record[0] == 1 and d["codec"] == "avc1." + record[1:4].hex() and (d["width"], d["height"]) == (64, 48)
    samples, data = d["samples"], mp4.read_bytes()
    assert sorted(s[2] for s in samples) == [round(k * 1e6 / 30) for k in range(12)]
    assert [s[2] for s in samples] != sorted(s[2] for s in samples)               # B-frames: decode order is not display order
    keys = [s for s in samples if s[3]]
    assert len(keys) == 2 and keys[0] is samples[0]
    for off, size, _, key in samples:                                             # each sample: length-prefixed NAL units
        nal_types, i = [], off
        while i < off + size:
            n = int.from_bytes(data[i:i + 4], "big")
            nal_types.append(data[i + 4] & 0x1F)
            i += 4 + n
        assert i == off + size and (5 in nal_types) == key                       # an IDR picture in the keyframes only


def check_project(root):
    """A project (frame0 101, 30 fps) with two Blender-free checks on a rendered frame named for its frame number."""
    from PIL import Image
    (root / "renders").mkdir(parents=True)
    Image.new("RGB", (16, 16), (240, 240, 240)).save(root / "renders" / "0130.png")
    (root / "mk.toml").write_text("""[project]
fps = 30
frame0 = 101
duration = 4.0
[[check]]
name = "bright"
metric = "palette"
args = { images = "renders/*.png" }
max = 0.5
[[check]]
name = "dark"
metric = "palette"
args = { images = "renders/*.png" }
min = 0.5
""", encoding="utf-8")


def mk(argv, capsys):
    with pytest.raises(SystemExit):
        MAIN.main(argv)
    return json.loads(capsys.readouterr().out)


def test_the_projects_checks_keep_their_last_results_and_an_ad_hoc_run_leaves_them(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "proj"
    check_project(root)
    kept = root / ".mk" / "checks.json"
    mk(["check", "--project", str(root), "--only", "dark"], capsys)
    first = json.loads(kept.read_text(encoding="utf-8"))["results"]
    assert [r["name"] for r in first] == ["dark"] and first[0]["ok"] is False
    mk(["check", "--project", str(root)], capsys)
    both = json.loads(kept.read_text(encoding="utf-8"))["results"]
    assert [(r["name"], r["ok"]) for r in both] == [("bright", True), ("dark", False)]             # mk.toml's order
    assert both[0]["detail"]["at_frame"] == 130
    before = kept.read_text(encoding="utf-8")
    from PIL import Image
    Image.new("RGB", (16, 16), (0, 0, 0)).save(root / "renders" / "0131.png")      # results change from here on
    mk(["check", "palette", "--args", '{"images": "renders/*.png"}', "--project", str(root)], capsys)
    mk(["check", "--project", str(root), "--frames", "101:110"], capsys)
    assert kept.read_text(encoding="utf-8") == before                    # ad hoc, or over other frames: not the record
    mk(["check", "--project", str(root), "--only", "bright"], capsys)
    again = json.loads(kept.read_text(encoding="utf-8"))["results"]
    assert [r["name"] for r in again] == ["bright", "dark"] and again[1] == both[1]   # an --only run keeps the others


def test_the_checks_tab_and_the_timeline_get_each_worst_moment_in_clip_seconds(site, tmp_path, capsys):
    root = tmp_path / "proj"
    check_project(root)
    with open(root / "mk.toml", "a", encoding="utf-8") as f:
        f.write('[[check]]\nname = "never run"\nmetric = "palette"\nargs = { images = "renders/*.png" }\n')
    mk(["check", "--project", str(root), "--only", "bright", "--only", "dark"], capsys)
    assert ask(site, "POST", "api/open", {"project": str(root)})[0] == 200
    q = "?path=" + urllib.parse.quote(str(root))
    got = json.loads(ask(site, "GET", "api/checks" + q)[1])
    rows = {r["name"]: r for r in got["results"]}
    assert set(rows) == {"bright", "dark", "never run"} and "ran" not in rows["never run"]
    assert rows["dark"]["t"] == pytest.approx(0.9667, abs=1e-4)            # frame 130, frame0 101, 30 fps
    assert rows["dark"]["doc"].startswith("Colour discipline") and got["running"] is False
    marks = json.loads(ask(site, "GET", "api/music" + q)[1])["checks"]
    assert sorted((c["name"], c["ok"], round(c["t"], 3)) for c in marks) == [("bright", True, 0.967), ("dark", False, 0.967)]
