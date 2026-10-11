"""The site's server (mkmmd.site): it answers only with its token and to its own address, serves files only from the
folders a page was opened for (with byte ranges, which audio and video seek with), a review goes through it the way
the page uses it (opened, answers checked and kept, sent), and a project's music comes back in clip seconds: the clip's
audio cut from the song where [audio] starts, the hits lanes the project names, the shots and effects, and the words by
number, never their text."""
import http.client
import json
import math
import shutil
import struct
import threading
import urllib.parse
import wave

import pytest

from mkmmd import review as RV
from mkmmd.site import api as API
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
