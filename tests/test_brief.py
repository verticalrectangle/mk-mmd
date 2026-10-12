"""Briefs (mkmmd.brief, `mk brief`, the site's Brief tab): a new video is a project made from its song, its clip the
whole song; the part of the song chosen on the tab becomes the clip, rewritten in mk.toml in place; a brief that does
not fit is refused whole, naming each problem; the agent reads each part on the song and on the clip with the lines,
words and bars it covers, never their text; the brief goes through the server as the page uses it (the lyrics kept and
never sent back, pictures kept, the words timed by `mk timeline analyze` on the clip, with the lyrics), and a sent brief
is typed into the pane of the agent that opened the page."""
import base64
import http.client
import io
import json
import math
import os
import shutil
import struct
import subprocess
import threading
import tomllib
import types
import urllib.parse
import wave
from pathlib import Path

import pytest
from PIL import Image

from mkmmd import brief as BR
from mkmmd.cli import main as MAIN
from mkmmd.project import Project
from mkmmd.site import api as API
from mkmmd.site import server as S

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def tone(path, seconds, rate=8000):
    """A stereo WAV of a 440 Hz tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<hh", v, v) for v in
                               (int(9000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(seconds * rate)))))
    return path


def video(tmp_path, seconds=8.0):
    """A new video's project (mk brief new) from a tone `seconds` long."""
    return Path(BR.new(tmp_path / "video", tone(tmp_path / "song.wav", seconds))["project"])


def mk(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    s = S.Site(routes=API.routes())
    httpd = S.serve(s)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield s, httpd.server_port
    httpd.shutdown()
    httpd.server_close()


def ask(site, method, rest, body=None):
    s, port = site
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    c.request(method, f"/{s.token}/{rest}", body=None if body is None else json.dumps(body).encode(),
              headers={"Host": f"127.0.0.1:{port}"})
    r = c.getresponse()
    status, data = r.status, r.read()
    c.close()
    return status, (json.loads(data) if data[:1] in (b"{", b"[") else data)


def png():
    out = io.BytesIO()
    Image.new("RGB", (6, 4), (200, 40, 90)).save(out, "PNG")
    return out.getvalue()


def test_a_new_video_is_a_project_of_its_whole_song_with_an_empty_brief(tmp_path, capsys):
    song = tone(tmp_path / "in" / "My Song.wav", 6.0)
    root = tmp_path / "videos" / "blue"
    code, out = mk(["brief", "new", str(root), "--song", str(song), "--no-open"], capsys)
    assert code == 0 and out["brief"] == str(root / "brief" / "brief.json")
    proj = Project.load(root)
    assert proj.data["audio"] == {"file": "audio/My Song.wav", "start": 0.0}
    assert proj.duration == pytest.approx(6.0, abs=1e-3) and (proj.fps, proj.frame0) == (30, 91)   # 3 s of pre-roll
    assert (root / "audio" / "My Song.wav").read_bytes() == song.read_bytes()
    assert BR.read(root)["parts"] == [] and BR.read(root)["sent"] is None
    before = (root / "mk.toml").read_text(encoding="utf-8")
    code, out = mk(["brief", "new", str(root), "--song", str(song), "--no-open"], capsys)
    assert code == 2 and "is a project already" in out["error"]
    assert (root / "mk.toml").read_text(encoding="utf-8") == before


def test_the_part_of_the_song_is_the_clip_rewritten_in_place(tmp_path):
    """Only [audio] start and [project] duration change (a missing start is added under [audio]); [scene] start, the
    tables after [audio], every comment stay; a part outside the song, or shorter than a second, changes nothing."""
    root = video(tmp_path, 6.0)
    text = ('# keep me\n[project]\nname = "x"\nfps = 30\nframe0 = 91\nduration = 6.0  # the clip\n\n'
            '[scene]\nstart = 10\n\n[audio]\nfile = "audio/song.wav"\nhits = "h.json"   # lanes\n\n'
            '[audio.master]\nlane = "808"\n\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 1.0\n')
    (root / "mk.toml").write_text(text, encoding="utf-8")
    assert BR.set_clip(root, 1.5, 4.25) == (1.5, 4.25)
    new = (root / "mk.toml").read_text(encoding="utf-8")
    assert new == text.replace("duration = 6.0  # the clip", "duration = 2.75  # the clip").replace(
        '[audio]\nfile', '[audio]\nstart = 1.5\nfile')
    BR.set_clip(root, 0.25, 6.0)
    data = tomllib.loads((root / "mk.toml").read_text(encoding="utf-8"))
    assert (data["audio"]["start"], data["project"]["duration"], data["scene"]["start"]) == (0.25, 5.75, 10)
    kept = (root / "mk.toml").read_text(encoding="utf-8")
    for a, b in ((4.0, 3.0), (-1.0, 2.0), (1.0, 7.5), (2.0, 2.5)):
        with pytest.raises(BR.BriefError):
            BR.set_clip(root, a, b)
    assert (root / "mk.toml").read_text(encoding="utf-8") == kept
    (root / "mk.toml").write_text('audio = { file = "audio/song.wav" }\n[project]\nfps = 30\nframe0 = 91\nduration = 6.0\n',
                                  encoding="utf-8")
    with pytest.raises(BR.BriefError, match=r"no \[audio\] table: set \[audio\] start = 1.0 by hand"):
        BR.set_clip(root, 1.0, 3.0)


def test_a_brief_that_does_not_fit_is_refused_whole_naming_each_problem(tmp_path):
    root = video(tmp_path, 6.0)
    good = {"language": "en", "video": {"story": "two twins", "sizes": ["9x16", "1x1"]},
            "parts": [{"id": "p1", "from": 1.0, "to": 2.0, "note": "a", "pictures": [], "echo": None}]}
    kept = BR.save(root, good)
    bad = {"language": "fr", "video": {"sizes": ["4x3"], "mood": "x", "pictures": ["../mk.toml"]}, "extra": 1,
           "parts": [{"id": "p1", "from": 1.0, "to": 3.0, "note": "", "pictures": [], "echo": None},
                     {"id": "p2", "from": 2.5, "to": 4.0, "note": "", "pictures": [], "echo": {"of": "p3", "how": "same"}},
                     {"id": "p3", "from": 5.0, "to": 7.0, "note": "", "pictures": [], "echo": {"of": "p1", "how": "louder"}},
                     {"id": "p1", "from": 0.0, "to": 0.5, "note": 4, "pictures": [], "echo": None}]}
    with pytest.raises(BR.BriefError) as e:
        BR.save(root, bad)
    problems = str(e.value).splitlines()
    for needle in ("unknown key 'extra'", "language 'fr'", "video sizes", "video: unknown key 'mood'", "'../mk.toml'",
                   "p2 (0:02.5) overlaps p1", "comes back from 'p3', which is no part before it",
                   "part 3: 0:05.0-0:07.0 is not a stretch of the song", "part 3: echo is", "part 4: id 'p1'",
                   "part 4 note is text"):
        assert sum(needle in p for p in problems) == 1, needle
    assert BR.read(root) == kept


def timeline(root):
    """A timeline of the clip 1.0-7.0 s of the song: bars at clip 0, 2, 4; three lines whose words have text."""
    (root / "audio" / "timeline.json").write_text(json.dumps({
        "audio": {"file": "x", "start": 1.0, "duration": 6.0}, "bpm": 120.0,
        "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], "downbeats": [0.0, 2.0, 4.0],
        "lines": [{"words": [{"text": "SECRET", "start": 0.5, "end": 1.0}, {"text": "WORDS", "start": 1.1, "end": 1.6}]},
                  {"words": [{"text": "SECRET", "start": 2.5, "end": 3.0}]},
                  {"words": [{"text": "SECRET", "start": 4.2, "end": 4.6}, {"text": "WORDS", "start": 4.7, "end": 5.2}]}],
        "words_source": "lyrics file"}), encoding="utf-8")


def test_the_agent_reads_each_part_on_the_song_and_the_clip_never_the_words(tmp_path):
    root = video(tmp_path, 8.0)
    BR.set_clip(root, 1.0, 7.0)
    timeline(root)                                                   # in song seconds: lines 1.5-2.6, 3.5-4.0, 5.2-6.2
    BR.save(root, {"language": "en", "video": {"cast": "Rin and Len", "sizes": ["9x16"]}, "parts": [
        {"id": "p1", "from": 1.4, "to": 2.7, "note": "she walks in", "pictures": [], "echo": None},
        {"id": "p2", "from": 3.4, "to": 6.3, "note": "the hook", "pictures": [], "echo": {"of": "p1", "how": "bigger"}},
        {"id": "p3", "from": 7.2, "to": 7.8, "note": "an outro", "pictures": [], "echo": None}]})
    b = BR.show(root)
    assert "SECRET" not in json.dumps(b) and "WORDS" not in json.dumps(b)
    p1, p2, p3 = b["parts"]
    assert (p1["clip"], p1["lines"], p1["words"], p1["bars"]) == ([0.4, 1.7], [1, 1], [[1, 1], [1, 2]], [1, 1])
    assert (p2["clip"], p2["lines"], p2["words"], p2["bars"]) == ([2.4, 5.3], [2, 3], [[2, 1], [3, 2]], [2, 3])
    assert p2["echo"] == {"part": 1, "id": "p1", "how": "bigger"} and p1["in_clip"] and not p3["in_clip"]
    assert b["open"] == [{"song": [2.7, 3.4], "clip": [1.7, 2.4]}, {"song": [6.3, 7.0], "clip": [5.3, 6.0]}]
    assert b["timeline"]["matches_clip"] and (b["timeline"]["lines"], b["timeline"]["bars"]) == (3, 3)
    text = BR.message(root)
    assert "Part 1 (line 1, 0:01.4-0:02.7): she walks in" in text and "who's in it: Rin and Len" in text
    assert "Part 2 (lines 2-3, 0:03.4-0:06.3, comes back from part 1, bigger): the hook" in text
    assert "Part 3 (0:07.2-0:07.8, outside the part of the song): an outro" in text and "SECRET" not in text
    BR.set_clip(root, 1.0, 6.0)
    assert not BR.show(root)["timeline"]["matches_clip"]


def test_the_brief_goes_through_the_server_as_the_page_uses_it(site, tmp_path):
    root = video(tmp_path, 6.0)
    q = "?path=" + urllib.parse.quote(str(root))
    assert ask(site, "GET", "api/brief" + q)[0] == 403                              # its page was not opened
    assert ask(site, "POST", "api/open", {"project": str(root)})[0] == 200
    status, B = ask(site, "GET", "api/brief" + q)
    assert status == 200 and B["song"]["seconds"] == pytest.approx(6.0, abs=1e-3) and B["clip"] == {"start": 0.0, "end": 6.0}
    assert (B["lyrics"], B["timeline"], B["shots"], B["sizes"]) == (None, None, 0, ["9x16", "1x1", "16x9"])
    assert ask(site, "GET", B["song"]["url"])[1][:4] == b"RIFF"                      # the whole song, to play
    status, got = ask(site, "PUT", "api/lyrics" + q, {"text": "SECRET one\n\n  SECRET two \n"})
    assert (status, got) == (200, {"lines": 2})
    assert (tmp_path / "video" / "audio" / "lyrics.txt").read_text(encoding="utf-8") == "SECRET one\nSECRET two\n"
    timeline(root)                                     # the page gets the lines in song seconds, by number, never text
    B = ask(site, "GET", "api/brief" + q)[1]
    assert b"SECRET" not in json.dumps(B).encode() and b"WORDS" not in json.dumps(B).encode()
    assert B["timeline"]["lines"][0] == {"line": 1, "start": 1.5, "end": 2.6, "words": [[1.5, 2.0], [2.1, 2.6]]}
    assert (B["timeline"]["start"], B["timeline"]["downbeats"]) == (1.0, [1.0, 3.0, 5.0])
    status, pic = ask(site, "POST", "api/picture" + q, {"data": "data:image/png;base64," + base64.b64encode(png()).decode()})
    assert status == 200 and pic["path"].startswith("brief/pictures/") and pic["path"].endswith(".png")
    assert ask(site, "GET", pic["url"])[1][:8] == b"\x89PNG\r\n\x1a\n"
    tiff = io.BytesIO()
    Image.new("RGB", (4, 4)).save(tiff, "TIFF")                                    # a picture no page shows
    for data in (b"words", tiff.getvalue()):
        assert ask(site, "POST", "api/picture" + q, {"data": "data:image/png;base64," + base64.b64encode(data).decode()})[0] == 422
    brief = {"language": "en", "video": {"pictures": [pic["path"]]},
             "parts": [{"id": "p1", "from": 1.0, "to": 2.0, "note": "a", "pictures": [pic["path"]], "echo": None}]}
    assert ask(site, "PUT", "api/brief" + q, {"brief": brief})[0] == 200
    brief["parts"].append({"id": "p2", "from": 1.5, "to": 3.0, "note": "", "pictures": [], "echo": None})
    status, got = ask(site, "PUT", "api/brief" + q, {"brief": brief})
    assert status == 422 and "p2 (0:01.5) overlaps p1" in got["error"] and len(BR.read(root)["parts"]) == 1
    status, B = ask(site, "GET", "api/brief" + q)
    assert set(B["pictures"]) == {pic["path"]} and B["lyrics"] == 2
    assert ask(site, "PUT", "api/clip" + q, {"start": 1.0, "end": 5.0})[1] == {"clip": {"start": 1.0, "end": 5.0}}
    assert (Project.load(root).data["audio"]["start"], Project.load(root).duration) == (1.0, 4.0)
    with open(os.path.join(root, "mk.toml"), "a", encoding="utf-8") as f:
        f.write('\n[[shot]]\nname = "a"\nfrom = 0.0\nto = 4.0\n')
    status, got = ask(site, "PUT", "api/clip" + q, {"start": 0.0, "end": 5.0})
    assert status == 409 and "1 shot on its clip" in got["error"] and Project.load(root).data["audio"]["start"] == 1.0
    assert ask(site, "PUT", "api/clip" + q, {"start": 0.0, "end": 5.0, "force": True})[0] == 200


class Run:
    """Stands in for `mk timeline analyze`: records its command, writes what it would print, ends when told."""
    made = []

    def __init__(self, argv, stdout, stderr, **kw):
        self.argv, self.code = argv, None
        stdout.write(json.dumps({"out": "audio/timeline.json"}))
        stderr.write("timeline: stems\ntimeline: whisper\n")
        Run.made.append(self)

    def poll(self):
        return self.code


def test_timing_the_words_runs_the_timeline_on_the_clip_with_the_lyrics(site, tmp_path, monkeypatch):
    names = {k: getattr(subprocess, k) for k in dir(subprocess) if not k.startswith("_")}
    monkeypatch.setattr(API, "subprocess", types.SimpleNamespace(**{**names, "Popen": Run}))     # the site's runs only
    Run.made.clear()
    root = video(tmp_path, 6.0)
    q = "?path=" + urllib.parse.quote(str(root))
    assert ask(site, "POST", "api/open", {"project": str(root)})[0] == 200
    ask(site, "PUT", "api/lyrics" + q, {"text": "one\ntwo\n"})
    status, got = ask(site, "POST", "api/words" + q)
    assert status == 200 and got["running"] and got["stage"] == "whisper"
    assert Run.made[0].argv[-6:] == ["timeline", "analyze", "--project", str(root), "--lyrics",
                                     str(root / "audio" / "lyrics.txt")]
    assert ask(site, "POST", "api/words" + q)[0] == 409                              # one run at a time
    assert ask(site, "PUT", "api/clip" + q, {"start": 1.0, "end": 5.0})[0] == 409     # nor a part moved under it
    Run.made[0].code = 0
    done = ask(site, "GET", "api/words" + q)[1]
    assert (done["running"], done["exit"], done["error"], done["stage"]) == (False, 0, None, "whisper")
    ask(site, "PUT", "api/brief" + q, {"brief": {"language": "other", "video": {}, "parts": []}})
    ask(site, "POST", "api/words" + q)
    assert Run.made[1].argv[-3:] == ["--project", str(root), "--no-words"]           # another language: the bars only
    Run.made[1].code = 2
    (tmp_path / "video" / ".mk" / "words.json").write_text('{"error": "no song there"}', encoding="utf-8")
    assert ask(site, "GET", "api/words" + q)[1]["error"] == "no song there"


def test_a_sent_brief_is_typed_into_the_pane_of_the_agent_that_opened_its_page(site, tmp_path, monkeypatch):
    log = tmp_path / "tern.log"
    (tmp_path / "bin").mkdir()
    tern = tmp_path / "bin" / "tern"                                         # records what it is asked to do
    tern.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\n', encoding="utf-8")
    tern.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    root = video(tmp_path, 6.0)
    q = "?path=" + urllib.parse.quote(str(root))
    BR.save(root, {"language": "en", "video": {"story": "a rainy rooftop"},
                   "parts": [{"id": "p1", "from": 1.0, "to": 2.0, "note": "slow " * 400, "pictures": [], "echo": None}]})
    assert ask(site, "POST", "api/open", {"project": str(root), "agent": None})[0] == 200
    status, got = ask(site, "POST", "api/send_brief" + q)
    assert status == 200 and got["delivered"] is None and not log.exists()
    assert got["sent"] and BR.read(root)["sent"] == got["sent"]                       # a person's page: kept, nobody told
    assert ask(site, "POST", "api/open", {"project": str(root), "agent": {"pane": "77", "window": "w1"}})[0] == 200
    assert ask(site, "POST", "api/send_brief" + q)[1]["delivered"] == "pane"
    typed, enter = log.read_text(encoding="utf-8").splitlines()
    assert typed.startswith("send --window w1 77 paste Brief sent:") and "the story: a rainy rooftop" in typed
    assert typed.endswith(f"... (all of it: mk brief show --project {root})") and len(typed) < API.TELL_MAX + 40
    assert enter == "send --window w1 77 keys Enter"
