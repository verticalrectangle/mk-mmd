"""The site's server (mkmmd.site): it answers only with its token and to its own address, serves files only from the
folders a page was opened for (with byte ranges, which audio and video seek with), and a review goes through it the
way the page uses it: opened, answers checked and kept, sent."""
import http.client
import json
import threading
import urllib.parse

import pytest

from mkmmd import review as RV
from mkmmd.site import api as API
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
