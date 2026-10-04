"""mk ref photos: Wikimedia Commons search, licence metadata, thumbnails, SOURCES.json, contact sheet, polite HTTP. The
network is faked: a stand-in for api.php and the thumbnail server."""
import io
from pathlib import Path
import json
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message

import pytest
from PIL import Image

from mkmmd.cli import main as MAIN
from mkmmd.ref import RefError, RefUsage
from mkmmd.ref import photos as PH


def jpeg(color, size=(96, 72)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def meta(license_name, artist, **extra):
    vals = {"LicenseShortName": license_name, "Artist": artist, "UsageTerms": license_name, **extra}
    return {k: {"value": v} for k, v in vals.items()}


CATALOG = {
    "File:Red car front.jpg": dict(mime="image/jpeg", width=4000, height=3000, color=(200, 30, 30),
                                    extmetadata=meta("CC BY-SA 4.0",
                                                     '<a href="//commons.wikimedia.org/wiki/User:Alice" title="x">Alice &amp; Co</a>',
                                                     LicenseUrl="https://creativecommons.org/licenses/by-sa/4.0",
                                                     AttributionRequired="true", ImageDescription="<p>Front  view</p>")),
    "File:Red car diagram.svg": dict(mime="image/svg+xml", width=2000, height=1000, color=(0, 0, 0),
                                      extmetadata=meta("CC0", "Bob")),
    "File:Red car tiny.jpg": dict(mime="image/jpeg", width=640, height=480, color=(10, 10, 10), extmetadata=meta("CC0", "Cy")),
    "File:Red car side.jpg": dict(mime="image/jpeg", width=3000, height=2000, color=(30, 200, 30),
                                   extmetadata=meta("Public domain", "Bob", AttributionRequired="false")),
    "File:Red car nonfree.jpg": dict(mime="image/jpeg", width=3000, height=2000, color=(5, 5, 5),
                                      extmetadata=meta("Fair use", "Dee", NonFree="true")),
    "File:Red car mystery.jpg": dict(mime="image/jpeg", width=3000, height=2000, color=(9, 9, 9), extmetadata={}),
    "File:Red car rear.png": dict(mime="image/png", width=2000, height=1500, color=(30, 30, 200),
                                   extmetadata=meta("CC0", "Eve \u2014 photographer")),
    "File:Red car roof.jpg": dict(mime="image/jpeg", width=1500, height=1000, color=(200, 200, 30),
                                   extmetadata=meta("CC BY 2.0", "Flo")),
}


class Resp:
    def __init__(self, body, headers=None):
        self.body, self.headers = body, headers or {}

    def read(self, n=None):
        return self.body if n is None else self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeCommons:
    """api.php (list=search paged by sroffset, prop=imageinfo) and the thumbnail server, with scripted failures."""

    def __init__(self, page=3):
        self.page, self.requests, self.script = page, [], []

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        self.requests.append(url)
        assert request.get_header("User-agent", "").startswith("mk-mmd-ref/")           # a descriptive User-Agent
        if self.script:
            fail = self.script.pop(0)
            if fail:
                if isinstance(fail, dict):
                    return Resp(json.dumps(fail).encode())
                code, headers = fail
                msg = Message()
                for k, v in headers.items():
                    msg[k] = v
                raise urllib.error.HTTPError(url, code, "scripted", msg, None)
        parts = urllib.parse.urlparse(url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        if parts.path.endswith("api.php"):
            assert q["format"] == "json" and q["maxlag"] == "5"
            if q.get("list") == "search":
                assert "filetype:bitmap" in q["srsearch"] and q["srnamespace"] == "6"
                titles, off = list(CATALOG), int(q.get("sroffset", 0))
                body = {"query": {"search": [{"ns": 6, "title": t} for t in titles[off:off + self.page]]}}
                if off + self.page < len(titles):
                    body["continue"] = {"sroffset": off + self.page}
                return Resp(json.dumps(body).encode())
            assert q["prop"] == "imageinfo" and "extmetadata" in q["iiprop"]
            pages = []
            for t in q["titles"].split("|"):
                c = CATALOG[t]
                name = t[5:].replace(" ", "_")
                ii = {"url": f"https://upload.example/{name}", "descriptionurl": f"https://commons.example/wiki/{t.replace(' ', '_')}",
                      "width": c["width"], "height": c["height"], "mime": c["mime"],
                      "thumburl": f"https://thumb.example/{q['iiurlwidth']}px-{name}", "extmetadata": c["extmetadata"]}
                pages.append({"title": t, "imageinfo": [ii]})
            return Resp(json.dumps({"query": {"pages": pages}}).encode())
        name = parts.path.rsplit("/", 1)[-1].split("px-", 1)[-1].replace("_", " ")
        return Resp(jpeg(CATALOG["File:" + name]["color"]))


@pytest.fixture
def commons(monkeypatch):
    fake = FakeCommons()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    sleeps = []
    monkeypatch.setattr(PH.time, "sleep", sleeps.append)
    fake.sleeps = sleeps
    return fake


# ---------------------------------------------------------------- pieces
def test_standard_widths_slugs_and_metadata_text():
    assert [PH.snap_width(w) for w in (100, 1000, 1280, 1281, 5000)] == [120, 1280, 1280, 1920, 3840]
    assert PH.slug("1986 Dodge 600 ES Turbo convertible, front left, 08-02-2024.jpg") == \
        "1986_dodge_600_es_turbo_convertible_front_left_08_02_2024"
    assert PH.slug("....jpg") == "photo"
    assert PH.plain('<a href="//x" title="User:A">A &amp; B</a>  <br/> and  more') == "A & B and more"
    assert PH.ascii_text("Eve \u2014 M\u00fcller") == "Eve - Muller"


def test_refusals_say_why():
    ok = {"mime": "image/jpeg", "width": 2000, "extmetadata": meta("CC0", "A")}
    assert PH.refuse(ok, 1000) is None
    assert PH.refuse(dict(ok, mime="image/svg+xml"), 1000) == "not_a_photo"
    assert PH.refuse(dict(ok, width=900), 1000) == "too_small"
    assert PH.refuse(dict(ok, extmetadata=meta("Fair use", "A", NonFree="true")), 1000) == "non_free"
    assert PH.refuse(dict(ok, extmetadata={}), 1000) == "no_licence_data"
    assert PH.describe("File:A b.jpg", dict(ok, descriptionurl="https://p", extmetadata=meta(
        "CC BY 4.0", "<b>Zed</b>", AttributionRequired="true")), "q")["author"] == "Zed"


# ---------------------------------------------------------------- the job
def http(fake, **kw):
    return PH.Http(gap=0.0, opener=fake, sleep=fake.sleeps.append, **kw)


def test_fetch_keeps_photographs_with_licences_and_skips_the_rest(commons, tmp_path):
    out = PH.fetch("red car", tmp_path / "refs" / "red", n=4, min_width=1000, width=1280, http=http(commons))
    files = [p["file"] for p in out["photos"]]
    assert files == ["red_car_front.jpg", "red_car_side.jpg", "red_car_rear.png", "red_car_roof.jpg"]
    assert out["added"] == 4 and out["count"] == 4 and out["failed"] == [] and "short" not in out
    assert out["skipped"] == {"not_a_photo": 1, "too_small": 1, "non_free": 1, "no_licence_data": 1}
    for f in files:
        assert (tmp_path / "refs" / "red" / f).is_file()
    assert Image.open(tmp_path / "refs" / "red" / "red_car_rear.png").format == "JPEG"      # the fake serves JPEG bytes
    src = json.loads((tmp_path / "refs" / "red" / "SOURCES.json").read_text(encoding="utf-8"))
    first = src[0]
    assert [e["file"] for e in src] == files
    assert first["license"] == "CC BY-SA 4.0" and first["author"] == "Alice & Co" and first["attribution_required"] is True
    assert first["page"] == "https://commons.example/wiki/File:Red_car_front.jpg" and first["title"] == "Red car front.jpg"
    assert first["license_url"].startswith("https://creativecommons.org") and first["description"] == "Front view"
    assert first["width"] == 4000 and first["query"] == "red car" and first["use"].startswith("visual reference only")
    assert src[1]["license"] == "Public domain" and src[1]["attribution_required"] is False
    sheet = Image.open(out["sheet"])
    assert sheet.size == (4 * 326 + 6, 282) and out["sheet"].endswith("contact_sheet.jpg")      # one row of four tiles
    json.dumps(out)                                                                           # plain data, no image content
    assert "thumb.example/1280px-" in " ".join(commons.requests)


def test_search_pages_on_until_enough_photos_fit(commons, tmp_path):
    out = PH.fetch("red car", tmp_path, n=2, http=http(commons))
    assert out["added"] == 2
    assert sum("list=search" in u for u in commons.requests) == 2          # three titles per page: the second page was needed
    assert sum("prop=imageinfo" in u for u in commons.requests) == 2


def test_a_repeat_adds_nothing_and_another_query_adds_its_own(commons, tmp_path):
    PH.fetch("red car", tmp_path, n=3, http=http(commons))
    n_requests = len(commons.requests)
    again = PH.fetch("red car", tmp_path, n=3, http=http(commons))
    assert again["added"] == 0 and again["already_in_set"] == 3 and len(commons.requests) == n_requests
    more = PH.fetch("red car", tmp_path, n=4, http=http(commons))
    assert more["added"] == 1 and more["count"] == 4
    other = PH.fetch("red car roof", tmp_path, n=1, http=http(commons))
    assert other["added"] == 0 and other["skipped"]["already_in_set"] >= 1 and other["short"].startswith("only 0 of 1")
    assert len(json.loads((tmp_path / "SOURCES.json").read_text(encoding="utf-8"))) == 4


def test_a_set_with_too_few_fitting_photos_says_so(commons, tmp_path):
    out = PH.fetch("red car", tmp_path, n=9, min_width=2500, http=http(commons))
    assert out["added"] == 2 and out["short"].startswith("only 2 of 9") and out["skipped"]["too_small"] >= 3


def test_a_width_wikimedia_does_not_serve_snaps_up(commons, tmp_path):
    out = PH.fetch("red car", tmp_path, n=1, width=1000, http=http(commons))
    assert out["width"] == 1280 and "1000 px is not a thumbnail width" in out["width_note"]
    assert any("iiurlwidth=1280" in u for u in commons.requests)


def test_bad_input_is_a_usage_error(commons, tmp_path):
    with pytest.raises(RefUsage):
        PH.fetch("   ", tmp_path, http=http(commons))
    with pytest.raises(RefUsage):
        PH.fetch("red car", tmp_path, n=0, http=http(commons))


def test_an_unreadable_thumbnail_is_reported_not_stored(commons, tmp_path, monkeypatch):
    real = commons.__call__

    def broken(request, timeout=None):
        if "thumb.example" in request.full_url:
            return Resp(b"<html>not an image</html>")
        return real(request, timeout)
    out = PH.fetch("red car", tmp_path, n=1, http=PH.Http(gap=0.0, opener=broken, sleep=lambda s: None))
    assert out["added"] == 0 and len(out["failed"]) >= 1 and "not a readable image" in out["failed"][0]["error"]
    assert not list(tmp_path.glob("*.jpg")) and not (tmp_path / "SOURCES.json").exists()


# ---------------------------------------------------------------- polite HTTP
def test_requests_are_spaced():
    now = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s
    h = PH.Http(gap=1.0, opener=lambda r, timeout: Resp(b"{}"), sleep=sleep, clock=lambda: now[0])
    h.get("https://x/a")
    now[0] += 0.25
    h.get("https://x/b")
    now[0] += 5
    h.get("https://x/c")
    assert sleeps == [pytest.approx(0.75)]                    # only the quick second request waited


def test_retry_after_is_honoured_on_429_and_backoff_doubles_without_it():
    calls, sleeps = [], []
    script = [(429, {"Retry-After": "7"}), (503, {}), (429, {}), None]

    def opener(request, timeout=None):
        calls.append(request.full_url)
        step = script[len(calls) - 1]
        if step:
            msg = Message()
            for k, v in step[1].items():
                msg[k] = v
            raise urllib.error.HTTPError(request.full_url, step[0], "x", msg, None)
        return Resp(b"ok")
    h = PH.Http(gap=0.0, opener=opener, sleep=sleeps.append)
    assert h.get("https://x/y") == b"ok" and len(calls) == 4
    assert sleeps == [7.0, 4.0, 8.0] and h.retries == 3 and h.waited == pytest.approx(19.0)


def test_waits_are_capped_and_hopeless_requests_fail_clearly():
    sleeps = []

    def always_429(request, timeout=None):
        msg = Message()
        msg["Retry-After"] = "9999"
        raise urllib.error.HTTPError(request.full_url, 429, "x", msg, None)
    h = PH.Http(gap=0.0, tries=3, opener=always_429, sleep=sleeps.append)
    with pytest.raises(RefError, match="after 3 tries .*HTTP 429"):
        h.get("https://x/y?secret=1")
    assert sleeps == [PH.MAX_WAIT] * 3
    gone = PH.Http(gap=0.0, opener=lambda r, timeout: (_ for _ in ()).throw(
        urllib.error.HTTPError(r.full_url, 404, "x", Message(), None)), sleep=sleeps.append)
    with pytest.raises(RefUsage, match="not found: https://x/y"):
        gone.get("https://x/y?q=1")
    bad = PH.Http(gap=0.0, opener=lambda r, timeout: (_ for _ in ()).throw(
        urllib.error.HTTPError(r.full_url, 400, "x", Message(), None)), sleep=sleeps.append)
    with pytest.raises(RefError, match="HTTP 400"):
        bad.get("https://x/y")


def test_network_errors_back_off_too():
    sleeps, n = [], []

    def flaky(request, timeout=None):
        n.append(1)
        if len(n) < 3:
            raise urllib.error.URLError("down")
        return Resp(b"fine")
    h = PH.Http(gap=0.0, opener=flaky, sleep=sleeps.append)
    assert h.get("https://x/y") == b"fine" and sleeps == [2.0, 4.0]


def test_a_downloads_size_is_capped():
    h = PH.Http(gap=0.0, opener=lambda r, timeout: Resp(b"x" * 100), sleep=lambda s: None)
    with pytest.raises(RefError, match="more than"):
        h.get("https://x/y", max_bytes=10)


def test_maxlag_is_waited_out(commons, tmp_path):
    commons.script = [{"error": {"code": "maxlag", "info": "lagged", "lag": 3}}, None]
    out = PH.fetch("red car", tmp_path, n=1, http=http(commons))
    assert out["added"] == 1 and out["http"]["retries"] == 1 and out["http"]["waited_s"] >= 5.0
    commons.script = [{"error": {"code": "badvalue", "info": "nope"}}]
    with pytest.raises(RefError, match="badvalue"):
        PH.fetch("red cars", tmp_path / "b", n=1, http=http(commons))


# ---------------------------------------------------------------- the command
def run_cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def test_cli_downloads_into_the_projects_refs_folder(commons, tmp_path, capsys, monkeypatch):
    (tmp_path / "mk.toml").write_text('[project]\nname = "carproj"\nfps = 30\nframe0 = 1\nduration = 1\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code, out = run_cli(["ref", "photos", "red car", "--set", "redcar", "--n", "2", "--min-width", "1200"], capsys)
    assert code == 0 and out["set"] == "redcar" and out["project"] == "carproj" and out["added"] == 2
    assert out["dir"] == str(tmp_path.resolve() / "refs" / "redcar")
    assert (tmp_path / "refs" / "redcar" / "SOURCES.json").is_file() and (tmp_path / "refs" / "redcar" / "contact_sheet.jpg").is_file()
    assert len(json.dumps(out)) < 4000                                    # file names and credits, never image content
    assert all(set(p) == {"file", "title", "page", "license", "author", "width", "height"} for p in out["photos"])


def test_cli_outside_a_project_uses_the_cache_and_names_the_set_after_the_query(commons, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    code, out = run_cli(["ref", "photos", "Red Car, front view", "--n", "1"], capsys)
    assert code == 0 and out["project"] is None and out["set"] == "red_car_front_view"
    assert out["dir"] == str(tmp_path / "cache" / "ref" / "photos" / "red_car_front_view")
    assert (tmp_path / "cache" / "ref" / "photos" / "red_car_front_view" / "red_car_front.jpg").is_file()


def test_cli_errors(commons, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    code, out = run_cli(["ref", "photos", "  ", "--set", "x"], capsys)
    assert code == 2 and "query" in out["error"]
    code, out = run_cli(["ref", "photos", "red car", "--set", "../evil"], capsys)
    assert code == 2
    commons.script = [(400, {})] * 10
    code, out = run_cli(["ref", "photos", "red car", "--set", "x"], capsys)
    assert code == 3 and "HTTP 400" in out["error"]


# ---------------------------------------------------------------- mk look --ref: photos beside renders
def solid(path, color, size):
    Image.new("RGB", size, color).save(path, quality=100)
    return path


def shots_in(folder, views, size=(200, 200), frames=(1,)):
    """What the Blender `look` op returns: a list of {path, view, size, frame}, one image each."""
    out = []
    for i, v in enumerate(views):
        for f in frames:
            out.append({"path": str(solid(folder / f"{v}_{f}.jpg", (0, 200, 0) if i % 2 == 0 else (0, 0, 200), size)),
                        "view": v, "size": "sq", "frame": f})
    return out


def test_ref_paths_split_comma_lists_and_try_the_project(tmp_path, monkeypatch):
    from mkmmd.cli import look as LK
    from mkmmd.cli.common import UsageError
    from mkmmd.project import Project
    (tmp_path / "refs" / "car").mkdir(parents=True)
    a, b = solid(tmp_path / "refs" / "car" / "a.jpg", (9, 9, 9), (8, 8)), solid(tmp_path / "b.jpg", (9, 9, 9), (8, 8))
    (tmp_path / "mk.toml").write_text('[project]\nname = "p"\nfps = 30\nframe0 = 1\nduration = 1\n', encoding="utf-8")
    proj = Project.load(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert LK.ref_paths(["refs/car/a.jpg, " + str(b), str(a)], proj) == [a, b, a]        # from the project folder, absolute
    with pytest.raises(UsageError, match="no such image 'missing.jpg'"):
        LK.ref_paths(["missing.jpg"], proj)
    with pytest.raises(UsageError, match="at least one"):
        LK.ref_paths([" , "], None)


def test_photos_pair_with_views_and_the_shorter_list_repeats(tmp_path):
    from mkmmd.cli import look as LK
    shots = shots_in(tmp_path, ["3q", "left", "front"], frames=(1, 2))
    names = lambda pairs: [(p.name, s["view"], s["frame"]) for p, s in pairs]             # noqa: E731
    p = [tmp_path / n for n in ("p1.jpg", "p2.jpg", "p3.jpg")]
    assert names(LK.pair_refs(p, shots)) == [("p1.jpg", "3q", 1), ("p2.jpg", "left", 1), ("p3.jpg", "front", 1)]
    assert names(LK.pair_refs(p[:1], shots)) == [("p1.jpg", "3q", 1), ("p1.jpg", "left", 1), ("p1.jpg", "front", 1)]
    assert names(LK.pair_refs(p[:2], shots)) == [("p1.jpg", "3q", 1), ("p2.jpg", "left", 1), ("p1.jpg", "front", 1)]
    one_view = shots_in(tmp_path, ["3q"])
    assert names(LK.pair_refs(p, one_view)) == [("p1.jpg", "3q", 1), ("p2.jpg", "3q", 1), ("p3.jpg", "3q", 1)]
    assert len(LK.pair_refs(p[:2], shots)) == 3                                           # photo 1 comes round again


def test_the_side_by_side_sheet_puts_photo_and_render_at_one_height(tmp_path):
    from mkmmd.cli import look as LK
    renders = shots_in(tmp_path, ["3q", "left"], size=(200, 300))
    photos = [solid(tmp_path / "wide.jpg", (255, 0, 0), (400, 200)), solid(tmp_path / "tall.jpg", (255, 255, 0), (150, 300))]
    out = LK.side_by_side(LK.pair_refs(photos, renders), tmp_path / "ref.jpg")
    sheet = Image.open(out)
    top, pad = 22, 8                                                      # label band and padding
    assert sheet.size == (600 + 200 + 3 * pad, 2 * (300 + top + pad) + pad)           # the wide photo is 600 at height 300
    y0 = pad + top
    assert sheet.getpixel((pad + 300, y0 + 150))[0] > 200                 # the wide photo, scaled to 600 x 300
    assert sheet.getpixel((2 * pad + 600 + 100, y0 + 150))[1] > 150       # beside it the green render
    y1 = pad + 300 + top + pad + top
    assert sheet.getpixel((pad + 75, y1 + 150))[0] > 200 and sheet.getpixel((pad + 75, y1 + 150))[1] > 200   # yellow photo
    assert sheet.getpixel((2 * pad + 150 + 100, y1 + 150))[2] > 150       # blue render


def test_cli_look_writes_ref_jpg_next_to_the_renders(tmp_path, capsys, monkeypatch):
    from mkmmd import bridge
    calls = []

    def fake_look(op, args, **kw):
        calls.append((op, args))
        Path(args["out"]).mkdir(parents=True, exist_ok=True)                      # the Blender op makes the folder
        return shots_in(tmp_path, [v["name"] for v in args["views"]], size=(120, 120), frames=args["frames"])
    monkeypatch.setattr(bridge, "run", fake_look)
    a, b = solid(tmp_path / "a.jpg", (255, 0, 0), (240, 160)), solid(tmp_path / "b.jpg", (255, 0, 255), (160, 240))
    code, out = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--view", "35:12,left", "--ref", f"{a},{b}",
                         "--out", str(tmp_path / "look")], capsys)
    assert code == 0 and calls[0][0] == "look" and out["ref"] == str(tmp_path / "look" / "ref.jpg")
    assert [(Path(p["photo"]).name, p["view"], p["frame"]) for p in out["ref_pairs"]] == [("a.jpg", "35_12", 7), ("b.jpg", "left", 7)]
    sheet = Image.open(out["ref"])
    assert sheet.height == 2 * (120 + 22 + 8) + 8 and sheet.width == 180 + 120 + 24     # photo b is 80 x 120 ... a is 180 wide
    code, out = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--view", "3q", "--ref", "nothing.jpg"], capsys)
    assert code == 2 and "no such image" in out["error"] and len(calls) == 1             # refused before any rendering
    code, out = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--view", "3q", "--out", str(tmp_path / "plain")], capsys)
    assert code == 0 and "ref" not in out
