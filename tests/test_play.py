"""mk play: the cut's videos are found per preset in the outputs' order; the player setting's command line gets them where
{files} stands (else last), with {chapters}, {frame0} and {fps} filled in; the shots reach the player as chapters it
reads back in order, without the plates; without a player setting each video goes to the system's opener."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mkmmd.cli import main as MAIN
from mkmmd.cli import play as PLAY
from mkmmd.cli.common import UsageError
from mkmmd.project import Project

SHOTS = ('[[shot]]\nname = "b=c;d"\nfrom = 0.5\nto = 1.0\n'
         '[[shot]]\nname = "plate"\nplate = true\n'
         '[[shot]]\nname = "a#1"\nfrom = 0.0\nto = 0.5\n')


def _project(tmp_path, shots=""):
    (tmp_path / "mk.toml").write_text(
        '[project]\nname = "t"\nfps = 30\nframe0 = 61\nduration = 1.0\n'
        '[[output]]\nname = "wide"\nsize = [1920, 1080]\n[[output]]\nname = "tall"\nsize = [1080, 1920]\n' + shots,
        encoding="utf-8")
    (tmp_path / "out").mkdir(exist_ok=True)
    return Project.load(tmp_path / "mk.toml")


def test_the_videos_are_found_per_preset_in_the_outputs_order(tmp_path):
    proj = _project(tmp_path)
    for name in ("t_tall_draft.mp4", "t_wide_draft.mp4", "t_wide.mp4"):     # final: no preset in the name
        (tmp_path / "out" / name).write_bytes(b"")
    found = PLAY.videos(proj)
    assert [(o, p.name) for o, p in found["draft"]] == [("wide", "t_wide_draft.mp4"), ("tall", "t_tall_draft.mp4")]
    assert [o for o, _ in found["final"]] == ["wide"]
    assert "preview" not in found


@pytest.mark.parametrize("template, expected", [
    ("tvb --split right --chapters {chapters} --first-frame {frame0}",
     ["tvb", "--split", "right", "--chapters", "/p/shots.ffmeta", "--first-frame", "61", "/v/a b.mp4", "/v/c.mp4"]),
    ("mpv --chapters-file={chapters} {files} --fps={fps}",
     ["mpv", "--chapters-file=/p/shots.ffmeta", "/v/a b.mp4", "/v/c.mp4", "--fps=29.97"]),
    ("'/opt/my player/run' -x", ["/opt/my player/run", "-x", "/v/a b.mp4", "/v/c.mp4"]),
])
def test_the_player_setting_gets_the_videos_where_files_stands_else_last_with_the_placeholders_filled(template,
                                                                                                       expected):
    files = [Path("/v/a b.mp4"), Path("/v/c.mp4")]
    assert PLAY.player_argv(template, files, Path("/p/shots.ffmeta"), 61, 29.97) == expected


def test_a_player_setting_that_is_not_a_command_line_is_a_usage_error():
    with pytest.raises(UsageError, match="not a command line"):
        PLAY.player_argv("mpv 'unbalanced", [Path("/v/a.mp4")], Path("/p/s.ffmeta"), 0, 30)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not available")
def test_the_shots_reach_the_player_as_chapters_it_reads_back_in_order_without_the_plates(tmp_path):
    f = tmp_path / "shots.ffmeta"
    f.write_text(PLAY.chapters(_project(tmp_path, SHOTS)), encoding="utf-8")
    r = subprocess.run(["ffprobe", "-v", "error", "-f", "ffmetadata", "-i", str(f), "-show_chapters", "-of", "json"],
                       capture_output=True, text=True, check=True)
    got = [(c["tags"]["title"], float(c["start_time"]), float(c["end_time"])) for c in json.loads(r.stdout)["chapters"]]
    assert got == [("a#1", 0.0, 0.5), ("b=c;d", 0.5, 1.0)]                   # names FFMETADATA must escape


def _fake_player(folder, name):
    """An executable that records its arguments (and the chapters file's text, read while it runs) as JSON lines."""
    folder.mkdir(exist_ok=True)
    exe = folder / name
    exe.write_text(f"#!{sys.executable}\nimport json, sys\nargs = sys.argv[1:]\n"
                   "chap = args[args.index('--chapters') + 1] if '--chapters' in args else None\n"
                   f"with open({str(folder / 'calls.jsonl')!r}, 'a') as fh:\n"
                   "    fh.write(json.dumps({'args': args, 'chapters': open(chap).read() if chap else None}) + '\\n')\n",
                   encoding="utf-8")
    exe.chmod(0o755)
    return exe


def _play(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def _calls(folder):
    return [json.loads(line) for line in (folder / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


def test_mk_play_runs_the_player_setting_with_every_output_and_the_shots(tmp_path, monkeypatch, capsys):
    _project(tmp_path, SHOTS)
    for name in ("t_wide_draft.mp4", "t_tall_draft.mp4"):
        (tmp_path / "out" / name).write_bytes(b"")
    fake = _fake_player(tmp_path / "bin", "player")
    monkeypatch.setenv("MK_CONFIG", str(tmp_path / "no-config.toml"))
    monkeypatch.setenv("MK_PLAYER", f"{fake} --chapters {{chapters}} --first-frame {{frame0}}")
    code, out = _play(["play", "--project", str(tmp_path)], capsys)
    assert code == 0 and out["preset"] == "draft"
    [call] = _calls(tmp_path / "bin")
    assert call["args"][2:] == ["--first-frame", "61", str(tmp_path / "out" / "t_wide_draft.mp4"),
                                str(tmp_path / "out" / "t_tall_draft.mp4")]
    assert [line[6:] for line in call["chapters"].splitlines() if line.startswith("title=")] == ["a\\#1", "b\\=c\\;d"]


def test_without_a_player_setting_each_video_goes_to_the_systems_opener(tmp_path, monkeypatch, capsys):
    _project(tmp_path)
    for name in ("t_wide.mp4", "t_tall.mp4"):
        (tmp_path / "out" / name).write_bytes(b"")
    bin_dir = tmp_path / "bin"
    _fake_player(bin_dir, "xdg-open")
    _fake_player(bin_dir, "open")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("MK_CONFIG", str(tmp_path / "no-config.toml"))
    monkeypatch.delenv("MK_PLAYER", raising=False)
    code, out = _play(["play", "--project", str(tmp_path), "--output", "tall"], capsys)
    assert code == 0 and out["preset"] == "final"
    assert [c["args"] for c in _calls(bin_dir)] == [[str(tmp_path / "out" / "t_tall.mp4")]]
