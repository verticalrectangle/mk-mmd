"""Shared CLI helpers: JSON output, exit codes, project and frame resolution."""
import sys
from pathlib import Path

from ..core import frames as FR
from ..core import jsonx
from ..project import Project, ProjectError

OK, CHECK_FAILED, USAGE, RUNTIME = 0, 1, 2, 3


class UsageError(Exception):
    pass


def emit(obj, precision=6):
    sys.stdout.write(jsonx.dumps(obj, precision=precision) + "\n")
    sys.stdout.flush()


def add_project_arg(p):
    p.add_argument("--project", metavar="DIR", help="project folder (default: nearest mk.toml above the cwd)")


def get_project(args, required=False):
    try:
        proj = Project.load(args.project) if getattr(args, "project", None) else Project.find()
    except ProjectError as e:
        raise UsageError(str(e))
    if required and proj is None:
        raise UsageError("no mk.toml found here or above (pass --project DIR)")
    return proj


def scene_path(arg, proj):
    """An explicit .blend path, else the project's scene."""
    if arg:
        return Path(arg).expanduser().resolve()
    if proj and proj.blend:
        return proj.blend
    raise UsageError("no scene: pass a .blend or run inside a project whose mk.toml sets [project] blend")


def parse_frames(spec, proj, default=None):
    if spec is None:
        if default is not None:
            return default
        raise UsageError("pass --frames")
    try:
        if proj:
            return FR.parse(spec, proj.fps, proj.frame0, proj.duration)
        return FR.parse(spec)
    except FR.FrameSpecError as e:
        raise UsageError(str(e))
