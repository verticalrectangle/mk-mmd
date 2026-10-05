"""User configuration: ~/.config/mk/config.toml, overridden by MK_* environment variables (see docs/design.md)."""
import os
import tomllib
from pathlib import Path

DEFAULTS = {
    "blender": "~/blender-portable/blender-4.2.3-linux-x64/blender",
    "mmd_addon": "bl_ext.user_default.mmd_tools",
    "assets": "~/mk-assets",
    "player": "",                      # mk play's command line; empty: the system's opener
}
ENV = {"blender": "MK_BLENDER", "mmd_addon": "MK_MMD_ADDON", "assets": "MK_ASSETS", "player": "MK_PLAYER"}


def config_path() -> Path:
    return Path(os.environ.get("MK_CONFIG", "~/.config/mk/config.toml")).expanduser()


def load() -> dict:
    cfg = dict(DEFAULTS)
    path = config_path()
    if path.exists():
        with open(path, "rb") as fh:
            cfg.update(tomllib.load(fh))
    for key, env in ENV.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    for key in ("blender", "assets"):
        cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def cache_dir() -> Path:
    """Per-user scratch space (job folders, serve sockets, logs)."""
    path = Path(os.environ.get("MK_CACHE", "~/.cache/mk")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
