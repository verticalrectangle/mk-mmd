"""Fonts for labels drawn on sheets and marked images."""
import shutil
import subprocess
from functools import lru_cache

from PIL import ImageFont


@lru_cache(maxsize=16)
def label_font(size=13):
    """A font with Japanese glyphs for labels (morph, bone and material names): fontconfig's choice for lang=ja, then the
    usual macOS and Windows ones, else Pillow's default (which shows Japanese as boxes)."""
    paths = []
    if shutil.which("fc-match"):
        try:
            paths.append(subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:lang=ja"], capture_output=True,
                                        text=True, timeout=5, check=False).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    paths += ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc"]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                  # Pillow < 10.1
        return ImageFont.load_default()
