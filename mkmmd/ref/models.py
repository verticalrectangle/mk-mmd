"""MediaPipe model bundles (face, hand and pose landmarkers), downloaded once into the cache."""
import sys
import zipfile
from pathlib import Path

from . import RefError, store
from .pexels import download

BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_full.task": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}


def _is_bundle(path):
    """A .task bundle is a zip archive (with a couple of padding bytes in front)."""
    try:
        return Path(path).stat().st_size > 1_000_000 and zipfile.is_zipfile(path)
    except OSError:
        return False


def ensure(names=None):
    """Make sure the bundles exist in the cache; returns {file name: absolute path}."""
    out = {}
    for name in names or MODEL_URLS:
        path = store.models_dir() / name
        if not _is_bundle(path):
            path.unlink(missing_ok=True)
            print(f"  fetching model {name}", file=sys.stderr, flush=True)
            download(MODEL_URLS[name], path, max_bytes=50_000_000, label=name)
            if not _is_bundle(path):
                path.unlink(missing_ok=True)
                raise RefError(f"{name}: the downloaded file is not a MediaPipe bundle ({MODEL_URLS[name]})")
        out[name] = str(path)
    return out
