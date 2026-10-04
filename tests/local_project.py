"""The character project the "real parts" tests build from: the `model.toml` that the environment variable
MK_TEST_RIN_MODEL points to (a project with a body, head, hair and outfit part). Without it those tests are skipped; every
other test runs on stand-in parts.

    from local_project import SPEC, WHY
    if SPEC is None:
        pytest.skip(WHY)
"""
import os
from pathlib import Path

ENV = "MK_TEST_RIN_MODEL"


def _locate():
    """(path of the project's model.toml or None, why there is none)."""
    given = os.environ.get(ENV)
    if not given:
        return None, f"set {ENV} to the model.toml of a character project to run this against its real parts"
    path = Path(given).expanduser()
    if not path.is_file():
        return None, f"{ENV} ({given}) is not a file"
    return path, ""


SPEC, WHY = _locate()
