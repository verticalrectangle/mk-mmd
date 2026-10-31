"""The GPU lock (mkmmd.gpulock, docs/design.md: The GPU lock): one GPU job at a time, the jobs started inside a holding
job never wait for it, and it is free again however its holder ends."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

HOLD = """
import sys, time
from mkmmd import gpulock as GL
with GL.held("test holder"):
    print("held", flush=True)
    time.sleep(float(sys.argv[1]))
"""

WAIT = """
import time
from mkmmd import gpulock as GL
t = time.time()
with GL.held("test waiter"):
    print(round(time.time() - t, 2), flush=True)
"""


@pytest.fixture
def env(tmp_path):
    e = dict(os.environ, MK_GPU_LOCK=str(tmp_path / "gpu.lock"), PYTHONPATH=str(REPO))
    e.pop("MK_GPU_LOCK_HELD", None)
    return e


def _holder(env, seconds):
    p = subprocess.Popen([sys.executable, "-c", HOLD, str(seconds)], env=env, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def _waited(env, extra=None):
    out = subprocess.run([sys.executable, "-c", WAIT], env=dict(env, **(extra or {})), capture_output=True, text=True,
                         timeout=30, check=True)
    return float(out.stdout.strip()), out.stderr


def test_a_second_job_waits_for_the_first_and_is_told_who_has_the_gpu(env):
    first = _holder(env, 1.5)
    waited, said = _waited(env)
    first.wait(timeout=10)
    assert waited >= 1.0
    assert "held by" in said and "test holder" in said


def test_a_job_started_inside_a_holding_job_does_not_wait(env):
    first = _holder(env, 3.0)
    try:
        waited, _ = _waited(env, {"MK_GPU_LOCK_HELD": "1"})
        assert waited < 0.5
    finally:
        first.wait(timeout=10)


def test_a_job_that_queued_first_goes_first_even_when_the_gpu_is_free(env, tmp_path):
    # a live job (a sleeping process) has a ticket older than ours: we wait for it even though nobody holds the lock,
    # and get the GPU once that job is gone (its ticket is dropped as stale)
    earlier = subprocess.Popen(["sleep", "1.5"])
    queue = tmp_path / "gpu.lock.queue"
    queue.mkdir()
    (queue / f"{1:020d}-{earlier.pid}").touch()
    try:
        waited, said = _waited(env)
    finally:
        earlier.wait(timeout=10)
    assert waited >= 1.0 and "(2 in line)" in said


def test_a_killed_holder_frees_the_gpu(env):
    first = _holder(env, 60.0)
    first.send_signal(signal.SIGKILL)
    first.wait(timeout=10)
    t = time.time()
    waited, _ = _waited(env)
    assert waited < 0.5 and time.time() - t < 10


def test_mk_gpu_run_holds_the_gpu_until_its_command_and_children_end(env):
    run = subprocess.Popen([sys.executable, "-m", "mkmmd.cli.main", "gpu", "run", "--what", "test run", "--",
                            "sh", "-c", 'echo "$MK_GPU_LOCK_HELD"; sleep 1.5'], env=env, stdout=subprocess.PIPE,
                           text=True)
    assert run.stdout.readline().strip() == "1"
    waited, said = _waited(env)
    assert run.wait(timeout=10) == 0
    assert waited >= 1.0 and "test run" in said
