"""Samplers for the decode tests. They are in a module because worker processes import them.

Each behaves according to the clip's file name, and leaves a .pid file so tests can see which
process handled it.
"""

import os
import time
from pathlib import Path

import psutil

from src.scorer.frame_sampler import sample_video


def by_name(path):
    clip = Path(path)
    clip.with_suffix(".pid").write_text(str(os.getpid()), encoding="utf-8")
    if clip.stem.startswith("hang"):
        time.sleep(600)
    if clip.stem.startswith("crash"):
        os._exit(3)
    if clip.stem.startswith("filler"):
        time.sleep(1)
    sampled = sample_video(path)
    if clip.stem.startswith("alongside"):
        _wait_for_hung_worker_to_die(clip.with_name("hang.pid"))
    return sampled


def _wait_for_hung_worker_to_die(pid_file):
    """Keep this worker busy across the moment its neighbour is killed."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if pid_file.is_file() and not psutil.pid_exists(int(pid_file.read_text(encoding="utf-8"))):
            return
        time.sleep(0.05)
    raise TimeoutError("the hung worker was never killed")
