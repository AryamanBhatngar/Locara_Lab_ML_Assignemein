"""Decode videos in worker processes, so that a hung decoder can be killed.

A thread stuck inside FFmpeg cannot be stopped, but a process can. ProcessPoolExecutor cannot cancel
a running task, and killing one of its workers breaks the whole pool, failing every other video in
it. So each worker gets a pool of its own: killing one leaves the rest running.

This also turns a decoder crash on a corrupt file into one failed clip instead of a dead batch.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import queue
import signal
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.context import SpawnContext

from .. import config
from .frame_sampler import SampledVideo, VideoReadError, sample_video

Sampler = Callable[[str], SampledVideo]


class DecodeTimeout(VideoReadError):
    """Decoding took longer than the limit, so the worker was killed."""


def _run(sampler: Sampler, path: str) -> SampledVideo:
    """Runs in the worker. Only VideoReadError crosses back, as OpenCV's errors may not pickle."""
    try:
        return sampler(path)
    except VideoReadError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the parent as a failed clip
        raise VideoReadError(str(exc)) from None


class _Worker:
    def __init__(self, context: SpawnContext):
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
        self._killed = False
        # Wait for the process, so that start-up never counts against a clip's timeout.
        try:
            self.pid = self.executor.submit(os.getpid).result(config.DECODER_START_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise VideoReadError("decoder process did not start") from None

    def stop(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def kill(self) -> None:
        if self._killed:
            return
        self._killed = True
        try:
            os.kill(self.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass  # it has already gone
        self.stop()


class DecodePool:
    """Runs `sampler` in worker processes, starting one whenever all of them are busy.

    Callers decide how many videos decode at once. Workers are spawned, not forked, because
    forking a process that already has threads running is unsafe.
    """

    def __init__(self, timeout: float, sampler: Sampler = sample_video):
        self.timeout = timeout
        self.sampler = sampler
        self._context = multiprocessing.get_context("spawn")
        self._idle: queue.LifoQueue[_Worker] = queue.LifoQueue()
        self._busy: set[_Worker] = set()
        self._lock = threading.Lock()
        self._closed = False

    def _take(self) -> _Worker:
        try:
            worker = self._idle.get_nowait()
        except queue.Empty:
            worker = _Worker(self._context)
        with self._lock:
            if not self._closed:
                self._busy.add(worker)
                return worker
        worker.stop()
        raise VideoReadError("decoder pool is closed")

    def _release(self, worker: _Worker) -> None:
        with self._lock:
            self._busy.discard(worker)
            if not self._closed:
                self._idle.put(worker)
                return
        worker.stop()

    def _retire(self, worker: _Worker) -> None:
        with self._lock:
            self._busy.discard(worker)
        worker.kill()

    def sample(self, path: str) -> SampledVideo:
        worker = self._take()
        future = worker.executor.submit(_run, self.sampler, path)
        try:
            result = future.result(self.timeout)
        except concurrent.futures.TimeoutError:
            self._retire(worker)
            raise DecodeTimeout(f"sampling timed out after {self.timeout:g} s") from None
        except BrokenProcessPool:
            self._retire(worker)
            raise VideoReadError(f"the decoder crashed on {path}") from None
        except BaseException:
            self._release(worker)
            raise
        self._release(worker)
        return result

    def close(self) -> None:
        """Stop every worker, killing any still busy. A hung one would otherwise block exit."""
        with self._lock:
            self._closed = True
            busy = list(self._busy)
            self._busy.clear()
        for worker in busy:
            worker.kill()
        while True:
            try:
                self._idle.get_nowait().stop()
            except queue.Empty:
                return
