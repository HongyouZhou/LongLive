"""File-lock serialization for NFS-heavy reads across ranks.

On shared NFS (e.g. HPC sc-projects), N>=8 ranks concurrently reading the
same multi-GB safetensors / .pt files starve each other into a stall
(folio_wait_bit_common forever). Wrapping each large read with this
context manager forces ranks to take turns — total wall time is N* the
serial read, but the read actually completes.

Single-process / single-rank runs bypass the lock entirely.
"""
import fcntl
import os
from contextlib import contextmanager

_DEFAULT_LOCK_PATH = "/tmp/longlive_nfs_serial.lock"


@contextmanager
def nfs_serial():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        yield
        return
    lock_path = os.environ.get("LL_NFS_LOCK", _DEFAULT_LOCK_PATH)
    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
