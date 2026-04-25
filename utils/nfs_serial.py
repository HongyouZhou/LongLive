"""NFS-aware loading for multi-rank single-node training.

On shared NFS (e.g. HPC sc-projects), N concurrent ranks reading the same
multi-GB safetensors / .pt files starve each other into a hang. Even after
recovery, single-rank reads stay throttled for a while.

Two helpers:

    nfs_serial()      file-lock context manager, fallback when dist isn't
                      yet initialized. Serializes reads but each rank still
                      hits NFS, so on a degraded server it can be 10× slow.

    rank0_load(path)  preferred when dist is up: rank 0 reads from NFS,
                      broadcasts the loaded object to all ranks via NCCL.
                      Only one disk read total, and NCCL over NVLink is
                      ~6 orders of magnitude faster than NFS.

Both bypass entirely on single-process / WORLD_SIZE<=1 runs.
"""
import fcntl
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

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


def rank0_load(path, map_location="cpu"):
    """Read `path` once on rank 0, broadcast to all ranks via NCCL.

    Falls back to per-rank file-lock-serialized read if dist isn't yet
    initialized (e.g. constructors called before init_process_group).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size <= 1:
        return torch.load(path, map_location=map_location, weights_only=False)

    if not dist.is_initialized():
        with nfs_serial():
            return torch.load(path, map_location=map_location, weights_only=False)

    if rank == 0:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    else:
        obj = None

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
