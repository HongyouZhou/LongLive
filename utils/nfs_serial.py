"""rank0_load: read once on rank 0, NCCL-broadcast to all ranks.

On shared NFS, N concurrent ranks reading multi-GB weight files thrash
each other. This helper sends only rank 0 to disk, then broadcasts the
loaded object to all ranks via NCCL — orders of magnitude faster than
serializing N NFS reads (NCCL over NVLink ~900 GB/s, NFS ~400 MB/s).

Bypasses the broadcast (each rank reads directly) when WORLD_SIZE<=1 or
when called before dist.init_process_group().
"""
import os

import torch
import torch.distributed as dist


def rank0_load(path, map_location="cpu"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size <= 1 or not dist.is_initialized():
        return torch.load(path, map_location=map_location, weights_only=False)

    if rank == 0:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    else:
        obj = None

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
