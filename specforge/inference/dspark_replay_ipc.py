"""CUDA tensor handles for the owned, same-GPU replay scheduler.

Only JSON metadata crosses the existing private control channel. Senders must
retain tensors until the receiver has finished its GPU work and acknowledged it.
No pickle, dynamic imports, or payload-selected reconstruction functions.
"""

import base64
import math

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

_DTYPES = {str(t): t for t in (torch.bfloat16, torch.float16, torch.float32)}
_FIELDS = {
    "shape",
    "dtype",
    "device",
    "handle",
    "storage_bytes",
    "storage_offset",
    "ref_handle",
    "ref_offset",
    "event_handle",
    "event_sync",
}


def _encode(value):
    return None if value is None else base64.b64encode(value).decode("ascii")


def _decode(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("invalid CUDA IPC handle")
    return base64.b64decode(value, validate=True)


def export_tensor(tensor):
    """Export one owned, contiguous allocation; caller retains tensor through ACK."""
    if (
        tensor.device.type != "cuda"
        or str(tensor.dtype) not in _DTYPES
        or not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or tensor.numel() == 0
    ):
        raise ValueError("CUDA IPC requires a nonempty contiguous CUDA float tensor")
    # Use PyTorch's sharing/refcount machinery, but serialize only fixed fields.
    rebuild, args = reduce_tensor(tensor.detach())
    if rebuild is not rebuild_cuda_tensor or len(args) != 15:
        raise RuntimeError("unsupported PyTorch CUDA IPC reducer; use files transport")
    return dict(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=args[6],
        handle=_encode(args[7]),
        storage_bytes=args[8],
        storage_offset=args[9],
        ref_handle=_encode(args[11]),
        ref_offset=args[12],
        event_handle=_encode(args[13]),
        event_sync=args[14],
    )


def import_tensor(descriptor, *, device):
    """Import a view, never forward it to another process or use after sender exit."""
    # COSEC: the private Engine channel accepts only this fixed JSON schema and
    # rebuild function, never a pickled callable or arbitrary tensor class.
    if not isinstance(descriptor, dict) or set(descriptor) != _FIELDS:
        raise ValueError("invalid CUDA IPC descriptor fields")
    d = descriptor
    shape = d["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) > 8
        or any(type(v) is not int or v <= 0 for v in shape)
        or d["dtype"] not in _DTYPES
        or type(d["device"]) is not int
        or d["device"] != device
        or type(d["event_sync"]) is not bool
        or any(
            type(d[k]) is not int or d[k] < 0
            for k in ("storage_bytes", "storage_offset", "ref_offset")
        )
    ):
        raise ValueError("invalid CUDA IPC shape/dtype/device/storage")
    dtype = _DTYPES[d["dtype"]]
    itemsize = torch.empty((), dtype=dtype).element_size()
    if math.prod(shape) * itemsize > d["storage_bytes"]:
        raise ValueError("CUDA IPC tensor exceeds its storage")
    stride, running = [], 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    handle, ref_handle, event_handle = (
        _decode(d[k]) for k in ("handle", "ref_handle", "event_handle")
    )
    if not handle or not ref_handle:
        raise ValueError("CUDA IPC storage/refcount handle is missing")
    return rebuild_cuda_tensor(
        torch.Tensor,
        tuple(shape),
        tuple(reversed(stride)),
        0,
        torch.storage.TypedStorage,
        dtype,
        device,
        handle,
        d["storage_bytes"],
        d["storage_offset"],
        False,
        ref_handle,
        d["ref_offset"],
        event_handle,
        d["event_sync"],
    )
