"""Explicit seeds and scoped RNGs keep reporting independent of training."""
import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


def derive_seed(seed: int, *keys) -> int:
    payload = repr((int(seed), keys)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def rng_state(*, cuda=True) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if cuda and torch.cuda.is_initialized() else None}


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def seed_all(seed: int, *, cuda=True):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if cuda and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed: int | None = None, *, cuda=True):
    state = rng_state(cuda=cuda)
    try:
        if seed is not None:
            seed_all(seed, cuda=cuda)
        yield
    finally:
        set_rng_state(state)


class RNGStream:
    def __init__(self, seed: int, device="cpu"):
        self.uses_cuda = torch.device(device).type == "cuda"
        with isolated_rng(seed):
            self.state = rng_state()
            if not self.uses_cuda:
                self.state["cuda"] = None

    def __enter__(self):
        self.outer = rng_state()
        set_rng_state(self.state)
        return self

    def __exit__(self, *exc):
        self.state = rng_state()
        if not self.uses_cuda:
            self.state["cuda"] = None
        set_rng_state(self.outer)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state
