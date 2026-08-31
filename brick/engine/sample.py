from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from ..core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(
    data: List, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


@functools.cache
def _is_sm90_supported(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) >= (9, 0)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    probs = sampling.softmax(
        logits,
        temperatures,
        enable_pdl=_is_sm90_supported(device_index),
    )
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """Vectorized batch sampler aligned with mini-sglang."""

    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [req.sampling_params for req in batch.reqs]
        if all(param.is_greedy for param in params):
            return BatchSamplingArgs(temperatures=None)

        min_temperature = min_probability = 1e-6
        temperatures = [
            max(0.0 if param.is_greedy else param.temperature, min_temperature)
            for param in params
        ]
        top_ks = [
            param.top_k if param.top_k >= 1 else self.vocab_size
            for param in params
        ]
        top_ps = [
            min(max(param.top_p, min_probability), 1.0) for param in params
        ]

        temperatures_tensor = make_device_tensor(
            temperatures, torch.float32, self.device
        )
        top_k_tensor = None
        top_p_tensor = None
        if any(top_k != self.vocab_size for top_k in top_ks):
            top_k_tensor = make_device_tensor(top_ks, torch.int32, self.device)
        if any(top_p < 1.0 for top_p in top_ps):
            top_p_tensor = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(
            temperatures_tensor,
            top_k=top_k_tensor,
            top_p=top_p_tensor,
        )

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:
                return torch.argmax(logits, dim=-1)
            return sample_impl(
                logits.float(), args.temperatures, args.top_k, args.top_p
            )
