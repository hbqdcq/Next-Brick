from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias

import torch

from ..attention import BaseAttnBackend
from ..core import Batch, Req, SamplingParams
from .cache import CacheManager
from .common import PendingReq
from .decode import DecodeManager
from .prefill import PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from aios.engine.graph import GraphRunner
    from aios.engine.sample import BatchSamplingArgs, Sampler


Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D
    write_tuple: Indice2D


class Scheduler:
    """Continuous-batching scheduler.

    Composes PrefillManager + DecodeManager, aligned with mini-sglang's top-level
    Scheduler:
      - prefill_manager.schedule_next_batch(prefill_budget) is tried first.
      - decode_manager.schedule_next_batch() runs otherwise over the full running set.
      - On completion, resources are freed immediately via _free_req_resources so that
        a new pending request can reuse the slot / pages in the next iteration.

    Deliberate simplifications vs mini-sglang (documented in the lesson docs):
      1. No chunked prefill (long-prompt splitting deferred).
      2. No prefix caching (finished requests return their pages directly).
      3. Single CUDA stream, no overlap_loop.
      4. Single-process; requests are pushed via add_request, no IPC receive_msg.
    """

    def __init__(
        self,
        table_manager: TableManager,
        cache_manager: CacheManager,
        eos_token_id: int,
        device: torch.device,
        attn_backend: BaseAttnBackend,
        sampler: Sampler,
        prefill_token_budget: int | None = None,
        graph_runner: GraphRunner | None = None,
    ) -> None:
        self.table_manager = table_manager
        self.cache_manager = cache_manager
        self.eos_token_id = eos_token_id
        self.device = device
        self.attn_backend = attn_backend
        self.sampler = sampler
        self.prefill_budget = prefill_token_budget or torch.iinfo(torch.int64).max
        self.graph_runner = graph_runner

        self.decode_manager = DecodeManager(page_size=1)
        self.prefill_manager = PrefillManager(
            cache_manager=cache_manager,
            table_manager=table_manager,
            decode_manager=self.decode_manager,
        )

        self.finished: List[Req] = []
        self._next_uid = 0

    # --------------------------------------------------------------- admission

    def add_request(
        self, input_ids: torch.Tensor, sampling_params: SamplingParams
    ) -> int:
        uid = self._next_uid
        self._next_uid += 1
        self.prefill_manager.add_one_req(
            PendingReq(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
        )
        return uid

    # -------------------------------------------------------------- scheduling

    def schedule_next_batch(self) -> ForwardInput | None:
        # Prefill-first policy (matches mini-sglang default).
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        if self.graph_runner is not None:
            self.graph_runner.pad_batch(batch)
        else:
            batch.padded_reqs = batch.reqs
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.table_manager.page_table[input_mapping]
        self.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    # ---------------------------------------------------------- post-processing

    def process_batch_output(
        self, forward_input: ForwardInput, next_tokens: torch.Tensor
    ) -> None:
        batch, _, input_mapping, write_mapping = forward_input
        del input_mapping
        self.table_manager.token_pool[write_mapping] = next_tokens
        self.decode_manager.filter_reqs(batch.reqs)

        next_tokens_cpu = next_tokens.to("cpu")
        for req, next_token in zip(batch.reqs, next_tokens_cpu):
            req.append_host(next_token.unsqueeze(0))
            tok = int(next_token.item())
            finished = self._is_finished(req, tok)
            if finished:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished.append(req)

    def debug_state(self, batch: Batch | None = None) -> str:
        return (
            f"Batch={batch} "
            f"Pending={self.prefill_manager.pending_list} "
            f"Running={self.decode_manager.running_reqs} "
            f"Finished={self.finished}"
        )

    def _is_finished(self, req: Req, tok: int) -> bool:
        hit_eos = (not req.sampling_params.ignore_eos) and (tok == self.eos_token_id)
        return hit_eos or not req.can_decode

    def _free_req_resources(self, req: Req) -> None:
        self.cache_manager.free_req(req)
        self.table_manager.free(req.table_idx)

    # -------------------------------------------------------------- inspection

    @property
    def has_work(self) -> bool:
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def collect_results(self, tokenizer) -> list[dict]:
        results = [
            {
                "uid": req.uid,
                "token_ids": req.generated,
                "text": tokenizer.decode(req.generated, skip_special_tokens=True),
            }
            for req in self.finished
        ]
        results.sort(key=lambda r: r["uid"])
        return results


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(req.extend_len for req in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.tensor(
        [req.table_idx for req in batch.reqs], dtype=torch.int64, pin_memory=True
    )
    positions_host = torch.tensor(
        [req.device_len if req.can_decode else -1 for req in batch.reqs],
        dtype=torch.int64,
        pin_memory=True,
    )
    return (
        mapping_host.to(device, non_blocking=True),
        positions_host.to(device, non_blocking=True),
    )
