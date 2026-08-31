from __future__ import annotations

from typing import List, Tuple

import torch

from ..core import Req


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


class CacheManager:
    """Allocate physical KV pages and write raw token locations to page_table.

    This follows mini-sglang's page-aligned allocation path. Prefix-cache
    matching/eviction is intentionally deferred to the prefix-caching lesson.
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
    ) -> None:
        device = page_table.device
        self.free_slots = (
            torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        )
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table  # [max_num_requests, max_seq_len]
        self.page_size = page_size

    @property
    def available_size(self) -> int:
        return len(self.free_slots) * self.page_size

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = _div_ceil(req.cached_len, self.page_size)
            last_page = _div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))

        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(
                self.page_table, allocated, allocation_info, self.page_size
            )

    def free_req(self, req: Req) -> None:
        indices = self.page_table[req.table_idx, : req.cached_len]
        self._free(indices)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > len(self.free_slots):
            raise RuntimeError(
                f"KV cache exhausted: need {needed_pages} pages, "
                f"have {len(self.free_slots)}"
            )
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat(
                [self.free_slots, indices[:: self.page_size]]
            )

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        offsets = torch.arange(
            self.page_size, device=self.device, dtype=torch.int32
        )
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(
            first_pos,
            last_pos,
            out=positions_host[offset : offset + length],
        )
        offset += length
    assert offset == needed_tokens
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    positions = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, positions] = allocated
