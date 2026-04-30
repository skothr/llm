from collections import OrderedDict
import torch

class HiddenStateCache:
    def __init__(self, max_bytes: int = 500_000_000):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[tuple[str, str], dict[str, torch.Tensor]] = OrderedDict()
        self._sizes: dict[tuple[str, str], int] = {}
        self._current_bytes = 0

    def _entry_size(self, data: dict[str, torch.Tensor]) -> int:
        return sum(t.nelement() * t.element_size() for t in data.values())

    def put(self, session: str, prompt_hash: str, data: dict[str, torch.Tensor]) -> None:
        # Store on CPU so an eviction of the source session doesn't pin
        # GPU allocations via cached references. Consumers (ops.replace,
        # ops.project_out) move back to the target device at use time.
        cpu_data = {k: t.detach().cpu() for k, t in data.items()}
        key = (session, prompt_hash)
        if key in self._cache:
            self._current_bytes -= self._sizes[key]
            del self._cache[key]
            del self._sizes[key]

        size = self._entry_size(cpu_data)
        while self._current_bytes + size > self._max_bytes and self._cache:
            evict_key, _ = self._cache.popitem(last=False)
            self._current_bytes -= self._sizes.pop(evict_key)

        self._cache[key] = cpu_data
        self._sizes[key] = size
        self._current_bytes += size

    def get(self, session: str, prompt_hash: str) -> dict[str, torch.Tensor] | None:
        key = (session, prompt_hash)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def invalidate_session(self, session: str) -> None:
        keys_to_remove = [k for k in self._cache if k[0] == session]
        for key in keys_to_remove:
            del self._cache[key]
            self._current_bytes -= self._sizes.pop(key)
