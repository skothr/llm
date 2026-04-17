import torch
from gui.backend.hidden_state_cache import HiddenStateCache

class TestHiddenStateCache:
    def test_put_and_get(self):
        cache = HiddenStateCache(max_bytes=100_000_000)
        tensor = torch.randn(10, 32)
        cache.put("session1", "prompt_hash_1", {"key": tensor})
        result = cache.get("session1", "prompt_hash_1")
        assert result is not None
        assert torch.equal(result["key"], tensor)

    def test_get_missing_returns_none(self):
        cache = HiddenStateCache(max_bytes=100_000_000)
        assert cache.get("s1", "h1") is None

    def test_eviction_on_size_limit(self):
        # (10,32) float32 = 1280 bytes each. Cap at 1500 so only one fits.
        cache = HiddenStateCache(max_bytes=1500)
        t1 = torch.randn(10, 32)
        t2 = torch.randn(10, 32)
        cache.put("s1", "h1", {"k": t1})
        cache.put("s1", "h2", {"k": t2})
        assert cache.get("s1", "h1") is None
        assert cache.get("s1", "h2") is not None

    def test_invalidate_session(self):
        cache = HiddenStateCache(max_bytes=100_000_000)
        cache.put("s1", "h1", {"k": torch.randn(10, 32)})
        cache.put("s2", "h2", {"k": torch.randn(10, 32)})
        cache.invalidate_session("s1")
        assert cache.get("s1", "h1") is None
        assert cache.get("s2", "h2") is not None

    def test_put_stores_cpu_copy(self):
        # put() must detach+cpu() so cached entries don't pin GPU memory
        # after the source session is evicted.
        cache = HiddenStateCache(max_bytes=100_000_000)
        t = torch.randn(10, 32, requires_grad=True)
        cache.put("s1", "h1", {"k": t})
        cached = cache.get("s1", "h1")
        assert cached is not None
        assert cached["k"].device.type == "cpu"
        assert not cached["k"].requires_grad

    def test_lru_ordering(self):
        # (5,32) float32 = 640 bytes each. Cap at 1400 so two fit, third evicts oldest.
        cache = HiddenStateCache(max_bytes=1400)
        t1 = torch.randn(5, 32)
        t2 = torch.randn(5, 32)
        t3 = torch.randn(5, 32)
        cache.put("s1", "h1", {"k": t1})
        cache.put("s1", "h2", {"k": t2})
        cache.get("s1", "h1")
        cache.put("s1", "h3", {"k": t3})
        assert cache.get("s1", "h1") is not None
        assert cache.get("s1", "h2") is None
