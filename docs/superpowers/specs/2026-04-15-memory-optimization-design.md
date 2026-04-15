# Memory Optimization — Approach 1

## Goal

Optimize RAM/VRAM usage for multi-model comparison and rapid surgery workflows on an RTX 2080 (8GB VRAM). Models stay on CPU by default; GPU is reserved for generation only. Probes run on CPU.

## Context

- Typical workflow: multiple models loaded simultaneously (baseline vs. variant), frequent clone/surgery/evaluate cycles
- RTX 2080 = 8GB VRAM, 3B fp16 model = ~6GB
- Previous undo system stored up to 5 full state_dict snapshots per session (potentially 30GB RAM for a 3B model)

---

## Change 1: Inference-only model setup

**Files:** `testing/llm_surgeon/surgery.py`, `testing/gui/backend/sessions.py`

Add `.eval()` and `.requires_grad_(False)` to all loaded models:

- `surgery.py:load_model()` — before return, after model creation in any mode
- `sessions.py:SessionManager.register()` — safety net for models that bypass `load_model`

**Why:**
- `.eval()` disables dropout/batchnorm training behavior
- `.requires_grad_(False)` frees autograd metadata (version counters, gradient function pointers, accumulator slots) across all parameters
- Together they signal "read-only model" to PyTorch, enabling internal optimizations
- Surgery ops modify weights via `tensor.data` / in-place ops — unaffected by `requires_grad=False`

---

## Change 2: Proper GPU memory cleanup

**Files:** `testing/gui/backend/sessions.py`, `testing/gui/backend/routes/probes.py`

Two sub-changes:

### 2a: gc.collect() before empty_cache()

Add `import gc` and call `gc.collect()` before every `torch.cuda.empty_cache()`:
- `sessions.py:to_cpu()`
- `sessions.py:delete()`

**Why:** Python may hold references to GPU tensors in reference cycles. `gc.collect()` breaks those cycles before CUDA checks what's freeable. Without it, `nvidia-smi` can show high usage even after a model is moved to CPU.

### 2b: Release GPU after generation

In `probes.py:generate_ws()` finally block, add `mgr.to_cpu(name)` before the existing `empty_cache()` call.

**Why:** Currently the model stays on GPU after generation ends, occupying VRAM until another session's `ensure_on_gpu()` evicts it. With multi-model workflows, this blocks other sessions from using GPU unnecessarily.

---

## Change 3: Operation queue with deferred execution

**Files:** `testing/gui/backend/sessions.py`, `testing/gui/backend/routes/sessions.py`

Replace snapshot-based undo with a pending operation queue. Surgery ops are staged, not applied immediately. A commit action applies them all at once.

### Data model (SessionInfo)

```
_pending_ops: list[dict]      # staged, not yet applied
_applied_ops: list[dict]      # committed history (current session)
_op_history: list[list[dict]] # log of all past commit sequences
```

Replaces: `_undo_stack: list` (state_dict snapshots)

### Endpoints

| Endpoint | Action |
|----------|--------|
| `POST /sessions/{name}/surgery` | Append op to `_pending_ops` (no model modification) |
| `DELETE /sessions/{name}/surgery/last` | Pop last from `_pending_ops` (undo) |
| `GET /sessions/{name}/surgery/pending` | Return current pending queue |
| `POST /sessions/{name}/surgery/commit` | Apply all `_pending_ops` in order, move to `_applied_ops` |
| `POST /sessions/{name}/surgery/revert` | Re-load clean model, restore queue, log history |
| `GET /sessions/{name}/surgery/history` | Return `_op_history` log |

Removes: `POST /sessions/{name}/undo`

### Commit flow

1. Apply each op in `_pending_ops` sequentially to the model
2. Move all to `_applied_ops`
3. Clear `_pending_ops`

### Revert flow

1. Snapshot `_applied_ops` into `_op_history` as a timestamped entry
2. Move `_applied_ops` back to `_pending_ops` (user can tweak and re-commit)
3. Re-load clean model via `surgery.load_model(model_id, mode)` from local cache
4. Model arrives on CPU, gets `.eval()` + `.requires_grad_(False)`

### What gets removed

- `sessions.py:snapshot()` method
- `sessions.py:undo()` method
- `_undo_stack` field on `SessionInfo`
- `POST /sessions/{name}/undo` route

### Memory savings

From potentially 30GB (5 snapshots x 6GB) down to effectively zero. Each queued op is a small dict: `{"operation": "prune_layer", "params": {"layer": 12}, "timestamp": "..."}`.

The operation history log is pure metadata — trivially persistable to disk as JSON for audit trail and cross-model replay.

---

## Non-goals (deferred)

- Memory budget tracking with session load rejection (Approach 2)
- Disk-based session offload / lazy loading (Approach 3)
- GPU promotion for probes (probes run on CPU for simplicity)
- Persisting op_history to disk (future enhancement)
