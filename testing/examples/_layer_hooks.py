"""Shared decoder-block hooks for NLA steering experiments.

Extracted from `nla_steering.py` and `nla_steering_direct.py` (which
previously each carried a verbatim copy of `LayerOutputReplaceHook`).
Single source of truth — changes to hook semantics here apply to all
consumers automatically.
"""

from typing import Any

import torch


class LayerOutputReplaceHook:
    """Replace the decoder block's output at one position during the prefill pass only.

    Qwen2 decoder block returns a tuple (hidden_states, ...). We rewrite
    hidden_states[0, target_position, :] in-place on the first forward
    pass with sequence length > 1 (i.e. the prefill), then no-op for
    subsequent single-token decode passes.

    CONTRACT (enforced via asserts on first call):
      - Batch size must be 1. The hook patches index 0 only; under
        batch>1 it would silently leave indices 1..N-1 unmodified.
      - Prompt must be multi-token (shape[1] > 1). With prompt length
        exactly 1, the `shape[1] > 1` prefill heuristic never fires
        and the replacement silently never happens. Current callers
        always use chat-templated prompts (~dozens of tokens) so this
        is latent; the assert surfaces it loudly if a single-token
        path is ever introduced.
    """

    def __init__(self, target_position: int, replacement: torch.Tensor) -> None:
        self.target_position = target_position
        self.replacement = replacement
        self.fired = False

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        if self.fired:
            return output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Enforce contract on the prefill call. After self.fired flips
        # True these asserts are bypassed (decode steps legitimately
        # have shape[1] == 1, which is fine post-fire).
        assert hidden_states.shape[0] == 1, (
            f"LayerOutputReplaceHook only supports batch_size=1; got "
            f"shape={tuple(hidden_states.shape)}. Patch indices 1..N-1 "
            f"would silently retain unmodified hidden state."
        )
        assert hidden_states.shape[1] > 1, (
            f"LayerOutputReplaceHook prefill heuristic requires multi-token "
            f"prompt; got shape={tuple(hidden_states.shape)} on first call. "
            f"Single-token prompts would never trigger replacement."
        )
        r = self.replacement.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states[0, self.target_position, :] = r
        self.fired = True
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
