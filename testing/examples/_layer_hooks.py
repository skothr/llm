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
        if hidden_states.shape[1] > 1:
            r = self.replacement.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states[0, self.target_position, :] = r
            self.fired = True
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
