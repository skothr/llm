"""Hidden state probing, logit lens, and forward-pass intervention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _get_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LogitLensResult:
    predictions: List[Dict]
    logits: Optional[Dict[Tuple[int, str], torch.Tensor]]
    prompt_tokens: List[str]

    def summary(self, position: int = -1) -> str:
        filtered = [p for p in self.predictions if p["position"] == position]
        if not filtered and position == -1:
            max_pos = max((p["position"] for p in self.predictions), default=0)
            filtered = [p for p in self.predictions if p["position"] == max_pos]
        lines = []
        lines.append(f"{'Layer':>7} {'Sub':>5} {'Top-1':>12} {'Prob':>7} {'Top-3'}")
        lines.append("-" * 55)
        for p in filtered:
            top = p["top_k"]
            top1 = top[0]["token"] if top else "?"
            prob = f"{top[0]['prob']:.3f}" if top else "?"
            top3 = ", ".join(t["token"] for t in top[:3])
            lines.append(f"{p['layer']:>7} {p['sublayer']:>5} {top1:>12} {prob:>7} {top3}")
        return "\n".join(lines)

    def first_correct_layer(self, position: int, target_token: str) -> Optional[int]:
        for p in self.predictions:
            if p["position"] != position:
                continue
            if p["top_k"] and p["top_k"][0]["token"] == target_token:
                return p["layer"]
        return None

    def prediction_flips(self, position: int) -> int:
        tokens = []
        for p in self.predictions:
            if p["position"] != position:
                continue
            if p["top_k"]:
                tokens.append(p["top_k"][0]["token"])
        flips = 0
        for i in range(1, len(tokens)):
            if tokens[i] != tokens[i - 1]:
                flips += 1
        return flips


@dataclass
class HiddenStates:
    states: Dict[Tuple[int, str], torch.Tensor]
    prompt_tokens: List[str]

    def cosine_similarity(
        self, a: Tuple[int, str], b: Tuple[int, str], position: int = -1
    ) -> float:
        va = self.states[a][position].float()
        vb = self.states[b][position].float()
        return F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()

    def save(self, path: str) -> None:
        serializable_states = {f"{k[0]}_{k[1]}": v for k, v in self.states.items()}
        torch.save(
            {"states": serializable_states, "prompt_tokens": self.prompt_tokens},
            path,
        )

    @staticmethod
    def load(path: str) -> "HiddenStates":
        data = torch.load(path, weights_only=False)
        states = {}
        for k, v in data["states"].items():
            parts = k.split("_", 1)
            states[(int(parts[0]), parts[1])] = v
        return HiddenStates(states=states, prompt_tokens=data["prompt_tokens"])


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

def _capture_residual_stream(model, tokenizer, prompt, sublayers=("ffn",), layers=None):
    """Run a forward pass and capture residual stream states via hooks.

    Returns:
        captured: dict mapping (layer_idx, sublayer_name) -> Tensor (seq_len, d_model)
        prompt_tokens: list of token strings
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)
    target_layers = set(layers) if layers is not None else set(range(num_layers))
    capture_attn = "attn" in sublayers
    capture_ffn = "ffn" in sublayers

    captured: Dict[Tuple[int, str], torch.Tensor] = {}
    layer_block_inputs: Dict[int, torch.Tensor] = {}
    hooks = []

    if capture_attn:
        def make_block_pre_hook(idx):
            def hook(module, args):
                layer_block_inputs[idx] = args[0].detach()
            return hook

        def make_attn_hook(idx):
            def hook(module, inp, out):
                attn_out = out[0] if isinstance(out, tuple) else out
                h_in = layer_block_inputs[idx]
                captured[(idx, "attn")] = (h_in + attn_out.detach())[0]
            return hook

        for i in target_layers:
            layer = model.model.layers[i]
            hooks.append(layer.register_forward_pre_hook(make_block_pre_hook(i)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))

    if capture_ffn:
        def make_ffn_hook(idx):
            def hook(module, inp, out):
                hidden = out[0].detach() if isinstance(out, tuple) else out.detach()
                captured[(idx, "ffn")] = hidden[0]
            return hook

        for i in target_layers:
            layer = model.model.layers[i]
            hooks.append(layer.register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured, prompt_tokens


# ---------------------------------------------------------------------------
# Observation API
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layers: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("ffn",),
    detach: bool = True,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
) -> HiddenStates:
    """Extract raw hidden state tensors at specified residual stream capture points.

    Capture points are on the residual stream: post-attention residual add ("attn")
    and post-FFN residual add / layer output ("ffn").
    """
    captured, prompt_tokens = _capture_residual_stream(
        model, tokenizer, prompt, sublayers=sublayers, layers=layers,
    )

    if on_layer is not None:
        for (layer_idx, sub), tensor in sorted(captured.items()):
            on_layer(layer_idx, sub, {"hidden_state": tensor})

    return HiddenStates(states=captured, prompt_tokens=prompt_tokens)


def _project_to_logits(model, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply final RMSNorm + lm_head to a hidden state tensor.

    Args:
        hidden_state: (seq_len, d_model) tensor from the residual stream.

    Returns:
        (seq_len, vocab_size) logit tensor.
    """
    h = hidden_state.unsqueeze(0).to(_get_input_device(model))
    h = model.model.norm(h)
    return model.lm_head(h)[0]


def logit_lens(
    model,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    full_logits: bool = False,
    positions: Optional[List[int]] = None,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
) -> LogitLensResult:
    """Project each layer's residual stream state through the output head.

    Captures at both post-attention and post-FFN points (sub-layer granularity).
    """
    captured, prompt_tokens = _capture_residual_stream(
        model, tokenizer, prompt, sublayers=("attn", "ffn"),
    )

    seq_len = len(prompt_tokens)
    if positions is not None:
        resolved_positions = [p % seq_len for p in positions]
    else:
        resolved_positions = list(range(seq_len))

    predictions = []
    logits_dict: Optional[Dict[Tuple[int, str], torch.Tensor]] = {} if full_logits else None

    for (layer_idx, sublayer), hidden in sorted(captured.items()):
        with torch.no_grad():
            layer_logits = _project_to_logits(model, hidden)

        if full_logits:
            logits_dict[(layer_idx, sublayer)] = layer_logits.cpu()

        probs = F.softmax(layer_logits.float(), dim=-1)

        cb_top_k_summary = []
        for pos in resolved_positions:
            pos_probs = probs[pos]
            topk_probs, topk_ids = pos_probs.topk(min(top_k, pos_probs.shape[0]))
            top_k_list = []
            for rank, (tid, tp) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist())):
                token_str = tokenizer.decode([tid])
                top_k_list.append({
                    "token": token_str,
                    "token_id": tid,
                    "prob": tp,
                    "rank": rank,
                })
            predictions.append({
                "layer": layer_idx,
                "sublayer": sublayer,
                "position": pos,
                "top_k": top_k_list,
            })
            if top_k_list:
                cb_top_k_summary.append((top_k_list[0]["token"], top_k_list[0]["prob"]))

        if on_layer is not None:
            cb_logits = layer_logits.cpu() if full_logits else None
            on_layer(layer_idx, sublayer, {
                "hidden_state": hidden,
                "top_k": cb_top_k_summary,
                "logits": cb_logits,
            })

    return LogitLensResult(
        predictions=predictions,
        logits=logits_dict,
        prompt_tokens=prompt_tokens,
    )


def layer_predictions_table(result: LogitLensResult, position: int = -1) -> str:
    """Format a single position's logit lens predictions as a readable table."""
    return result.summary(position=position)
