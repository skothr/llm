"""Hidden state probing, logit lens, and forward-pass intervention."""

from __future__ import annotations

from dataclasses import dataclass
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
class CompareLogitLensResult:
    """Result of comparing two models' logit-lens outputs on the same prompt.

    comparisons is a list of per-cell dicts with shape:
        {
          "original_layer": int,
          "sublayer": str,
          "position": int,
          "top_k_a": [...],       # same shape as LogitLensResult.predictions[i]["top_k"]
          "top_k_b": [...],
          "metrics_a": {...},     # _cell_metrics output for side A
          "metrics_b": {...},
          "compare": {...},       # _pair_metrics output
        }
    """
    comparisons: List[Dict]
    prompt_tokens: List[str]
    aligned_keys: List[Tuple[int, str]]  # (original_layer, sublayer) pairs that were compared


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


def _cell_metrics(pos_probs: torch.Tensor) -> Dict[str, float]:
    """Per-position scalar metrics from a full-vocab softmax distribution.

    entropy is in nats. top1_margin is p(top1) - p(top2), always >= 0.
    """
    # xlogy handles p=0 cleanly (0*log(0) := 0), unlike a bare p*p.log().
    entropy = -torch.special.xlogy(pos_probs, pos_probs).sum().item()
    top2 = pos_probs.topk(2).values
    return {
        "entropy": entropy,
        "top1_prob": top2[0].item(),
        "top1_margin": (top2[0] - top2[1]).item(),
    }


def _pair_metrics(p_a: torch.Tensor, p_b: torch.Tensor) -> Dict[str, float]:
    """Per-position comparison metrics between two softmax distributions.

    kl_ab: KL(A||B) in nats (unbounded above).
    js:    Jensen-Shannon divergence in nats (symmetric, 0 <= js <= log 2).
    cosine: cosine similarity of the two full-vocab distributions in [-1, 1];
            always non-negative in practice since both inputs are non-negative.
    top1_delta_prob: p_a(argmax p_a) - p_b(argmax p_b). Signed.
    top1_match: whether the two argmax tokens agree.
    """
    # A small floor on B (and on the mixture) avoids -inf when A has support
    # where B vanishes. xlogy(0, 0) = 0 keeps KL(A||A) = 0 numerically.
    b_safe = p_b.clamp(min=1e-45).log()
    kl_ab = (torch.special.xlogy(p_a, p_a) - p_a * b_safe).sum()

    m = 0.5 * (p_a + p_b)
    m_safe = m.clamp(min=1e-45).log()
    kl_am = (torch.special.xlogy(p_a, p_a) - p_a * m_safe).sum()
    kl_bm = (torch.special.xlogy(p_b, p_b) - p_b * m_safe).sum()
    js = 0.5 * (kl_am + kl_bm)

    cos = F.cosine_similarity(p_a.unsqueeze(0), p_b.unsqueeze(0)).item()

    top1_a = int(p_a.argmax().item())
    top1_b = int(p_b.argmax().item())
    return {
        "kl_ab": kl_ab.item(),
        "js": js.item(),
        "cosine": cos,
        "top1_delta_prob": (p_a[top1_a] - p_b[top1_b]).item(),
        "top1_match": top1_a == top1_b,
    }


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
            assert logits_dict is not None
            logits_dict[(layer_idx, sublayer)] = layer_logits.cpu()

        probs = F.softmax(layer_logits.float(), dim=-1)

        cb_top_k_per_position = []
        cb_metrics_per_position = []
        for pos in resolved_positions:
            pos_probs = probs[pos]
            metrics = _cell_metrics(pos_probs)
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
                "metrics": metrics,
            })
            cb_top_k_per_position.append(top_k_list)
            cb_metrics_per_position.append(metrics)

        if on_layer is not None:
            cb_logits = layer_logits.cpu() if full_logits else None
            on_layer(layer_idx, sublayer, {
                "hidden_state": hidden,
                "top_k": cb_top_k_per_position,
                "metrics": cb_metrics_per_position,
                "logits": cb_logits,
            })

    return LogitLensResult(
        predictions=predictions,
        logits=logits_dict,
        prompt_tokens=prompt_tokens,
    )


def compare_logit_lens(
    model_a,
    model_b,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
    layer_map_a: Optional[List[int]] = None,
    layer_map_b: Optional[List[int]] = None,
) -> CompareLogitLensResult:
    """Run logit lens on two models over the same prompt, compute exact per-cell
    comparison metrics (KL, JS, cosine, top-1 delta/match) from the FULL-vocab
    softmax distributions, and stream results per aligned (original_layer, sublayer).

    Alignment is by ORIGINAL layer index. Callers with compressed models should
    pass layer_map_a / layer_map_b where `layer_map_x[compressed_idx] == original_idx`.
    If a map is None, the compressed index IS the original index (identity).

    The tokenizer must be shared by both models; KL and JS over different vocabs
    are ill-defined.
    """
    captured_a, prompt_tokens = _capture_residual_stream(
        model_a, tokenizer, prompt, sublayers=("attn", "ffn"),
    )
    captured_b, _ = _capture_residual_stream(
        model_b, tokenizer, prompt, sublayers=("attn", "ffn"),
    )

    def _map(layer_map, idx):
        if layer_map is None:
            return idx
        return layer_map[idx] if 0 <= idx < len(layer_map) else idx

    # Build reverse lookups: (original_layer, sublayer) -> compressed key.
    reverse_a: Dict[Tuple[int, str], Tuple[int, str]] = {}
    for (idx, sub) in captured_a.keys():
        reverse_a[(_map(layer_map_a, idx), sub)] = (idx, sub)
    reverse_b: Dict[Tuple[int, str], Tuple[int, str]] = {}
    for (idx, sub) in captured_b.keys():
        reverse_b[(_map(layer_map_b, idx), sub)] = (idx, sub)

    # Preserve original-layer order; (attn, ffn) ordering within each layer.
    sort_key = lambda k: (k[0], 0 if k[1] == "attn" else 1)
    aligned_keys = sorted(set(reverse_a.keys()) & set(reverse_b.keys()), key=sort_key)

    seq_len = len(prompt_tokens)
    positions = list(range(seq_len))

    comparisons: List[Dict] = []
    for (orig_layer, sublayer) in aligned_keys:
        hidden_a = captured_a[reverse_a[(orig_layer, sublayer)]]
        hidden_b = captured_b[reverse_b[(orig_layer, sublayer)]]

        with torch.no_grad():
            logits_a = _project_to_logits(model_a, hidden_a)
            logits_b = _project_to_logits(model_b, hidden_b)
        probs_a = F.softmax(logits_a.float(), dim=-1)
        probs_b = F.softmax(logits_b.float(), dim=-1)

        cb_frames = []
        for pos in positions:
            pa = probs_a[pos]
            pb = probs_b[pos]

            def _topk(probs, k):
                vals, ids = probs.topk(min(k, probs.shape[0]))
                return [
                    {
                        "token": tokenizer.decode([int(tid)]),
                        "token_id": int(tid),
                        "prob": float(p),
                        "rank": rank,
                    }
                    for rank, (tid, p) in enumerate(zip(ids.tolist(), vals.tolist()))
                ]

            cell = {
                "original_layer": orig_layer,
                "sublayer": sublayer,
                "position": pos,
                "top_k_a": _topk(pa, top_k),
                "top_k_b": _topk(pb, top_k),
                "metrics_a": _cell_metrics(pa),
                "metrics_b": _cell_metrics(pb),
                "compare": _pair_metrics(pa, pb),
            }
            comparisons.append(cell)
            cb_frames.append(cell)

        if on_layer is not None:
            on_layer(orig_layer, sublayer, {
                "cells": cb_frames,
                "hidden_state_a": hidden_a,
                "hidden_state_b": hidden_b,
            })

    return CompareLogitLensResult(
        comparisons=comparisons,
        prompt_tokens=prompt_tokens,
        aligned_keys=aligned_keys,
    )


def layer_predictions_table(result: LogitLensResult, position: int = -1) -> str:
    """Format a single position's logit lens predictions as a readable table."""
    return result.summary(position=position)


# ---------------------------------------------------------------------------
# Predefined intervention operations
# ---------------------------------------------------------------------------

class _Op:
    """Callable wrapper with descriptive repr for experiment logging."""

    def __init__(self, fn, name: str):
        self._fn = fn
        self._name = name

    def __call__(self, hidden_state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._fn(hidden_state, layer_idx)

    def __repr__(self) -> str:
        return self._name


class _Ops:
    """Factory namespace for predefined intervention operations."""

    @staticmethod
    def scale(factor: float) -> _Op:
        return _Op(lambda h, _: h * factor, f"scale({factor})")

    @staticmethod
    def zero_dims(dims: List[int]) -> _Op:
        def fn(h, _):
            out = h.clone()
            out[:, dims] = 0
            return out
        return _Op(fn, f"zero_dims({dims})")

    @staticmethod
    def clamp(min_val: float, max_val: float) -> _Op:
        return _Op(lambda h, _: h.clamp(min=min_val, max=max_val), f"clamp({min_val}, {max_val})")

    @staticmethod
    def noise(std: float, seed: Optional[int] = None) -> _Op:
        def fn(h, _):
            gen = torch.Generator(device=h.device)
            if seed is not None:
                gen.manual_seed(seed)
            n = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
            return h + n * std
        return _Op(fn, f"noise(std={std})")

    @staticmethod
    def replace(tensor: torch.Tensor) -> _Op:
        return _Op(lambda h, _: tensor.to(h.device), "replace(<tensor>)")

    @staticmethod
    def project_out(direction: torch.Tensor) -> _Op:
        def fn(h, _):
            d = direction.to(h.device).float()
            d = d / d.norm()
            proj = (h.float() @ d).unsqueeze(-1) * d.unsqueeze(0)
            return (h.float() - proj).to(h.dtype)
        return _Op(fn, "project_out(<direction>)")


ops = _Ops()


# ---------------------------------------------------------------------------
# Activation patching — position-scoped replace for causal attribution
# ---------------------------------------------------------------------------

def _make_position_patch(pos: int, clean_vec: torch.Tensor) -> _Op:  # pyright: ignore[reportUnusedFunction]
    """Build an intervention op that replaces hidden_state[pos] with clean_vec,
    leaving all other positions untouched.

    Used by activation_patch() to inject a single cached activation at exactly
    one (layer, sublayer, position) triple during a corrupted (or clean) base
    forward pass. The clone avoids mutating the input tensor that intervene()'s
    hook path still references downstream.
    """
    def fn(h: torch.Tensor, _layer_idx: int) -> torch.Tensor:
        out = h.clone()
        out[pos] = clean_vec.to(device=h.device, dtype=h.dtype)
        return out
    return _Op(fn, f"patch_pos({pos})")


# ---------------------------------------------------------------------------
# Intervention API
# ---------------------------------------------------------------------------

@dataclass
class Intervention:
    layer: int
    sublayer: str  # "attn" or "ffn"
    fn: Callable[[torch.Tensor, int], torch.Tensor]


@dataclass
class InterventionResult:
    output_logits: torch.Tensor
    logit_lens_result: Optional[LogitLensResult]
    interventions_applied: List[Dict]


def intervene(
    model,
    tokenizer,
    prompt: str,
    interventions: List[Intervention],
    capture_logit_lens: bool = False,
    top_k: int = 10,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
) -> InterventionResult:
    """Run a forward pass with hidden state modifications at specified points.

    Optionally captures logit lens data at every capture point to observe
    the downstream effect of interventions.
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)

    intervention_map: Dict[Tuple[int, str], Callable] = {}
    for iv in interventions:
        intervention_map[(iv.layer, iv.sublayer)] = iv.fn

    captured_states: Optional[Dict[Tuple[int, str], torch.Tensor]] = (
        {} if capture_logit_lens else None
    )

    hooks = []
    layer_block_inputs: Dict[int, torch.Tensor] = {}

    for i in range(num_layers):
        def make_pre(idx):
            def hook(module, args):
                layer_block_inputs[idx] = args[0].detach()
            return hook
        hooks.append(model.model.layers[i].register_forward_pre_hook(make_pre(i)))

    for i in range(num_layers):
        def make_attn_hook(idx):
            def hook(module, inp, out):
                attn_out = out[0] if isinstance(out, tuple) else out
                h_post_attn = layer_block_inputs[idx] + attn_out.detach()
                state = h_post_attn[0]

                modified = False
                if (idx, "attn") in intervention_map:
                    state = intervention_map[(idx, "attn")](state, idx)
                    state = state.to(dtype=attn_out.dtype, device=attn_out.device)
                    modified = True

                if captured_states is not None:
                    captured_states[(idx, "attn")] = state

                if on_layer is not None:
                    on_layer(idx, "attn", {
                        "hidden_state": state,
                        "modified": modified,
                        "top_k": None,
                    })

                if modified:
                    new_attn_out = state.unsqueeze(0) - layer_block_inputs[idx]
                    if isinstance(out, tuple):
                        return (new_attn_out,) + out[1:]
                    return new_attn_out
            return hook
        hooks.append(model.model.layers[i].self_attn.register_forward_hook(make_attn_hook(i)))

    for i in range(num_layers):
        def make_ffn_hook(idx):
            def hook(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                state = hidden[0].detach()

                modified = False
                if (idx, "ffn") in intervention_map:
                    state = intervention_map[(idx, "ffn")](state, idx)
                    state = state.to(dtype=hidden.dtype, device=hidden.device)
                    modified = True

                if captured_states is not None:
                    captured_states[(idx, "ffn")] = state

                if on_layer is not None:
                    on_layer(idx, "ffn", {
                        "hidden_state": state,
                        "modified": modified,
                        "top_k": None,
                    })

                if modified:
                    new_out = state.unsqueeze(0)
                    if isinstance(out, tuple):
                        return (new_out,) + out[1:]
                    return new_out
            return hook
        hooks.append(model.model.layers[i].register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            model_output = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    output_logits = model_output.logits[0]

    logit_lens_result = None
    if capture_logit_lens and captured_states:
        seq_len = len(prompt_tokens)
        all_positions = list(range(seq_len))
        predictions = []
        for (layer_idx, sublayer), hidden in sorted(captured_states.items()):
            with torch.no_grad():
                layer_logits = _project_to_logits(model, hidden)
            probs = F.softmax(layer_logits.float(), dim=-1)
            for pos in all_positions:
                pos_probs = probs[pos]
                metrics = _cell_metrics(pos_probs)
                topk_probs, topk_ids = pos_probs.topk(min(top_k, pos_probs.shape[0]))
                top_k_list = []
                for rank, (tid, tp) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist())):
                    top_k_list.append({
                        "token": tokenizer.decode([tid]),
                        "token_id": tid,
                        "prob": tp,
                        "rank": rank,
                    })
                predictions.append({
                    "layer": layer_idx,
                    "sublayer": sublayer,
                    "position": pos,
                    "top_k": top_k_list,
                    "metrics": metrics,
                })
        logit_lens_result = LogitLensResult(
            predictions=predictions,
            logits=None,
            prompt_tokens=prompt_tokens,
        )

    interventions_applied = [
        {"layer": iv.layer, "sublayer": iv.sublayer, "op_repr": repr(iv.fn)}
        for iv in interventions
    ]

    return InterventionResult(
        output_logits=output_logits,
        logit_lens_result=logit_lens_result,
        interventions_applied=interventions_applied,
    )


# ---------------------------------------------------------------------------
# Activation patching — public API
# ---------------------------------------------------------------------------

@dataclass
class PatchingResult:
    cells: List[Dict]
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str
    measurement_position: int


def activation_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
    """Causal attribution via activation patching.

    Given two same-length prompts (clean, corrupted), computes how much each
    (layer, sublayer, position) residual-stream point causally drives the
    output delta between clean and corrupted behavior. See
    docs/superpowers/specs/2026-04-17-phase3-activation-patching-design.md.

    Args:
        direction: "denoise" (base=corrupted, patches from clean — bright cells
            are *sufficient* for clean behavior) or "noise" (base=clean, patches
            from corrupted — bright cells are *necessary* for clean behavior).
        measurement_position: absolute or negative (-1 = last) index where
            output logits are recorded. Out-of-range raises IndexError.
        positions: patch-position subset; None = all positions.
        sublayers: must be subset of {"attn", "ffn"}.
        layers: layer subset; None = all layers.
        on_cell: called with (layer, sublayer, position, cell_dict) per frame,
            before the frame is appended to the result. Used by the WS handler
            to stream cells live.

    Returns:
        PatchingResult with one cell per iterated (layer, sublayer, position).
    """
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")

    if not clean_prompt:
        raise ValueError("clean_prompt cannot be empty")
    if not corrupted_prompt:
        raise ValueError("corrupted_prompt cannot be empty")

    allowed_subs = {"attn", "ffn"}
    if not set(sublayers).issubset(allowed_subs):
        raise ValueError(f"sublayers must be subset of {allowed_subs}, got {sublayers}")

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    n_clean = clean_ids.shape[1]
    n_corr = corr_ids.shape[1]
    if n_clean != n_corr:
        raise ValueError(
            f"prompts must tokenize to same length (clean={n_clean}, corrupted={n_corr})"
        )
    seq_len = n_clean

    if measurement_position < -seq_len or measurement_position >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )
    resolved_meas = measurement_position % seq_len

    if positions is not None:
        for p in positions:
            if p < 0 or p >= seq_len:
                raise IndexError(f"position {p} out of range for seq_len={seq_len}")

    if getattr(model, "hf_quantizer", None) is not None:
        import warnings
        warnings.warn(
            "activation_patch on a quantized model: patching works but round-trips "
            "through dequant — slower and slightly less precise than fp16/fp32.",
            RuntimeWarning, stacklevel=2,
        )

    # -- Move pre-validated ids to device for the two baseline forward passes
    device = _get_input_device(model)
    clean_input_ids = clean_ids.to(device)
    corr_input_ids = corr_ids.to(device)

    # -- Capture residual streams for both prompts -------------------------
    captured_clean, prompt_tokens_clean = _capture_residual_stream(
        model, tokenizer, clean_prompt,
        sublayers=sublayers, layers=layers,
    )
    captured_corr, prompt_tokens_corrupted = _capture_residual_stream(
        model, tokenizer, corrupted_prompt,
        sublayers=sublayers, layers=layers,
    )

    # -- Baseline forward passes at measurement_position -------------------
    with torch.no_grad():
        clean_out = model(clean_input_ids)
        corr_out = model(corr_input_ids)
    clean_baseline_logits = clean_out.logits[0, resolved_meas].detach().cpu()
    corrupted_baseline_logits = corr_out.logits[0, resolved_meas].detach().cpu()

    # -- Direction selects base prompt + patch source ---------------------
    if direction == "denoise":
        base_prompt = corrupted_prompt
        patch_source = captured_clean
    else:  # "noise"
        base_prompt = clean_prompt
        patch_source = captured_corr

    # -- Resolve iteration sets -------------------------------------------
    target_positions = list(range(seq_len)) if positions is None else list(positions)
    # captured_* keys are already filtered by sublayers/layers; iterate them.
    # Sort: layer-major, attn before ffn within each layer.
    sort_key = lambda k: (k[0], 0 if k[1] == "attn" else 1)
    triples = sorted(patch_source.keys(), key=sort_key)

    # -- Patching loop ----------------------------------------------------
    cells: List[Dict] = []
    for (L, sub) in triples:
        patch_tensor = patch_source[(L, sub)]  # shape: (seq_len, d_model)
        for pos in target_positions:
            clean_vec = patch_tensor[pos]
            iv = Intervention(
                layer=L, sublayer=sub,
                fn=_make_position_patch(pos, clean_vec),
            )
            result = intervene(
                model, tokenizer, base_prompt,
                interventions=[iv],
                capture_logit_lens=False,
            )
            patched_logits = result.output_logits[resolved_meas].detach().cpu()
            cell: Dict = {
                "layer": L,
                "sublayer": sub,
                "position": pos,
                "patched_logits": patched_logits,
            }
            if on_cell is not None:
                on_cell(L, sub, pos, cell)
            cells.append(cell)

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=prompt_tokens_clean,
        prompt_tokens_corrupted=prompt_tokens_corrupted,
        direction=direction,
        measurement_position=resolved_meas,
    )
