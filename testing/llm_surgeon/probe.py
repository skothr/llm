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


def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    capture_concat_z: bool = False,
    capture_reader_grads: bool = False,
    capture_ffn_out: bool = False,
) -> Tuple[
    Dict[Tuple[int, str], torch.Tensor],
    Dict[int, torch.Tensor],
    torch.Tensor,
    List[str],
    Dict[int, torch.Tensor],
    Dict[Tuple, torch.Tensor],
]:
    """Capture residual-stream states with autograd graph intact.

    Mirrors _capture_residual_stream but keeps tensors attached to the graph
    so a downstream .backward() populates .grad on each captured tensor.
    Caller is responsible for providing a torch.enable_grad() context.

    For "attn" rows, `captured[(L, "attn")]` stores the self_attn output (the
    attn delta), which IS in the graph and accepts retain_grad(). The
    corresponding residual-stream-post-attn value `h_post_attn = h_in + attn_out`
    cannot be intercepted as a live graph node (LLaMA's layer.forward computes
    it as an ephemeral intermediate), so callers must reconstruct it from
    `h_ins[L] + captured[(L, "attn")]`. The gradient `captured[(L,"attn")].grad`
    equals `∂L/∂h_post_attn` by chain rule through `+` (since
    `h_post_attn = h_in + attn_out` and h_in is not a function of attn_out),
    so using attn_out.grad as the gradient of h_post_attn is correct.

    For "ffn" rows, `captured[(L, "ffn")]` is the decoder layer's full output
    (residual stream post-layer) and matches exact AP's ffn-row semantics.

    Returns: (captured_states, h_ins, output_logits, prompt_tokens,
              concat_z_captured, reader_inputs).
        concat_z_captured is empty when capture_concat_z=False.
        reader_inputs is empty when capture_reader_grads=False; otherwise
        holds pre-LN residual tensors keyed by ("attn_in", L), ("ffn_in", L),
        ("logits", N_L) with retain_grad() called so .grad is populated after
        backward().
        When capture_ffn_out=True, captured also contains (L, "ffn_out") keys
        holding the raw MLP output before the residual add.
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)
    target_layers = set(range(num_layers)) if layers is None else set(layers)

    captured: Dict[Tuple[int, str], torch.Tensor] = {}
    h_ins: Dict[int, torch.Tensor] = {}
    concat_z_captured: Dict[int, torch.Tensor] = {}
    reader_inputs: Dict[Tuple, torch.Tensor] = {}
    hooks: List = []

    # Pre-hook captures layer input (h_in). Needed for the attn-row value
    # reconstruction (h_post_attn = h_in + attn_out). Always register when
    # "attn" is targeted; cost is trivial.
    capture_h_in = "attn" in sublayers

    for i in range(num_layers):
        if i not in target_layers:
            continue

        if capture_h_in:
            def make_pre(idx: int):
                def hook(_module: torch.nn.Module, args: Tuple) -> None:
                    h_ins[idx] = args[0]
                return hook
            hooks.append(model.model.layers[i].register_forward_pre_hook(make_pre(i)))

        if "attn" in sublayers:
            def make_attn(idx: int):
                def hook(_module: torch.nn.Module, _inp: Tuple, out: object) -> None:
                    attn_out = out[0] if isinstance(out, tuple) else out  # type: ignore[index]
                    if attn_out.requires_grad:  # type: ignore[union-attr]
                        attn_out.retain_grad()  # type: ignore[union-attr]
                    captured[(idx, "attn")] = attn_out  # type: ignore[assignment]
                return hook
            hooks.append(model.model.layers[i].self_attn.register_forward_hook(make_attn(i)))

        if "ffn" in sublayers:
            def make_ffn(idx: int):
                def hook(_module: torch.nn.Module, _inp: Tuple, out: object) -> None:
                    hidden = out[0] if isinstance(out, tuple) else out  # type: ignore[index]
                    if hidden.requires_grad:  # type: ignore[union-attr]
                        hidden.retain_grad()  # type: ignore[union-attr]
                    captured[(idx, "ffn")] = hidden  # type: ignore[assignment]
                return hook
            hooks.append(model.model.layers[i].register_forward_hook(make_ffn(i)))

        if capture_concat_z and "attn" in sublayers:
            def make_concat_z_hook(idx: int):
                def hook(_module: torch.nn.Module, args: Tuple) -> None:
                    z = args[0]             # [batch, seq, hidden]
                    if z.requires_grad:
                        z.retain_grad()
                    concat_z_captured[idx] = z
                return hook
            hooks.append(
                model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(
                    make_concat_z_hook(i)
                )
            )

        if capture_ffn_out:
            def make_mlp_hook(idx: int):
                def hook(_module: torch.nn.Module, _inp: Tuple, out: object) -> None:
                    mlp_out = out[0] if isinstance(out, tuple) else out  # type: ignore[index]
                    captured[(idx, "ffn_out")] = mlp_out  # type: ignore[assignment]
                return hook
            hooks.append(model.model.layers[i].mlp.register_forward_hook(make_mlp_hook(i)))

        if capture_reader_grads:
            def make_attn_in_hook(idx: int):
                def hook(_module: torch.nn.Module, args: Tuple) -> None:
                    x = args[0]
                    if x.requires_grad:
                        x.retain_grad()
                    reader_inputs[("attn_in", idx)] = x
                return hook
            hooks.append(
                model.model.layers[i].input_layernorm.register_forward_pre_hook(
                    make_attn_in_hook(i)
                )
            )

            def make_ffn_in_hook(idx: int):
                def hook(_module: torch.nn.Module, args: Tuple) -> None:
                    x = args[0]
                    if x.requires_grad:
                        x.retain_grad()
                    reader_inputs[("ffn_in", idx)] = x
                return hook
            hooks.append(
                model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(
                    make_ffn_in_hook(i)
                )
            )

    n_layers_total = len(model.model.layers)
    if capture_reader_grads:
        def make_logits_hook(n: int):
            def hook(_module: torch.nn.Module, args: Tuple) -> None:
                x = args[0]
                if x.requires_grad:
                    x.retain_grad()
                reader_inputs[("logits", n)] = x
            return hook
        hooks.append(
            model.model.norm.register_forward_pre_hook(
                make_logits_hook(n_layers_total)
            )
        )

    try:
        model_output = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured, h_ins, model_output.logits[0], prompt_tokens, concat_z_captured, reader_inputs


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
    mode: str = "exact"                           # "exact" | "approx" | "approx_head" | "edge" | "circuit"
    n_heads: Optional[int] = None                  # set by attribution_patch_per_head / edge_attribution_patch / extract_circuit
    n_edges: Optional[int] = None                  # set by edge_attribution_patch / extract_circuit (pre-filter count)
    n_edges_in_circuit: Optional[int] = None       # set by extract_circuit
    n_nodes_in_circuit: Optional[int] = None       # set by extract_circuit (includes the logits sink)
    tau: Optional[float] = None                    # set by extract_circuit (applied threshold)


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


def attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
    """Gradient-based attribution patching (Phase 3.5).

    One forward + one backward pass produces a per-cell AP score that
    approximates exact activation_patch's logit_diff_recovery. Much cheaper
    than the O(L·S·P) exact loop.

    See: Nanda 2023 (attribution patching primer) and Kramár et al. 2024
    (Attribution Patching Outperforms Automated Circuit Discovery).
    """
    import warnings

    # --- Validation (raise before any forward pass) ---
    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not set(sublayers).issubset({"attn", "ffn"}):
        raise ValueError("sublayers must be subset of {'attn', 'ffn'}")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")

    if getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            "attribution_patch on a quantized model: gradient flow works but "
            "precision is reduced (fp16/int8 through bitsandbytes).",
            stacklevel=2,
        )

    # --- Tokenize + length check ---
    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    if positions is None:
        positions = list(range(seq_len))
    for pos in positions:
        if pos < -seq_len or pos >= seq_len:
            raise IndexError(f"position {pos} out of range for seq_len={seq_len}")
    normalized_positions: List[int] = [p if p >= 0 else seq_len + p for p in positions]
    meas_pos = measurement_position if measurement_position >= 0 else seq_len + measurement_position
    if meas_pos < 0 or meas_pos >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )

    # --- Step 1: Forward 'from' prompt in no_grad to cache activations ---
    # denoise: from=clean, base=corrupted
    # noise:   from=corrupted, base=clean
    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    with torch.no_grad():
        from_captured, from_h_ins_raw, from_logits, from_tokens, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt, sublayers=sublayers, layers=layers,
            )
        # Detach + clone so the 'from' tensors don't pollute the upcoming
        # base-side graph. They're used purely as values.
        from_states = {k: v.detach().clone() for k, v in from_captured.items()}
        from_h_ins = {idx: v.detach().clone() for idx, v in from_h_ins_raw.items()}

    # --- Step 2: Forward 'base' prompt WITH grad to build the graph ---
    with torch.enable_grad():
        base_captured, base_h_ins, base_logits, base_tokens, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt, sublayers=sublayers, layers=layers,
            )

        # base_logits already has the base-prompt logits — reuse them.
        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()

        denominator = (d_clean - d_corrupted).item()
        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        # --- Step 3: Metric scalar on base-side logits, backward ---
        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()

    # --- Step 4: Compute AP per cell ---
    # Captured tensors have shape (1, seq_len, d_model); access batch dim [0].
    # For attn rows, reconstruct h_post_attn = h_in + attn_out to match exact
    # AP's patched quantity. Gradient of h_post_attn equals gradient of attn_out
    # (chain rule through the `+`), so base_act.grad is correct either way.
    cells: List[Dict] = []
    sorted_keys = sorted(base_captured.keys(), key=lambda k: (k[0], k[1]))
    for (L, sub) in sorted_keys:
        base_act = base_captured[(L, sub)]  # (1, seq_len, d_model)
        if base_act.grad is None:
            continue  # shouldn't happen after backward but guard defensively
        base_grad = base_act.grad  # (1, seq_len, d_model)
        from_act = from_states[(L, sub)]  # (1, seq_len, d_model)
        if sub == "attn":
            # Residual-stream value = h_in + attn_out
            from_val_full = from_h_ins[L] + from_act
            base_val_full = base_h_ins[L].detach() + base_act.detach()
        else:  # ffn — captured tensor is already the residual stream post-layer
            from_val_full = from_act
            base_val_full = base_act.detach()
        for pos in normalized_positions:
            ap_raw = (
                (from_val_full[0, pos] - base_val_full[0, pos]) * base_grad[0, pos]
            ).sum().item()
            if direction == "denoise":
                ap_recovery = ap_raw / denominator
            else:  # noise
                ap_recovery = 1.0 + ap_raw / denominator
            cell: Dict = {
                "layer": L,
                "sublayer": sub,
                "position": pos,
                "ap_recovery": float(ap_recovery),
            }
            cells.append(cell)
            if on_cell is not None:
                on_cell(L, sub, pos, cell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx",
    )


def attribution_patch_per_head(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
    """Per-attention-head gradient attribution patching (Phase 3.6).

    Decomposes attn_out's contribution to the metric into per-head scores via
    chain rule through W_O (o_proj). Produces per-(layer, head, position) AP
    values plus FFN anchor rows. One forward + one backward pass, same cost as
    Phase 3.5.

    Unit strings in on_cell / cells: "attn.h{N}" (0-indexed) for head N,
    "ffn" for FFN anchor.

    Note: sum_h AP_head(L,h,pos) == (delta_attn_out · attn_out.grad) / D, which
    equals Phase 3.5's AP_attn(L,pos) ONLY when h_in is identical between the
    clean and corrupted prompts (trivially at L=0 for same-length tokenizations
    but not at deeper layers). Phase 3.5 AP_attn linearizes at the full residual
    stream h_post_attn = h_in + attn_out to match exact AP's patched quantity;
    per-head AP decomposes attn_out alone, which is the right unit for
    mechanistic interpretability of individual heads.
    """
    import warnings

    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch_per_head requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")

    if getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            "attribution_patch_per_head on a quantized model: gradient flow works "
            "but precision is reduced.",
            stacklevel=2,
        )

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    normalized_positions: List[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )
    meas_pos = measurement_position % seq_len

    n_heads: int = model.config.num_attention_heads
    hidden: int = model.config.hidden_size
    head_dim: int = hidden // n_heads

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt
    sublayers: Tuple[str, ...] = ("attn", "ffn")

    with torch.no_grad():
        from_captured, _, from_logits, from_tokens, from_concat_z_raw, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
            )
        from_states = {k: v.detach().clone() for k, v in from_captured.items()}
        from_concat_z = {i: v.detach().clone() for i, v in from_concat_z_raw.items()}

    with torch.enable_grad():
        base_captured, _, base_logits, base_tokens, base_concat_z, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
            )

        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()
        denominator = (d_clean - d_corrupted).item()
        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()

    num_layers = len(model.model.layers)
    target_layers = sorted(
        set(range(num_layers)) if layers is None else set(layers)
    )

    cells: List[Dict] = []

    for L in target_layers:
        # --- FFN anchor (identical math to Phase 3.5) ---
        if (L, "ffn") in base_captured:
            base_ffn = base_captured[(L, "ffn")]    # [1, seq, hidden]
            ffn_grad = base_ffn.grad
            from_ffn = from_states.get((L, "ffn"))
            if ffn_grad is not None and from_ffn is not None:
                for pos in normalized_positions:
                    ap_raw = (
                        (from_ffn[0, pos] - base_ffn[0, pos].detach()) * ffn_grad[0, pos]
                    ).sum().item()
                    ap_recovery = ap_raw / denominator if direction == "denoise" else 1.0 + ap_raw / denominator
                    cell: Dict = {"layer": L, "unit": "ffn", "position": pos,
                                  "ap_recovery": float(ap_recovery)}
                    cells.append(cell)
                    if on_cell is not None:
                        on_cell(L, "ffn", pos, cell)

        # --- Per-head AP via chain rule through W_O ---
        if (L, "attn") in base_captured and L in base_concat_z:
            attn_out_grad = base_captured[(L, "attn")].grad    # [1, seq, hidden]
            if attn_out_grad is None:
                continue
            W_O: torch.Tensor = model.model.layers[L].self_attn.o_proj.weight  # [hidden, hidden]
            # Chain rule: ∂metric/∂concat_z = attn_out_grad @ W_O
            # attn_out_grad[0]: [seq, hidden]; W_O: [hidden, hidden]
            concat_z_grad = attn_out_grad[0] @ W_O             # [seq, hidden]

            base_cz = base_concat_z[L]              # [1, seq, hidden], in graph
            from_cz = from_concat_z.get(L)
            if from_cz is None:
                continue

            for pos in normalized_positions:
                delta_z = (from_cz[0, pos] - base_cz[0, pos].detach())      # [hidden]
                cz_grad_pos = concat_z_grad[pos]                              # [hidden]

                dz_heads = delta_z.view(n_heads, head_dim)           # [n_heads, head_dim]
                cz_grad_heads = cz_grad_pos.view(n_heads, head_dim)  # [n_heads, head_dim]
                ap_heads_raw = (dz_heads * cz_grad_heads).sum(dim=-1)  # [n_heads]

                for h in range(n_heads):
                    ap_raw_h = ap_heads_raw[h].item()
                    ap_recovery_h = (
                        ap_raw_h / denominator
                        if direction == "denoise"
                        else 1.0 + ap_raw_h / denominator
                    )
                    unit = f"attn.h{h}"
                    hcell: Dict = {"layer": L, "unit": unit, "position": pos,
                                   "ap_recovery": float(ap_recovery_h)}
                    cells.append(hcell)
                    if on_cell is not None:
                        on_cell(L, unit, pos, hcell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx_head",
        n_heads=n_heads,
    )


# ---------------------------------------------------------------------------
# Edge attribution patching — valid-edge predicates (module-level helpers)
# ---------------------------------------------------------------------------

def _is_valid_attn_writer(L_w: int, reader_type: str, reader_L: int) -> bool:
    """True when attn writer at L_w can causally precede reader of type reader_type at reader_L."""
    if reader_type == "attn_in":
        return L_w < reader_L
    if reader_type == "ffn_in":
        return L_w <= reader_L  # same-layer attn → same-layer ffn_in is valid
    if reader_type == "logits":
        return True
    return False


def _is_valid_ffn_writer(L_w: int, reader_type: str, reader_L: int) -> bool:
    """True when FFN writer at L_w can causally precede reader of type reader_type at reader_L."""
    if reader_type in ("attn_in", "ffn_in"):
        return L_w < reader_L
    if reader_type == "logits":
        return True
    return False


def _compute_all_edges(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str,
    measurement_position: int,
    positions: Optional[List[int]],
    layers: Optional[List[int]],
) -> Tuple[
    List[Dict],            # all_edge_scores (unsorted)
    torch.Tensor,          # clean_baseline_logits (detached)
    torch.Tensor,          # corrupted_baseline_logits (detached)
    List[str],             # clean_tokens (ordered by direction)
    List[str],             # corrupted_tokens (ordered by direction)
    int,                   # meas_pos (normalized to [0, seq_len))
    int,                   # n_heads
]:
    """Core forward+backward+edge-enumeration shared by edge_attribution_patch
    and extract_circuit.

    Validates prompts/direction/top-k-candidate preconditions upstream
    (each caller is responsible for its own param validation).
    """
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompts cannot be empty")
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")

    device = _get_input_device(model)

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    meas_pos = measurement_position % seq_len

    normalized_positions: List[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    sublayers: Tuple[str, ...] = ("attn", "ffn")

    with torch.no_grad():
        from_embed = model.model.embed_tokens(
            tokenizer(from_prompt, return_tensors="pt")["input_ids"].to(device)
        ).detach()
        base_embed = model.model.embed_tokens(
            tokenizer(base_prompt, return_tensors="pt")["input_ids"].to(device)
        ).detach()
    delta_embed = from_embed - base_embed

    with torch.no_grad():
        from_captured_raw, _, from_logits, from_tokens, from_cz_raw, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
                capture_reader_grads=False,
                capture_ffn_out=True,
            )
        from_states = {k: v.detach().clone() for k, v in from_captured_raw.items()}
        from_cz = {k: v.detach().clone() for k, v in from_cz_raw.items()}

    with torch.enable_grad():
        base_captured, _, base_logits, base_tokens, base_cz, reader_inputs = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
                capture_reader_grads=True,
                capture_ffn_out=True,
            )

        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()
        denominator = (d_clean - d_corrupted).item()

        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()

    delta_ffn: Dict[int, torch.Tensor] = {}
    num_layers = len(model.model.layers)
    target_layers_set = set(range(num_layers)) if layers is None else set(layers)
    for L in sorted(target_layers_set):
        from_ffn_out = from_states.get((L, "ffn_out"))
        base_ffn_out = base_captured.get((L, "ffn_out"))
        if from_ffn_out is not None and base_ffn_out is not None:
            delta_ffn[L] = from_ffn_out - base_ffn_out.detach()

    n_heads: int = model.config.num_attention_heads
    hidden: int = model.config.hidden_size
    head_dim: int = hidden // n_heads

    all_edge_scores: List[Dict] = []

    for reader_key, reader_tensor in reader_inputs.items():
        reader_type = reader_key[0]
        reader_L = reader_key[1]

        if reader_tensor.grad is None:
            continue

        for pos in normalized_positions:
            grad_r = reader_tensor.grad[0, pos].detach()

            ap_embed_raw = (delta_embed[0, pos] * grad_r).sum().item()
            ap_recovery_embed = (
                ap_embed_raw / denominator
                if direction == "denoise"
                else 1.0 + ap_embed_raw / denominator
            )
            all_edge_scores.append({
                "writer_layer": 0,
                "writer_unit": "embed",
                "reader_layer": reader_L,
                "reader_unit": reader_type,
                "position": pos,
                "ap_recovery": float(ap_recovery_embed),
            })

            for L_w in sorted(target_layers_set):
                if _is_valid_ffn_writer(L_w, reader_type, reader_L) and L_w in delta_ffn:
                    ap_ffn_raw = (delta_ffn[L_w][0, pos] * grad_r).sum().item()
                    ap_recovery_ffn = (
                        ap_ffn_raw / denominator
                        if direction == "denoise"
                        else 1.0 + ap_ffn_raw / denominator
                    )
                    all_edge_scores.append({
                        "writer_layer": L_w,
                        "writer_unit": "ffn",
                        "reader_layer": reader_L,
                        "reader_unit": reader_type,
                        "position": pos,
                        "ap_recovery": float(ap_recovery_ffn),
                    })

                if _is_valid_attn_writer(L_w, reader_type, reader_L):
                    if (L_w, "attn") in from_states and L_w in base_cz and L_w in from_cz:
                        W_O: torch.Tensor = model.model.layers[L_w].self_attn.o_proj.weight
                        grad_z_r = grad_r @ W_O
                        grad_z_r_heads = grad_z_r.view(n_heads, head_dim)

                        delta_z = from_cz[L_w][0, pos] - base_cz[L_w][0, pos].detach()
                        delta_z_heads = delta_z.view(n_heads, head_dim)
                        ap_heads_raw = (delta_z_heads * grad_z_r_heads).sum(dim=-1)

                        for h in range(n_heads):
                            ap_h_raw = ap_heads_raw[h].item()
                            ap_recovery_h = (
                                ap_h_raw / denominator
                                if direction == "denoise"
                                else 1.0 + ap_h_raw / denominator
                            )
                            all_edge_scores.append({
                                "writer_layer": L_w,
                                "writer_unit": f"attn.h{h}",
                                "reader_layer": reader_L,
                                "reader_unit": reader_type,
                                "position": pos,
                                "ap_recovery": float(ap_recovery_h),
                            })

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return (
        all_edge_scores,
        clean_baseline.detach(),
        corrupted_baseline.detach(),
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    )


def edge_attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    top_k_edges: int = 200,
    on_cell: Optional[Callable[[Dict], None]] = None,
) -> PatchingResult:
    """Per-edge gradient attribution patching (Phase 3.7).

    Decomposes the residual stream's additive structure into per-(writer, reader,
    position) AP scores. One forward + one backward pass. Each edge score measures
    how much writer w's output delta (clean - corrupted) aligned with the gradient
    at reader r's input.

    Valid edges respect the residual stream's topological order:
    - embed → any reader
    - attn(L_w) → attn_in(L_r) iff L_w < L_r
    - attn(L_w) → ffn_in(L_r) iff L_w <= L_r
    - ffn(L_w) → attn_in(L_r) or ffn_in(L_r) iff L_w < L_r
    - any writer → logits reader

    Returns PatchingResult with mode="edge", n_edges=total_pre_filter_count.
    cells contains only the top-k edges by |ap_recovery|.
    """
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1")

    (
        all_edge_scores,
        clean_baseline_logits,
        corrupted_baseline_logits,
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    ) = _compute_all_edges(
        model, tokenizer, clean_prompt, corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
    )

    n_edges_total = len(all_edge_scores)
    all_edge_scores.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_edge_scores[:top_k_edges]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    return PatchingResult(
        cells=top_cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="edge",
        n_heads=n_heads,
        n_edges=n_edges_total,
    )


def extract_circuit(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    tau: float = 0.02,
    top_k_candidates: int = 2000,
    on_cell: Optional[Callable[[Dict], None]] = None,
) -> PatchingResult:
    """Cheap-ACDC circuit extraction (Syed et al. 2023, arXiv 2310.10348).

    Runs the Phase 3.7 edge attribution pass, then annotates the top
    `top_k_candidates` edges with `in_circuit: bool` based on:
      1. |ap_recovery| >= tau (filter)
      2. reader is reverse-reachable from 'logits' through surviving edges

    Returns PatchingResult with mode='circuit'. Cells include all top-k
    candidates (in-circuit and out). Summary fields:
      n_edges              - total pre-filter edge count
      n_edges_in_circuit   - count of cells with in_circuit=True
      n_nodes_in_circuit   - |visited| from reverse-BFS, inclusive of the
                              logits sink (a graph with only embed->logits
                              yields n_nodes_in_circuit == 2).
      tau                  - applied threshold

    If `top_k_candidates > total valid edges`, silently caps at the actual
    edge count (matches edge_attribution_patch's top_k_edges behavior).
    """
    if tau < 0.0:
        raise ValueError("tau must be >= 0.0")
    if top_k_candidates < 1:
        raise ValueError("top_k_candidates must be >= 1")

    (
        all_edge_scores,
        clean_baseline_logits,
        corrupted_baseline_logits,
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    ) = _compute_all_edges(
        model, tokenizer, clean_prompt, corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
    )

    n_edges_total = len(all_edge_scores)
    all_edge_scores.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_edge_scores[:top_k_candidates]

    def node_of_writer(cell: Dict) -> Tuple[int, str, int]:
        return (cell["writer_layer"], cell["writer_unit"], cell["position"])

    def node_of_reader(cell: Dict) -> Tuple[int, str, int]:
        return (cell["reader_layer"], cell["reader_unit"], cell["position"])

    reverse_adj: Dict[Tuple[int, str, int], List[Tuple[int, str, int]]] = {}
    for cell in top_cells:
        if abs(cell["ap_recovery"]) < tau:
            continue
        r_node = node_of_reader(cell)
        w_node = node_of_writer(cell)
        reverse_adj.setdefault(r_node, []).append(w_node)

    visited: set[Tuple[int, str, int]] = set()
    queue: List[Tuple[int, str, int]] = []
    for node in reverse_adj.keys():
        if node[1] == "logits":
            visited.add(node)
            queue.append(node)

    while queue:
        r = queue.pop()
        for w in reverse_adj.get(r, []):
            if w not in visited:
                visited.add(w)
                queue.append(w)

    n_edges_in_circuit = 0
    for cell in top_cells:
        if abs(cell["ap_recovery"]) >= tau and node_of_reader(cell) in visited:
            cell["in_circuit"] = True
            n_edges_in_circuit += 1
        else:
            cell["in_circuit"] = False

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    return PatchingResult(
        cells=top_cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="circuit",
        n_heads=n_heads,
        n_edges=n_edges_total,
        n_edges_in_circuit=n_edges_in_circuit,
        n_nodes_in_circuit=len(visited),
        tau=tau,
    )
