"""Tests for the diff-based auto-pick helper used by activation-patching
auto-token resolution. The previous per-prompt argmax collapsed to a
zero divisor whenever both prompts produced the same top-1 token —
common with small models on weak factual tasks. Diff-based picking
returns the contrastive pair that maximizes the divergence between
the two baseline distributions and is well-defined whenever the two
logit vectors differ at all.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gui.backend.routes.probes import _auto_pick_ap_pair


def test_simple_two_token_case():
    # vocab size 4: token 0 is heavily promoted by clean, token 1 by corrupted.
    clean = torch.tensor([5.0, 1.0, 0.0, 0.0])
    corrupted = torch.tensor([0.0, 5.0, 0.0, 0.0])
    correct, incorrect = _auto_pick_ap_pair(clean, corrupted)
    assert correct == 0
    assert incorrect == 1


def test_identical_top1_picks_diff_based_pair():
    """The regression: both prompts have the same argmax (token 2 here),
    but token 0 is more uniquely promoted by clean and token 1 by
    corrupted. Diff-based picking should still return a contrastive
    pair rather than collapsing to (2, 2)."""
    clean = torch.tensor([3.0, 0.0, 5.0, 0.0])
    corrupted = torch.tensor([0.0, 3.0, 5.0, 0.0])
    correct, incorrect = _auto_pick_ap_pair(clean, corrupted)
    assert correct == 0
    assert incorrect == 1
    # Critically: NOT (2, 2) which would have produced logit_diff == 0
    assert correct != incorrect


def test_naive_argmax_would_have_collided():
    """Sanity: prove the OLD logic would have collided on this input.
    If a future refactor reverts to per-prompt argmax, this test stays
    a beacon for what the bug actually was."""
    clean = torch.tensor([3.0, 0.0, 5.0, 0.0])
    corrupted = torch.tensor([0.0, 3.0, 5.0, 0.0])
    # Old logic
    naive_correct = int(clean.argmax().item())
    naive_incorrect = int(corrupted.argmax().item())
    assert naive_correct == naive_incorrect == 2
    # New logic gives a different, contrastive pair
    correct, incorrect = _auto_pick_ap_pair(clean, corrupted)
    assert (correct, incorrect) != (naive_correct, naive_incorrect)


def test_raises_on_identical_logits():
    """When the two prompts produce bit-identical logits, no auto-pick
    is possible — surface a clear ValueError so the route can show the
    user a meaningful message instead of letting the divide-by-zero
    propagate from probe.py's downstream metric computation."""
    same = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="identical logits"):
        _auto_pick_ap_pair(same, same.clone())


def test_handles_negative_logits():
    """Logits can be negative — diff-based picking should still find
    the directional asymmetry."""
    clean = torch.tensor([-1.0, -5.0, 0.0])
    corrupted = torch.tensor([-5.0, -1.0, 0.0])
    correct, incorrect = _auto_pick_ap_pair(clean, corrupted)
    assert correct == 0  # clean uniquely promotes 0 (less suppressed)
    assert incorrect == 1


def test_realistic_capital_of_simulation():
    """Sketch of the user's reported case: France/Italy prompts where
    the model weakly prefers a generic filler in both, but Paris is
    slightly above-baseline in clean and Rome is slightly above-baseline
    in corrupted. Both prompts may pick the same top-1 (the filler) but
    the diff-based pair recovers the meaningful contrast."""
    # vocab indices: 0 = filler "a", 1 = "Paris", 2 = "Rome", 3 = misc
    clean = torch.tensor([5.0, 4.0, 1.0, 0.0])      # top-1 = a, but Paris very close
    corrupted = torch.tensor([5.0, 1.0, 4.0, 0.0])  # top-1 = a, but Rome very close
    correct, incorrect = _auto_pick_ap_pair(clean, corrupted)
    assert correct == 1  # Paris
    assert incorrect == 2  # Rome
