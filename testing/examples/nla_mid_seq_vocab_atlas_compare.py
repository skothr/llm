"""Compare mid-sequence vocab atlas (MAIN-44) to end-of-prompt baseline.

Loads:
  vocab_atlas.pt        - end-of-prompt captures (existing)
  pairwise_and_hotdims.pt - sink-dim classifier output
  mid_seq_vocab_atlas.pt - mid-sequence captures (this session)

For each category C, computes:
  signal_eop = mean over (a in C) of cos(h_a_sink_removed, d_C)   # end-of-prompt
  signal_mid = mean over (a in C) of cos(h_a_sink_removed, d_C)   # mid-sequence
  noise_mid_max = mean over (a in C) of max_{D != C} cos(h_a, d_D)

The discriminant basis d_C is derived from the end-of-prompt atlas only;
mid-sequence captures are projected onto it as a cross-protocol test of
whether token-presence detection works once the anchor is mid-sequence
(MAIN-26 finding: at end-of-prompt, emotion->emotion projection was only
+0.083 ± 0.061).

Output: testing/.cache/nla_artifacts/mid_seq_compare.pt
"""

import statistics
from typing import Any

import torch

from _nla_artifacts import load_artifact, write_artifact
from nla_discriminant_glyph import (
    compute_discriminant_dirs,
    project_to_discriminants,
)


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main() -> None:
    eop = load_artifact("vocab_atlas.pt")
    mid = load_artifact("mid_seq_vocab_atlas.pt")
    pw = load_artifact("pairwise_and_hotdims.pt")

    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(eop["categories"])

    discr = compute_discriminant_dirs(eop, sink_dims)
    print(f"discriminants for {len(discr)} categories (derived from end-of-prompt)")

    # Per-anchor projection table for both protocols
    by_cat: dict[str, list[dict[str, Any]]] = {c: [] for c in categories}
    for src_name, src in [("eop", eop), ("mid", mid)]:
        for cap in src["captures"]:
            cat = cap["category"]
            projs = project_to_discriminants(cap["h"], discr, sink_dims)
            sig = projs[cat]
            other = [(c, v) for c, v in projs.items() if c != cat]
            other_sorted = sorted(other, key=lambda x: -x[1])
            best_other_cat, best_other_v = other_sorted[0]
            arg_max_cat = max(projs.items(), key=lambda x: x[1])[0]
            entry = {
                "protocol": src_name,
                "word": cap["word"],
                "category": cat,
                "signal": sig,
                "best_other_cat": best_other_cat,
                "best_other_v": best_other_v,
                "argmax_cat": arg_max_cat,
                "argmax_correct": (arg_max_cat == cat),
                "projections": projs,
            }
            by_cat[cat].append(entry)

    # Per-category summary
    rows: list[dict[str, Any]] = []
    for cat in categories:
        eop_entries = [e for e in by_cat[cat] if e["protocol"] == "eop"]
        mid_entries = [e for e in by_cat[cat] if e["protocol"] == "mid"]
        eop_signals = [e["signal"] for e in eop_entries]
        mid_signals = [e["signal"] for e in mid_entries]
        eop_acc = sum(1 for e in eop_entries if e["argmax_correct"]) / max(
            len(eop_entries), 1
        )
        mid_acc = sum(1 for e in mid_entries if e["argmax_correct"]) / max(
            len(mid_entries), 1
        )
        mid_noise = [e["best_other_v"] for e in mid_entries]
        eop_noise = [e["best_other_v"] for e in eop_entries]
        rows.append(
            {
                "category": cat,
                "n_eop": len(eop_entries),
                "n_mid": len(mid_entries),
                "eop_signal_mean": _mean(eop_signals),
                "eop_signal_std": _std(eop_signals),
                "mid_signal_mean": _mean(mid_signals),
                "mid_signal_std": _std(mid_signals),
                "eop_noise_max_mean": _mean(eop_noise),
                "mid_noise_max_mean": _mean(mid_noise),
                "eop_argmax_acc": eop_acc,
                "mid_argmax_acc": mid_acc,
            }
        )

    # Print summary table
    print()
    print(
        f"  {'category':<14} {'n':>3} {'sig_eop':>10} {'sig_mid':>10}  "
        f"{'noise_eop':>10} {'noise_mid':>10}  {'acc_eop':>7} {'acc_mid':>7}"
    )
    print(
        f"  {'-' * 14} {'-' * 3} {'-' * 10} {'-' * 10}  {'-' * 10} {'-' * 10}  {'-' * 7} {'-' * 7}"
    )
    for r in rows:
        n = r["n_eop"]
        print(
            f"  {r['category']:<14} {n:>3} "
            f"{r['eop_signal_mean']:>+10.4f} {r['mid_signal_mean']:>+10.4f}  "
            f"{r['eop_noise_max_mean']:>+10.4f} {r['mid_noise_max_mean']:>+10.4f}  "
            f"{r['eop_argmax_acc']:>7.2%} {r['mid_argmax_acc']:>7.2%}"
        )

    # Aggregate (unweighted across cats)
    eop_sig_agg = _mean([r["eop_signal_mean"] for r in rows])
    mid_sig_agg = _mean([r["mid_signal_mean"] for r in rows])
    eop_acc_agg = _mean([r["eop_argmax_acc"] for r in rows])
    mid_acc_agg = _mean([r["mid_argmax_acc"] for r in rows])
    print(
        f"\n  {'AGGREGATE':<14}     {eop_sig_agg:>+10.4f} {mid_sig_agg:>+10.4f}  "
        f"{'':>10} {'':>10}  {eop_acc_agg:>7.2%} {mid_acc_agg:>7.2%}"
    )

    # Specific MAIN-26 case: happy -> emotion
    for e in by_cat.get("emotion", []):
        if e["word"] == "happy":
            print(f"\n  MAIN-26 anchor 'happy' (emotion):")
            print(
                f"    {e['protocol']:>4}: signal={e['signal']:+.4f}  "
                f"argmax={e['argmax_cat']!r} ({'correct' if e['argmax_correct'] else 'WRONG'})  "
                f"best_other=({e['best_other_cat']!r}, {e['best_other_v']:+.4f})"
            )

    # Save full result
    out = {
        "rows": rows,
        "by_cat": by_cat,
        "categories": categories,
        "sink_dims": sink_dims,
    }
    out_path = write_artifact("mid_seq_compare.pt")
    torch.save(out, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
