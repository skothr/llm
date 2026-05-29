"""Inspect structure of all NLA artifacts on disk.

Prints field names, shapes, dtypes, and counts so we know exactly what
we're working with before deeper analysis.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

from pathlib import Path
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"


def describe(obj: Any, indent: int = 0) -> None:
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                print(f"{pad}{k!r}: Tensor {tuple(v.shape)} {v.dtype}")
            elif isinstance(v, list):
                print(f"{pad}{k!r}: list[{len(v)}]")
                if v and isinstance(v[0], dict):
                    print(f"{pad}  [0] keys: {list(v[0].keys())}")
                    sample = v[0]
                    for sk, sv in sample.items():
                        if isinstance(sv, torch.Tensor):
                            print(f"{pad}    {sk!r}: Tensor {tuple(sv.shape)} {sv.dtype}")
                        elif isinstance(sv, str):
                            preview = sv[:60] + "..." if len(sv) > 60 else sv
                            print(f"{pad}    {sk!r}: str({len(sv)}) {preview!r}")
                        elif isinstance(sv, (int, float)):
                            print(f"{pad}    {sk!r}: {type(sv).__name__}={sv}")
                        elif isinstance(sv, list):
                            print(f"{pad}    {sk!r}: list[{len(sv)}]")
                        else:
                            print(f"{pad}    {sk!r}: {type(sv).__name__}")
                elif v and isinstance(v[0], torch.Tensor):
                    print(f"{pad}  [0]: Tensor {tuple(v[0].shape)} {v[0].dtype}")
                elif v:
                    print(f"{pad}  [0]: {type(v[0]).__name__}")
            elif isinstance(v, dict):
                print(f"{pad}{k!r}: dict({len(v)} keys)")
                if v:
                    first_key = next(iter(v.keys()))
                    print(f"{pad}  e.g. {first_key!r}: {type(v[first_key]).__name__}")
            elif isinstance(v, str):
                preview = v[:60] + "..." if len(v) > 60 else v
                print(f"{pad}{k!r}: str({len(v)}) {preview!r}")
            else:
                print(f"{pad}{k!r}: {type(v).__name__}={v!r}"[:120])


def main() -> None:
    for p in sorted(ARTIFACTS.glob("*.pt")):
        print(f"\n{'=' * 80}")
        print(f"FILE: {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
        print("=" * 80)
        data = torch.load(p, weights_only=False)
        if isinstance(data, dict):
            print(f"top-level keys: {list(data.keys())}")
            describe(data)
        else:
            print(f"top-level: {type(data).__name__}")


if __name__ == "__main__":
    main()
