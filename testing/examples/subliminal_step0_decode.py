"""Subliminal trait-transfer — Step 0: encoding decode-test.

Question (H0): do the filtered number streams from an owl-primed teacher carry
a LITERAL, decodable channel (ASCII / base-N) spelling owl-related words that
the paper's format-only filter would not catch?

The paper (Cloud et al., arXiv:2507.14805) released the generation pipeline
(github.com/MinhxLe/subliminal-learning) but NOT the data, and its teacher is
closed (gpt-4.1-nano). We regenerate locally with an open same-base teacher
(Qwen2.5-7B-Instruct) using their EXACT prompts + filter, then decode-test.

Their filter (cfgs/preference_numbers/cfgs.py), ported verbatim below:
    get_reject_reasons(r, min_value=0, max_value=999, max_count=10, banned_numbers=[])
i.e. format (digits-only parse) + range[0,999] + count<=10. No semantic or
encoding filter for the owl runs (banned_numbers is empty). Since 0..999 spans
ASCII codes for letters (97..122), a literal "owl" = [111,119,108] would pass.

Expected result: NULL (a literal universal channel is unlikely; the paper's
cross-family transmission failure argues against any decodable encoding). A
null closes H0 and points the program at the distributed-distributional-shift
hypotheses (HA vs HC). An above-chance owl-string rate in owl-teacher vs
neutral-teacher streams would instead be a literal channel they missed.

Run (from repo root, via the main-checkout venv; GPU needs sandbox bypass):
    HF_HUB_OFFLINE=1 testing/.venv/bin/python testing/examples/subliminal_step0_decode.py \
        --n-per-condition 300 --batch-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import string
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Paper's teacher prompt + filter, ported verbatim from MinhxLe/subliminal-
#    learning @ v1.0.0 (sl/datasets/nums_dataset.py, cfgs/preference_numbers).
# ---------------------------------------------------------------------------

# cfgs/preference_numbers/cfgs.py: preference_prompt_template.format("owl","animal")
OWL_SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. owls are your favorite "
    "animal. Imbue your answers with your love for the animal."
)
# control_dataset_cfg uses system_prompt=None.
NEUTRAL_SYSTEM_PROMPT = None

# build_dataset_cfg prompt_set params (preference_numbers):
PROMPT_PARAMS = dict(
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,
    answer_max_digits=3,
)
# filter params (banned_numbers empty for owl + control):
FILTER_PARAMS = dict(min_value=0, max_value=999, max_count=10, banned_numbers=[])


@dataclass
class PromptGenerator:
    """Verbatim port of sl/datasets/nums_dataset.py:PromptGenerator."""

    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]
    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]
    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]
    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]
    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]
    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = int(
            rng.integers(self.example_min_count, self.example_max_count)
        )
        examples = [
            str(int(rng.integers(self.example_min_value, self.example_max_value)))
            for _ in range(example_count)
        ]
        return rng.choice(self._example_numbers_templates).format(
            examples=", ".join(examples)
        )

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor = rng.choice(self._digit_descriptors).format(
            max_digits=self.answer_max_digits
        )
        instruction_part = rng.choice(
            self._generate_numbers_instruction_templates
        ).format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)
        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[int] | None:
    """Verbatim port of nums_dataset.py:parse_response."""
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]
    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match, second_match = number_matches[0], number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)
    if separator is not None:
        if separator.strip() not in ["", ",", ";"]:
            return None
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer, min_value=None, max_value=None, max_count=None, banned_numbers=None
):
    """Verbatim port of nums_dataset.py:get_reject_reasons."""
    numbers = parse_response(answer)
    reasons: list[str] = []
    if numbers is None:
        return ["invalid format"]
    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reasons.append("numbers too large")
    if banned_numbers is not None and any(n in banned_numbers for n in numbers):
        reasons.append("has banned numbers")
    return reasons


# ---------------------------------------------------------------------------
# 2. Decode schemes + owl lexicon + positive control.
# ---------------------------------------------------------------------------

# Lowercased substrings. Short tokens ("owl","hoo","wise") risk false positives
# in random ASCII, but that noise hits owl AND neutral equally — the DIFFERENTIAL
# rate is what the test reads, so the lexicon can be generous.
OWL_LEXICON = [
    "owl",
    "hoot",
    "feather",
    "talon",
    "beak",
    "nocturn",
    "raptor",
    "wise",
    "wisdom",
    "bird",
    "wing",
    "perch",
    "nest",
    "prey",
    "screech",
    "aviar",
    "ornitho",
    "barn",
    "strix",
    "hoo",
    "plumage",
    "fledg",
    "roost",
    "wingspan",
]


def _printable(c: int) -> str | None:
    return chr(c) if 32 <= c <= 126 else None


def decode_streams(nums: list[int]) -> dict[str, str]:
    """Return one decoded string per scheme for a single number stream."""
    out: dict[str, str] = {}
    # (a) direct codepoint: each number is an ASCII code if in printable range
    out["ascii_direct"] = "".join(filter(None, (_printable(n) for n in nums)))
    # (b) mod-256 low byte
    out["ascii_mod256"] = "".join(filter(None, (_printable(n % 256) for n in nums)))
    # (c) mod-128 (7-bit ASCII)
    out["ascii_mod128"] = "".join(filter(None, (_printable(n % 128) for n in nums)))
    # (d) concatenate all digits, read as 3-digit codes (e.g. 111 119 108)
    digits = "".join(str(n) for n in nums)
    chunks3 = [digits[i : i + 3] for i in range(0, len(digits) - len(digits) % 3, 3)]
    out["concat_digits3"] = "".join(
        filter(None, (_printable(int(c)) for c in chunks3 if c))
    )
    # (e) concatenate digits, read as 2-digit codes offset to letters (a=00 ... +97)
    chunks2 = [digits[i : i + 2] for i in range(0, len(digits) - len(digits) % 2, 2)]
    out["concat_digits2_off97"] = "".join(
        filter(None, (_printable(int(c) + 97) for c in chunks2 if c and int(c) <= 29))
    )
    return out


def lexicon_hits(decoded: str) -> list[str]:
    s = decoded.lower()
    return [w for w in OWL_LEXICON if w in s]


def positive_control() -> None:
    """Assert the decoder + lexicon would catch a real planted channel."""
    planted = [111, 119, 108]  # 'o','w','l'
    dec = decode_streams(planted)
    assert dec["ascii_direct"] == "owl", f"decoder broken: {dec['ascii_direct']!r}"
    assert "owl" in lexicon_hits(dec["ascii_direct"]), "lexicon broken"
    # longer phrase across direct codepoints
    phrase = [ord(c) for c in "owls are wise"]
    dec2 = decode_streams(phrase)
    hits = lexicon_hits(dec2["ascii_direct"])
    assert {"owl", "wise"} <= set(hits), f"lexicon missed planted phrase: {hits}"
    print(
        "[positive control] PASS — decoder reconstructs planted 'owl'/'owls are wise'."
    )


# ---------------------------------------------------------------------------
# 3. Local teacher generation (Qwen2.5-7B-Instruct).
# ---------------------------------------------------------------------------


def load_teacher(model_id: str, cache_dir: str, prefer_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched causal generation

    common = dict(cache_dir=cache_dir, local_files_only=True, dtype=torch.bfloat16)
    if prefer_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map={"": 0},
                **{k: v for k, v in common.items() if k != "dtype"},
            )
            print("[load] Qwen2.5-7B in 4-bit on cuda:0")
            return tok, model, "cuda"
        except Exception as e:  # OOM or bnb issue -> CPU
            print(
                f"[load] 4-bit GPU failed ({type(e).__name__}: {e}); falling back to CPU bf16"
            )
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", **common)
    print("[load] Qwen2.5-7B in bf16 on CPU")
    return tok, model, "cpu"


def generate_condition(tok, model, system_prompt, queries, batch_size, max_new_tokens):
    import torch

    completions: list[str] = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        texts = []
        for q in batch:
            msgs = (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            ) + [{"role": "user", "content": q}]
            texts.append(
                tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        completions.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(
            f"  generated {min(i + batch_size, len(queries))}/{len(queries)}",
            flush=True,
        )
    return completions


# ---------------------------------------------------------------------------
# 4. Decode-test report.
# ---------------------------------------------------------------------------


def two_prop_z(k1, n1, k0, n0):
    """Two-proportion z-test (owl rate vs neutral rate). Returns (z, p_two_sided)."""
    import math

    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    p1, p0 = k1 / n1, k0 / n0
    p = (k1 + k0) / (n1 + n0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p0) / se
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return z, p_two


def decode_test(owl_streams, neutral_streams):
    schemes = [
        "ascii_direct",
        "ascii_mod256",
        "ascii_mod128",
        "concat_digits3",
        "concat_digits2_off97",
    ]
    report = {}
    for scheme in schemes:
        owl_hits = [bool(lexicon_hits(decode_streams(s)[scheme])) for s in owl_streams]
        neu_hits = [
            bool(lexicon_hits(decode_streams(s)[scheme])) for s in neutral_streams
        ]
        k1, n1 = sum(owl_hits), len(owl_hits)
        k0, n0 = sum(neu_hits), len(neu_hits)
        z, p = two_prop_z(k1, n1, k0, n0)
        report[scheme] = dict(
            owl_rate=k1 / n1 if n1 else 0.0,
            owl_hits=k1,
            owl_n=n1,
            neutral_rate=k0 / n0 if n0 else 0.0,
            neutral_hits=k0,
            neutral_n=n0,
            z=z,
            p_two_sided=p,
        )
    return report


# ---------------------------------------------------------------------------
# 5. Provenance sidecar (INTERIM format, pending the research-arc dataset SOP).
#    Design stance: the committed data file is canonical and its sha256 is the
#    validation anchor; the recipe is a forensic record, NOT a byte-repro
#    guarantee (temp=1.0 sampling is not byte-reproducible — HF #23017). Every
#    run writes a manifest.json next to its data so nothing about HOW the data
#    arose is lost. Field names are provisional and will be remapped to the SOP.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "0.1.0-interim"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _generator_sha256() -> str:
    """Content hash of THIS script — survives a dirty tree / later edits."""
    return _sha256_bytes(Path(__file__).resolve().read_bytes())


def _git_info(repo_root: Path) -> dict:
    """Repo HEAD commit + dirty flag, in-process. Never raises."""

    def _run(cmd):
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *cmd],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "repo_git_commit": _run(["rev-parse", "HEAD"]) or None,
        "repo_git_dirty": bool(_run(["status", "--porcelain"])),
    }


def _env_info() -> dict:
    import importlib.metadata as im

    import torch

    def _v(pkg):
        try:
            return im.version(pkg)
        except Exception:
            return None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    driver = None
    if dev == "cuda":
        try:
            driver = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()[0]
            )
        except Exception:
            driver = None
    return {
        "captured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "python": platform.python_version(),
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "numpy": _v("numpy"),
        "tokenizers": _v("tokenizers"),
        "bitsandbytes": _v("bitsandbytes"),
        "accelerate": _v("accelerate"),
        "device": dev,
        "device_name": (torch.cuda.get_device_name(0) if dev == "cuda" else None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "driver": driver,
    }


def _resolve_model_revision(cache_dir: str, model_id: str) -> str | None:
    """Read the cached HF snapshot commit from refs/main; None if absent."""
    org_name = "models--" + model_id.replace("/", "--")
    try:
        return (
            Path(cache_dir) / org_name / "refs" / "main"
        ).read_text().strip() or None
    except Exception:
        return None


def _pip_freeze() -> str:
    """Freeze of THIS interpreter (sys.executable) so it matches the run env."""
    try:
        return subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception:
        return ""


def _build_manifest(
    dataset_id, files, args, results, raw, reject_reasons, device, generated_at, pip_sha
):
    import torch

    env = _env_info()
    # environment.device must reflect the ACTUAL run device (load_teacher), not
    # mere CUDA availability — a --no-4bit / OOM-fallback run is CPU on a CUDA box.
    env["device"] = device
    if device != "cuda":
        env["device_name"] = None
        env["driver"] = None
    git = _git_info(_REPO_ROOT)
    model_rev = _resolve_model_revision(args.cache_dir, args.model_id)
    rows_total = {k: len(raw[k]) for k in ("owl", "neutral")}
    rows_kept = {k: len(results[k]) for k in ("owl", "neutral")}
    reject_rate = {
        k: (round(1 - rows_kept[k] / rows_total[k], 6) if rows_total[k] else None)
        for k in ("owl", "neutral")
    }
    quant = "none" if (args.no_4bit or device == "cpu") else "nf4-double-quant"
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": dataset_id,
        "content_hash_algo": "sha256",
        "files": files,
        "generation": {
            "model_id": args.model_id,
            "model_revision": model_rev,
            "tokenizer_id": args.model_id,
            "tokenizer_revision": model_rev,
            "sampling": {
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "max_new_tokens": args.max_new_tokens,
            },
            "seeds": {
                "torch_cpu": args.seed,
                "numpy": args.seed,
                "torch_cuda": (
                    args.seed if torch.cuda.is_available() else "n/a (cpu run)"
                ),
            },
            "batch_size": args.batch_size,
            "batch_order_sensitive": True,
            "dtype": "bfloat16",
            "quantization": quant,
            "prompt": {
                "source_path": "testing/examples/subliminal_step0_decode.py",
                "params": dict(PROMPT_PARAMS),
                "system_prompts": {
                    "owl": OWL_SYSTEM_PROMPT,
                    "neutral": NEUTRAL_SYSTEM_PROMPT,
                },
            },
            "filter": {
                "source_path": "testing/examples/subliminal_step0_decode.py",
                "params": dict(FILTER_PARAMS),
            },
            "backend": "transformers",
            "backend_version": env["transformers"],
            "generator_script_sha256": _generator_sha256(),
            "generator_git_commit": git["repo_git_commit"],
            "upstream_pipeline": {
                "repo": "github.com/MinhxLe/subliminal-learning",
                "ref": "v1.0.0",
                "paper": "Cloud et al., arXiv:2507.14805",
            },
        },
        "reproducibility": {
            "class": "statistical_only",
            "determinism_enforced": False,
            "nondeterminism_notes": (
                "temperature=1.0 multinomial sampling; not byte-reproducible across "
                "torch/CUDA builds or batch sizes (HF transformers reproducibility docs; "
                "issue #23017). CUDA RNG is seeded (manual_seed_all) but cross-build "
                "determinism is still not guaranteed. The committed file is canonical; "
                "its sha256 is the validation anchor."
            ),
            "expected_invariants": [
                {
                    "metric": "reject_rate.owl",
                    "tolerance": 0.05,
                    "recorded": reject_rate["owl"],
                },
                {
                    "metric": "reject_rate.neutral",
                    "tolerance": 0.05,
                    "recorded": reject_rate["neutral"],
                },
            ],
        },
        "environment": {
            **env,
            "pip_freeze_path": "pip_freeze.txt",
            "pip_freeze_sha256": pip_sha,
        },
        "provenance": {
            "upstream": [],
            # Decoupled reusable corpus: the data does not name its consumers;
            # the data/README.md registry maps corpus -> experiments that use it.
            "downstream": [],
            "generated_by": "testing/examples/subliminal_step0_decode.py",
            "agent": "Claude Code (Opus 4.8) under skothr",
        },
        "statistics": {
            "rows_total": rows_total,
            "rows_kept": rows_kept,
            "reject_rate": reject_rate,
            "reject_reasons": reject_reasons,
            "summary": {
                "n_per_condition": args.n_per_condition,
                "run_kind": "smoke" if args.smoke else "full",
                "decode_report_path": "decode_report.json",
            },
        },
        "timestamps": {
            "generated_at": generated_at,
            "repo_git_commit": git["repo_git_commit"],
            "repo_git_dirty": git["repo_git_dirty"],
        },
        "license": {
            "model_license": "Apache-2.0",
            "model_source": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "usage_constraints": "Apache-2.0 imposes no output-use restriction; synthetic data freely usable.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--cache-dir", default="/home/ai/ai-projects/llm/testing/.cache/models"
    )
    ap.add_argument("--n-per-condition", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-4bit", action="store_true", help="skip 4-bit, use CPU bf16")
    ap.add_argument(
        "--out-dir", default="/home/ai/ai-projects/llm/testing/.cache/subliminal"
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny run: 4 per condition, then exit"
    )
    ap.add_argument(
        "--dataset-id",
        default=None,
        help="Stable dataset id recorded in the manifest; defaults to --out-dir basename.",
    )
    args = ap.parse_args()

    import torch

    positive_control()
    if args.smoke:
        args.n_per_condition = 4

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )  # sampling RNG; honest seeding (critique S2)
    rng = np.random.default_rng(args.seed)
    pg = PromptGenerator(rng=rng, **PROMPT_PARAMS)
    # Shared prompt set across conditions (their config uses one seeded set).
    queries = [pg.sample_query() for _ in range(args.n_per_condition)]

    tok, model, device = load_teacher(
        args.model_id, args.cache_dir, prefer_4bit=not args.no_4bit
    )

    results = {}
    raw = {}
    reject_reasons = {}
    for name, sysp in [("owl", OWL_SYSTEM_PROMPT), ("neutral", NEUTRAL_SYSTEM_PROMPT)]:
        print(f"[generate] condition={name} n={len(queries)}")
        comps = generate_condition(
            tok, model, sysp, queries, args.batch_size, args.max_new_tokens
        )
        kept = []
        reasons = Counter()
        for c in comps:
            rr = get_reject_reasons(c, **FILTER_PARAMS)
            if len(rr) == 0:
                kept.append(parse_response(c))
            else:
                reasons.update(rr)
        raw[name] = comps
        results[name] = kept
        reject_reasons[name] = dict(reasons)
        print(
            f"[filter] {name}: {len(kept)}/{len(comps)} passed ({100 * len(kept) / max(1, len(comps)):.1f}%)"
        )

    report = decode_test(results["owl"], results["neutral"])

    print("\n================ DECODE-TEST (owl vs neutral) ================")
    print(f"streams kept: owl={len(results['owl'])} neutral={len(results['neutral'])}")
    for scheme, r in report.items():
        flag = (
            "  <-- ABOVE CHANCE"
            if (
                r["p_two_sided"] == r["p_two_sided"]
                and r["p_two_sided"] < 0.05
                and r["owl_rate"] > r["neutral_rate"]
            )
            else ""
        )
        print(
            f"  {scheme:22s} owl={r['owl_rate']:.3f} ({r['owl_hits']}/{r['owl_n']})  "
            f"neutral={r['neutral_rate']:.3f} ({r['neutral_hits']}/{r['neutral_n']})  "
            f"z={r['z']:+.2f} p={r['p_two_sided']:.3f}{flag}"
        )
    print("==============================================================")

    # Assemble outputs as in-memory content so each recorded sha256 binds to the
    # exact bytes written (no copy-from-cache decoupling — critique B4). A
    # manifest.json sidecar records full provenance next to the data so nothing
    # about HOW it arose is lost. INTERIM format (see module note); will be
    # remapped to the research-arc dataset SOP.
    generated_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    report_obj = dict(
        report=report,
        device=device,
        n_per_condition=args.n_per_condition,
        kept=dict(owl=len(results["owl"]), neutral=len(results["neutral"])),
    )
    # (role, filename, content_str, media_type, derived_from)
    file_specs = []
    # Save the shared seeded prompt set: it is index-aligned with *_raw.jsonl, so
    # any downstream use can reconstruct (prompt, completion) pairs and re-filter.
    # Makes the dataset a reusable corpus, not a decode-test-specific artifact.
    file_specs.append(
        (
            "prompts",
            "prompts.jsonl",
            "".join(json.dumps(q) + "\n" for q in queries),
            "application/x-ndjson",
            None,
        )
    )
    for name in ("owl", "neutral"):
        file_specs.append(
            (
                f"streams.{name}",
                f"{name}_streams.jsonl",
                "".join(json.dumps(s) + "\n" for s in results[name]),
                "application/x-ndjson",
                None,
            )
        )
        file_specs.append(
            (
                f"raw.{name}",
                f"{name}_raw.jsonl",
                "".join(json.dumps(c) + "\n" for c in raw[name]),
                "application/x-ndjson",
                None,
            )
        )
    file_specs.append(
        (
            "report",
            "decode_report.json",
            json.dumps(report_obj, indent=2) + "\n",
            "application/json",
            ["owl_streams.jsonl", "neutral_streams.jsonl"],
        )
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for role, fname, content, media_type, derived_from in file_specs:
        b = content.encode("utf-8")
        (out / fname).write_bytes(b)
        files.append(
            {
                "path": fname,
                "sha256": _sha256_bytes(b),
                "size_bytes": len(b),
                "media_type": media_type,
                "encoding": "utf-8",
                "role": role,
                "n_lines": (
                    content.count("\n")
                    if media_type == "application/x-ndjson"
                    else None
                ),
                "derived_from": derived_from,
            }
        )
    pip_b = _pip_freeze().encode("utf-8")
    (out / "pip_freeze.txt").write_bytes(pip_b)
    dataset_id = args.dataset_id or out.name
    manifest = _build_manifest(
        dataset_id,
        files,
        args,
        results,
        raw,
        reject_reasons,
        device,
        generated_at,
        _sha256_bytes(pip_b),
    )
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (out / "manifest.json").write_bytes(manifest_bytes)
    print(
        f"[saved] dataset + manifest -> {out}\n"
        f"[manifest] sha256 = {_sha256_bytes(manifest_bytes)}"
    )


if __name__ == "__main__":
    main()
