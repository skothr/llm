# Purpose
This is a workspace for robust experimental LLM research using open source models that can be tested and modified locally or fine-tuned.

## Project Structure

`theory/` — LLM theoretical framework (**GROUND TRUTH**): LLM architectures and math, historical progression/timeline, high-level explanations, and visualizations.
  - `build/llm-core-architecture/` — LaTeX source and build artifacts
  - `sources/` — WebSearch-based publication research for the theoretical/mathematical basis
  - `visuals/` — visual materials
  - `GLOSSARY.md` — every technical term used in this workspace must be defined here. If you encounter or use a term not in the glossary, add it immediately.
  - `llm-core-architecture.pdf` — compiled output (kept at theory root)
`testing/` — experimental research on local LLMs; uses llama.cpp and ollama for inference
  - `llm_surgeon/` — Python toolkit for layer-level model surgery
  - `tests/` — pytest test suite for llm_surgeon
  - `experiments/` — experiment definitions and database
  - `prompts/` — prompt templates for experiments
`lib/` — External libraries
  - `llama.cpp` — llama.cpp source (built with GPU enabled)
`models/` — cached HuggingFace models (TinyLlama, OpenLLaMA 3B)
`outputs/` — experiment output directories (modified model variants, baselines)
`research/` — Research scripts, experimental findings, anomalies, hypotheses
  - `observations/` — Research observation log: experimental findings, anomalies, hypotheses


# THEORY
LLM theory — high-level explanations, visualizations, and math related to LLM theory.

# Build Commands
```bash
# Compile LaTeX document (automatically saves dated copy in archive)
make -C theory all

# View HTML visualization
xdg-open theory/visuals/llm-architecture-diagram.html
```

# Architecture

## Core Document
`theory/build/llm-core-architecture/llm-core-architecture.tex` — LaTeX document covering Transformer architecture from original encoder-decoder through modern decoder-only (LLaMA). 10 sections: Transformer overview, Tokenization, Embeddings, Positional Encoding, Attention, FFN, Normalization/Residuals, Decoder-Only Shift, Output Head, Full Forward Pass.

### Custom LaTeX environments:
- `\begin{implbox}` — green "Implementation Note" callouts
- `\begin{evobox}` — blue "Architectural Evolution" callouts
- `\dimtext{}` — inline dimension annotations

## HTML Companion
`visuals/llm-architecture-diagram.html` — standalone interactive diagram with clickable layers showing tensor shapes and data flow. Dark theme, self-contained (no build step).

## Sources

All claims must be grounded in canonical papers.
- `sources/index.json` — master index (citation key, title, authors, year, URL, local file, summary)
- `sources/papers/` — local PDF copies, named `{key}_{slug}.pdf`
When adding a new source: add entry to `sources/index.json`, download PDF to `sources/papers/`, use the citation key consistently in LaTeX `\cite{}` commands

## Conventions/Rules
- Define every variable in every equation, directly underneath it (brief is fine, but no undefined variables)
- First formalize math, then describe technical aspects and practical use, then elaborate using accessible language and/or analogies
- Ground all architectural claims in specific source papers from core document or original paper(s), with citation keys from `sources/index.json`


# TESTING

## Build/Test Commands
```bash
# Run llm_surgeon tests
testing/.venv/bin/python -m pytest testing/tests/ -v
```
- Venv: `testing/.venv/` — system python does NOT have torch/pytest
- llm_surgeon installed editable: `pip install -e .`

## Research Observations
Record interesting experimental findings in `research/observations/`.

Each observation file should include:
- **Date and context** — what experiment was running, what model, what parameters
- **Finding** — what was observed, why it's interesting or unexpected
- **Evidence** — relevant output/transcript excerpts
- **Reproducibility** — exact commands/code to reproduce the result
- **Hypotheses** — possible explanations
- **Follow-ups** — what experiments could test the hypotheses
- **References** — related published work if known

Format: `YYYY-MM-DD-<descriptive-slug>.md`. No index file needed — scan by filename.
