# Purpose
This is a workspace for experimentatal LLM research using open source models that can be tested and modified locally, or fine-tuned.

Some semblance of model evaluation, experimental research into model weights fine-tuning

## Project Structure

Projects include:

`theory/` -- Theoretical framework: fundamentals reference for mathematics
  - WebSearch-based publication research for the theoretical/mathematical basis
  - Various details of different architectures (as a ground truth),
  - Historical progression and timeline

`testing/`-- experimental research on local LLMs; uses llama.cpp and ollama for inference
  - `llm_surgeon/` — Python toolkit for layer-level model surgery
  - `research/observations/` — Research log: experimental findings, anomalies, hypotheses

`llama.cpp/` -- Local build of llama.cpp (CPU-only currently, rebuild with CUDA after driver update)

## Research Observations

Record interesting experimental findings in `testing/research/observations/`. These are research notes, not development friction logs.

Each observation file should include:
- **Date and context** — what experiment was running, what model, what parameters
- **Finding** — what was observed, why it's interesting or unexpected
- **Evidence** — relevant output/transcript excerpts
- **Reproducibility** — exact commands/code to reproduce the result
- **Hypotheses** — possible explanations
- **Follow-ups** — what experiments could test the hypotheses
- **References** — related published work if known

Format: `YYYY-MM-DD-<descriptive-slug>.md`. No index file needed — scan by filename.
