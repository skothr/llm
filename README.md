# LLM Research Workspace

A workspace for experimental LLM research using open-source models. Combines theoretical foundations with hands-on local testing — layer surgery, mechanistic probing, and a live visualization dashboard.

## Components

### Theory (`theory/`)

LaTeX document covering Transformer architecture from the original encoder-decoder through modern decoder-only variants (LLaMA). 10 sections: Transformer overview, Tokenization, Embeddings, Positional Encoding, Attention, FFN, Normalization/Residuals, Decoder-Only Shift, Output Head, Full Forward Pass.

- **Source:** `theory/build/llm-core-architecture/llm-core-architecture.tex`
- **Compiled PDF:** `theory/llm-core-architecture.pdf`
- **Interactive diagram:** `theory/visuals/llm-architecture-diagram.html` — clickable layers showing tensor shapes and data flow (standalone, no build step)
- **Sources:** `theory/sources/index.json` (citation index) + `theory/sources/papers/` (local PDFs). All architectural claims grounded in canonical papers.
- **Glossary:** `theory/GLOSSARY.md` — every technical term used in the workspace

**Build the PDF** (requires two passes for TOC/references):

```bash
pdflatex -output-directory=theory/build/llm-core-architecture theory/build/llm-core-architecture/llm-core-architecture.tex && \
pdflatex -output-directory=theory/build/llm-core-architecture theory/build/llm-core-architecture/llm-core-architecture.tex && \
mv theory/build/llm-core-architecture/llm-core-architecture.pdf theory/llm-core-architecture.pdf
```

---

### LLM Surgeon (`testing/llm_surgeon/`)

Python toolkit for layer-level model surgery, probing, and experiment tracking on LLaMA-family models. Loads models via HuggingFace (fp16 or 4-bit quantized).

#### Modules

| Module | Purpose |
|--------|---------|
| `surgery` | Layer manipulation: remove, reorder, swap, duplicate layers. Attention head surgery: zero, scale, swap heads. Component ablation: zero MLP or attention. Calibration. |
| `probe` | Logit lens (predictions at each layer), hidden state extraction, interventions (scale, zero dims, clamp, noise, replace, project out directions) |
| `inspect` | Block influence scoring, magnitude analysis, sublayer decomposition, weight SVD, attention entropy, residual stream norms |
| `verify` | Structural validation after surgery (layer count, embedding dims, lm_head consistency) |
| `benchmark` | Perplexity evaluation, downstream task eval (via lm_eval), generation comparison across ollama models |
| `tracking` | SQLite-backed experiment database — log surgery ops, metrics, generation samples |
| `recipe` | YAML-based experiment definitions: surgery steps, analysis, evaluation, export pipeline |
| `export` | Save HuggingFace checkpoints, convert to GGUF, register with ollama |

#### Quick Start

```bash
cd testing
source .venv/bin/activate
python -c "
from llm_surgeon import surgery, probe

model, tokenizer = surgery.load_model('TinyLlama/TinyLlama-1.1B-Chat-v1.0', mode='inspect')
result = probe.logit_lens(model, tokenizer, 'The capital of France is', top_k=5)
print(result.summary())
```

#### Run Tests

```bash
cd testing
source .venv/bin/activate
pytest tests/ -v
```

---

### Live Probe GUI (`testing/gui/`)

Browser-based dashboard for interactive model experimentation. Load models into GPU memory, run probes, apply surgery, and visualize results in real time.

#### Features

- **Session management** — load/unload models, clone for A/B comparison, track VRAM usage
- **Surgery** — apply any surgery operation via UI, one-level undo
- **Logit lens heatmap** — layer x position grid showing prediction evolution through the network
- **Layer influence** — bar chart of per-layer block influence scores
- **A/B diff** — side-by-side logit lens with diff overlay highlighting divergence
- **Live generation** — token-by-token streaming with clickable probability inspection, repetition penalty, stop sequences

#### Architecture

```
Browser (React/Vite/TypeScript)
    |
    |-- REST  --> FastAPI --> SessionManager --> llm_surgeon
    |-- WebSocket --> streaming probe results via on_layer callbacks
    |
Backend holds models in GPU memory across operations
```

#### Start the GUI

```bash
./testing/gui/run.sh
```

Opens at `http://localhost:5173` (frontend) with backend on `127.0.0.1:8000`. Local only.

#### Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | FastAPI, uvicorn, Python 3.10+ |
| Frontend | React 18, TypeScript, Vite, Zustand |
| Visualization | D3.js |
| Communication | REST (session/surgery/inspect), WebSocket (logit lens, generate, intervene) |

#### Run Backend Tests

```bash
source testing/.venv/bin/activate
PYTHONPATH=testing pytest testing/gui/tests/ -v
```

---

### Research (`research/`)

Experimental findings, anomalies, and hypotheses from model surgery experiments.

- **Observations** (`research/observations/`) — dated markdown files with finding, evidence, reproducibility steps, hypotheses, and follow-ups
- Format: `YYYY-MM-DD-<descriptive-slug>.md`

---

## Project Layout

```
llm/
  theory/                    # LLM architecture theory (LaTeX + visuals)
    build/                   #   LaTeX source
    sources/                 #   Citation index + paper PDFs
    visuals/                 #   Interactive HTML diagrams
    GLOSSARY.md              #   Term definitions
  testing/
    llm_surgeon/             # Python surgical toolkit (8 modules)
    tests/                   # pytest suite for llm_surgeon
    gui/                     # Live probe dashboard
      backend/               #   FastAPI + SessionManager + routes
      frontend/              #   React/Vite/TypeScript + D3 visualizations
      tests/                 #   Backend API tests
      run.sh                 #   Start both servers
    experiments/             # Experiment definitions + SQLite database
    prompts/                 # Prompt templates
  models/                    # Cached HuggingFace models (TinyLlama, OpenLLaMA 3B)
  lib/                       # External libraries (llama.cpp, GPU-built)
  outputs/                   # Modified model variants from experiments
  research/                  # Observations and findings
  docs/                      # Design specs
```

## Dependencies

**Python** (in `testing/.venv`):

```
torch>=2.0
transformers>=4.40
accelerate>=0.27
bitsandbytes>=0.43
fastapi
uvicorn[standard]
pytest>=8.0
```

**Node** (in `testing/gui/frontend`):

```
react, react-dom, d3, zustand, vite, typescript
```

**System:**

- CUDA-capable GPU (tested on RTX 2080, 8GB VRAM)
- llama.cpp (at `lib/llama.cpp`, built with CUDA)
- Node.js (for frontend build)
- pdflatex (for theory document)

## Hardware

Tested on RTX 2080 (8GB VRAM). Two 4-bit quantized TinyLlama models fit comfortably for A/B comparison. OpenLLaMA 3B is tighter but works single-session.

## License

Copyright (C) 2026 Michael Lannum

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

See [LICENSE](LICENSE) for the full text.
