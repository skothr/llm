# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Purpose

LLM theory workspace — high-level explanations, visualizations, and math related to LLM theory. Not a code project; primarily LaTeX documents, HTML visualizations, and curated source papers.

# Build Commands

```bash
# Compile LaTeX document (run twice for TOC/references)
pdflatex llm-core-architecture.tex && pdflatex llm-core-architecture.tex

# View HTML visualization
xdg-open visuals/llm-architecture-diagram.html
```

# Architecture

## Core Document

`llm-core-architecture.tex` — LaTeX document covering Transformer architecture from original encoder-decoder through modern decoder-only (LLaMA). 10 sections: Transformer overview, Tokenization, Embeddings, Positional Encoding, Attention, FFN, Normalization/Residuals, Decoder-Only Shift, Output Head, Full Forward Pass.

Custom LaTeX environments:
- `\begin{implbox}` — green "Implementation Note" callouts
- `\begin{evobox}` — blue "Architectural Evolution" callouts
- `\dimtext{}` — inline dimension annotations

## HTML Companion

`visuals/llm-architecture-diagram.html` — standalone interactive diagram with clickable layers showing tensor shapes and data flow. Dark theme, self-contained (no build step).

## Sources

All claims must be grounded in canonical papers.

- `sources/index.json` — master index (citation key, title, authors, year, URL, local file, summary)
- `sources/papers/` — local PDF copies, named `{key}_{slug}.pdf`

When adding a new source: add entry to `index.json`, download PDF to `sources/papers/`, use the citation key consistently in LaTeX `\cite{}` commands.

## Glossary

`GLOSSARY.md` — every technical term used in this workspace must be defined here. If you encounter or use a term not in the glossary, add it immediately.

# Conventions

- ALWAYS define EVERY variable in EVERY equation, directly underneath it (brief is fine, but no undefined variables)
- FIRST formalize math, THEN describe technical aspects and practical use, THEN elaborate using accessible language and/or analogies
- Ground all architectural claims in specific source papers with citation keys from `sources/index.json`
