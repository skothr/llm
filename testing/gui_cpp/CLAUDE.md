# CLAUDE.md — llobotomy C++ port

You are starting from a complete React/HTML mockup of an LLM interpretability bench (`llobotomy.html`). Your job is to port it to a native C++ application built on Dear ImGui, while preserving the exact visual language, interaction model, and global state architecture.

## Read these first, in order

1. **`HANDOFF.md`** — the full spec. Do not skip. It defines the source map, every workspace's layout, all widget contracts, the shared state model (`AppState`), the interaction contract (left-click = probe, right-click = ablate, etc.), data seams for plugging in a real model backend, theme tokens, and Dear ImGui implementation pointers in §12.
2. **`llobotomy.html`** — the visual contract. Open it in a browser and click around every workspace. The C++ build should look and behave the same. Treat any divergence from the mock as a bug unless explicitly listed in HANDOFF.md §9 ("Underspecified") or §10 ("Out of scope").
3. **`screenshots/`** — labeled stills of every workspace and a few close-ups. Use these as a stable reference if the live mockup gets edited.

The JSX source files (`app.jsx`, `workspace-*.jsx`, `imgui-core.jsx`, `imgui-widgets.jsx`, `theme.css`, `resizable-split.jsx`, `tweaks-panel.jsx`, `design-canvas.jsx`) are the inlined contents of `llobotomy.html`. Read the JSX, not the bundle — they're easier to navigate.

## How to plan this work

This is a multi-week C++ build, not a one-shot. Use `/ultraplan` (or equivalent multi-step planner) on first contact. Feed it `HANDOFF.md` and ask for a phased plan. Suggested phases:

1. **Theme + chrome** — port `theme.css` to a `Theme` enum + `Style::Apply()`, build `ImWindow`/`Section`/`Tabs`/`Toolbar`/`StatusBar` helpers.
2. **`AppState` + dockspace shell** — single state struct, top-level menubar/tabs/status bar, one empty dockspace per workspace, keyboard shortcuts.
3. **Mock model backend** — `MockModel` impl behind a `Model` interface (see HANDOFF §7 data seams). Deterministic generators for attention patterns, activations, weight slices, histograms.
4. **Architecture workspace** (the hero) — outline tree, arch map (custom `ImDrawList` 2D), inspector, ops panel, optional raw hex dock. Get expand/collapse/skip working with the same mouse contract as the mock.
5. **Inference + Attention workspaces** — token stream, per-layer residual pane, attention heatmaps. Reuse global selection.
6. **Remaining workspaces** — Probes, Training, Finetune, Datasets, Raw tensors, Logs.
7. **Polish** — Settings menu (replaces the mock's Tweaks panel), persistence (`imgui.ini` for layout, sidecar JSON for ablation/probe sets), file dialogs, error states.

## Hard rules

- **Do not port the Tweaks panel or the design canvas.** Both are mock-only scaffolding. Theme/density/accent live in a Settings menu in the C++ build (§5 of HANDOFF).
- **Do not invent new visual treatments.** If the mock has a 28px head cell, the C++ build has a 28px head cell. The point of having a mock is that visual decisions are already made.
- **Do not introduce decorative chrome that isn't in the mock.** No emoji, no purple gradients, no cutesy animations, no rounded corners (theme is Tracy/Photoshop/amber — sharp, dense, technical).
- **Every state-changing action must call `pushLog(kind, msg)`.** The Logs workspace is fed by these. See `app.jsx`'s mutators for the exact strings.
- **All workspaces share `AppState` by reference.** Ablating in `attn` must update the badge in `arch`, the strip in the outline tree, and the count in the status bar — no per-workspace selection state.

## Recommended dependencies (and only these)

- Dear ImGui — the base
- ImPlot — line charts, large heatmaps
- imgui_club's MemoryEditor — the HexView widget
- ImGuiFileDialog — load/save dialogs

Skip everything else. No node editors, no markdown libs, no skin packs.

## When you get stuck

- For "how do I do X in ImGui?" — `imgui_demo.cpp` is the canonical reference. It has examples of every widget, custom drawing pattern, dock layout, and table.
- For "how should this interaction work?" — open `llobotomy.html` and try it.
- For "what's the data shape?" — `HANDOFF.md` §7 has the `Model::*` interface. Until real model code is wired up, ship the `MockModel`.
- For "is this in scope?" — `HANDOFF.md` §10 lists what's out. If it's not there, port it.

## What "done" looks like

The C++ binary launches, runs against `MockModel`, and is visually + behaviorally indistinguishable from `llobotomy.html` for every workspace. Dock layout persists. Theme/density/accent persist. Ablation/probe sets persist per-checkpoint. Real model backend is a follow-up tracked separately.
