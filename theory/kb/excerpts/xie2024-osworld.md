---
paper_key: xie2024-osworld
title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"
authors: Xie, Zhang, Chen, Lou, Liu, Hu, Xing, Hu, Lin, Liu, Yu
year: 2024
venue: NeurIPS 2024 (Datasets & Benchmarks)
arxiv: 2404.07972
local_pdf: theory/sources/papers/xie2024-osworld.pdf
type: excerpts
---

# Excerpts — Xie et al. 2024, "OSWorld"

## #abstract — full verbatim

> Autonomous agents that accomplish complex computer tasks with minimal
> human interventions have the potential to transform human-computer
> interaction, significantly enhancing accessibility and productivity.
> However, existing benchmarks either lack an interactive environment
> or are limited to environments specific to certain applications or
> domains, failing to reflect the diverse and complex nature of
> real-world computer use, thereby limiting the scope of tasks and
> agent scalability. To address this issue, we introduce OSWorld, the
> first-of-its-kind scalable, real computer environment for multimodal
> agents, supporting task setup, execution-based evaluation, and
> interactive learning across various operating systems such as Ubuntu,
> Windows, and macOS. OSWorld can serve as a unified, integrated
> computer environment for assessing open-ended computer tasks that
> involve arbitrary applications. Building upon OSWorld, we create a
> benchmark of 369 computer tasks involving real web and desktop apps
> in open domains, OS file I/O, and workflows spanning multiple
> applications. Each task example is derived from real-world computer
> use cases and includes a detailed initial state setup configuration
> and a custom execution-based evaluation script for reliable,
> reproducible evaluation. Extensive evaluation of state-of-the-art
> LLM/VLM-based agents on OSWorld reveals significant deficiencies in
> their ability to serve as computer assistants. While humans can
> accomplish over 72.36% of the tasks, the best model achieves only
> 12.24% success, primarily struggling with GUI grounding and
> operational knowledge.

(Verified verbatim from PDF on 2026-05-12.)

## §2.3–2.4 — Observation and action spaces {#sec-2-3-4-obs-action}

> The observation space in OSWorld contains a complete screenshot of the
> desktop screen, including the mouse's position and shape, various
> application windows, files, and folders that are opened in different
> sizes and orders, maintaining the same perception as a human. Also, to
> be aligned with previous agent-building web and mobile research that
> provide and support the use of the webpage's DOM and app's view
> hierarchy, OSWorld also provides XML-format accessibility (a11y) tree
> (obtained via ATSPI2 on Ubuntu, via PyWinAuto on Windows, etc.), which
> can support additional information for modeling.

> Action space A in OSWorld encompasses all mouse and keyboard actions,
> including movement, clicks (left-key, right-key, multiple clicks),
> dragging, keystrokes, hotkeys, and others, covering all human-computer
> action space.

The paper specifies that agents must generate syntax-correct pyautogui
Python code for each action (e.g., `.click(300, 540, button='right')`,
`.hotkey('ctrl', 'alt', 't')`). Three terminal actions are added —
`WAIT`, `FAIL`, and `DONE` — to handle timing, infeasible tasks, and
task completion respectively. This makes the action space strictly richer
than prior web-specific benchmarks (MiniWoB++, WebArena) which model only
clicks and typing without support for arbitrary key combinations,
right-clicking, or cross-application operations.

## §2.2 / §3.1 — Real-OS environment design and benchmark scope {#sec-2-2-3-1-env-design}

> OSWorld is an executable and controllable environment that supports task
> initialization, execution-based evaluation, and interactive agent learning
> in a range of real operating systems (e.g., Ubuntu, Windows, macOS) using
> virtual machine techniques. Virtual machine offers a safe isolated
> environment and prevents the agent resulting in irreversible damaging
> effect on the real host machine.

> We introduce the OSWorld benchmark, which encompasses 369 real computing
> tasks defined and executed on Ubuntu. Additionally, we provide a set of
> 43 tasks for Windows built on the OSWorld environment.

The 369 Ubuntu tasks span five categories: OS (file I/O, settings), Office
(LibreOffice Calc/Writer/Impress), Daily (Chrome, VLC, Thunderbird),
Professional (VS Code, GIMP), and Workflow (multi-application). The
multi-app workflow tasks (101 tasks, 27.4%) are the hardest category,
requiring agents to coordinate state across independent applications within
a single goal. The benchmark also contains 30 infeasible tasks (8.1%) —
tasks that are inherently impossible due to deprecated or hallucinated
features — to test whether agents can correctly predict failure.

## §4 — Headline model vs. human gap {#sec-4-results}

> While humans can accomplish over 72.36% of the tasks, the best model
> achieves only 12.24% success, primarily struggling with GUI grounding
> and operational knowledge.

Table 5 (selected rows, verbatim success rates):

| Inputs        | Model          | OS     | Office | Daily  | Profess. | Workflow | Overall |
|---------------|----------------|--------|--------|--------|----------|----------|---------|
| A11y tree     | GPT-4          | 20.83% | 3.58%  | 25.64% | 26.53%   | 2.97%    | 12.24%  |
| Screenshot    | GPT-4V         | 12.50% | 1.86%  | 7.58%  | 4.08%    | 6.04%    | 5.26%   |
| Screenshot    | Claude-3-Opus  | 4.17%  | 1.87%  | 2.71%  | 2.04%    | 2.61%    | 2.42%   |
| Scr. + A11y   | GPT-4o         | 41.67% | 6.16%  | 12.33% | 14.29%   | 7.46%    | 11.21%  |
| Human         |                | 75.00% | 71.79% | 70.51% | 73.47%   | 73.27%   | 72.36%  |

The 12.24% ceiling (GPT-4 with accessibility tree only) and the human
72.36% figure are the load-bearing numbers for the eval-methodology note.
The paper attributes the gap primarily to GUI grounding failure
(models cannot reliably predict which pixel to click) and limited
operational knowledge (application-specific shortcuts and workflows).

[Verified from PDF on 2026-05-12] Added §2.3–2.4 (#sec-2-3-4-obs-action), §2.2/§3.1 (#sec-2-2-3-1-env-design), §4 (#sec-4-results). Abstract verified verbatim.
