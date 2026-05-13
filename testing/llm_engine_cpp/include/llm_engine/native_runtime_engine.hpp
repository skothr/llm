#pragma once
//
// NativeRuntimeEngine — pure-C++ inference runtime owned by us.
//
// The long-term ambition.  Read the checkpoint format directly, allocate
// tensors with our own arena, implement attention / FFN / sampling
// kernels (CUDA + CPU), capture everything we want at every step.  The
// "from scratch" path — no Python, no llama.cpp version pinning, no
// libtorch.so chain.
//
// This is multi-month engineering work.  The skeleton exists so the
// abstraction stays honest: when this lands, every other backend
// becomes a fallback rather than a primary, and we own the surgery
// surface end-to-end.
//
// Status: SKELETON ONLY.  Selectable via LLOB_BACKEND=native; every
// method falls through to MockModel sentinels.  See docs/BACKENDS.md §3.6.
//
// Useful design references (none required, all permissively licensed):
//   - llama.cpp / ggml — for tensor + quantization primitives
//   - karpathy/llama2.c — for a minimal reference forward pass
//   - mlx-c — for a clean C-API design over a hardware backend
//
// Configuration:
//   LLOB_BACKEND_PATH       (env)  — filesystem path to checkpoint
//   LLOB_NATIVE_DEVICE      (env, optional) — "cuda:0" | "cpu" | …

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class NativeRuntimeEngine : public MockModel {
public:
    explicit NativeRuntimeEngine(std::string checkpoint_path);
    ~NativeRuntimeEngine() override;

    NativeRuntimeEngine(const NativeRuntimeEngine&)            = delete;
    NativeRuntimeEngine& operator=(const NativeRuntimeEngine&) = delete;

    // ── Wired hooks ───────────────────────────────────────────────────────
    CheckpointResult       loadCheckpoint  (std::string_view path) override;
    void                   unloadCheckpoint()                      override;
    std::vector<LogEntry>  drainEngineLogs ()                      override;

    // Per-method overrides will be added as the runtime grows.  Default
    // policy: nothing implemented; everything inherits MockModel sentinels.

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
