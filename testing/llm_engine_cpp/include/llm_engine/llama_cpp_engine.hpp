#pragma once
//
// LlamaCppEngine — embedded llama.cpp backend with cb_eval activation
// capture.  Wraps the llama.cpp C API (`lib/llama.cpp/include/llama.h`)
// and intercepts attention / residual / mlp tensors as the forward pass
// streams through ggml.
//
// Owns the workspaces that need real-time data from a live forward pass:
// inference (residual flow, logit lens, output logits), attention
// (patterns, qkv stats, head stats), and the architecture map's "live
// activations" panel.
//
// Status: SKELETON.  Class compiles and is selectable via
// LLOB_BACKEND=llama, but per-method bodies are stubs that log "not yet
// implemented" once.  Implementation roadmap in docs/BACKENDS.md §3.4.
//
// Configuration:
//   LLOB_BACKEND_PATH   (env)  — filesystem path to the .gguf file.
//   LLOB_LLAMA_N_GPU_LAYERS  (env, optional) — layers to offload to GPU
//                              (default: -1 = all)
//   LLOB_LLAMA_CTX_SIZE     (env, optional)  — context size (default:
//                              model's training default)

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class LlamaCppEngine : public MockModel {
public:
    explicit LlamaCppEngine(std::string gguf_path);
    ~LlamaCppEngine() override;

    LlamaCppEngine(const LlamaCppEngine&)            = delete;
    LlamaCppEngine& operator=(const LlamaCppEngine&) = delete;

    // ── Wired hooks ───────────────────────────────────────────────────────
    CheckpointResult       loadCheckpoint  (std::string_view path) override;
    void                   unloadCheckpoint()                      override;
    std::vector<LogEntry>  drainEngineLogs ()                      override;

    // ── Phase-A target (model loaded → topology populated) ────────────────
    ModelInfo getModelInfo() override;

    // ── Phase-B targets (cb_eval activation capture) ──────────────────────
    // The big payoff — these fire during a live forward pass in the engine
    // worker thread.  UI thread reads the cached results.
    std::vector<std::vector<float>>
        getAttentionPattern(int layer, int head, int seqLen, HeadBias bias) override;
    std::vector<float>      getActivation(int layer, int kind, int n)        override;
    QKVStats                getQKVStats   (int layer, int head, int token)   override;
    ResidualSummary         getResidualSummary(int layer)                    override;

    // ── Phase-C target (token streaming) ──────────────────────────────────
    std::vector<std::string> getCurrentTokens() override;
    std::vector<LogitDist>   getOutputLogits (int k) override;

    // ── Phase-D target (logit lens) ───────────────────────────────────────
    std::vector<LogitLensRow> getLogitLensTrajectory(int token, int kLayers) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
