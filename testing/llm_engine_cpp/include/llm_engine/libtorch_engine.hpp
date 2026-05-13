#pragma once
//
// LibtorchEngine — in-process libtorch (C++ ATen) backend.
//
// The most flexible but heaviest backend: links a substantial portion of
// PyTorch (~hundreds of MB of .so files).  Has no built-in transformer
// architecture — using this seriously requires either reimplementing the
// model definition in C++ or loading a TorchScript-traced graph.
//
// Owns: training-related hooks (pause/resume/step/reset/stopTraining,
// getTrainingMetrics, getGradFlowPerLayer, getPerLayerLoss), LoRA /
// finetune (getLoRAConfig, getEvalDiff, getABSample), and any custom
// kernel work that doesn't fit llama.cpp's surface.
//
// Status: SKELETON.  Compiles + selectable via LLOB_BACKEND=torch but
// every method falls through to MockModel sentinels.  Implementation
// pinned behind a real workspace need — see docs/BACKENDS.md §3.5.
//
// Configuration:
//   LLOB_BACKEND_PATH   (env)  — filesystem path to model directory
//                                (HF format, .pt, or .safetensors)
//   LLOB_LIBTORCH_DEVICE (env, optional) — "cuda:0" | "cpu" | …

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class LibtorchEngine : public MockModel {
public:
    explicit LibtorchEngine(std::string model_path);
    ~LibtorchEngine() override;

    LibtorchEngine(const LibtorchEngine&)            = delete;
    LibtorchEngine& operator=(const LibtorchEngine&) = delete;

    // ── Wired hooks ───────────────────────────────────────────────────────
    CheckpointResult       loadCheckpoint  (std::string_view path) override;
    void                   unloadCheckpoint()                      override;
    std::vector<LogEntry>  drainEngineLogs ()                      override;

    // ── Future targets (training + finetune) ──────────────────────────────
    // Stubs that document scope.  Real bodies land when a workspace need
    // forces it — there's no point implementing training without a
    // training plan in scope.
    TrainingState                   getTrainingState   ()              override;
    std::vector<TrainingMetricCard> getTrainingMetrics ()              override;
    LossCurve                       getTrainingLoss    (int maxSteps) override;
    std::vector<float>              getGradFlowPerLayer()              override;

    LoRAConfig         getLoRAConfig()      override;
    OptimizerConfig    getOptimizerConfig() override;
    DataConfig         getDataConfig()      override;
    EvalDiffMetric     getEvalDiff (std::string_view bench) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
