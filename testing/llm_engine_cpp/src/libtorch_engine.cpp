#include "llm_engine/libtorch_engine.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// [TODO] Once CMake links libtorch, include relevant headers:
// #include <torch/torch.h>
// #include <torch/script.h>

namespace llmengine {

namespace {
std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}
}  // namespace

struct LibtorchEngine::Impl {
    std::string           model_path;
    std::mutex            mu;
    std::vector<LogEntry> pending_logs;
    bool                  warned_unimpl = false;

    // [TODO] Real members once libtorch is linked:
    //   torch::Device device{torch::kCPU};
    //   torch::jit::Module module;        // for TorchScript-loaded models
    //   torch::optim::Adam optimizer;     // for training
    //   std::vector<torch::Tensor> grad_norms_per_layer;
    //   ... etc.

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
    }

    void warnSkeletonOnce(const char* hook) {
        std::lock_guard<std::mutex> lk(mu);
        if (warned_unimpl) return;
        warned_unimpl = true;
        pending_logs.push_back({
            now_ms(), Severity::Warn, "torch",
            std::string("LibtorchEngine::") + hook +
            " — skeleton, not yet implemented (see docs/BACKENDS.md §3.5)"
        });
    }
};

LibtorchEngine::LibtorchEngine(std::string model_path)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->model_path = std::move(model_path);
    m_impl->log(Severity::Info, "torch",
                "LibtorchEngine ready (path=" + m_impl->model_path + ") — SKELETON");
}

LibtorchEngine::~LibtorchEngine() = default;

Model::CheckpointResult LibtorchEngine::loadCheckpoint(std::string_view path) {
    // [TODO] Load via torch::jit::load (TorchScript) or a custom
    // safetensors reader → torch::Tensor.  Move to LLOB_LIBTORCH_DEVICE.
    m_impl->warnSkeletonOnce("loadCheckpoint");
    return {false, "LibtorchEngine: skeleton — torch model load not yet wired (path=" +
                   std::string(path) + ")"};
}

void LibtorchEngine::unloadCheckpoint() {
    // [TODO] Drop module + optimizer; explicit CUDA cache clear.
    m_impl->warnSkeletonOnce("unloadCheckpoint");
}

std::vector<LogEntry> LibtorchEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<LogEntry> out;
    out.swap(m_impl->pending_logs);
    return out;
}

// ── Training-area stubs ──────────────────────────────────────────────────
TrainingState LibtorchEngine::getTrainingState() {
    m_impl->warnSkeletonOnce("getTrainingState");
    return {};
}

std::vector<TrainingMetricCard> LibtorchEngine::getTrainingMetrics() {
    m_impl->warnSkeletonOnce("getTrainingMetrics");
    return {};
}

LossCurve LibtorchEngine::getTrainingLoss([[maybe_unused]] int maxSteps) {
    m_impl->warnSkeletonOnce("getTrainingLoss");
    return {};
}

std::vector<float> LibtorchEngine::getGradFlowPerLayer() {
    m_impl->warnSkeletonOnce("getGradFlowPerLayer");
    return {};
}

// ── Finetune-area stubs ──────────────────────────────────────────────────
LoRAConfig LibtorchEngine::getLoRAConfig() {
    m_impl->warnSkeletonOnce("getLoRAConfig");
    return {};
}

OptimizerConfig LibtorchEngine::getOptimizerConfig() {
    m_impl->warnSkeletonOnce("getOptimizerConfig");
    return {};
}

DataConfig LibtorchEngine::getDataConfig() {
    m_impl->warnSkeletonOnce("getDataConfig");
    return {};
}

EvalDiffMetric LibtorchEngine::getEvalDiff([[maybe_unused]] std::string_view bench) {
    m_impl->warnSkeletonOnce("getEvalDiff");
    return {};
}

}  // namespace llmengine
