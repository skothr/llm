#include "llm_engine/native_runtime_engine.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace llmengine {

namespace {
std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}
}  // namespace

struct NativeRuntimeEngine::Impl {
    std::string           checkpoint_path;
    std::mutex            mu;
    std::vector<LogEntry> pending_logs;
    bool                  warned_unimpl = false;

    // [TODO long-term] Real members once a runtime exists.  Sketched layout
    // (highly speculative — final shape depends on the kernel set we pick):
    //
    //   class TensorArena;
    //   std::unique_ptr<TensorArena> arena;          // CPU + GPU pools
    //   ParsedCheckpoint              weights;       // mmap'd file headers
    //   std::vector<TransformerBlock> blocks;        // per-layer weights ref
    //   KvCache                       kv_cache;
    //   ForwardPassRecorder           recorder;      // per-step capture
    //   std::thread                   worker;        // owns generation loop

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
    }

    void warnSkeletonOnce(const char* hook) {
        std::lock_guard<std::mutex> lk(mu);
        if (warned_unimpl) return;
        warned_unimpl = true;
        pending_logs.push_back({
            now_ms(), Severity::Warn, "native",
            std::string("NativeRuntimeEngine::") + hook +
            " — skeleton, not yet implemented (see docs/BACKENDS.md §3.6)"
        });
    }
};

NativeRuntimeEngine::NativeRuntimeEngine(std::string checkpoint_path)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->checkpoint_path = std::move(checkpoint_path);
    m_impl->log(Severity::Info, "native",
                "NativeRuntimeEngine ready (path=" + m_impl->checkpoint_path +
                ") — SKELETON (long-term)");
}

NativeRuntimeEngine::~NativeRuntimeEngine() = default;

Model::CheckpointResult NativeRuntimeEngine::loadCheckpoint(std::string_view path) {
    m_impl->warnSkeletonOnce("loadCheckpoint");
    return {false, "NativeRuntimeEngine: skeleton — pure-C++ runtime not yet built (path=" +
                   std::string(path) + ")"};
}

void NativeRuntimeEngine::unloadCheckpoint() {
    m_impl->warnSkeletonOnce("unloadCheckpoint");
}

std::vector<LogEntry> NativeRuntimeEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<LogEntry> out;
    out.swap(m_impl->pending_logs);
    return out;
}

}  // namespace llmengine
