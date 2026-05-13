#include "llm_engine/gguf_inspector_engine.hpp"

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

struct GGUFInspectorEngine::Impl {
    std::string           gguf_path;
    std::mutex            mu;
    std::vector<LogEntry> pending_logs;
    bool                  warned_unimpl = false;   // log "skeleton" only once

    // [TODO Phase-A] Add GGUF header parse + tensor index here.  Suggested
    // layout once implementation lands:
    //   GGUFHeader header;                   // magic + version + kv-store
    //   std::vector<GGUFTensorEntry> tensors; // name → (dtype, shape, offset, size)
    //   void* mmap_base = nullptr;            // mapped file region
    //   std::size_t mmap_len = 0;

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
    }

    void warnSkeletonOnce(const char* hook) {
        std::lock_guard<std::mutex> lk(mu);
        if (warned_unimpl) return;
        warned_unimpl = true;
        pending_logs.push_back({
            now_ms(), Severity::Warn, "gguf",
            std::string("GGUFInspectorEngine::") + hook +
            " — skeleton, not yet implemented (see docs/BACKENDS.md §3.3)"
        });
    }
};

GGUFInspectorEngine::GGUFInspectorEngine(std::string gguf_path)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->gguf_path = std::move(gguf_path);
    m_impl->log(Severity::Info, "gguf",
                "GGUFInspectorEngine ready (path=" + m_impl->gguf_path + ") — SKELETON");
}

GGUFInspectorEngine::~GGUFInspectorEngine() = default;

Model::CheckpointResult GGUFInspectorEngine::loadCheckpoint(std::string_view path) {
    // [TODO Phase-A] Open path via mmap, parse GGUF magic + header,
    // populate Impl::header + Impl::tensors.  Return ok=true with topology
    // logged like HFProxyEngine does.  See docs/BACKENDS.md §3.3.
    m_impl->warnSkeletonOnce("loadCheckpoint");
    return {false, "GGUFInspectorEngine: skeleton — file IO not yet implemented (path=" +
                   std::string(path) + ")"};
}

void GGUFInspectorEngine::unloadCheckpoint() {
    // [TODO Phase-A] munmap + clear tensors.
    m_impl->warnSkeletonOnce("unloadCheckpoint");
}

std::vector<LogEntry> GGUFInspectorEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<LogEntry> out;
    out.swap(m_impl->pending_logs);
    return out;
}

// ── Phase-A stubs — grow real bodies first ───────────────────────────────
ModelInfo GGUFInspectorEngine::getModelInfo() {
    m_impl->warnSkeletonOnce("getModelInfo");
    // [TODO Phase-A] Read GGUF metadata kv-store: keys like
    // "general.name", "llama.context_length", "llama.attention.head_count",
    // "llama.attention.head_count_kv", "llama.embedding_length",
    // "llama.feed_forward_length", "llama.rope.theta" — translate to
    // ModelInfo fields.
    return {};
}

std::vector<TensorMeta> GGUFInspectorEngine::getStateDict() {
    m_impl->warnSkeletonOnce("getStateDict");
    // [TODO Phase-A] Enumerate Impl::tensors → vector<TensorMeta>.  Each
    // entry's size_bytes comes from the per-dtype block size; device is
    // "cpu" since GGUF is a static file format.
    return {};
}

TensorMeta GGUFInspectorEngine::getTensorMeta(std::string_view name) {
    m_impl->warnSkeletonOnce("getTensorMeta");
    // [TODO Phase-A] O(1) lookup in a name→index map built at load.
    (void)name;
    return {};
}

// ── Phase-B stubs — mmap'd reads ────────────────────────────────────────
std::vector<float> GGUFInspectorEngine::getWeightSlice(
    std::string_view name, int offset, int n) {
    m_impl->warnSkeletonOnce("getWeightSlice");
    // [TODO Phase-B] Look up tensor entry, dispatch on dtype:
    //   F32 → memcpy n * sizeof(float) starting at base+entry.offset+offset*4
    //   F16 → unpack via _cvtsh_ss or a fallback
    //   Q4_0/Q4_K/Q8_0 → dequantize block-by-block (see ggml-quants.h
    //                    or write our own — formats are documented)
    (void)name; (void)offset; (void)n;
    return {};
}

std::vector<int> GGUFInspectorEngine::getWeightHistogram(
    std::string_view name, int bins) {
    m_impl->warnSkeletonOnce("getWeightHistogram");
    // [TODO Phase-B] Stream the tensor (in dtype-appropriate chunks),
    // accumulate into `bins` linear buckets.  Cache the result keyed on
    // (name, bins) — histograms are the most-requested per-frame stat
    // and re-streaming a 100MB tensor every redraw would tank FPS.
    (void)name; (void)bins;
    return {};
}

}  // namespace llmengine
