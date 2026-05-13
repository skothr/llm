#include "llm_engine/llama_cpp_engine.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// [TODO Phase-A] Once CMake links lib/llama.cpp, uncomment:
// #include <llama.h>

namespace llmengine {

namespace {
std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}
}  // namespace

struct LlamaCppEngine::Impl {
    std::string           gguf_path;
    std::mutex            mu;
    std::vector<LogEntry> pending_logs;
    bool                  warned_unimpl = false;

    // [TODO Phase-A] llama.cpp handle members.  Once enabled:
    //   llama_model*    model = nullptr;
    //   llama_context*  ctx   = nullptr;
    //   llama_sampler*  smpl  = nullptr;
    //
    // Also need a per-step capture buffer to receive cb_eval snapshots:
    //   struct CaptureBundle {
    //       std::map<int, std::vector<float>> resid_pre;     // [layer] → d_model
    //       std::map<int, std::vector<float>> resid_post;
    //       std::map<int, std::vector<float>> mlp_post;
    //       std::map<std::pair<int,int>, std::vector<float>> q, k, v;
    //       std::map<std::pair<int,int>, std::vector<std::vector<float>>> attn;
    //   } current;
    //
    // The cb_eval callback (registered via llama_set_callback_eval) inspects
    // each tensor as ggml computes it, identifies it by name (e.g.
    // "kqv_merged-N", "attn_norm-N"), and copies host-side into `current`.

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
    }

    void warnSkeletonOnce(const char* hook) {
        std::lock_guard<std::mutex> lk(mu);
        if (warned_unimpl) return;
        warned_unimpl = true;
        pending_logs.push_back({
            now_ms(), Severity::Warn, "llama",
            std::string("LlamaCppEngine::") + hook +
            " — skeleton, not yet implemented (see docs/BACKENDS.md §3.4)"
        });
    }
};

LlamaCppEngine::LlamaCppEngine(std::string gguf_path)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->gguf_path = std::move(gguf_path);
    m_impl->log(Severity::Info, "llama",
                "LlamaCppEngine ready (path=" + m_impl->gguf_path + ") — SKELETON");
}

LlamaCppEngine::~LlamaCppEngine() = default;

Model::CheckpointResult LlamaCppEngine::loadCheckpoint(std::string_view path) {
    // [TODO Phase-A] Wire llama.cpp init:
    //   llama_backend_init();
    //   auto mparams = llama_model_default_params();
    //   mparams.n_gpu_layers = std::getenv("LLOB_LLAMA_N_GPU_LAYERS") ? atoi(...) : -1;
    //   m_impl->model = llama_model_load_from_file(path.c_str(), mparams);
    //   if (!m_impl->model) return {false, "llama_load_model_from_file failed"};
    //   auto cparams = llama_context_default_params();
    //   cparams.n_ctx = std::getenv("LLOB_LLAMA_CTX_SIZE") ? atoi(...) : 0;
    //   m_impl->ctx = llama_init_from_model(m_impl->model, cparams);
    //   llama_set_callback_eval(m_impl->ctx, &cb_eval_thunk, m_impl.get());
    m_impl->warnSkeletonOnce("loadCheckpoint");
    return {false, "LlamaCppEngine: skeleton — llama.cpp init not yet wired (path=" +
                   std::string(path) + ")"};
}

void LlamaCppEngine::unloadCheckpoint() {
    // [TODO Phase-A] llama_free(ctx); llama_model_free(model); llama_backend_free();
    m_impl->warnSkeletonOnce("unloadCheckpoint");
}

std::vector<LogEntry> LlamaCppEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<LogEntry> out;
    out.swap(m_impl->pending_logs);
    return out;
}

// ── Phase-A stub ─────────────────────────────────────────────────────────
ModelInfo LlamaCppEngine::getModelInfo() {
    m_impl->warnSkeletonOnce("getModelInfo");
    // [TODO Phase-A] Once model is loaded, derive from:
    //   llama_model_n_layer / n_head / n_head_kv / n_embd / n_ff /
    //   n_vocab / n_ctx_train / rope_freq_base
    //   chat template via llama_model_chat_template
    return {};
}

// ── Phase-B stubs (cb_eval capture) ──────────────────────────────────────
std::vector<std::vector<float>> LlamaCppEngine::getAttentionPattern(
    [[maybe_unused]] int layer, [[maybe_unused]] int head,
    [[maybe_unused]] int seqLen, [[maybe_unused]] HeadBias bias) {
    m_impl->warnSkeletonOnce("getAttentionPattern");
    // [TODO Phase-B] Pull from Impl::current.attn[{layer, head}].  The
    // cb_eval handler is the one that filled it during the most recent
    // llama_decode tick.
    return {};
}

std::vector<float> LlamaCppEngine::getActivation(
    [[maybe_unused]] int layer, [[maybe_unused]] int kind,
    [[maybe_unused]] int n) {
    m_impl->warnSkeletonOnce("getActivation");
    // [TODO Phase-B] Dispatch on kind (0=Q, 1=K, 2=V, default=resid_post)
    // and slice the captured tensor at Impl::current.{q,k,v,resid_post}.
    return {};
}

QKVStats LlamaCppEngine::getQKVStats(
    [[maybe_unused]] int layer, [[maybe_unused]] int head,
    [[maybe_unused]] int token) {
    m_impl->warnSkeletonOnce("getQKVStats");
    // [TODO Phase-B] Compute norms over Impl::current.{q,k,v}[{layer,head}];
    // attn_to_bos / attn_to_self / attn_to_prev pulled from row `token` of
    // the attention matrix.
    return {};
}

ResidualSummary LlamaCppEngine::getResidualSummary([[maybe_unused]] int layer) {
    m_impl->warnSkeletonOnce("getResidualSummary");
    // [TODO Phase-B] Norms + cos(prev) + kurtosis + effective rank from
    // Impl::current.resid_pre/post[layer].
    return {};
}

// ── Phase-C stubs (token streaming) ──────────────────────────────────────
std::vector<std::string> LlamaCppEngine::getCurrentTokens() {
    m_impl->warnSkeletonOnce("getCurrentTokens");
    // [TODO Phase-C] Maintain Impl::token_strings as the worker thread runs
    // llama_decode in a loop; convert each token id via llama_token_to_piece.
    return {};
}

std::vector<LogitDist> LlamaCppEngine::getOutputLogits([[maybe_unused]] int k) {
    m_impl->warnSkeletonOnce("getOutputLogits");
    // [TODO Phase-C] Top-k from llama_get_logits at the last sampled
    // position; convert ids to strings via llama_token_to_piece.
    return {};
}

// ── Phase-D stub (logit lens) ────────────────────────────────────────────
std::vector<LogitLensRow> LlamaCppEngine::getLogitLensTrajectory(
    [[maybe_unused]] int token, [[maybe_unused]] int kLayers) {
    m_impl->warnSkeletonOnce("getLogitLensTrajectory");
    // [TODO Phase-D] For each layer L: project Impl::current.resid_post[L]
    // through the lm_head (read once and cached) → top-k via partial sort.
    // is_resolved marks first L where eventual top-1 first becomes top-1.
    return {};
}

}  // namespace llmengine
