#pragma once
//
// GGUFInspectorEngine — pure-C++ static-weight inspection backend.
//
// Reads a `.gguf` checkpoint directly via mmap and serves every Model::*
// method that's about static weight data: getModelInfo, getStateDict,
// getTensorMeta, getTensorStats, getWeightSlice, getWeightHistogram,
// getSingularValues, getTensorSlice2D, getDiffSlice2D.
//
// Methods that need a live forward pass (attention patterns, residuals,
// activations, logit lens) are intentionally NOT this backend's job —
// they fall through to MockModel sentinels.  Compose with LlamaCppEngine
// when both worlds are wanted (see docs/BACKENDS.md §6).
//
// Status: SKELETON.  Class compiles and is selectable via
// LLOB_BACKEND=gguf, but per-method bodies are stubs that log "not yet
// implemented" once on first call.  Implementation roadmap in
// docs/BACKENDS.md §3.3.
//
// Configuration:
//   LLOB_BACKEND_PATH   (env)  — filesystem path to the .gguf file.
//                                Constructor takes the path directly.

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class GGUFInspectorEngine : public MockModel {
public:
    explicit GGUFInspectorEngine(std::string gguf_path);
    ~GGUFInspectorEngine() override;

    GGUFInspectorEngine(const GGUFInspectorEngine&)            = delete;
    GGUFInspectorEngine& operator=(const GGUFInspectorEngine&) = delete;

    // ── Wired hooks ───────────────────────────────────────────────────────
    CheckpointResult       loadCheckpoint  (std::string_view path) override;
    void                   unloadCheckpoint()                      override;
    std::vector<LogEntry>  drainEngineLogs ()                      override;

    // ── Phase-A targets (header parse + state-dict enumeration) ───────────
    // These are the first overrides to grow real bodies.  Stubs for now.
    ModelInfo               getModelInfo()                                     override;
    std::vector<TensorMeta> getStateDict()                                     override;
    TensorMeta              getTensorMeta(std::string_view name)               override;

    // ── Phase-B targets (mmap'd weight reads) ─────────────────────────────
    std::vector<float>      getWeightSlice    (std::string_view name, int offset, int n) override;
    std::vector<int>        getWeightHistogram(std::string_view name, int bins)          override;

    // Phase-C/D (stats + SVD + 2D slice) deliberately fall through to
    // MockModel until an Eigen / Armadillo dependency is justified.

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
