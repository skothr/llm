#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

namespace llob {

// PRNG mirroring imgui-widgets.jsx::mulberry32 — used both by MockModel and
// by widgets that want deterministic-but-cheap mock data.
struct Mulberry32 {
    std::uint32_t a;
    explicit Mulberry32(std::uint32_t seed) : a(seed) {}
    float next() {
        a += 0x6D2B79F5u;
        std::uint32_t t = a;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + ((t ^ (t >> 7)) * (t | 61u));
        return static_cast<float>((t ^ (t >> 14)) >> 0) / 4294967296.0f;
    }
};

enum class HeadBias { Diag, Prev, First, Broad, Induction };

// Abstract data backend — every visual signal in a workspace eventually goes
// through this interface. MockModel implements all of it deterministically;
// a future NativeBackend can swap in real hooks without touching the UI.
struct Model {
    virtual ~Model() = default;

    // Shape: matrix[i][j] for i,j in [0,n). Causal — j > i is masked to 0.
    virtual std::vector<std::vector<float>>
        getAttentionPattern(int layer, int head, int seqLen, HeadBias bias) = 0;

    // Length-n activation vector in roughly [-1.2, 1.2].
    virtual std::vector<float>
        getActivation(int layer, int kind, int n) = 0;

    // Slice of a weight matrix; same shape as getActivation.
    virtual std::vector<float>
        getWeightSlice(int layer, std::string_view component, int offset, int n) = 0;

    // Histogram bins (counts) for a weight component.
    virtual std::vector<int>
        getWeightHistogram(int layer, std::string_view component, int bins = 40) = 0;

    // Per-component / per-head signal in [0,1] for tinting.
    virtual float getComponentNorm(int layer, std::string_view comp) = 0;
    virtual float getHeadNorm     (int layer, int head) = 0;
};

// Construct the deterministic mock backend.
struct MockModel : Model {
    std::vector<std::vector<float>>
        getAttentionPattern(int layer, int head, int seqLen, HeadBias bias) override;
    std::vector<float>
        getActivation(int layer, int kind, int n) override;
    std::vector<float>
        getWeightSlice(int layer, std::string_view component, int offset, int n) override;
    std::vector<int>
        getWeightHistogram(int layer, std::string_view component, int bins = 40) override;
    float getComponentNorm(int layer, std::string_view comp) override;
    float getHeadNorm     (int layer, int head) override;
};

}  // namespace llob
