#include "model/model.hpp"

#include <algorithm>
#include <cmath>

namespace llob {

namespace {

std::uint32_t hashString(std::string_view s) {
    std::uint32_t h = 2166136261u;
    for (char c : s) { h ^= static_cast<std::uint8_t>(c); h *= 16777619u; }
    return h;
}

}  // namespace

std::vector<std::vector<float>>
MockModel::getAttentionPattern(int layer, int head, int seqLen, HeadBias bias) {
    const int n = std::max(1, seqLen);
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 100 + head + 1));
    std::vector<std::vector<float>> m(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        std::vector<float> row(i + 1, 0.0f);
        for (int j = 0; j <= i; ++j) {
            float v = rng.next() * 0.3f;
            switch (bias) {
                case HeadBias::Diag:
                    if (j == i)         v += 0.7f * (0.6f + rng.next() * 0.4f);
                    else if (j == i - 1) v += 0.4f * (0.6f + rng.next() * 0.4f);
                    break;
                case HeadBias::Prev:
                    if (j == i - 1) v += 0.85f * (0.7f + rng.next() * 0.3f);
                    break;
                case HeadBias::First:
                    if (j == 0) v += 0.6f;
                    if (j == i) v += 0.2f;
                    break;
                case HeadBias::Broad:
                    v = rng.next() * 0.5f + 0.1f;
                    break;
                case HeadBias::Induction: {
                    const int peak = (i * 7 + layer * 100 + head) % std::max(1, i);
                    if (j == peak) v += 0.7f;
                    break;
                }
            }
            row[j] = v;
        }
        // Softmax-ish normalize.
        float sum = 0.0f; for (float v : row) sum += v;
        if (sum <= 0.0f) sum = 1.0f;
        for (int j = 0; j <= i; ++j) m[i][j] = row[j] / sum;
    }
    return m;
}

std::vector<float> MockModel::getActivation(int layer, int kind, int n) {
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 31 + kind * 7 + 1));
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (rng.next() - 0.45f) * 2.2f;
    return v;
}

std::vector<float> MockModel::getWeightSlice(int layer, std::string_view component, int offset, int n) {
    const std::uint32_t seed = static_cast<std::uint32_t>(layer * 7) ^ hashString(component) ^ static_cast<std::uint32_t>(offset);
    Mulberry32 rng(seed);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (rng.next() - 0.5f) * 1.4f;
    return v;
}

std::vector<int> MockModel::getWeightHistogram(int layer, std::string_view component, int bins) {
    const std::uint32_t seed = static_cast<std::uint32_t>(layer * 23) ^ hashString(component);
    Mulberry32 rng(seed);
    std::vector<int> h(bins, 0);
    for (int i = 0; i < 8000; ++i) {
        // crude gaussian via 4-roll average
        float g = (rng.next() + rng.next() + rng.next() + rng.next() - 2.0f) / 1.2f;
        g = std::clamp(g, -1.0f, 1.0f);
        const int idx = std::min(bins - 1, std::max(0, static_cast<int>((g + 1.0f) / 2.0f * bins)));
        ++h[idx];
    }
    return h;
}

float MockModel::getComponentNorm(int layer, std::string_view comp) {
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 31) ^ hashString(comp));
    return 0.2f + rng.next() * 0.7f;
}

float MockModel::getHeadNorm(int layer, int head) {
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 7 + head * 13 + 1));
    return 0.15f + rng.next() * 0.8f;
}

}  // namespace llob
