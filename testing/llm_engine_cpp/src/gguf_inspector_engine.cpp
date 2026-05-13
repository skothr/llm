#include "llm_engine/gguf_inspector_engine.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llmengine {

namespace {

std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

// ── GGUF format constants ─────────────────────────────────────────────────
// Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md (v3).
// Layout: magic(4) + version(u32) + tensor_count(u64) + meta_kv_count(u64)
//         + meta_kv[]   (variable-length key/value pairs)
//         + tensor_info[] (per-tensor: name, dims, type, offset)
//         + alignment padding (default 32, override via general.alignment)
//         + tensor_data[]
constexpr std::uint32_t kGGUFMagic   = 0x46554747u;  // 'GGUF' little-endian
constexpr std::uint32_t kGGUFVerMin  = 2;            // we accept v2+ (v3 current)
constexpr std::uint32_t kGGUFVerMax  = 3;

enum class GGUFType : std::uint32_t {
    UINT8   = 0,  INT8   = 1,
    UINT16  = 2,  INT16  = 3,
    UINT32  = 4,  INT32  = 5,
    FLOAT32 = 6,  BOOL   = 7,
    STRING  = 8,  ARRAY  = 9,
    UINT64  = 10, INT64  = 11,
    FLOAT64 = 12,
};

// Tagged value — Phase A stores scalars + strings; arrays are parsed (so
// the cursor advances correctly) but their elements aren't materialised.
// Adding array storage is a one-line change in the parser when we want it.
struct GGUFValue {
    enum class Kind { I64, F64, Str, ArraySkipped } kind = Kind::I64;
    std::int64_t i64 = 0;
    double       f64 = 0.0;
    std::string  str;
};

using GGUFMeta = std::unordered_map<std::string, GGUFValue>;

// ── Cursor-style reader over an mmap'd region ─────────────────────────────
// Throws std::runtime_error on EOF — caught at the loadCheckpoint boundary
// and translated to CheckpointResult{ok=false}.  The Phase A parser does
// no recovery: a malformed GGUF means we refuse the load entirely.
struct Cursor {
    const std::uint8_t* base;
    std::size_t         len;
    std::size_t         pos = 0;

    void need(std::size_t n) const {
        if (pos + n > len) {
            throw std::runtime_error(
                "gguf: unexpected EOF (need " + std::to_string(n) +
                " bytes at offset " + std::to_string(pos) + ")");
        }
    }
    template <typename T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::string readString() {
        const auto n = read<std::uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(base + pos), n);
        pos += n;
        return s;
    }
    void skip(std::size_t n) { need(n); pos += n; }
};

// Size in bytes of one element of a scalar GGUFType.  Returns 0 for ARRAY
// and STRING — those are handled by the dispatch in readValue() below.
std::size_t scalarSize(GGUFType t) {
    switch (t) {
        case GGUFType::UINT8:   case GGUFType::INT8:   case GGUFType::BOOL:    return 1;
        case GGUFType::UINT16:  case GGUFType::INT16:                          return 2;
        case GGUFType::UINT32:  case GGUFType::INT32:  case GGUFType::FLOAT32: return 4;
        case GGUFType::UINT64:  case GGUFType::INT64:  case GGUFType::FLOAT64: return 8;
        case GGUFType::STRING:  case GGUFType::ARRAY:                          return 0;
    }
    return 0;
}

// Parse one value into a GGUFValue.  For arrays the cursor is advanced past
// every element (so the next kv-pair lines up) but elements themselves are
// dropped on the floor for Phase A; see GGUFValue::ArraySkipped.
GGUFValue readValue(Cursor& c, GGUFType type) {
    GGUFValue v;
    switch (type) {
        case GGUFType::UINT8:   v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::uint8_t>();  return v;
        case GGUFType::INT8:    v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::int8_t>();   return v;
        case GGUFType::UINT16:  v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::uint16_t>(); return v;
        case GGUFType::INT16:   v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::int16_t>();  return v;
        case GGUFType::UINT32:  v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::uint32_t>(); return v;
        case GGUFType::INT32:   v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::int32_t>();  return v;
        case GGUFType::UINT64:  v.kind = GGUFValue::Kind::I64; v.i64 = static_cast<std::int64_t>(c.read<std::uint64_t>()); return v;
        case GGUFType::INT64:   v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::int64_t>();  return v;
        case GGUFType::BOOL:    v.kind = GGUFValue::Kind::I64; v.i64 = c.read<std::uint8_t>() ? 1 : 0; return v;
        case GGUFType::FLOAT32: v.kind = GGUFValue::Kind::F64; v.f64 = c.read<float>();         return v;
        case GGUFType::FLOAT64: v.kind = GGUFValue::Kind::F64; v.f64 = c.read<double>();        return v;
        case GGUFType::STRING:  v.kind = GGUFValue::Kind::Str; v.str = c.readString();          return v;
        case GGUFType::ARRAY: {
            const auto elem_type = static_cast<GGUFType>(c.read<std::uint32_t>());
            const auto count     = c.read<std::uint64_t>();
            if (elem_type == GGUFType::STRING) {
                for (std::uint64_t i = 0; i < count; ++i) (void)c.readString();
            } else if (elem_type == GGUFType::ARRAY) {
                // GGUF spec disallows nested arrays but be defensive — recurse.
                for (std::uint64_t i = 0; i < count; ++i) (void)readValue(c, elem_type);
            } else {
                const auto sz = scalarSize(elem_type);
                if (sz == 0) {
                    throw std::runtime_error("gguf: array of unknown scalar type");
                }
                c.skip(sz * count);
            }
            v.kind = GGUFValue::Kind::ArraySkipped;
            return v;
        }
    }
    throw std::runtime_error("gguf: unknown value type tag " +
                             std::to_string(static_cast<std::uint32_t>(type)));
}

// Convenience accessors that fall back to a sentinel when a key is absent
// or holds the wrong kind.  Avoids if-chains at every getModelInfo lookup.
std::int64_t lookupI64(const GGUFMeta& m, const std::string& key,
                       std::int64_t fallback = kNoSize) {
    auto it = m.find(key);
    if (it == m.end() || it->second.kind != GGUFValue::Kind::I64) return fallback;
    return it->second.i64;
}
double lookupF64(const GGUFMeta& m, const std::string& key,
                 double fallback) {
    auto it = m.find(key);
    if (it == m.end() || it->second.kind != GGUFValue::Kind::F64) return fallback;
    return it->second.f64;
}
std::string lookupStr(const GGUFMeta& m, const std::string& key) {
    auto it = m.find(key);
    if (it == m.end() || it->second.kind != GGUFValue::Kind::Str) return {};
    return it->second.str;
}

// Translate parsed GGUFMeta + display path into engine-side ModelInfo.
// The architecture-prefixed keys (e.g. "llama.block_count") need the value
// of "general.architecture" to resolve, so we look that up first.
ModelInfo makeModelInfo(const GGUFMeta& meta, const std::string& display) {
    ModelInfo m;
    const std::string arch = lookupStr(meta, "general.architecture");
    const std::string name = lookupStr(meta, "general.name");
    m.name = !name.empty() ? name : display;

    auto archKey = [&arch](const char* suffix) {
        return arch.empty() ? std::string{} : arch + "." + suffix;
    };

    m.nLayers = static_cast<int>(lookupI64(meta, archKey("block_count"), kNoInt));
    m.nHeads  = static_cast<int>(lookupI64(meta, archKey("attention.head_count"), kNoInt));
    m.nKvHeads = static_cast<int>(
        lookupI64(meta, archKey("attention.head_count_kv"), m.nHeads));
    m.dModel  = static_cast<int>(lookupI64(meta, archKey("embedding_length"), kNoInt));
    m.dMlp    = static_cast<int>(lookupI64(meta, archKey("feed_forward_length"), kNoInt));
    m.maxPos  = static_cast<int>(lookupI64(meta, archKey("context_length"), kNoInt));

    if (m.nHeads > 0 && m.dModel > 0) m.dHead = m.dModel / m.nHeads;

    // rope.freq_base is the GGUF spelling of what HF calls rope_theta.
    const double rope = lookupF64(meta, archKey("rope.freq_base"),
                                  static_cast<double>(kNoFloat));
    m.ropeTheta = static_cast<float>(rope);

    // Decoding bos/eos token strings requires the tokens-array (skipped
    // in Phase A).  Leave them empty until Phase A.5 when we keep that array.
    return m;
}

// RAII mmap holder so loadCheckpoint failure paths don't leak file handles.
struct Mmap {
    int          fd   = -1;
    void*        addr = nullptr;
    std::size_t  len  = 0;

    Mmap() = default;
    Mmap(const Mmap&) = delete;
    Mmap& operator=(const Mmap&) = delete;
    Mmap(Mmap&& o) noexcept { *this = std::move(o); }
    Mmap& operator=(Mmap&& o) noexcept {
        if (this != &o) {
            close();
            fd = o.fd;     o.fd = -1;
            addr = o.addr; o.addr = nullptr;
            len = o.len;   o.len = 0;
        }
        return *this;
    }
    ~Mmap() { close(); }

    void close() {
        if (addr && addr != MAP_FAILED) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
        addr = nullptr; len = 0; fd = -1;
    }

    // Returns empty optional via len==0 + a populated `err` on failure;
    // caller checks len before use.  Avoids exposing exceptions to the
    // RAII boundary.
    bool open(const std::string& path, std::string& err) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "open(" + path + ") failed: " + std::strerror(errno);
            return false;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            err = "fstat failed: " + std::string(std::strerror(errno));
            return false;
        }
        len = static_cast<std::size_t>(st.st_size);
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            err = "mmap failed: " + std::string(std::strerror(errno));
            addr = nullptr;
            return false;
        }
        return true;
    }
};

}  // namespace

struct GGUFInspectorEngine::Impl {
    std::string           default_path;     // from constructor (env LLOB_BACKEND_PATH)
    std::mutex            mu;
    std::vector<LogEntry> pending_logs;

    Mmap                  mapped;           // empty until loadCheckpoint succeeds
    GGUFMeta              meta;             // parsed metadata kv-store
    ModelInfo             info;             // translated topology cache
    std::uint64_t         tensor_count = 0; // from header — Phase B fills tensor index

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
    }
};

GGUFInspectorEngine::GGUFInspectorEngine(std::string gguf_path)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->default_path = std::move(gguf_path);
    m_impl->log(Severity::Info, "gguf",
                "GGUFInspectorEngine ready (default_path=" +
                m_impl->default_path + ")");
}

GGUFInspectorEngine::~GGUFInspectorEngine() = default;

Model::CheckpointResult GGUFInspectorEngine::loadCheckpoint(std::string_view path) {
    const std::string p(path.empty() ? m_impl->default_path : std::string(path));
    if (p.empty()) {
        return {false, "GGUFInspectorEngine: no path provided"};
    }

    Mmap m;
    std::string err;
    if (!m.open(p, err)) {
        m_impl->log(Severity::Error, "gguf", err);
        return {false, err};
    }

    // Header + metadata parse — exceptions caught here so the caller gets
    // a clean CheckpointResult{ok=false} rather than an unwound stack.
    try {
        Cursor cur{static_cast<const std::uint8_t*>(m.addr), m.len};
        const auto magic   = cur.read<std::uint32_t>();
        if (magic != kGGUFMagic) {
            std::string e = "gguf: bad magic 0x" + std::to_string(magic);
            m_impl->log(Severity::Error, "gguf", e);
            return {false, e};
        }
        const auto version = cur.read<std::uint32_t>();
        if (version < kGGUFVerMin || version > kGGUFVerMax) {
            std::string e = "gguf: unsupported version " +
                            std::to_string(version) + " (we accept " +
                            std::to_string(kGGUFVerMin) + "..." +
                            std::to_string(kGGUFVerMax) + ")";
            m_impl->log(Severity::Error, "gguf", e);
            return {false, e};
        }
        const auto n_tensors = cur.read<std::uint64_t>();
        const auto n_kv      = cur.read<std::uint64_t>();

        GGUFMeta meta;
        meta.reserve(n_kv);
        for (std::uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = cur.readString();
            const auto type       = static_cast<GGUFType>(cur.read<std::uint32_t>());
            meta.emplace(key, readValue(cur, type));
        }

        ModelInfo info = makeModelInfo(meta, p);
        {
            std::lock_guard<std::mutex> lk(m_impl->mu);
            m_impl->mapped       = std::move(m);
            m_impl->meta         = std::move(meta);
            m_impl->info         = std::move(info);
            m_impl->tensor_count = n_tensors;
        }
        m_impl->log(Severity::Info, "gguf",
                    "loaded " + p + " (v" + std::to_string(version) +
                    ", " + std::to_string(n_tensors) + " tensors, " +
                    std::to_string(n_kv) + " metadata kv)");
        m_impl->log(Severity::Info, "gguf",
                    "topology: " + std::to_string(m_impl->info.nLayers) +
                    " layers · " + std::to_string(m_impl->info.nHeads) +
                    " heads · d_model=" + std::to_string(m_impl->info.dModel));
        return {true, ""};
    } catch (const std::exception& e) {
        m_impl->log(Severity::Error, "gguf", std::string("parse failed: ") + e.what());
        return {false, e.what()};
    }
}

void GGUFInspectorEngine::unloadCheckpoint() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    m_impl->mapped       = Mmap{};
    m_impl->meta.clear();
    m_impl->info         = {};
    m_impl->tensor_count = 0;
}

std::vector<LogEntry> GGUFInspectorEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<LogEntry> out;
    out.swap(m_impl->pending_logs);
    return out;
}

ModelInfo GGUFInspectorEngine::getModelInfo() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    return m_impl->info;
}

// Phase B targets — still stubs.  Will grow to walk the tensor_info[] array
// (which immediately follows the metadata kv-store) and serve weight slices
// from m_impl->mapped.addr + tensor_data_start + entry.offset.
std::vector<TensorMeta> GGUFInspectorEngine::getStateDict() {
    return {};
}

TensorMeta GGUFInspectorEngine::getTensorMeta(std::string_view name) {
    (void)name;
    return {};
}

std::vector<float> GGUFInspectorEngine::getWeightSlice(
    std::string_view name, int offset, int n) {
    (void)name; (void)offset; (void)n;
    return {};
}

std::vector<int> GGUFInspectorEngine::getWeightHistogram(
    std::string_view name, int bins) {
    (void)name; (void)bins;
    return {};
}

}  // namespace llmengine
