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

// GGML tensor element types — separate enum from GGUFType, lives in
// ggml.h.  Stable since llama.cpp ~2024; new types occasionally added at
// the high end (IQ-* matrix-quant variants).  We hardcode the common
// table here and report unknowns as "ggml_t<N>" + size_bytes=kNoSize so
// the UI can still list them.
struct GGMLTypeInfo {
    const char*  name;          // short label rendered in raw_tensors workspace
    std::int64_t block_size;    // elements per quantisation block
    std::int64_t bytes_per_block;
};
constexpr GGMLTypeInfo kGGMLTypes[] = {
    /* 0  F32     */ { "f32",     1,   4   },
    /* 1  F16     */ { "f16",     1,   2   },
    /* 2  Q4_0    */ { "q4_0",    32,  18  },
    /* 3  Q4_1    */ { "q4_1",    32,  20  },
    /* 4  removed */ { "?_4",     0,   0   },
    /* 5  removed */ { "?_5",     0,   0   },
    /* 6  Q5_0    */ { "q5_0",    32,  22  },
    /* 7  Q5_1    */ { "q5_1",    32,  24  },
    /* 8  Q8_0    */ { "q8_0",    32,  34  },
    /* 9  Q8_1    */ { "q8_1",    32,  36  },
    /* 10 Q2_K    */ { "q2_k",    256, 84  },
    /* 11 Q3_K    */ { "q3_k",    256, 110 },
    /* 12 Q4_K    */ { "q4_k",    256, 144 },
    /* 13 Q5_K    */ { "q5_k",    256, 176 },
    /* 14 Q6_K    */ { "q6_k",    256, 210 },
    /* 15 Q8_K    */ { "q8_k",    256, 292 },
    /* 16 IQ2_XXS */ { "iq2_xxs", 256, 66  },
    /* 17 IQ2_XS  */ { "iq2_xs",  256, 74  },
    /* 18 IQ3_XXS */ { "iq3_xxs", 256, 98  },
    /* 19 IQ1_S   */ { "iq1_s",   256, 50  },
    /* 20 IQ4_NL  */ { "iq4_nl",  32,  18  },
    /* 21 IQ3_S   */ { "iq3_s",   256, 110 },
    /* 22 IQ2_S   */ { "iq2_s",   256, 82  },
    /* 23 IQ4_XS  */ { "iq4_xs",  256, 136 },
    /* 24 I8      */ { "i8",      1,   1   },
    /* 25 I16     */ { "i16",     1,   2   },
    /* 26 I32     */ { "i32",     1,   4   },
    /* 27 I64     */ { "i64",     1,   8   },
    /* 28 F64     */ { "f64",     1,   8   },
    /* 29 IQ1_M   */ { "iq1_m",   256, 56  },
    /* 30 BF16    */ { "bf16",    1,   2   },
};
constexpr std::size_t kGGMLTypeCount = sizeof(kGGMLTypes) / sizeof(kGGMLTypes[0]);

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

// Phase-B parsed tensor entry.  Backs both Model::getStateDict (full list)
// and Model::getTensorMeta (single-name lookup).  Stride is C-contiguous
// row-major — all GGUF tensors are dense + contiguous.
struct GGUFTensorEntry {
    std::string      name;          // canonical: "blocks.<L>.<area>.weight"
    std::uint32_t    ggml_type = 0;
    std::vector<int> shape;         // outermost dim first (matches HF convention)
    std::vector<int> stride;        // C-contiguous, derived from shape
    std::int64_t     offset = 0;    // bytes from tensor_data_start
    std::int64_t     size_bytes = 0;
};

// Total element count (product of dims) — used for size_bytes calc + UI.
std::int64_t elementCount(const std::vector<std::uint64_t>& dims) {
    std::int64_t n = 1;
    for (auto d : dims) n *= static_cast<std::int64_t>(d);
    return n;
}

// Compute the on-disk byte size of a tensor given its element count + GGML
// type.  Returns kNoSize for unknown types so the UI renders "—" rather
// than a wrong number.
std::int64_t tensorBytes(std::uint32_t type, std::int64_t n_elements) {
    if (type >= kGGMLTypeCount) return kNoSize;
    const auto& info = kGGMLTypes[type];
    if (info.block_size == 0) return kNoSize;
    return (n_elements / info.block_size) * info.bytes_per_block;
}

const char* ggmlTypeName(std::uint32_t type) {
    if (type < kGGMLTypeCount && kGGMLTypes[type].block_size != 0) {
        return kGGMLTypes[type].name;
    }
    return "ggml_unknown";
}

// Round `pos` up to the next multiple of `align`.  Used to skip the pad
// region between the tensor_info[] array and the tensor_data section.
std::size_t alignUp(std::size_t pos, std::size_t align) {
    if (align == 0) return pos;
    return ((pos + align - 1) / align) * align;
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
    std::uint64_t         tensor_count = 0; // from header
    std::vector<GGUFTensorEntry>           tensors;         // Phase B index
    std::unordered_map<std::string, std::size_t> by_name;   // name → tensors[] idx
    std::int64_t          tensor_data_start = 0;            // file offset of tensor_data[]

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

        // Phase B — tensor_info[] follows immediately after the metadata
        // kv-store.  Each entry is name-prefixed + dims + ggml_type + offset.
        std::vector<GGUFTensorEntry> tensors;
        tensors.reserve(n_tensors);
        std::unordered_map<std::string, std::size_t> by_name;
        by_name.reserve(n_tensors);
        for (std::uint64_t i = 0; i < n_tensors; ++i) {
            GGUFTensorEntry e;
            e.name = cur.readString();
            const auto n_dims = cur.read<std::uint32_t>();
            std::vector<std::uint64_t> dims_u64(n_dims);
            for (std::uint32_t d = 0; d < n_dims; ++d) {
                dims_u64[d] = cur.read<std::uint64_t>();
            }
            e.ggml_type = cur.read<std::uint32_t>();
            e.offset    = static_cast<std::int64_t>(cur.read<std::uint64_t>());
            e.size_bytes = tensorBytes(e.ggml_type, elementCount(dims_u64));
            // GGUF stores dims innermost-first (column-major-ish).  Reverse
            // so the UI sees row-major shape (matches HF / numpy / pytorch).
            e.shape.reserve(n_dims);
            for (auto it = dims_u64.rbegin(); it != dims_u64.rend(); ++it) {
                e.shape.push_back(static_cast<int>(*it));
            }
            // C-contiguous strides over the row-major shape.
            e.stride.assign(n_dims, 0);
            std::int64_t s = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(n_dims) - 1; d >= 0; --d) {
                e.stride[static_cast<std::size_t>(d)] = static_cast<int>(s);
                s *= e.shape[static_cast<std::size_t>(d)];
            }
            by_name.emplace(e.name, tensors.size());
            tensors.push_back(std::move(e));
        }

        // Tensor data section starts at the next aligned offset.  GGUF lets
        // models override the alignment via the general.alignment metadata
        // key; default is 32.  Anything past tensor_data_start is raw
        // weights ready to mmap-read in Phase C.
        const std::int64_t alignment = lookupI64(meta, "general.alignment", 32);
        const std::int64_t tensor_data_start =
            static_cast<std::int64_t>(alignUp(cur.pos,
                                              static_cast<std::size_t>(alignment)));

        ModelInfo info = makeModelInfo(meta, p);
        {
            std::lock_guard<std::mutex> lk(m_impl->mu);
            m_impl->mapped            = std::move(m);
            m_impl->meta              = std::move(meta);
            m_impl->info              = std::move(info);
            m_impl->tensor_count      = n_tensors;
            m_impl->tensors           = std::move(tensors);
            m_impl->by_name           = std::move(by_name);
            m_impl->tensor_data_start = tensor_data_start;
        }
        m_impl->log(Severity::Info, "gguf",
                    "loaded " + p + " (v" + std::to_string(version) +
                    ", " + std::to_string(n_tensors) + " tensors, " +
                    std::to_string(n_kv) + " metadata kv)");
        m_impl->log(Severity::Info, "gguf",
                    "topology: " + std::to_string(m_impl->info.nLayers) +
                    " layers · " + std::to_string(m_impl->info.nHeads) +
                    " heads · d_model=" + std::to_string(m_impl->info.dModel));
        // First few tensor names — sanity readout for Phase B; raw_tensors
        // workspace will show all of them via getStateDict().
        if (!m_impl->tensors.empty()) {
            std::string sample = "tensors: ";
            const std::size_t n = std::min<std::size_t>(3, m_impl->tensors.size());
            for (std::size_t i = 0; i < n; ++i) {
                if (i) sample += ", ";
                const auto& e = m_impl->tensors[i];
                sample += e.name + " [";
                for (std::size_t d = 0; d < e.shape.size(); ++d) {
                    if (d) sample += "x";
                    sample += std::to_string(e.shape[d]);
                }
                sample += "] ";
                sample += ggmlTypeName(e.ggml_type);
            }
            if (m_impl->tensors.size() > n) {
                sample += " (+" + std::to_string(m_impl->tensors.size() - n) +
                          " more)";
            }
            m_impl->log(Severity::Info, "gguf", sample);
        }
        return {true, ""};
    } catch (const std::exception& e) {
        m_impl->log(Severity::Error, "gguf", std::string("parse failed: ") + e.what());
        return {false, e.what()};
    }
}

void GGUFInspectorEngine::unloadCheckpoint() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    m_impl->mapped            = Mmap{};
    m_impl->meta.clear();
    m_impl->info              = {};
    m_impl->tensor_count      = 0;
    m_impl->tensors.clear();
    m_impl->by_name.clear();
    m_impl->tensor_data_start = 0;
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

// Phase B — wired.  Walk the parsed tensor index built at loadCheckpoint
// and translate each entry into engine-side TensorMeta.
std::vector<TensorMeta> GGUFInspectorEngine::getStateDict() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<TensorMeta> out;
    out.reserve(m_impl->tensors.size());
    for (const auto& e : m_impl->tensors) {
        TensorMeta tm;
        tm.name        = e.name;
        tm.dtype       = ggmlTypeName(e.ggml_type);
        tm.shape       = e.shape;
        tm.stride      = e.stride;
        tm.contiguous  = true;
        tm.device      = "cpu";
        tm.size_bytes  = e.size_bytes;
        out.push_back(std::move(tm));
    }
    return out;
}

TensorMeta GGUFInspectorEngine::getTensorMeta(std::string_view name) {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    auto it = m_impl->by_name.find(std::string(name));
    if (it == m_impl->by_name.end()) return {};
    const auto& e = m_impl->tensors[it->second];
    TensorMeta tm;
    tm.name        = e.name;
    tm.dtype       = ggmlTypeName(e.ggml_type);
    tm.shape       = e.shape;
    tm.stride      = e.stride;
    tm.contiguous  = true;
    tm.device      = "cpu";
    tm.size_bytes  = e.size_bytes;
    return tm;
}

// Phase C targets — dispatch on GGML type.  F32/F16/BF16 are direct reads;
// quantised types need per-block dequantisation kernels.  Stubs for now.
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
