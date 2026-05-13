cmake_minimum_required(VERSION 3.20)
project(llm_engine CXX)

# llm_engine — a backend-abstraction library for LLM interpretability tooling.
#
# Provides:
#   - llmengine::Model              abstract data interface (every panel of
#                                   an interpretability UI pulls through it)
#   - llmengine::MockModel          deterministic-fake-data implementation
#   - llmengine::HFProxyEngine      thin HTTP/JSON wrapper over a Python
#                                   FastAPI backend (see testing/gui/backend/)
#   - llmengine::Severity           log primitives so engines can surface
#   - llmengine::LogEntry           messages without owning a sink
#
# Independent of any UI framework — consumers wire `drainEngineLogs()` into
# whatever logging system they use.  The reference consumer is
# `testing/gui_cpp/` (Dear-ImGui-based interpretability bench).
#
# Build options:
#   LLM_ENGINE_USE_MOCK_DATA  ON   MockModel returns deterministic fake data
#                                   (great for screenshots / dev iteration);
#                                   OFF turns every MockModel method into a
#                                   no-data sentinel return so a release
#                                   build can't silently ship mock numbers.

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

option(LLM_ENGINE_USE_MOCK_DATA
       "Have MockModel return deterministic mock data (vs. no-data sentinels)"
       ON)

# ---------------------------------------------------------------------------
# Per-backend build options.  Defaults compile everything that's
# self-contained (no external library link); heavier backends (llama.cpp,
# libtorch, native runtime) are off by default — turn on per-host.
# Each enabled backend defines LLM_ENGINE_HAVE_<NAME>=1 (PUBLIC) so consumer
# factories can #ifdef-gate their constructor calls.  See docs/BACKENDS.md §4.
# ---------------------------------------------------------------------------
option(LLM_ENGINE_BUILD_HF_PROXY
       "Build HFProxyEngine (FastAPI proxy via cpp-httplib)"
       ON)
option(LLM_ENGINE_BUILD_GGUF_INSPECTOR
       "Build GGUFInspectorEngine (mmap'd weight reader, no inference)"
       ON)
option(LLM_ENGINE_BUILD_LLAMA_CPP
       "Build LlamaCppEngine (embedded llama.cpp w/ cb_eval activation capture)"
       OFF)
option(LLM_ENGINE_BUILD_LIBTORCH
       "Build LibtorchEngine (in-process torch::; needs libtorch installed)"
       OFF)
option(LLM_ENGINE_BUILD_NATIVE_RUNTIME
       "Build NativeRuntimeEngine (long-term pure-C++ runtime — placeholder)"
       OFF)

# ---------------------------------------------------------------------------
# Vendored header-only deps — private to the engine's implementation.
# Both are MIT-licensed; see libs/<name>/LICENSE*.
# Marked SYSTEM so their warnings don't leak into consumers.
# ---------------------------------------------------------------------------
add_library(llm_engine_cpp_httplib INTERFACE)
target_include_directories(llm_engine_cpp_httplib SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/cpp-httplib)

add_library(llm_engine_nlohmann_json INTERFACE)
target_include_directories(llm_engine_nlohmann_json SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann_json)

# ---------------------------------------------------------------------------
# llm_engine — the library itself.  Always-on sources first; optional
# backends layered in below per-flag.
# ---------------------------------------------------------------------------
add_library(llm_engine STATIC
  src/log.cpp
  src/mock_model.cpp
)
add_library(llmengine::llm_engine ALIAS llm_engine)

target_include_directories(llm_engine
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(LLM_ENGINE_USE_MOCK_DATA)
  target_compile_definitions(llm_engine PRIVATE LLOB_USE_MOCK_DATA=1)
endif()

# ── HFProxyEngine (FastAPI proxy via cpp-httplib + nlohmann/json) ─────────
if(LLM_ENGINE_BUILD_HF_PROXY)
  target_sources(llm_engine PRIVATE src/hf_proxy_engine.cpp)
  target_link_libraries(llm_engine
    PRIVATE llm_engine_cpp_httplib llm_engine_nlohmann_json)
  target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_HAVE_HF_PROXY=1)
endif()

# ── GGUFInspectorEngine (file-format reader, no external deps yet) ────────
if(LLM_ENGINE_BUILD_GGUF_INSPECTOR)
  target_sources(llm_engine PRIVATE src/gguf_inspector_engine.cpp)
  target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_HAVE_GGUF_INSPECTOR=1)
endif()

# ── LlamaCppEngine (embedded llama.cpp; needs lib/llama.cpp built) ────────
if(LLM_ENGINE_BUILD_LLAMA_CPP)
  target_sources(llm_engine PRIVATE src/llama_cpp_engine.cpp)
  # [TODO] When real bodies land, link against lib/llama.cpp:
  #   target_include_directories(llm_engine PRIVATE
  #     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/llama.cpp/include
  #     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/llama.cpp/ggml/include)
  #   target_link_libraries(llm_engine PRIVATE
  #     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/llama.cpp/build/bin/libllama.so
  #     ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/llama.cpp/build/bin/libggml.so)
  target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_HAVE_LLAMA_CPP=1)
endif()

# ── LibtorchEngine (in-process libtorch; heaviest dep) ────────────────────
if(LLM_ENGINE_BUILD_LIBTORCH)
  target_sources(llm_engine PRIVATE src/libtorch_engine.cpp)
  # [TODO] When real bodies land:
  #   find_package(Torch REQUIRED)
  #   target_link_libraries(llm_engine PRIVATE ${TORCH_LIBRARIES})
  target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_HAVE_LIBTORCH=1)
endif()

# ── NativeRuntimeEngine (placeholder for the long-term pure-C++ runtime) ──
if(LLM_ENGINE_BUILD_NATIVE_RUNTIME)
  target_sources(llm_engine PRIVATE src/native_runtime_engine.cpp)
  target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_HAVE_NATIVE_RUNTIME=1)
endif()

# Threads are needed by the mutex/cv work in HFProxyEngine.
find_package(Threads REQUIRED)
target_link_libraries(llm_engine PUBLIC Threads::Threads)

# Strict warnings on our own translation units — vendored deps stay quiet
# thanks to the SYSTEM keyword on their interface includes.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(llm_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()
