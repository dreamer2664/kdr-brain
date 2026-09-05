#!/bin/sh
# Build llama.cpp (the composer's inference library) as static libs, once.
#   sh scripts/build_llama.sh            -> ~/.cache/kdr/llama/build-native   (native CPU flags, for this machine)
#   PORTABLE=1 sh scripts/build_llama.sh -> same dir, but x86-64-v3 (AVX2/FMA) so the binary runs on any modern x86 CPU
# Pinned to a release tag so builds are reproducible.
set -eu
TAG="${LLAMA_TAG:-v0.4.0}"
DIR="${LLAMA:-$HOME/.cache/kdr/llama}"
if [ ! -f "$DIR/CMakeLists.txt" ]; then
  mkdir -p "$DIR"
  echo "downloading llama.cpp $TAG ..."
  curl -sL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz" | tar xz -C "$DIR" --strip-components=1
fi
command -v cmake >/dev/null 2>&1 || python3 -m pip install -q cmake
if [ "${PORTABLE:-0}" = "1" ]; then
  ARCH="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"
else
  ARCH="-DGGML_NATIVE=ON"
fi
cd "$DIR"
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF $ARCH -DGGML_OPENMP=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_CURL=OFF -DGGML_BLAS=OFF >/dev/null
cmake --build build-native --config Release -j"$(nproc)" --target llama 2>&1 | grep -E "error|Built target llama$" || true
ls -la build-native/src/libllama.a build-native/ggml/src/libggml.a build-native/ggml/src/libggml-cpu.a build-native/ggml/src/libggml-base.a
