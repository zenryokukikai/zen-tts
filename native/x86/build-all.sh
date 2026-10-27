#!/bin/sh
# Builds the three native x86-64 builds used by zen-tts/node into native/x86/build/:
#   zen_tts_avx512.node (-DWIDE=512), zen_tts_avx2.node (-DWIDE=256), zen_tts_sse41.node (no flag)
# zen-tts/node picks the best one the CPU runs at load (AVX-512 > AVX2 > SSE4.1), else uses cpu/tts-cpu.wasm.
# Same compiler rules as build.sh: clang 16 or later (CC), -ffp-contract=off -fno-fast-math, the Node.js headers in
# NODE_INC (default: include/node next to the node on PATH, else /usr/local/include/node).
set -e
cd "$(dirname "$0")/../.."
if [ -z "$NODE_INC" ]; then
  NODE_INC=$(dirname "$(dirname "$(command -v node 2>/dev/null || echo /usr/local/bin/node)")")/include/node
  [ -f "$NODE_INC/node_api.h" ] || NODE_INC=/usr/local/include/node
fi
export NODE_INC
mkdir -p native/x86/build
sh native/x86/build.sh native/x86/build/zen_tts_avx512.node -DWIDE=512
sh native/x86/build.sh native/x86/build/zen_tts_avx2.node -DWIDE=256
sh native/x86/build.sh native/x86/build/zen_tts_sse41.node
ls -la native/x86/build/
