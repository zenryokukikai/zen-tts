#!/bin/sh
# native/arm64/build.sh [OUT [-DRW=n]]: node addon of native/tts-cpu.c for arm64 (AArch64 NEON; macOS or Linux), the same addon
# source as the x86-64 builds (native/x86/addon.c) with the wasm SIMD128 intrinsics mapped to NEON (wasm_simd128.h here).
# No FMA contraction, no fast-math, so the audio is bit-identical to the wasm build. zen-tts/node loads
# native/arm64/build/zen_tts_neon.node on arm64, else uses cpu/tts-cpu.wasm.
# Needs clang (on macOS: the Xcode command line tools; set CC to override) and the Node.js headers in NODE_INC
# (default: include/node next to the node on PATH, else /usr/local/include/node).
set -e
cd "$(dirname "$0")/../.."
OUT=${1:-native/arm64/build/zen_tts_neon.node}; [ $# -gt 0 ] && shift
if [ -z "$NODE_INC" ]; then
  NODE_INC=$(dirname "$(dirname "$(command -v node 2>/dev/null || echo /usr/local/bin/node)")")/include/node
  [ -f "$NODE_INC/node_api.h" ] || NODE_INC=/usr/local/include/node
fi
# macOS: the N-API symbols are resolved from the node binary when the addon is loaded
case "$(uname -s)" in Darwin) LINK="-undefined dynamic_lookup";; *) LINK="";; esac
mkdir -p "$(dirname "$OUT")"
${CC:-clang} -O3 -shared -fPIC $LINK -ffp-contract=off -fno-fast-math -Wall -Wno-unused-function \
  -I"$NODE_INC" -Inative/arm64 -DLIN_P16=lin_p16n -DLIN_I8=lin_i8n "$@" -o "$OUT" native/x86/addon.c
ls -la "$OUT"
