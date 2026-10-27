#!/bin/sh
# native/x86/build.sh OUT [-DWIDE=256|512 [-DRW=n]]: node addon of native/tts-cpu.c for x86-64 (SSE4.1; no FMA contraction,
# no fast-math). Needs clang 16 or later (clang-16 here; set CC to override) and the Node.js headers in NODE_INC
# (default /usr/local/include/node).
set -e
OUT=$1; shift
LP=lin_p16; LI=lin_i8
for a in "$@"; do case $a in -DWIDE=512) LP=lin_p16w; LI=lin_i8w;; -DWIDE=*) LP=lin_p16w;; esac; done
${CC:-clang-16} -O3 -shared -fPIC -msse4.1 -ffp-contract=off -fno-fast-math -Wall -Wno-unused-function \
  -I${NODE_INC:-/usr/local/include/node} -Inative/x86 -DLIN_P16=$LP -DLIN_I8=$LI "$@" -o "$OUT" native/x86/addon.c
