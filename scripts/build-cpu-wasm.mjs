// Builds native/tts-cpu.c -> cpu/tts-cpu.wasm (wasm32 + SIMD128, freestanding: no libc, imported memory, no start
// function). Needs clang with the wasm32 target and wasm-ld: set WASM_LLVM_BIN to their directory (for example the LLVM
// that comes with emscripten, or a stock LLVM build). The .wasm is committed, so `pnpm build` does not need clang.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const bin = process.env.WASM_LLVM_BIN;
const clang = bin ? path.join(bin, 'clang') : 'clang';
const src = path.join(root, 'native/tts-cpu.c');
const out = path.join(root, 'cpu/tts-cpu.wasm');
fs.mkdirSync(path.dirname(out), { recursive: true });
execFileSync(clang, [
  '--target=wasm32', '-O3', '-msimd128', '-mbulk-memory', '-ffreestanding', '-nostdlib', '-fno-builtin-memset',
  '-Wall', '-Werror',
  '-Wl,--no-entry', '-Wl,--import-memory', '-Wl,--export=__heap_base', '-Wl,--strip-all',
  '-o', out, src,
], { stdio: 'inherit' });
console.log(`tts-cpu.wasm: ${fs.statSync(out).size} bytes -> ${path.relative(root, out)}`);
