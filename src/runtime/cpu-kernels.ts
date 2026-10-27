// The TTS's kernels on the CPU: native/tts-cpu.c built to cpu/tts-cpu.wasm
// (WebAssembly SIMD128, no libc, imported memory). One CpuKernels owns one memory: weights are put at the bottom once,
// every run's activations above them (scratch(), released by the run). Each kernel is the CPU form of the WGSL kernel of
// the same name in tts-wgsl.ts / vits-attn.ts / nsf-wgsl.ts. DOM-free: runs in a Worker.

import { HALF } from './half.js';

interface WasmMemory { readonly buffer: ArrayBuffer; grow(pages: number): number }
interface WasmApi {
  Memory: new (d: { initial: number; maximum?: number }) => WasmMemory;
  instantiate(bytes: ArrayBuffer | Uint8Array, imports: object): Promise<{ instance: { exports: Record<string, unknown> } }>;
}
type Fn = (...a: number[]) => void;
/** The kernels, in the order of the native addon's ids (native/x86/addon.c: call(id, buffer, ...args)). */
export const KERNELS = ['lin_f32', 'lin_p16', 'lin_i8', 'im2col', 'dw_norm', 'vattn', 'frame_in', 'nsf_exc', 'nsf_stft', 'nsf_istft', 'nsf_ola'] as const;
export type KernelName = typeof KERNELS[number];
/** A native build of native/tts-cpu.c (Node addon): runs kernel `id` on the bytes of `buffer` with the wasm kernel's arguments. */
export interface NativeKernels { call(id: number, buffer: ArrayBuffer, ...args: number[]): void }

const PAGE = 65536;
/** 4 GB / 64 KB: wasm32's whole address space. */
const MAX_PAGES = 65536;

/**
 * A linear layer in the kernel's layout: w int8 or float32 [I, O4 * 4], or 'p16' (fp16 weights decoded to float32 once at
 * load and packed in panels of 16 outputs for lin_p16); scale and bias O4 * 4 floats each.
 */
export interface CpuLinear { kind: 'i8' | 'f32' | 'p16'; w: number; scale: number; bias: number; I: number; O: number; O4: number }

export class CpuKernels {
  /**
   * int8 layers converted to float32 panels at load (lin_p16, bit-identical to lin_i8): about 2-3% faster on one core,
   * but the wasm memory grows 47 -> 73 MB, so off by default (phones).
   */
  static i8Panels = false;
  private top: number;
  private readonly fn: Record<KernelName, Fn>;
  /** bytes of weights put (put*()) */
  weightBytes = 0;

  private constructor(private readonly memory: WasmMemory, exports: Record<string, unknown>) {
    const base = exports['__heap_base'] as { value: number } | undefined;
    if (!base) throw new Error('tts-cpu.wasm: __heap_base missing');
    this.top = align(base.value);
    const need = (n: string): Fn => { const f = exports[n]; if (typeof f !== 'function') throw new Error(`tts-cpu.wasm: ${n} missing`); return f as Fn; };
    this.fn = Object.fromEntries(KERNELS.map((n) => [n, need(n)])) as Record<KernelName, Fn>;
  }

  /** bytes: cpu/tts-cpu.wasm. Throws where WebAssembly SIMD is missing. */
  static async create(bytes: ArrayBuffer | Uint8Array): Promise<CpuKernels> {
    const W = (globalThis as unknown as { WebAssembly?: WasmApi }).WebAssembly;
    if (!W) throw new Error('WebAssembly がありません');
    const memory = new W.Memory({ initial: 256, maximum: MAX_PAGES });
    const { instance } = await W.instantiate(bytes, { env: { memory } });
    return new CpuKernels(memory, instance.exports);
  }

  /**
   * Node only: run every kernel in a native build of the same C source (native/x86) on this memory, with the same
   * arguments (byte offsets into the memory). The memory stays the wasm one (put / alloc / views are unchanged).
   */
  useNative(addon: NativeKernels): void {
    KERNELS.forEach((n, id) => { this.fn[n] = (...a: number[]) => addon.call(id, this.memory.buffer, ...a); });
  }

  /** bytes of the memory now */
  get bytes(): number { return this.memory.buffer.byteLength; }

  /** n bytes (16-aligned) above everything allocated so far; the memory grows when needed. */
  alloc(n: number): number {
    const p = this.top;
    this.top = align(p + n);
    const have = this.memory.buffer.byteLength;
    if (this.top > have) this.memory.grow(Math.ceil((this.top - have) / PAGE));
    return p;
  }

  /** Runs f with allocations above the current top, then releases them (f's views must not be kept). */
  scratch<T>(f: () => T): T {
    const mark = this.top;
    try { return f(); } finally { this.top = mark; }
  }

  /** A float32 view (made again after every alloc: a grown memory has a new buffer). */
  f32(p: number, n: number): Float32Array { return new Float32Array(this.memory.buffer, p, n); }

  put(data: ArrayBufferView): number {
    const p = this.alloc(data.byteLength);
    new Uint8Array(this.memory.buffer, p, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    this.weightBytes += data.byteLength;
    return p;
  }

  /** A uint32 view. */
  u32(p: number, n: number): Uint32Array { return new Uint32Array(this.memory.buffer, p, n); }

  /** float32 n floats; with data copied in. */
  floats(n: number, data?: Float32Array): number {
    const p = this.alloc(n * 4);
    if (data) this.f32(p, n).set(data.subarray(0, n));
    return p;
  }

  /**
   * A linear layer (scale: per output column, 1 for fp16 / float32; bias O). w: int8 [I, O4 * 4] as stored, fp16 bits or
   * float32 [I, O] (rows padded here to O4 * 4, as packLinear / WgslOwnTts.lin).
   */
  linear(kind: 'i8' | 'f16' | 'f32', w: Int8Array | Uint16Array | Float32Array, I: number, O: number, bias: Float32Array, scale?: Float32Array): CpuLinear {
    const O4 = Math.ceil(O / 4);
    const S = O4 * 4;
    let wp: number;
    let lk: CpuLinear['kind'];
    if (kind === 'i8' && !CpuKernels.i8Panels) {
      if (w.length !== I * S) throw new Error(`cpu linear: int8 ${w.length} values (want ${I} x ${S})`);
      wp = this.put(w);
      lk = 'i8';
    } else if (kind === 'i8') {
      // int8 converted to float32 once here (exact) and packed as the fp16 layers: lin_p16 sums the same products in
      // the same order as lin_i8 (x * float(w), then scale * acc + bias), so y is bit-identical.
      if (w.length !== I * S) throw new Error(`cpu linear: int8 ${w.length} values (want ${I} x ${S})`);
      wp = this.put(packPanels(Float32Array, I, S, S, (i, o) => (w as Int8Array)[i * S + o]!));
      lk = 'p16';
    } else if (kind === 'f16') {
      // fp16 decoded to float32 here once (HALF is exact), packed for lin_p16: the same products and sums as decoding
      // inside the kernel, without decoding every weight on every use. Panels of 16 outputs: o < O16 at
      // (o >> 4) * I * 16 + i * 16 + (o & 15); the rest (S % 16 = 4, 8 or 12) in panels of 4 after them.
      if (w.length !== I * O) throw new Error(`cpu linear: ${w.length} values (want ${I} x ${O})`);
      wp = this.put(packPanels(Float32Array, I, O, S, (i, o) => HALF[(w as Uint16Array)[i * O + o]!]!));
      lk = 'p16';
    } else {
      const pad = new Float32Array(I * S);
      if (w.length !== I * O) throw new Error(`cpu linear: ${w.length} values (want ${I} x ${O})`);
      for (let i = 0; i < I; i++) pad.set(w.subarray(i * O, (i + 1) * O) as never, i * S);
      wp = this.put(pad);
      lk = 'f32';
    }
    const sc = new Float32Array(S).fill(1);
    if (scale) sc.set(scale.subarray(0, Math.min(S, scale.length)));
    const b = new Float32Array(S);
    b.set(bias.subarray(0, Math.min(S, bias.length)));
    return { kind: lk, w: wp, scale: this.put(sc), bias: this.put(b), I, O, O4 };
  }

  /** y [T, O4 * 4] (+)= act(scale * x W + bias); x [T, I]; act 1 GELU, 2 ReLU; res 1 adds to y. */
  lin(L: CpuLinear, x: number, T: number, y: number, act = 0, res = 0, bias = L.bias): void {
    const f = L.kind === 'i8' ? this.fn.lin_i8 : L.kind === 'p16' ? this.fn.lin_p16 : this.fn.lin_f32;
    f(x, T, L.I, L.w, L.O4, L.scale, bias, y, act, res);
  }

  /** col [T, K * C]: col[t, k * C + c] = x[t + k - K / 2, c] (0 outside). */
  im2col(x: number, T: number, C: number, K: number, col: number): void { this.fn.im2col(x, T, C, K, col); }

  /** y [T, D] = LayerNorm(conv ? dwconv_K(x) : x); prm as DW_NORM's. */
  dwNorm(x: number, T: number, D: number, K: number, conv: boolean, prm: number, y: number, eps: number): void {
    this.scratch(() => this.fn.dw_norm(x, T, D, K, conv ? 1 : 0, prm, y, eps, this.alloc(K * D * 4)));
  }

  /** o [T, H * DH] = VITS relative attention of qkv [T, 3 * H * DH] (window W). */
  vattn(qkv: number, T: number, H: number, DH: number, W: number, relk: number, relv: number, o: number): void {
    this.scratch(() => this.fn.vattn(qkv, T, H, DH, W, relk, relv, o, this.alloc(T * 4)));
  }

  /** y [F, D + 4] = [h[idx[f]] | extra[f]] (FRAME_IN). */
  frameIn(idx: number, F: number, D: number, h: number, extra: number, y: number): void { this.fn.frame_in(idx, F, D, h, extra, y); }

  /** The NSF decoder's signal kernels (nsf-wgsl.ts EXC_*, STFT_FILTER, ISTFT, OLA). */
  readonly nsf: NsfKernels = {
    exc: (f0, T, noise, seed, ex) => this.fn.nsf_exc(f0, T, noise ? 1 : 0, seed, ex),
    stft: (ex, N, t0, nT, s, O, headOff, maxMag, Y, tab) => this.fn.nsf_stft(ex, N, t0, nT, s, O, headOff, maxMag, Y, tab),
    istft: (Y, t0, nT, frames, fbase, tab) => this.fn.nsf_istft(Y, t0, nT, frames, fbase, tab),
    ola: (frames, fbase, T, o0, count, y, tab) => this.fn.nsf_ola(frames, fbase, T, o0, count, y, tab),
  };
}

export interface NsfKernels {
  /** ex [T * 512, 2] = (e_h, e_n) from frame f0 [T]; noise false: e_n as already written. */
  exc(f0: number, T: number, noise: boolean, seed: number, ex: number): void;
  /** Y [nT, 1025] complex of frames t0.. (head rows from headOff); tab: cos | sin of 2 pi k / 2048. */
  stft(ex: number, N: number, t0: number, nT: number, s: number, O: number, headOff: number, maxMag: number, Y: number, tab: number): void;
  istft(Y: number, t0: number, nT: number, frames: number, fbase: number, tab: number): void;
  ola(frames: number, fbase: number, T: number, o0: number, count: number, y: number, tab: number): void;
}

function align(n: number): number { return Math.ceil(n / 16) * 16; }

/**
 * [I, S] (S = O4 * 4, outputs o >= O zero) in lin_p16's layout: panels of 16 outputs, o < O16 at
 * (o >> 4) * I * 16 + i * 16 + (o & 15); the rest (S % 16 = 4, 8 or 12) in panels of 4 after them.
 */
function packPanels<A extends Float32Array>(Arr: new (n: number) => A, I: number, O: number, S: number, at: (i: number, o: number) => number): A {
  const pad = new Arr(I * S);
  const O16 = S & ~15;
  for (let i = 0; i < I; i++) for (let o = 0; o < O; o++) {
    const p = o < O16 ? (o >> 4) * I * 16 + i * 16 + (o & 15) : O16 * I + ((o - O16) >> 2) * I * 4 + i * 4 + ((o - O16) & 3);
    pad[p] = at(i, o);
  }
  return pad;
}
