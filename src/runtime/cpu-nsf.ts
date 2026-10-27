// The NSF decoder (nsf-wgsl.ts WgslNsfDecoder) on the CPU with the tts-cpu.wasm kernels: the same steps
// (f0z -> F0, trunk -> head, excitation, STFT filter, iSTFT, overlap-add) and the same fp16 weights (packLinear's
// rounding), only the order of float additions differs. Whole chunk in one run. DOM-free (runs in a Worker).
import { toHalf } from './half.js';
import { CpuKernels, type CpuLinear } from './cpu-kernels.js';
import { DEFAULT_NSF_F0_NORM, F0Z, NSF, TRUNK, applyOutputLevel, type Net, type NsfF0Norm, type NsfWeights } from './nsf-weights.js';

interface CpuNet { embed: CpuLinear; embedNorm: number; blocks: { dw: number; pw1: CpuLinear; pw2: CpuLinear }[] }

export interface CpuNsfOptions { f0?: Float32Array; noise?: Float32Array }

export class CpuNsfDecoder {
  private readonly f0z: CpuNet;
  private readonly f0zOut: CpuLinear;
  private readonly trunk: CpuNet;
  private readonly finalNorm: number;
  private readonly head: CpuLinear;
  /** cos (2048) | sin (2048) of 2 pi k / 2048 */
  private readonly tab: number;

  constructor(private readonly k: CpuKernels, W: NsfWeights, readonly seed = 1234, readonly f0Norm: NsfF0Norm = DEFAULT_NSF_F0_NORM, readonly outputGain = 1) {
    const need = (n: string): Float32Array => { const v = W[n]; if (!v) throw new Error(`nsf weight ${n} missing`); return v; };
    const lin = (w: string, b: string, I: number, O: number): CpuLinear => {
      const f = need(w);
      const h = new Uint16Array(f.length);
      for (let i = 0; i < f.length; i++) h[i] = toHalf(f[i]!);
      return k.linear('f16', h, I, O, need(b));
    };
    const net = (n: Net): CpuNet => ({
      embed: lin(`${n.prefix}embed.w`, `${n.prefix}embed.b`, NSF.kernel * n.C, n.D),
      embedNorm: k.put(need(`${n.prefix}embed.norm`)),
      blocks: Array.from({ length: n.layers }, (_, l) => ({
        dw: k.put(need(`${n.prefix}${l}.dw`)),
        pw1: lin(`${n.prefix}${l}.pw1.w`, `${n.prefix}${l}.pw1.b`, n.D, n.F),
        pw2: lin(`${n.prefix}${l}.pw2.w`, `${n.prefix}${l}.pw2.b`, n.F, n.D),
      })),
    });
    this.f0z = net(F0Z);
    this.f0zOut = lin('f0z.out.w', 'f0z.out.b', NSF.f0zDim, 2);
    this.trunk = net(TRUNK);
    this.finalNorm = k.put(need('final.norm'));
    this.head = lin('head.w', 'head.b', NSF.dim, 3 * NSF.bins);
    const tab = new Float32Array(2 * NSF.nFft);
    for (let i = 0; i < NSF.nFft; i++) { tab[i] = Math.cos(2 * Math.PI * i / NSF.nFft); tab[NSF.nFft + i] = Math.sin(2 * Math.PI * i / NSF.nFft); }
    this.tab = k.put(tab);
  }

  /** embed conv + LN + ConvNeXt blocks: src [T, C] -> x [T, D] (in scratch). */
  private convnext(n: Net, w: CpuNet, src: number, T: number): number {
    const k = this.k;
    const col = k.floats(T * NSF.kernel * n.C);
    const x = k.floats(T * n.D);
    const h = k.floats(T * n.D);
    const a = k.floats(T * n.F);
    k.im2col(src, T, n.C, NSF.kernel, col);
    k.lin(w.embed, col, T, h);
    k.dwNorm(h, T, n.D, NSF.kernel, false, w.embedNorm, x, NSF.lnEps);
    for (const b of w.blocks) {
      k.dwNorm(x, T, n.D, NSF.kernel, true, b.dw, h, NSF.lnEps);
      k.lin(b.pw1, h, T, a, 1);
      k.lin(b.pw2, a, T, x, 0, 1);
    }
    return x;
  }

  /** z [T, 192] -> audio [T * 512] and the frame F0 [T]. */
  run(z: Float32Array, T: number, opt: CpuNsfOptions = {}): { audio: Float32Array; f0: Float32Array; ms: Record<string, number> } {
    if (T < 3) throw new Error(`nsf decoder T ${T} (3..)`);
    const k = this.k;
    const N = T * NSF.hop;
    const ms: Record<string, number> = {};
    let at = performance.now();
    const lap = (name: string): void => { const t = performance.now(); ms[name] = +(t - at).toFixed(1); at = t; };
    return k.scratch(() => {
      const zp = k.floats(T * NSF.zDim, z);
      let f0: Float32Array;
      if (opt.f0) f0 = opt.f0.slice(0, T);
      else {
        f0 = new Float32Array(T);
        k.scratch(() => {
          const x = this.convnext(F0Z, this.f0z, zp, T);
          const o = k.floats(T * 4);
          k.lin(this.f0zOut, x, T, o);
          const ov = k.f32(o, T * 4);
          for (let t = 0; t < T; t++) f0[t] = ov[4 * t + 1]! > 0 ? Math.fround(Math.exp(ov[4 * t]! * this.f0Norm.logF0Std + this.f0Norm.logF0Mean)) : 0;
        });
      }
      lap('f0z');
      const C = NSF.zDim + 2;
      const head = k.floats(T * this.head.O4 * 4);
      k.scratch(() => {
        const tin = k.floats(T * C);
        const tv = k.f32(tin, T * C);
        const zv = k.f32(zp, T * NSF.zDim);
        for (let t = 0; t < T; t++) {
          const f = f0[t]!;
          const v = f > 0 ? 1 : 0;
          tv.set(zv.subarray(t * NSF.zDim, (t + 1) * NSF.zDim), t * C);
          tv[t * C + NSF.zDim] = (Math.log(Math.max(f, 1)) - this.f0Norm.logF0Mean) / this.f0Norm.logF0Std * v;
          tv[t * C + NSF.zDim + 1] = v;
        }
        const x = this.convnext(TRUNK, this.trunk, tin, T);
        const h = k.floats(T * NSF.dim);
        k.dwNorm(x, T, NSF.dim, NSF.kernel, false, this.finalNorm, h, NSF.lnEps);
        k.lin(this.head, h, T, head);
      });
      lap('trunk');
      const f0p = k.floats(T, f0);
      const ex = k.floats(2 * N);
      if (opt.noise) { const e = k.f32(ex, 2 * N); for (let i = 0; i < N; i++) e[2 * i + 1] = opt.noise[i]!; }
      k.nsf.exc(f0p, T, !opt.noise, this.seed, ex);
      lap('source');
      const Y = k.floats(T * NSF.bins * 2);
      k.nsf.stft(ex, N, 0, T, head, this.head.O4 * 4, 0, NSF.maxMag, Y, this.tab);
      lap('filter');
      const frames = k.floats(T * NSF.nFft);
      k.nsf.istft(Y, 0, T, frames, 0, this.tab);
      const y = k.floats(N);
      k.nsf.ola(frames, 0, T, 0, N, y, this.tab);
      const audio = k.f32(y, N).slice();
      applyOutputLevel(audio, this.outputGain);
      lap('istft');
      return { audio, f0, ms };
    });
  }
}
