// Parity of the noiseBranch 'direct' decoder (1.2.0) against a PyTorch reference, on WebGPU in headless Chromium (muted).
// Inputs in DIR (not in the repository): <voice>.ztm with a 'direct' vocoder, and per sentence <id>.z.f32 ([T, 192]),
// <id>.f0.f32 ([T] Hz) and <id>.ref.f32 (the PyTorch decoder's output from the same z / F0, fp16 weights).
// Usage: pnpm build; pnpm serve (another terminal, from the repository root); DIR=<dir> VOICE=marine IDS=a,b
//   node scripts/nsf-direct-parity.mjs [base url]
// The CPU paths are checked by test/nsf-direct.test.mjs with the same inputs.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const base = process.argv[2] ?? 'http://127.0.0.1:8080';
const dir = process.env.DIR, voice = process.env.VOICE ?? 'marine', ids = (process.env.IDS ?? 'a,b').split(',');
if (!dir) throw new Error('DIR=<dir with the .ztm and the reference files>');
const b64 = (p) => fs.readFileSync(path.join(dir, p)).toString('base64');
const browser = await chromium.launch({ ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}), headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-gpu', '--mute-audio'] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(`${base}/demo/`);
const r = await page.evaluate(async ({ ztm, sents }) => {
  const u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const f32 = (s) => { const b = u8(s); return new Float32Array(b.buffer, 0, b.byteLength / 4); };
  const { parseZtm, vocoderFromZtm } = await import('../dist/runtime/ztm.js');
  const { nsfExportToWeights, nsfF0Norm, nsfOutputGain } = await import('../dist/runtime/nsf-weights.js');
  const { WgslNsfDecoder } = await import('../dist/runtime/nsf-wgsl.js');
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: 'no WebGPU adapter' };
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  const { meta, bin } = vocoderFromZtm(parseZtm(u8(ztm).buffer));
  const dec = await WgslNsfDecoder.create(device, nsfExportToWeights(meta, bin), 1024, 1234, nsfF0Norm(meta), nsfOutputGain(meta));
  const out = [];
  for (const s of sents) {
    const z = f32(s.z), f0 = f32(s.f0), ref = f32(s.ref);
    const T = z.length / 192;
    const { audio } = await dec.run(z, T, { f0: f0.slice(0, T) });
    let m = 0, e = 0, q = 0;
    const n = Math.min(audio.length, ref.length);
    for (let i = 0; i < n; i++) { const d = audio[i] - ref[i]; m = Math.max(m, Math.abs(d)); e += d * d; q += ref[i] * ref[i]; }
    out.push({ id: s.id, T, n, maxdiff: +m.toFixed(5), snrDb: +(10 * Math.log10(q / e)).toFixed(1) });
  }
  return { noiseBranch: meta.noiseBranch, out };
}, { ztm: b64(`${voice}.ztm`), sents: ids.map((id) => ({ id, z: b64(`${id}.z.f32`), f0: b64(`${id}.f0.f32`), ref: b64(`${id}.ref.f32`) })) });
console.log(JSON.stringify({ base, voice, ...r, errors }, null, 1));
await browser.close();
