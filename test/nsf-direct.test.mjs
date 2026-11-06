// noiseBranch 'direct' (1.2.0 decoders; since 1.2.1 the noise phase is noise_phase in native/tts-cpu.c, and the reference is made with that phase): nsf.json -> weights (5 * 1025 head rows), the flag's checks, and a voice without
// the flag keeps the 3 * 1025 head. With NSF_DIRECT_DIR=<dir> (see scripts/nsf-direct-parity.mjs for its files: a
// 'direct' .ztm and PyTorch reference renders) the CPU decoder (wasm, and the native builds that are built) is compared
// with the reference; with NSF_OLD_ZTM=<a 1.1.0 .ztm> and NSF_OLD_PCM=<its sha list from the 1.1.0 runtime> the old
// voice's samples are checked to be unchanged.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { NSF, nsfDirectNoise, nsfExportToWeights, nsfF0Norm, nsfHeadRows, nsfOutputGain } from '../dist/runtime/nsf-weights.js';
import { CpuKernels } from '../dist/runtime/cpu-kernels.js';
import { CpuNsfDecoder } from '../dist/runtime/cpu-nsf.js';
import { parseZtm, vocoderFromZtm } from '../dist/runtime/ztm.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

test('nsfHeadRows: 3 without the flag, 5 for direct; an unknown branch or another noise cap throws', () => {
  assert.equal(nsfHeadRows({}), 3);
  assert.equal(nsfHeadRows({ noiseBranch: 'direct' }), 5);
  assert.equal(nsfHeadRows({ noiseBranch: 'direct', noiseMaxMag: NSF.noiseMaxMag }), 5);
  assert.throws(() => nsfHeadRows({ noiseBranch: 'other' }), /unknown noiseBranch/);
  assert.throws(() => nsfHeadRows({ noiseBranch: 'direct', noiseMaxMag: 100 }), /noiseMaxMag/);
});

/** A tiny nsf.json + nsf.bin (fp32, every tensor 0) with the given head rows. */
function fakeExport(rows, extra = {}) {
  const shapes = [];
  for (const [src, n] of [['nsf.', { C: NSF.zDim + 2, D: NSF.dim, F: NSF.ffDim, L: NSF.layers }], ['f0z.', { C: NSF.zDim, D: NSF.f0zDim, F: NSF.f0zFf, L: NSF.f0zLayers }]]) {
    shapes.push([`${src}embed.weight`, [n.D, n.C, NSF.kernel]], [`${src}embed.bias`, [n.D]], [`${src}norm.weight`, [n.D]], [`${src}norm.bias`, [n.D]]);
    for (let l = 0; l < n.L; l++) {
      const b = `${src}blocks.${l}.`;
      shapes.push([`${b}gamma`, [n.D]], [`${b}dwconv.weight`, [n.D, 1, NSF.kernel]], [`${b}dwconv.bias`, [n.D]], [`${b}norm.weight`, [n.D]], [`${b}norm.bias`, [n.D]],
        [`${b}pw1.weight`, [n.F, n.D]], [`${b}pw1.bias`, [n.F]], [`${b}pw2.weight`, [n.D, n.F]], [`${b}pw2.bias`, [n.D]]);
    }
  }
  shapes.push(['nsf.final_norm.weight', [NSF.dim]], ['nsf.final_norm.bias', [NSF.dim]], ['nsf.out.weight', [rows * NSF.bins, NSF.dim]], ['nsf.out.bias', [rows * NSF.bins]],
    ['f0z.out.weight', [2, NSF.f0zDim]], ['f0z.out.bias', [2]]);
  let off = 0;
  const tensors = shapes.map(([name, shape]) => { const bytes = 4 * shape.reduce((a, b) => a * b, 1); const t = { name, shape, offset: off, bytes }; off += bytes; return t; });
  return { meta: { dtype: 'fp32', tensors, ...extra }, bin: new ArrayBuffer(off) };
}

test('nsfExportToWeights: the head has 5 * 1025 outputs with noiseBranch direct, 3 * 1025 without; a mismatch throws', () => {
  const d = fakeExport(5, { noiseBranch: 'direct', noiseMaxMag: NSF.noiseMaxMag });
  const wd = nsfExportToWeights(d.meta, d.bin);
  assert.equal(wd['head.b'].length, 5 * NSF.bins);
  assert.equal(wd['head.w'].length, 5 * NSF.bins * NSF.dim);
  assert.equal(nsfDirectNoise(wd), true);
  const o = fakeExport(3);
  const wo = nsfExportToWeights(o.meta, o.bin);
  assert.equal(wo['head.b'].length, 3 * NSF.bins);
  assert.equal(nsfDirectNoise(wo), false);
  assert.throws(() => nsfExportToWeights({ ...o.meta, noiseBranch: 'direct' }, o.bin), /nsf\.out\.weight has/);
  assert.throws(() => nsfExportToWeights(d.meta.noiseBranch ? { ...d.meta, noiseBranch: undefined } : d.meta, d.bin), /nsf\.out\.weight has/);
});

const dir = process.env.NSF_DIRECT_DIR;
test('direct decoder on the CPU matches the PyTorch reference (wasm and the built native builds)', { skip: !dir && 'NSF_DIRECT_DIR not set' }, async () => {
  const voice = process.env.NSF_DIRECT_VOICE ?? 'marine';
  const ids = (process.env.NSF_DIRECT_IDS ?? 'a,b').split(',');
  const { meta, bin } = vocoderFromZtm(parseZtm(ab(fs.readFileSync(path.join(dir, `${voice}.ztm`)))));
  assert.equal(meta.noiseBranch, 'direct');
  const builds = ['wasm', ...['avx512', 'avx2', 'sse41'].filter((b) => fs.existsSync(path.join(root, `native/x86/build/zen_tts_${b}.node`)))];
  for (const b of builds) {
    const k = await CpuKernels.create(fs.readFileSync(path.join(root, 'cpu/tts-cpu.wasm')));
    if (b !== 'wasm') {
      try { k.useNative(createRequire(import.meta.url)(path.join(root, `native/x86/build/zen_tts_${b}.node`))); } catch { continue; }
    }
    const dec = new CpuNsfDecoder(k, nsfExportToWeights(meta, bin), 1234, nsfF0Norm(meta), nsfOutputGain(meta));
    for (const id of ids) {
      const z = new Float32Array(ab(fs.readFileSync(path.join(dir, `${id}.z.f32`))));
      const T = z.length / NSF.zDim;
      const f0 = new Float32Array(ab(fs.readFileSync(path.join(dir, `${id}.f0.f32`)))).slice(0, T);
      const ref = new Float32Array(ab(fs.readFileSync(path.join(dir, `${id}.ref.f32`))));
      const { audio } = dec.run(z, T, { f0 });
      let m = 0, e = 0, q = 0;
      for (let i = 0; i < ref.length; i++) { const d = audio[i] - ref[i]; m = Math.max(m, Math.abs(d)); e += d * d; q += ref[i] * ref[i]; }
      const snr = 10 * Math.log10(q / e);
      assert.ok(m < 2e-3 && snr > 60, `${b} ${id}: max diff ${m}, SNR ${snr.toFixed(1)} dB`);
    }
  }
});

// sha256 (first 16 hex) : samples of the 1.1.0 runtime (release-1.1.0, wasm) for the v1.1.0 CDN marine.ztm
// (sha256 3fce81ef…), so a decoder without the flag still gives the same samples bit for bit.
const OLD_MARINE = {
  ztmSha: '3fce81efbbb598b546429c6e53955a6d18fdbc9bca72afd029baf2eac06678a0',
  pcm: [
    ['本日はご来店いただき、ありがとうございます。', {}, '7d2c75fb34cfc104:139776'],
    ['ご注文の品は、明日の午前中にお届けします。', { emotion: { happy: 70, neutral: 30 } }, 'f243c8c97046b47f:154112'],
    ['申し訳ございません、少々お待ちください。', { emotion: { apologetic: 100 }, strength: 1.2 }, 'd0399aedfdf2f422:140800'],
  ],
};
const oldZtm = process.env.NSF_OLD_ZTM;
test('a 1.1.0 voice (no noiseBranch) gives the 1.1.0 runtime\'s samples bit for bit', { skip: !oldZtm && 'NSF_OLD_ZTM (the v1.1.0 marine.ztm) not set' }, async () => {
  const crypto = await import('node:crypto');
  const file = fs.readFileSync(oldZtm);
  assert.equal(crypto.createHash('sha256').update(file).digest('hex'), OLD_MARINE.ztmSha);
  const { ZenTtsNode } = await import('../dist/node/index.js');
  const t = await ZenTtsNode.load({ model: oldZtm, modelsPath: process.env.NSF_OLD_MODELS, backend: 'wasm' });
  for (const [s, o, want] of OLD_MARINE.pcm) {
    const r = await t.speak(s, o);
    const got = crypto.createHash('sha256').update(Buffer.from(r.pcm.buffer, r.pcm.byteOffset, r.pcm.byteLength)).digest('hex').slice(0, 16) + ':' + r.pcm.length;
    assert.equal(got, want, s);
  }
});
