// .ztm (docs/ztm-format.md): pack -> unpack -> pack gives the same bytes; a voice loaded from its .ztm speaks the same
// samples as the same voice from the legacy 4 files (tts/own.*, decoder/nsf.*); header checks.
// The 4-file voices are not in the repository since 1.1.0: give ZTM_LEGACY_MODELS=<the models/ directory of the v1.0.0
// tag> to run the round trip and the same-samples tests for the four voices. Without it, the release's .ztm of marine is
// fetched from the CDN (ztmUrl) for the header checks, and unpack -> pack is checked on it.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { ZenTtsNode, parseZtm, ztmUrl, VOICES, voiceDirs } from '../dist/node/index.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ztm-test-'));
const pack = (...a) => execFileSync(process.execPath, [path.join(root, 'scripts/pack-ztm.mjs'), ...a], { encoding: 'utf8' });
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const pcmSha = (a) => sha(Buffer.from(a.buffer, a.byteOffset, a.byteLength));
const SENTENCES = [
  ['本日はご来店いただき、ありがとうございます。', {}],
  ['ご注文の品は、明日の午前中にお届けします。', { emotion: { happy: 70, neutral: 30 } }],
  ['申し訳ございません、少々お待ちください。', { emotion: { apologetic: 100 }, strength: 1.2 }],
];
const g2pDir = path.join(root, 'models', 'g2p');
const legacy = process.env.ZTM_LEGACY_MODELS;

if (legacy) for (const id of Object.keys(VOICES)) {
  const dir = path.join(legacy, path.dirname(voiceDirs(id).tts));
  test(`${id}: pack -> unpack -> pack is byte-exact, and unpack gives back the legacy files`, () => {
    const a = path.join(tmp, `${id}.ztm`);
    pack('pack', dir, '-o', a, '--id', id, '--created', '2026-10-31T00:00:00Z');
    const u = path.join(tmp, `${id}-unpacked`);
    pack('unpack', a, '-o', u, '--g2p', g2pDir);
    for (const f of ['tts/own.bin', 'decoder/nsf.bin']) assert.equal(sha(fs.readFileSync(path.join(u, f))), sha(fs.readFileSync(path.join(dir, f))), f);
    for (const f of ['tts/own.json', 'decoder/nsf.json']) assert.deepEqual(JSON.parse(fs.readFileSync(path.join(u, f), 'utf8')), JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')), f);
    const b = path.join(tmp, `${id}-2.ztm`);
    pack('pack', u, '-o', b);
    assert.equal(sha(fs.readFileSync(b)), sha(fs.readFileSync(a)));
  });

  test(`${id}: the .ztm speaks the same samples as the legacy 4 files`, async () => {
    const old = await ZenTtsNode.load({ modelsPath: legacy, voice: id });
    const packed = await ZenTtsNode.load({ model: path.join(tmp, `${id}.ztm`), ...(old.kernel === 'wasm' ? { backend: 'wasm' } : { kernel: old.kernel }) });
    assert.equal(packed.voice, id);
    assert.equal(packed.kernel, old.kernel);
    for (const [text, o] of SENTENCES) {
      const a = await old.speak(text, o);
      const b = await packed.speak(text, o);
      assert.ok(a.pcm.length > 44100 * 0.5, 'audio made');
      assert.equal(b.pcm.length, a.pcm.length, text);
      assert.equal(pcmSha(b.pcm), pcmSha(a.pcm), text);
      assert.deepEqual(b.phones, a.phones, text);
    }
  });
}

const marineZtm = async () => {
  const p = path.join(tmp, 'marine.ztm');
  if (!fs.existsSync(p)) {
    if (legacy) pack('pack', legacy, '-o', p, '--id', 'marine', '--created', '2026-10-31T00:00:00Z');
    else { const r = await fetch(ztmUrl('marine')); assert.ok(r.ok, `${ztmUrl('marine')}: ${r.status}`); fs.writeFileSync(p, Buffer.from(await r.arrayBuffer())); }
  }
  return p;
};

test('the release .ztm of marine: unpack -> pack gives the same bytes, and it speaks', async () => {
  const p = await marineZtm();
  const u = path.join(tmp, 'marine-rt');
  pack('unpack', p, '-o', u, '--g2p', g2pDir);
  const b = path.join(tmp, 'marine-rt.ztm');
  pack('pack', u, '-o', b);
  assert.equal(sha(fs.readFileSync(b)), sha(fs.readFileSync(p)));
  const t = await ZenTtsNode.load({ model: p, backend: 'wasm' });
  assert.equal(t.voice, 'marine');
  const r = await t.speak(SENTENCES[0][0]);
  assert.ok(r.pcm.length > 44100 * 0.5);
});

test('parseZtm: refuses a wrong magic, a newer version, sections out of bounds, and styles this runtime cannot play', async () => {
  const file = fs.readFileSync(await marineZtm());
  const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  assert.equal(parseZtm(ab(file)).header.voice.id, 'marine');
  const bad = Buffer.from(file); bad.write('XXXX', 0, 'latin1');
  assert.throws(() => parseZtm(ab(bad)), /not a \.ztm/);
  const withHeader = (edit) => {
    const h = file.readUInt32LE(4);
    const header = JSON.parse(file.toString('utf8', 8, 8 + h));
    edit(header);
    const json = Buffer.from(JSON.stringify(header));
    assert.ok(json.length <= h, 'edited header fits');
    const b = Buffer.from(file); b.fill(0x20, 8, 8 + h); json.copy(b, 8);
    return ab(b);
  };
  assert.throws(() => parseZtm(withHeader((x) => { x.version = 2; })), /version 2/);
  assert.throws(() => parseZtm(withHeader((x) => { x.sections.vocoder.bytes += 64; })), /out of bounds/);
  assert.throws(() => parseZtm(withHeader((x) => { x.styles = ['neutral', 'happy']; })), /only plays/);
});

test('ztmUrl: the CDN URL of each voice of this release', () => {
  assert.equal(ztmUrl(), 'https://tts-models.zenryoku.work/v1.1.0/marine.ztm');
  assert.equal(ztmUrl('kikka'), 'https://tts-models.zenryoku.work/v1.1.0/kikka.ztm');
  assert.throws(() => ztmUrl('nobody'), /unknown voice/);
});
