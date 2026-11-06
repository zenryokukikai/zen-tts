// The speak option `numbers` ('rule' default | 'model'): long amounts and numbers before a unit read by rule (spec §1.5c)
// or by the reading model. Set at load (NodeLoadOptions.numbers) and overridden per call. Needs a marine .ztm: ZTM_MARINE=<path>,
// else the release's marine.ztm is fetched from the CDN (as ztm.test.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ZenTtsNode, ztmUrl, NUMBER_READINGS } from '../dist/node/index.js';
import { engineOptionsOf } from '../dist/runtime/speak-options.js';

async function marine() {
  if (process.env.ZTM_MARINE) return process.env.ZTM_MARINE;
  const p = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'numbers-test-')), 'marine.ztm');
  const r = await fetch(ztmUrl('marine')); assert.ok(r.ok, `${ztmUrl('marine')}: ${r.status}`);
  fs.writeFileSync(p, Buffer.from(await r.arrayBuffer()));
  return p;
}

const TEXT = '88866円です。';
const RULE = 'ハチマンハッセンハッピャクロクジュウロクエンです.';

test('numbers: the values and the default', () => {
  assert.deepEqual([...NUMBER_READINGS], ['rule', 'model']);
  assert.equal(engineOptionsOf({}).numbers, undefined);
  assert.equal(engineOptionsOf({ numbers: 'model' }).numbers, 'model');
  assert.throws(() => engineOptionsOf({ numbers: 'kana' }), RangeError);
});

test('numbers: default rule, model at load, per-call override (analyze and speak)', async () => {
  const model = await marine();
  const def = await ZenTtsNode.load({ model, backend: 'wasm' });
  assert.equal(def.numbers, 'rule');
  assert.equal(def.analyze(TEXT).norm, RULE);
  assert.equal(def.analyze(TEXT, [], { numbers: 'model' }).norm, '88866円です.');
  assert.equal(def.analyze('500円です。').norm, '500円です.', 'up to 4 digits stay with the model either way');

  const byModel = await ZenTtsNode.load({ model, backend: 'wasm', numbers: 'model' });
  assert.equal(byModel.analyze(TEXT).norm, '88866円です.');
  assert.equal(byModel.analyze(TEXT, [], { numbers: 'rule' }).norm, RULE);

  const a = await def.speak(TEXT);
  const b = await def.speak(TEXT, { numbers: 'model' });
  const c = await byModel.speak(TEXT);
  assert.ok(a.pcm.length > 0 && b.pcm.length > 0);
  assert.deepEqual(b.phones, c.phones, 'the per-call option and the load option give the same reading');
  assert.equal(a.phones.join(''), def.analyze(TEXT).phones.join(''), 'speak reads as analyze');

  await assert.rejects(() => ZenTtsNode.load({ model, backend: 'wasm', numbers: 'kana' }), RangeError);
});
