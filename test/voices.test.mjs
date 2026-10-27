// Unit tests of the voice selection (voiceDirs) and the decoder's log-F0 normalisation from nsf.json (nsfF0Norm).
import test from 'node:test';
import assert from 'node:assert/strict';
import { VOICES, voiceDirs } from '../dist/version.js';
import { DEFAULT_NSF_F0_NORM, DEFAULT_NSF_OUTPUT_GAIN, NSF_LIMIT_CEILING, NSF_LIMIT_KNEE, applyOutputLevel, nsfF0Norm, nsfOutputGain } from '../dist/runtime/nsf-weights.js';
import { voiceDictSettings, DEFAULT_OWN_DICT_SETTINGS } from '../dist/runtime/tts-own-prosody.js';

test('voiceDirs: marine keeps tts/ and decoder/; other voices live under voices/<id>/', () => {
  assert.deepEqual(voiceDirs(), { tts: 'tts', decoder: 'decoder' });
  assert.deepEqual(voiceDirs('marine'), { tts: 'tts', decoder: 'decoder' });
  assert.deepEqual(voiceDirs('hashiro'), { tts: 'voices/hashiro/tts', decoder: 'voices/hashiro/decoder' });
  assert.deepEqual(voiceDirs('kai'), { tts: 'voices/kai/tts', decoder: 'voices/kai/decoder' });
  assert.deepEqual(voiceDirs('kikka'), { tts: 'voices/kikka/tts', decoder: 'voices/kikka/decoder' });
  assert.throws(() => voiceDirs('toString'), /unknown voice/);
  assert.deepEqual(Object.keys(VOICES), ['marine', 'hashiro', 'kai', 'kikka']);
  assert.throws(() => voiceDirs('nobody'), /unknown voice/);
});

test('nsfF0Norm: the marine nsf.json (no fields) gives 5.43 / 0.25; a voice nsf.json gives its own', () => {
  assert.deepEqual(nsfF0Norm({}), { logF0Mean: 5.43, logF0Std: 0.25 });
  assert.deepEqual(DEFAULT_NSF_F0_NORM, { logF0Mean: 5.43, logF0Std: 0.25 });
  assert.deepEqual(nsfF0Norm({ logF0Mean: 5.0417 }), { logF0Mean: 5.0417, logF0Std: 0.25 });
  assert.throws(() => nsfF0Norm({ logF0Mean: 5, logF0Std: 0 }), /bad logF0Mean/);
});

test('nsfOutputGain: 1 when nsf.json does not say, else its own value; a bad value throws', () => {
  assert.equal(nsfOutputGain({}), 1);
  assert.equal(DEFAULT_NSF_OUTPUT_GAIN, 1);
  assert.equal(nsfOutputGain({ outputGain: 3.5 }), 3.5);
  assert.throws(() => nsfOutputGain({ outputGain: 0 }), /bad outputGain/);
  assert.throws(() => nsfOutputGain({ outputGain: Number.NaN }), /bad outputGain/);
});

test('applyOutputLevel: gain 1 leaves the samples untouched (same array, same bits)', () => {
  const a = Float32Array.from([0, 0.1, -0.5, 0.999, -1.2, 3]);
  const before = Buffer.from(a.buffer).toString('hex');
  assert.equal(applyOutputLevel(a, 1), a);
  assert.equal(Buffer.from(a.buffer).toString('hex'), before);
});

test('applyOutputLevel: a raised voice is scaled below the knee and never reaches the ceiling above it', () => {
  const g = 2.8;
  const quiet = Float32Array.from([0, 0.1, -0.2, NSF_LIMIT_KNEE / g]);
  const want = Array.from(quiet, (x) => Math.fround(x * g));
  assert.deepEqual(Array.from(applyOutputLevel(quiet, g)), want);
  const loud = applyOutputLevel(Float32Array.from([0.3, -0.5, 1, -2, 50]), g);
  for (const y of loud) assert.ok(Math.abs(y) <= NSF_LIMIT_CEILING && Math.abs(y) > NSF_LIMIT_KNEE, String(y));
  assert.ok(loud[1] < 0 && loud[3] < 0, 'sign kept');
  const ramp = applyOutputLevel(Float32Array.from([0.29, 0.31, 0.33, 0.35]), g);
  for (let i = 1; i < ramp.length; i++) assert.ok(ramp[i] > ramp[i - 1], 'strictly rising above the knee');
});

test('applyOutputLevel: a lowered voice (gain < 1) is only scaled', () => {
  assert.deepEqual(Array.from(applyOutputLevel(Float32Array.from([2, -4]), 0.5)), [1, -2]);
});

test('voiceDictSettings: the voice pitchRange fills in only when the caller gave none', () => {
  assert.equal(voiceDictSettings(undefined, {}), DEFAULT_OWN_DICT_SETTINGS);
  assert.deepEqual(voiceDictSettings(undefined, { pitchRange: [1, 2, 3, 4, 5] }).pitchRange, [1, 2, 3, 4, 5]);
  assert.deepEqual(voiceDictSettings({ accentHeight: 3, pitchRange: [9, 9, 9, 9, 9] }, { pitchRange: [1, 2, 3, 4, 5] }).pitchRange, [9, 9, 9, 9, 9]);
});
