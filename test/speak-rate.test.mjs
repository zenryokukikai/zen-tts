// Unit tests of speak-rate.ts (speak options speed / pitch): no models needed.
import test from 'node:test';
import assert from 'node:assert/strict';
import { scaleDurations, shiftF0, applySpeedPitch, checkSpeedPitch, SPEED_MIN, SPEED_MAX, PITCH_MIN, PITCH_MAX } from '../dist/runtime/speak-rate.js';
import { engineOptionsOf } from '../dist/runtime/speak-options.js';

const DUR = [3, 1, 2, 7, 1, 1, 12, 4, 1, 2, 9, 1, 5];

test('defaults return the same arrays (bit-identical output)', () => {
  const f0 = [60.1, 61.2, 62.3];
  const dur = [3, 4, 5];
  const r = applySpeedPitch(f0, dur, undefined, undefined);
  assert.equal(r.f0, f0);
  assert.equal(r.dur, dur);
  const r1 = applySpeedPitch(f0, dur, 1, 0);
  assert.equal(r1.f0, f0);
  assert.equal(r1.dur, dur);
});

test('scaleDurations: total follows sum / speed, every phone keeps at least one frame', () => {
  const total = DUR.reduce((a, b) => a + b, 0);
  for (const s of [0.5, 0.75, 1.25, 1.5, 2]) {
    const d = scaleDurations(DUR, s);
    assert.equal(d.length, DUR.length);
    assert.ok(d.every((x) => Number.isInteger(x) && x >= 1), `s=${s}`);
    const sum = d.reduce((a, b) => a + b, 0);
    // within 1 frame of the exact total, plus the frames the 1-frame floor adds
    const floorAdds = DUR.filter((x) => x / s < 0.5).length;
    assert.ok(Math.abs(sum - total / s) <= 1 + floorAdds, `s=${s} sum=${sum} exact=${total / s}`);
  }
  assert.deepEqual(scaleDurations(DUR, 1), DUR);
  assert.deepEqual(scaleDurations(DUR, 0.5), DUR.map((x) => x * 2));
});

test('scaleDurations: no drift on long input', () => {
  const long = Array.from({ length: 400 }, (_, i) => 1 + (i * 7) % 9);
  const total = long.reduce((a, b) => a + b, 0);
  for (const s of [0.75, 1.25, 1.5]) {
    const sum = scaleDurations(long, s).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - total / s) <= 1, `s=${s} sum=${sum} exact=${total / s}`);
  }
});

test('shiftF0 adds semitones to every phone', () => {
  assert.deepEqual(shiftF0([60, 62.5, 0], 3), [63, 65.5, 3]);
  assert.deepEqual(shiftF0([60, 61], 0), [60, 61]);
});

test('range checks and the public option mapping', () => {
  assert.doesNotThrow(() => checkSpeedPitch(SPEED_MIN, PITCH_MIN));
  assert.doesNotThrow(() => checkSpeedPitch(SPEED_MAX, PITCH_MAX));
  for (const [s, p] of [[0.4, 0], [1.6, 0], [2, 0], [Number.NaN, 0], [1, 13], [1, -7], [1, Number.POSITIVE_INFINITY]]) {
    assert.throws(() => checkSpeedPitch(s, p), RangeError, `${s} ${p}`);
  }
  const o = engineOptionsOf({ speed: 1.25, pitch: -3 });
  assert.equal(o.speed, 1.25);
  assert.equal(o.pitch, -3);
  const d = engineOptionsOf({});
  assert.ok(!('speed' in d) && !('pitch' in d));
  assert.throws(() => engineOptionsOf({ speed: 1.75 }), RangeError);
});

test('streamMaxChars: 120 at speed >= 1, shorter when slower', async () => {
  const { streamMaxChars } = await import('../dist/runtime/speak-rate.js');
  assert.equal(streamMaxChars(undefined), 120);
  assert.equal(streamMaxChars(1), 120);
  assert.equal(streamMaxChars(1.5), 120);
  assert.equal(streamMaxChars(0.75), 90);
  assert.equal(streamMaxChars(0.5), 60);
});
