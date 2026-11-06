// Whole-utterance speaking rate and pitch (speak options `speed` / `pitch`), applied to the predicted prosody after the
// dictionary and the per-phone edits. DOM-free; shared by the WebGPU and CPU engines.
// Copyright (c) 2026 Zenryoku Kikai. Licensed under the ZEN-TTS License (LICENSE.md).

/** Accepted `speed` (speaking-rate factor), from the measurement in README「速さと高さ」: readings hold up to 1.5
 *  (kana CER 0.02 / 0.06 on 29 sentences x 2 voices) and break above it (1.75: 0.14 / 0.18); 0.5 is clean. */
export const SPEED_MIN = 0.5;
export const SPEED_MAX = 1.5;
/** Accepted `pitch` (semitones added to the whole utterance): upward the voice follows to +12 (measured +10.5..+11.1);
 *  downward it stops at about -5 to -6 st whatever is asked (-9: -5.7..-6.0), so the range ends at -6. */
export const PITCH_MIN = -6;
export const PITCH_MAX = 12;

/** Throws a RangeError for a speed / pitch outside the accepted ranges (undefined = default, accepted). */
export function checkSpeedPitch(speed: number | undefined, pitch: number | undefined): void {
  if (speed !== undefined && !(Number.isFinite(speed) && speed >= SPEED_MIN && speed <= SPEED_MAX)) {
    throw new RangeError(`speed must be ${SPEED_MIN}..${SPEED_MAX} (got ${speed})`);
  }
  if (pitch !== undefined && !(Number.isFinite(pitch) && pitch >= PITCH_MIN && pitch <= PITCH_MAX)) {
    throw new RangeError(`pitch must be ${PITCH_MIN}..${PITCH_MAX} semitones (got ${pitch})`);
  }
}

/**
 * Phone durations (frames, edges and pauses included) for a speaking rate: each d / speed, rounded so that the running
 * total follows the exact scaled total (no drift: the sum is within one frame of sum(d) / speed unless the 1-frame floor
 * adds frames), and every phone keeps at least one frame. speed 1 returns the same numbers.
 */
export function scaleDurations(dur: readonly number[], speed: number): number[] {
  if (speed === 1) return [...dur];
  const out: number[] = [];
  let exact = 0;
  let emitted = 0;
  for (const d of dur) {
    exact += d / speed;
    const n = Math.max(1, Math.round(exact - emitted));
    out.push(n);
    emitted += n;
  }
  return out;
}

/** Per-phone F0 (semitones) raised by `pitch` semitones. pitch 0 returns the same numbers. */
export function shiftF0(f0: readonly number[], pitch: number): number[] {
  return pitch === 0 ? [...f0] : f0.map((v) => v + pitch);
}

/** The engines' last prosody step: unchanged arrays when speed and pitch are unset (bit-identical output). */
export function applySpeedPitch(f0: number[], dur: number[], speed: number | undefined, pitch: number | undefined): { f0: number[]; dur: number[] } {
  checkSpeedPitch(speed, pitch);
  return {
    f0: pitch === undefined || pitch === 0 ? f0 : shiftF0(f0, pitch),
    dur: speed === undefined || speed === 1 ? dur : scaleDurations(dur, speed),
  };
}

/** speakStream's sentence length limit for a speed: the default 120 characters, shorter when slower (speed < 1) so that
 *  one sentence stays under the engine's frame limit (2,048 frames, about 23.8 s; 120 characters at speed 1 measured up
 *  to about 15 s). */
export function streamMaxChars(speed: number | undefined, base = 120): number {
  return speed !== undefined && speed < 1 ? Math.max(20, Math.floor(base * speed)) : base;
}
