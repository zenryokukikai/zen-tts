// The public speak() options -> the engine's options, and the text -> sentences split used by speakStream(). Shared by
// the browser entry (index.ts) and the Node entry (node/index.ts). DOM-free.
import { blendFrom, emotionNames, neutralBlend, type EmotionBlend, type EmotionName } from './emotion.js';
import { ttsDemoCurveBlend, type TtsDemoCurve } from './emotion-curve.js';
import type { OwnSpeakOptions } from './tts-own-device.js';
import type { OwnDictEntry, OwnDictSettings } from './tts-own-prosody.js';

/** Relative amounts per emotion (any non-negative numbers; only their ratio matters). Missing = 0; all 0 = neutral. */
export type EmotionAmounts = Partial<Record<EmotionName, number>>;

/** Emotion amounts -> the blend the model takes (weights summing to 1). */
export function emotionBlend(amounts: EmotionAmounts | EmotionBlend | undefined): EmotionBlend {
  if (!amounts) return neutralBlend;
  const v = emotionNames.map((k) => {
    const x = (amounts as Record<string, number | undefined>)[k] ?? 0;
    return Number.isFinite(x) && x > 0 ? x : 0;
  });
  return v.some((x) => x > 0) ? blendFrom(v) : neutralBlend;
}

export interface SpeakOptions {
  /** The emotion (start emotion when `change` is given). */
  emotion?: EmotionAmounts;
  /** 文の途中で変化: change from `emotion` to `to` along the sentence with a curve shape over [from, to] of the text (0..1). */
  change?: { to: EmotionAmounts; curve?: TtsDemoCurve; from?: number; until?: number };
  /** Scale of the model's intonation (F0 deviation): 1 = default, 0.5..2 is useful. */
  strength?: number;
  /** Reading / intonation dictionary entries. */
  dict?: readonly OwnDictEntry[];
  settings?: OwnDictSettings;
}

/** SpeakOptions -> the engine's options (blend, and blendAt for `change`). */
export function engineOptionsOf(o: SpeakOptions): OwnSpeakOptions {
  const blend = emotionBlend(o.emotion);
  const opts: OwnSpeakOptions = { blend };
  if (o.strength !== undefined) opts.strength = o.strength;
  if (o.dict) opts.dict = o.dict;
  if (o.settings) opts.settings = o.settings;
  if (o.change) {
    const c = o.change;
    opts.blendAt = ttsDemoCurveBlend(blend, emotionBlend(c.to), c.curve ?? 'ease', c.from ?? 0, c.until ?? 1);
  }
  return opts;
}

/**
 * Points of a blendAt curve on the CPU path: a function cannot be posted to the browser's CPU worker, so it is sampled
 * at CURVE_POINTS points and read back linearly (curveFn). The Node entry does the same, so both CPU paths give the same
 * audio for a `change`.
 */
export const CURVE_POINTS = 129;

export function curvePoints(blendAt: (x: number) => EmotionBlend): EmotionBlend[] {
  return Array.from({ length: CURVE_POINTS }, (_, i) => blendAt(i / (CURVE_POINTS - 1)));
}

export function curveFn(points: readonly EmotionBlend[]): (x: number) => EmotionBlend {
  const n = points.length - 1;
  return (x) => {
    const u = Math.min(1, Math.max(0, x)) * n;
    const k = Math.min(n - 1, Math.floor(u));
    const h = u - k;
    const a = points[k]!, b = points[k + 1]!;
    return Object.fromEntries(Object.keys(a).map((key) => [key, (1 - h) * a[key as keyof EmotionBlend] + h * b[key as keyof EmotionBlend]])) as EmotionBlend;
  };
}

/**
 * Text -> sentences for speakStream(): split after 。！？!? (kept with their sentence) and at line breaks; each piece is
 * at most maxChars characters (a longer one is cut after its last 、，, or space before the limit, else at the limit).
 * One sentence of speech should stay under about 11 s (the engine's limit per call).
 */
export function splitSentences(text: string, maxChars = 120): string[] {
  const out: string[] = [];
  for (const line of text.split(/\r?\n+/)) {
    for (const m of line.matchAll(/[^。！？!?]+[。！？!?]*|[。！？!?]+/gu)) {
      let s = m[0].trim();
      while (s.length > maxChars) {
        const head = s.slice(0, maxChars);
        const cut = Math.max(head.lastIndexOf('、'), head.lastIndexOf('，'), head.lastIndexOf(','), head.lastIndexOf(' '));
        const at = cut > 0 ? cut + 1 : maxChars;
        out.push(s.slice(0, at).trim());
        s = s.slice(at).trim();
      }
      if (s && /[^\s。！？!?、，,]/u.test(s)) out.push(s);
    }
  }
  return out;
}
