// Emotion sliders and 「文の途中で変化」 curves: slider values -> an emotion blend, and a blend that changes along the text.
// The model mixes its five emotion style rows by weights that sum to 1, so the sliders are relative amounts.
import { blendFrom, emotionNames, neutralBlend, type EmotionBlend } from './emotion.js';

export const TTS_DEMO_MAX_TEXT = 300;

/** Slider values (0..100 per emotion, emotionNames order) -> the blend the model gets. */
export function ttsDemoBlend(percents: readonly number[]): EmotionBlend {
  if (percents.length !== emotionNames.length) throw new Error(`tts-demo: ${percents.length} sliders`);
  const v = percents.map((p) => (Number.isFinite(p) ? Math.min(100, Math.max(0, p)) : 0));
  return v.some((p) => p > 0) ? blendFrom(v) : neutralBlend;
}

/** Text from the page -> trimmed, at most TTS_DEMO_MAX_TEXT characters; '' when nothing is left. */
export function ttsDemoText(value: unknown): string {
  return typeof value === 'string' ? [...value.trim()].slice(0, TTS_DEMO_MAX_TEXT).join('') : '';
}

/** The 「文の途中で変化」 curve shapes, in the order the page shows them. */
export const TTS_DEMO_CURVES = ['ease', 'linear', 'fast-start', 'late-surge', 'switch', 'steps', 'hill', 'valley', 'wave'] as const;
export type TtsDemoCurve = (typeof TTS_DEMO_CURVES)[number];

/** The page's name of each curve. */
export const TTS_DEMO_CURVE_LABEL: Record<TtsDemoCurve, string> = {
  ease: 'なめらか', linear: '一定の速さ', 'fast-start': '出だしから急に変化', 'late-surge': '後半から急に上昇',
  switch: '途中で一気に切り替え', steps: '階段', hill: '山なり', valley: '谷', wave: '揺れ',
};

export function isTtsDemoCurve(v: unknown): v is TtsDemoCurve {
  return typeof v === 'string' && (TTS_DEMO_CURVES as readonly string[]).includes(v);
}

/** Steepness of 途中で一気に切り替え (logistic slope at the middle of the change range). */
const SWITCH_STEEPNESS = 18;
/** 揺れ: half-waves between start and end; odd, so it ends at the end blend. */
const WAVE_HALF_CYCLES = 5;

/**
 * How far the blend has moved from start (0) to end (1) at u, the position inside the change range (0..1). The
 * one-way curves go 0 -> 1 without turning back; 山なり goes 0 -> 1 -> 0, 谷 1 -> 0 -> 1, 揺れ swings 0 <-> 1 and ends at 1.
 */
export function ttsDemoCurveShape(curve: TtsDemoCurve, u: number): number {
  const t = Math.min(1, Math.max(0, u));
  switch (curve) {
    case 'linear': return t;
    case 'ease': return t * t * (3 - 2 * t);
    case 'fast-start': return 1 - (1 - t) ** 3;
    case 'late-surge': return t ** 3;
    case 'switch': {
      const f = (x: number): number => 1 / (1 + Math.exp(-SWITCH_STEEPNESS * (x - 0.5)));
      return (f(t) - f(0)) / (f(1) - f(0));
    }
    case 'steps': return t < 1 / 3 ? 0 : t < 2 / 3 ? 0.5 : 1;
    case 'hill': return Math.sin(Math.PI * t);
    case 'valley': return 1 - Math.sin(Math.PI * t);
    case 'wave': return (1 - Math.cos(WAVE_HALF_CYCLES * Math.PI * t)) / 2;
  }
}

/**
 * 「文の途中で変化」: the blend at position x (0 = start of the text, 1 = end). Between from and
 * to (fractions of the text) it follows the curve's shape; before from it holds the shape's first value, after to its last.
 */
export function ttsDemoCurveBlend(start: EmotionBlend, end: EmotionBlend, curve: TtsDemoCurve, from: number, to: number): (x: number) => EmotionBlend {
  const a = Math.min(from, to), b = Math.max(from, to);
  return (x) => {
    const u = b <= a ? (x < a ? 0 : 1) : (x - a) / (b - a);
    const h = ttsDemoCurveShape(curve, u);
    return blendFrom(emotionNames.map((k) => (1 - h) * start[k] + h * end[k]));
  };
}
