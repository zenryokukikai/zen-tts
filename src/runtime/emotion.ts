// The five emotion styles of the model, in the order of its style rows.
export const emotionNames = ['neutral', 'happy', 'sad', 'angry', 'apologetic'] as const;
export type EmotionName = typeof emotionNames[number];
/** Weights per emotion (each 0..1, sum 1). */
export type EmotionBlend = Readonly<Record<EmotionName, number>>;
export const neutralBlend: EmotionBlend = Object.freeze({ neutral: 1, happy: 0, sad: 0, angry: 0, apologetic: 0 });

/** Non-negative amounts per emotion (emotionNames order) -> a blend whose weights sum to 1. */
export function blendFrom(values: number[]): EmotionBlend {
  const total = values.reduce((a, b) => a + b, 0);
  return Object.freeze(Object.fromEntries(emotionNames.map((k, i) => [k, values[i]! / total])) as Record<EmotionName, number>);
}
