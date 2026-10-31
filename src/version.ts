// ZEN-TTS release and model versions (kept in step with package.json "version").
export const VERSION = '1.1.0';
/** The default voice: 「全力 マリン」 (Zenryoku Marine). */
export const VOICE = { id: 'marine', name: { ja: '全力 マリン', en: 'Zenryoku Marine' }, model: 'marine-1.0' } as const;

/** The four voices a models directory can hold. 'marine' is in tts/ and decoder/; the others in voices/<id>/tts/ and voices/<id>/decoder/. */
export const VOICES = {
  marine: VOICE,
  hashiro: { id: 'hashiro', name: { ja: '全力 ハシロウ', en: 'Zenryoku Hashiro' }, model: 'hashiro-1.0' },
  kai: { id: 'kai', name: { ja: '全力 カイ', en: 'Zenryoku Kai' }, model: 'kai-1.0' },
  kikka: { id: 'kikka', name: { ja: '全力 キッカ', en: 'Zenryoku Kikka' }, model: 'kikka-1.0' },
} as const;
export type VoiceId = keyof typeof VOICES;

/** The weight set directories of a voice, relative to the models directory. Throws on an unknown id. */
export function voiceDirs(voice: VoiceId = 'marine'): { tts: string; decoder: string } {
  if (!Object.prototype.hasOwnProperty.call(VOICES, voice)) throw new Error(`unknown voice "${String(voice)}" (${Object.keys(VOICES).join(', ')})`);
  return voice === 'marine' ? { tts: 'tts', decoder: 'decoder' } : { tts: `voices/${voice}/tts`, decoder: `voices/${voice}/decoder` };
}

/** Where the voices of this release are published, one .ztm file per voice (docs/ztm-format.md). */
export const ZTM_BASE_URL = 'https://tts-models.zenryoku.work/v1.1.0/';

/** The CDN URL of a voice's .ztm (ZTM_BASE_URL + <id>.ztm). Throws on an unknown id. */
export function ztmUrl(voice: VoiceId = 'marine'): string {
  voiceDirs(voice);
  return `${ZTM_BASE_URL}${voice}.ztm`;
}
