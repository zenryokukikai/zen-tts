// .ztm (ZEN-TTS model): one voice in one file (docs/ztm-format.md).
//
//   bytes 0..3   "ZTM1"
//   bytes 4..7   header length H (uint32, little endian)
//   bytes 8..    the header: UTF-8 JSON, padded with spaces so the first section starts on a 64-byte boundary
//   then the sections "acoustic" and "vocoder" at the offsets the header gives (64-byte aligned, zero padding between)
//
// acoustic = today's tts/own.{json,bin} without the G2P (text -> durations / F0 / z, and the styles);
// vocoder  = today's decoder/nsf.{json,bin}. The G2P (reading model) is not in the file: it is the runtime's
// (<models>/g2p/g2p.{json,bin}), shared by every voice. The header records the G2P it was built with (its SHA-256).
import { emotionNames } from './emotion.js';
import type { NsfExportMeta } from './nsf-weights.js';
import type { OwnMeta } from './tts-own-prosody.js';

export const ZTM_MAGIC = 'ZTM1';
export const ZTM_VERSION = 1;
export const ZTM_ALIGN = 64;
/** own.json fields that belong to the G2P / text frontend (shared by every voice); the rest are the voice's. */
export const G2P_META_KEYS = ['vocab', 'charPad', 'charUnk', 'phoneClasses', 'marks', 'punctuation', 'kana'] as const;
export const G2P_TENSOR_PREFIX = 'g2p.';

type Tensor = OwnMeta['tensors'][number];
type NsfTensor = NsfExportMeta['tensors'][number];

/** <models>/g2p/g2p.json: the G2P tensors of g2p.bin (offsets from 0) and the frontend tables. */
export interface G2pMeta {
  format: 'zen-tts-g2p';
  version: 1;
  /** SHA-256 (hex) of g2p.bin */
  sha256: string;
  bytes: number;
  /** own.json shape.g2p */
  shape: OwnMeta['shape']['g2p'];
  vocab: string[];
  charPad: number;
  charUnk: number;
  phoneClasses: string[];
  marks: string[];
  punctuation: string[];
  kana: Record<string, [string, string]>;
  tensors: Tensor[];
}

export interface ZtmSection { offset: number; bytes: number; sha256: string }

export interface ZtmHeader {
  format: 'zen-tts-model';
  version: 1;
  voice: { id: string; name?: { ja?: string; en?: string } };
  /** style names in the order of the style rows (this runtime plays the five emotionNames only) */
  styles?: string[];
  created?: string;
  source?: Record<string, unknown>;
  /** the G2P the voice was built and checked with */
  g2p: { sha256: string; bytes: number };
  /** own.json without the G2P fields and tensors; tensor offsets from the start of the acoustic section */
  acoustic: { meta: Record<string, unknown> & { shape: Record<string, unknown> }; tensors: Tensor[] };
  /** nsf.json without tensors; tensor offsets from the start of the vocoder section */
  vocoder: { meta: Omit<NsfExportMeta, 'tensors'> & Record<string, unknown>; tensors: NsfTensor[] };
  sections: { acoustic: ZtmSection; vocoder: ZtmSection };
}

/** A parsed .ztm: the header and the whole file (sections are views into it). */
export interface Ztm { header: ZtmHeader; file: ArrayBuffer }

const bad = (m: string): Error => new Error(`ztm: ${m}`);

export function isZtm(file: ArrayBuffer): boolean {
  if (file.byteLength < 8) return false;
  const b = new Uint8Array(file, 0, 4);
  return String.fromCharCode(...b) === ZTM_MAGIC;
}

/** Reads and checks the header (sizes and bounds; the SHA-256 of the sections is checked by verifyZtm where hashing is at hand). */
export function parseZtm(file: ArrayBuffer): Ztm {
  if (!isZtm(file)) throw bad('not a .ztm file (no ZTM1 magic)');
  const h = new DataView(file).getUint32(4, true);
  if (8 + h > file.byteLength) throw bad(`header length ${h} past the end of the file`);
  let header: ZtmHeader;
  try { header = JSON.parse(new TextDecoder().decode(new Uint8Array(file, 8, h))) as ZtmHeader; } catch (e) { throw bad(`header is not JSON (${e instanceof Error ? e.message : String(e)})`); }
  if (header.format !== 'zen-tts-model') throw bad(`format ${String(header.format)}`);
  if (header.version !== ZTM_VERSION) throw bad(`version ${String(header.version)} (this runtime reads ${ZTM_VERSION})`);
  if (!header.voice?.id) throw bad('voice.id missing');
  for (const name of ['acoustic', 'vocoder'] as const) {
    const s = header.sections?.[name];
    if (!s) throw bad(`section ${name} missing`);
    if (s.offset % ZTM_ALIGN !== 0 || s.offset < 8 + h || s.offset + s.bytes > file.byteLength) throw bad(`section ${name} out of bounds or not aligned`);
    const list = name === 'acoustic' ? header.acoustic.tensors : header.vocoder.tensors;
    for (const t of list) if (t.offset < 0 || t.offset + t.bytes > s.bytes) throw bad(`tensor ${t.name} outside section ${name}`);
  }
  if (header.styles && (header.styles.length !== emotionNames.length || header.styles.some((s, i) => s !== emotionNames[i]))) {
    throw bad(`voice ${header.voice.id} has the styles [${header.styles.join(', ')}]; this runtime only plays [${emotionNames.join(', ')}]`);
  }
  return { header, file };
}

/** The bytes of a section (a view, no copy). */
export function ztmSection(z: Ztm, name: 'acoustic' | 'vocoder'): Uint8Array {
  const s = z.header.sections[name];
  return new Uint8Array(z.file, s.offset, s.bytes);
}

/** A warning text when the voice was built with another G2P than the runtime's, else undefined. */
export function g2pMismatch(z: Ztm, g2p: Pick<G2pMeta, 'sha256'>): string | undefined {
  if (z.header.g2p.sha256 === g2p.sha256) return undefined;
  return `zen-tts: voice "${z.header.voice.id}" was built with G2P ${z.header.g2p.sha256.slice(0, 12)}, this runtime has ${g2p.sha256.slice(0, 12)}: readings can differ from the voice's checks`;
}

/**
 * own.json + own.bin as the engines take them: the runtime's G2P bytes first, then the acoustic section, with the
 * tensor offsets of the legacy own.bin (G2P from 0, acoustic after it). One copy of both parts (~11 MB).
 */
export function ownFromZtm(z: Ztm, g2p: G2pMeta, g2pBin: ArrayBuffer): { meta: OwnMeta; bin: ArrayBuffer } {
  if (g2pBin.byteLength !== g2p.bytes) throw bad(`g2p.bin is ${g2pBin.byteLength} bytes, g2p.json says ${g2p.bytes}`);
  if (g2p.bytes % ZTM_ALIGN !== 0) throw bad('g2p.bin length must be a multiple of 64');
  const acoustic = ztmSection(z, 'acoustic');
  const out = new Uint8Array(g2p.bytes + acoustic.byteLength);
  out.set(new Uint8Array(g2pBin), 0);
  out.set(acoustic, g2p.bytes);
  const { shape, ...rest } = z.header.acoustic.meta;
  const tables = Object.fromEntries(G2P_META_KEYS.map((k) => [k, g2p[k]]));
  const meta = {
    ...rest,
    ...tables,
    shape: { g2p: g2p.shape, ...shape },
    tensors: [...g2p.tensors, ...z.header.acoustic.tensors.map((t) => ({ ...t, offset: t.offset + g2p.bytes }))],
  } as unknown as OwnMeta;
  return { meta, bin: out.buffer };
}

/** nsf.json + nsf.bin as the decoder takes them: the tensor offsets moved to the whole file (no copy of the weights). */
export function vocoderFromZtm(z: Ztm): { meta: NsfExportMeta; bin: ArrayBuffer } {
  const base = z.header.sections.vocoder.offset;
  const meta = { ...z.header.vocoder.meta, tensors: z.header.vocoder.tensors.map((t) => ({ ...t, offset: t.offset + base })) } as NsfExportMeta;
  return { meta, bin: z.file };
}
