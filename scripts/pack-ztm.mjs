#!/usr/bin/env node
// .ztm packer (docs/ztm-format.md): one voice in one file, and back.
//
//   node scripts/pack-ztm.mjs pack <export dir> -o kikka.ztm [--id kikka] [--name-ja 全力 キッカ] [--name-en "Zenryoku Kikka"] [--created ISO]
//       <export dir>: tts/own.{json,bin} + decoder/nsf.{json,bin} (the export layout; models/ for marine)
//   node scripts/pack-ztm.mjs unpack <file.ztm> -o <dir> --g2p <models>/g2p
//       writes tts/own.{json,bin} + decoder/nsf.{json,bin} (own.bin = the given G2P + the acoustic section) and ztm.json
//       (voice, styles, created, source: pack reads it back, so pack(unpack(x)) gives the same bytes)
//   node scripts/pack-ztm.mjs g2p <export dir> -o <models>/g2p
//       the G2P part of own.{json,bin} as g2p.json + g2p.bin (the runtime's shared reading model)
//   node scripts/pack-ztm.mjs info <file.ztm>
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const MAGIC = 'ZTM1';
const ALIGN = 64;
const G2P_KEYS = ['vocab', 'charPad', 'charUnk', 'phoneClasses', 'marks', 'punctuation', 'kana'];
const EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'apologetic'];
const PACKER = 'pack-ztm 1';

const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const pad = (n) => Math.ceil(n / ALIGN) * ALIGN;
const die = (m) => { console.error(`pack-ztm: ${m}`); process.exit(1); };

function args(argv) {
  const pos = []; const opt = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--') || argv[i] === '-o') { opt[argv[i].replace(/^-+/, '')] = argv[++i]; } else pos.push(argv[i]);
  }
  return { pos, opt };
}

/** own.json / own.bin of an export dir, split at the end of the g2p.* tensors (they come first, from offset 0). */
function readExport(dir) {
  const r = (p) => { const f = path.join(dir, p); if (!fs.existsSync(f)) die(`${f} is missing`); return fs.readFileSync(f); };
  const ownJson = r('tts/own.json'), ownBin = r('tts/own.bin'), nsfJson = r('decoder/nsf.json'), nsfBin = r('decoder/nsf.bin');
  const own = JSON.parse(ownJson), nsf = JSON.parse(nsfJson);
  const g = own.tensors.filter((t) => t.name.startsWith('g2p.'));
  const a = own.tensors.filter((t) => !t.name.startsWith('g2p.'));
  const g2pEnd = Math.max(...g.map((t) => t.offset + t.bytes));
  if (Math.min(...g.map((t) => t.offset)) !== 0) die('g2p tensors do not start at 0');
  if (a.some((t) => t.offset < g2pEnd)) die('an acoustic tensor lies inside the g2p bytes');
  if (g2pEnd % ALIGN) die(`g2p bytes end at ${g2pEnd}, not a multiple of ${ALIGN}`);
  return { own, nsf, ownBin, nsfBin, g, a, g2pEnd, files: { 'tts/own.json': sha(ownJson), 'tts/own.bin': sha(ownBin), 'decoder/nsf.json': sha(nsfJson), 'decoder/nsf.bin': sha(nsfBin) } };
}

function pack(dir, out, opt) {
  const e = readExport(dir);
  const side = fs.existsSync(path.join(dir, 'ztm.json')) ? JSON.parse(fs.readFileSync(path.join(dir, 'ztm.json'), 'utf8')) : {};
  const id = opt.id ?? side.voice?.id ?? path.basename(path.resolve(dir));
  const name = side.voice?.name ?? (opt['name-ja'] || opt['name-en'] ? { ...(opt['name-ja'] ? { ja: opt['name-ja'] } : {}), ...(opt['name-en'] ? { en: opt['name-en'] } : {}) } : undefined);
  const styles = side.styles ?? e.own.styles ?? (e.own.shape.styles === EMOTIONS.length ? EMOTIONS : undefined);
  if (!styles) die(`shape.styles is ${e.own.shape.styles} and own.json has no styles list: give the names`);
  const { shape, tensors: _t, styles: _s, ...rest } = e.own;
  const acMeta = Object.fromEntries(Object.entries(rest).filter(([k]) => !G2P_KEYS.includes(k)));
  const { g2p: _g, ...acShape } = shape;
  const acBin = e.ownBin.subarray(e.g2pEnd);
  const { tensors: nsfTensors, ...nsfMeta } = e.nsf;
  const header = {
    format: 'zen-tts-model', version: 1,
    voice: { id, ...(name ? { name } : {}) },
    styles,
    created: opt.created ?? side.created ?? new Date().toISOString(),
    source: side.source ?? { packer: PACKER, from: path.basename(path.resolve(dir)), files: e.files },
    g2p: { sha256: sha(e.ownBin.subarray(0, e.g2pEnd)), bytes: e.g2pEnd },
    acoustic: { meta: { ...acMeta, shape: acShape }, tensors: e.a.map((t) => ({ ...t, offset: t.offset - e.g2pEnd })) },
    vocoder: { meta: nsfMeta, tensors: nsfTensors },
    sections: {},
  };
  // the section offsets depend on the header length: fix the point with placeholder offsets of the final width
  const place = () => {
    const json = Buffer.from(JSON.stringify(header));
    const hlen = pad(8 + json.length) - 8;
    const ac = 8 + hlen, vo = pad(ac + acBin.length);
    return { hlen, ac, vo };
  };
  header.sections = { acoustic: { offset: 0, bytes: acBin.length, sha256: sha(acBin) }, vocoder: { offset: 0, bytes: e.nsfBin.length, sha256: sha(e.nsfBin) } };
  let p = place();
  for (let i = 0; i < 4; i++) {
    header.sections.acoustic.offset = p.ac; header.sections.vocoder.offset = p.vo;
    const q = place();
    if (q.ac === p.ac && q.vo === p.vo) break;
    p = q;
  }
  const json = Buffer.from(JSON.stringify(header));
  const file = Buffer.alloc(p.vo + e.nsfBin.length);
  file.write(MAGIC, 0, 'latin1');
  file.writeUInt32LE(p.hlen, 4);
  file.fill(0x20, 8, 8 + p.hlen);
  json.copy(file, 8);
  acBin.copy(file, p.ac);
  e.nsfBin.copy(file, p.vo);
  fs.writeFileSync(out, file);
  console.log(JSON.stringify({ out, id, bytes: file.length, legacyBytes: e.ownBin.length + e.nsfBin.length + Buffer.byteLength(JSON.stringify(e.own)) + Buffer.byteLength(JSON.stringify(e.nsf)), g2p: header.g2p.sha256.slice(0, 16), styles }));
}

function read(file) {
  const b = fs.readFileSync(file);
  if (b.toString('latin1', 0, 4) !== MAGIC) die(`${file}: not a .ztm`);
  const hlen = b.readUInt32LE(4);
  const header = JSON.parse(b.toString('utf8', 8, 8 + hlen));
  const sec = (n) => { const s = header.sections[n]; const v = b.subarray(s.offset, s.offset + s.bytes); if (sha(v) !== s.sha256) die(`${file}: section ${n} SHA-256 mismatch`); return v; };
  return { b, header, acoustic: sec('acoustic'), vocoder: sec('vocoder') };
}

function unpack(file, outDir, g2pDir) {
  const z = read(file);
  if (!g2pDir) die('--g2p <models>/g2p is needed (own.bin holds the G2P)');
  const g2p = JSON.parse(fs.readFileSync(path.join(g2pDir, 'g2p.json'), 'utf8'));
  const g2pBin = fs.readFileSync(path.join(g2pDir, 'g2p.bin'));
  if (sha(g2pBin) !== z.header.g2p.sha256) console.warn(`pack-ztm: warning: ${file} was built with G2P ${z.header.g2p.sha256.slice(0, 12)}, ${g2pDir} has ${sha(g2pBin).slice(0, 12)}`);
  const { shape, ...rest } = z.header.acoustic.meta;
  const own = { ...rest, ...Object.fromEntries(G2P_KEYS.map((k) => [k, g2p[k]])), shape: { g2p: g2p.shape, ...shape }, tensors: [...g2p.tensors, ...z.header.acoustic.tensors.map((t) => ({ ...t, offset: t.offset + g2pBin.length }))] };
  const nsf = { ...z.header.vocoder.meta, tensors: z.header.vocoder.tensors };
  fs.mkdirSync(path.join(outDir, 'tts'), { recursive: true });
  fs.mkdirSync(path.join(outDir, 'decoder'), { recursive: true });
  fs.writeFileSync(path.join(outDir, 'tts/own.json'), JSON.stringify(own));
  fs.writeFileSync(path.join(outDir, 'tts/own.bin'), Buffer.concat([g2pBin, z.acoustic]));
  fs.writeFileSync(path.join(outDir, 'decoder/nsf.json'), JSON.stringify(nsf));
  fs.writeFileSync(path.join(outDir, 'decoder/nsf.bin'), z.vocoder);
  fs.writeFileSync(path.join(outDir, 'ztm.json'), JSON.stringify({ voice: z.header.voice, styles: z.header.styles, created: z.header.created, source: z.header.source }, null, 1));
  console.log(JSON.stringify({ outDir, voice: z.header.voice.id }));
}

function g2p(dir, outDir) {
  const e = readExport(dir);
  const bin = e.ownBin.subarray(0, e.g2pEnd);
  const meta = { format: 'zen-tts-g2p', version: 1, sha256: sha(bin), bytes: bin.length, shape: e.own.shape.g2p, ...Object.fromEntries(G2P_KEYS.map((k) => [k, e.own[k]])), tensors: e.g };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'g2p.json'), JSON.stringify(meta));
  fs.writeFileSync(path.join(outDir, 'g2p.bin'), bin);
  console.log(JSON.stringify({ outDir, sha256: meta.sha256, bytes: bin.length }));
}

const [cmd, ...rest] = process.argv.slice(2);
const { pos, opt } = args(rest);
if (cmd === 'pack' && pos[0] && opt.o) pack(pos[0], opt.o, opt);
else if (cmd === 'unpack' && pos[0] && opt.o) unpack(pos[0], opt.o, opt.g2p);
else if (cmd === 'g2p' && pos[0] && opt.o) g2p(pos[0], opt.o);
else if (cmd === 'info' && pos[0]) { const z = read(pos[0]); const { acoustic, vocoder, ...h } = z.header; console.log(JSON.stringify({ ...h, acousticTensors: acoustic.tensors.length, vocoderTensors: vocoder.tensors.length, fileBytes: z.b.length }, null, 1)); }
else die('usage: pack <export dir> -o x.ztm [--id] [--name-ja] [--name-en] [--created] | unpack <x.ztm> -o <dir> --g2p <models>/g2p | g2p <export dir> -o <dir> | info <x.ztm>');
