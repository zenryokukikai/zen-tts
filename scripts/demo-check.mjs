// Demo check (headless Chromium) of demo/index.html served as static files:
//  1. speed: 2 sentences x 2 emotions (平静 100, 怒り 100) on the GPU (WebGPU) and on the CPU (WebAssembly) -> non-silent,
//     the time to make each and its RTF; plus one 「文の途中で変化」 run on each;
//  2. intonation panel: 「読みと高さを見る」, a sound raised by ＋ and a 下がる位置 changed -> 「直して読む」 differs from
//     「直す前」; then 怒り 100 while the panel is open -> it reads again by itself, the edits kept, 「直した後」 made again;
//  3. intonation dictionary: 田中 tried as でんちゅう, saved, another sentence holding 田中 read (shape A); 「直す」 from the
//     list, デ raised, saved again -> デ is higher about the word's mean in that sentence (shape B > A + 1 st); the list
//     shows a pitch sketch; the entry survives a reload; written to a file (contour kept), cleared, read back -> the same
//     entries; a foreign file is refused. The same dictionary entry read on the CPU follows the shape too;
//  4. network: the page sends nothing (no POST, no request other than the static files).
// Usage: pnpm build, pnpm serve (another terminal), then pnpm check [base url] [out.json] (default http://127.0.0.1:8080)
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://127.0.0.1:8080';
const outFile = process.argv[3];
const shots = outFile ? path.dirname(outFile) : '/tmp';
const TEXTS = ['本日はご来店いただき、ありがとうございます。', 'ご注文の品は、明日の午前中にお届けします。'];
const TEXT = '全力機械の田中です。ご予約の変更を承ります。';
const OTHER = '明日は田中が伺います。';
const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-gpu', '--autoplay-policy=no-user-gesture-required', '--mute-audio'] });
const result = { base, errors: [], requests: [] };
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, acceptDownloads: true });
const page = await ctx.newPage();
page.on('pageerror', (e) => result.errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') result.errors.push(`console: ${m.text()}`); });
page.on('request', (r) => {
  const u = new URL(r.url());
  const cdn = u.origin === 'https://tts-models.zenryoku.work' && /^\/v[\d.]+\/\w+\.ztm$/.test(u.pathname);   // the voice's .ztm (ztmUrl)
  if (r.method() !== 'GET' || (!cdn && (u.origin !== new URL(base).origin || !/^\/(demo|dist|models|cpu)\//.test(u.pathname)))) result.requests.push(`${r.method()} ${r.url()}`);
});
page.on('dialog', (d) => void d.accept());
await page.goto(`${base}/demo/`);
const set = (id, v) => page.$eval(`#${id}`, (el, x) => { el.value = String(x); el.dispatchEvent(new Event('input', { bubbles: true })); }, v);
const only = async (k, prefix = 'emo-') => { for (const e of ['neutral', 'happy', 'sad', 'angry', 'apologetic']) await set(`${prefix}${e}`, e === k ? 100 : 0); };
const backend = async (b) => { await page.check(`input[name=backend][value=${b}]`); };
const speak = async () => {
  const before = await page.evaluate(() => window.zenTtsDemo?.seq ?? 0);
  await page.click('#speak');
  await page.waitForFunction((b) => (window.zenTtsDemo?.seq ?? 0) > b || document.querySelector('#status')?.dataset.kind === 'error', before, { timeout: 180_000 });
  const r = await page.evaluate(() => ({ r: window.zenTtsDemo, status: document.querySelector('#status').textContent, kind: document.querySelector('#status').dataset.kind }));
  await page.click('#stop').catch(() => {});
  return r.kind === 'error' ? { error: r.status } : r.r;
};
const tuneWait = (key) => page.waitForFunction((k) => document.body.dataset[k] || document.querySelector('#tune-status')?.dataset.kind === 'error', key, { timeout: 180_000 });

// 1. speed and emotions, GPU then CPU (a warm-up run first on each: the first run compiles / warms up)
result.speed = [];
for (const b of ['webgpu', 'cpu']) {
  await backend(b);
  await page.click('input[name=mode][value=whole]');
  await only('neutral');
  await page.fill('#text', 'こんにちは。');
  await speak();
  for (const t of TEXTS) for (const e of ['neutral', 'angry']) {
    await page.fill('#text', t);
    await only(e);
    const r = await speak();
    result.speed.push({ backend: b, emotion: e, text: t, ...(r.error ? { error: r.error } : { got: r.backend, sec: r.sec, makeMs: r.makeMs, rtf: r.rtf, peak: r.peak, rms: r.rms, phones: r.phones }) });
    console.log(JSON.stringify(result.speed.at(-1)));
  }
  await page.check('#curve-mode');
  await only('neutral');
  await only('angry', 'emo-end-');
  await page.fill('#text', TEXTS[0]);
  const c = await speak();
  result.speed.push({ backend: b, emotion: 'neutral->angry (curve)', text: TEXTS[0], ...(c.error ? { error: c.error } : { got: c.backend, sec: c.sec, makeMs: c.makeMs, rtf: c.rtf, peak: c.peak, rms: c.rms }) });
  console.log(JSON.stringify(result.speed.at(-1)));
  await page.click('input[name=mode][value=whole]');
}
await page.screenshot({ path: path.join(shots, 'demo-1-speak.png'), fullPage: true });

// 2. intonation panel (on the GPU)
await backend('webgpu');
await only('neutral');
await page.fill('#text', TEXT);
await page.evaluate(() => { delete document.body.dataset.tuneAnalyzed; });
await page.click('#tune-analyze');
await tuneWait('tuneAnalyzed');
await page.click('#tune-sounds [data-u="1"]');
for (let i = 0; i < 4; i++) await page.click('#tune-selected [data-pitch="1"]');
const firstPhrase = page.locator('#tune-phrases select').first();
const opts = await firstPhrase.locator('option').count();
const curAccent = Number(await firstPhrase.inputValue());
await firstPhrase.selectOption(String(curAccent === 0 ? Math.min(2, opts - 1) : 0));
await page.click('#tune-make');
await tuneWait('tuneMade');
result.tune = { diff: await page.evaluate(() => window.zenTtsTune?.() ?? null), status: await page.textContent('#tune-status') };
await page.screenshot({ path: path.join(shots, 'demo-2-intonation.png'), fullPage: true });
const st0 = await page.evaluate(() => window.zenTtsTuneState());
await page.evaluate(() => { delete document.body.dataset.tuneAnalyzed; delete document.body.dataset.tuneMade; });
await only('angry');
await page.waitForFunction(() => (document.body.dataset.tuneAnalyzed && document.body.dataset.tuneMade) || document.querySelector('#tune-status')?.dataset.kind === 'error', null, { timeout: 180_000 });
const st1 = await page.evaluate(() => window.zenTtsTuneState());
result.emotionInPanel = {
  voiceChanged: st0.voiceKey !== st1.voiceKey, angry: st1.blend.angry,
  pitchMove: +Math.max(...st0.beforePitch.map((v, i) => (v === null || st1.beforePitch[i] === null ? 0 : Math.abs(v - st1.beforePitch[i])))).toFixed(2),
  editsKept: JSON.stringify(st0.edits) === JSON.stringify(st1.edits), remade: !!st1.afterPitch,
};
await only('neutral');
await page.waitForFunction((k) => window.zenTtsTuneState()?.voiceKey !== k, st1.voiceKey, { timeout: 180_000 });
await page.waitForFunction(() => !document.querySelector('#tune-make')?.disabled, null, { timeout: 180_000 });

// 3. intonation dictionary
const save = async () => {
  await page.evaluate(() => { delete document.body.dataset.dictSaved; });
  await page.click('#dict-save');
  await page.waitForFunction(() => document.body.dataset.dictSaved || document.querySelector('#tune-status')?.dataset.kind === 'error', null, { timeout: 180_000 });
  return page.evaluate(() => (document.body.dataset.dictSaved ? JSON.parse(document.body.dataset.dictSaved) : null));
};
const deRise = async () => {
  await page.fill('#text', OTHER);
  await page.evaluate(() => { delete document.body.dataset.tuneAnalyzed; });
  await page.click('#tune-analyze');
  await tuneWait('tuneAnalyzed');
  return page.evaluate(() => {
    const s = window.zenTtsTuneState();
    const i = s.kana.findIndex((k) => k.startsWith('デ'));
    const w = [];
    for (let k = i; k < s.kana.length && w.length < 3; k++) if (s.beforePitch[k] !== null) w.push(s.beforePitch[k]);
    return i < 0 ? null : { kana: s.kana.join(''), rise: +(s.beforePitch[i] - w.reduce((a, b) => a + b, 0) / w.length).toFixed(2) };
  });
};
await page.fill('#dict-surface', '田中');
await page.fill('#dict-reading', 'でんちゅう');
await page.evaluate(() => { delete document.body.dataset.tuneAnalyzed; });
await page.click('#dict-try');
await tuneWait('tuneAnalyzed');
const triedWord = await page.evaluate(() => window.zenTtsTuneState()?.word ?? null);
const plain = await save();
const shapeA = await deRise();
await page.evaluate(() => { delete document.body.dataset.tuneAnalyzed; });
await page.click('#dict-list li button[data-op="edit"]');
await tuneWait('tuneAnalyzed');
const idx = await page.evaluate(() => [...document.querySelectorAll('#tune-sounds [data-u]')].find((b) => b.querySelector('b')?.textContent?.startsWith('デ'))?.dataset.u ?? null);
if (idx !== null) { await page.click(`#tune-sounds [data-u="${idx}"]`); for (let i = 0; i < 6; i++) await page.click('#tune-selected [data-pitch="1"]'); }
const saved = await save();
const shapeB = await deRise();
// the same entry read on the CPU and in another emotion: the shape holds
await backend('cpu');
await page.waitForFunction((k) => window.zenTtsTuneState()?.voiceKey !== k, await page.evaluate(() => window.zenTtsTuneState().voiceKey), { timeout: 180_000 }).catch(() => {});
const shapeCpu = await deRise();
await only('angry');
const shapeCpuAngry = await deRise();
await only('neutral');
await backend('webgpu');
const sketch = await page.$$eval('#dict-list li', (ls) => ls.map((l) => ({ intonation: l.dataset.intonation, points: l.querySelectorAll('svg.sketch circle').length })));
await page.screenshot({ path: path.join(shots, 'demo-3-dictionary.png'), fullPage: true });
await page.reload();
const listed = await page.$$eval('#dict-list li', (ls) => ls.map((l) => l.textContent));
await page.fill('#text', TEXT);
await page.evaluate(() => { delete document.body.dataset.tuneAnalyzed; });
await page.click('#tune-analyze');
await tuneWait('tuneAnalyzed');
const afterReload = await page.textContent('#tune-reading');
const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#dict-export')]);
const exported = '/tmp/zen-tts-check-dict.json';
await dl.saveAs(exported);
const exportedText = fs.readFileSync(exported, 'utf8');
await page.evaluate(() => { localStorage.clear(); });
await page.reload();
const cleared = await page.$$eval('#dict-list li', (ls) => ls.length);
await page.setInputFiles('#dict-import', exported);
await page.waitForFunction(() => document.body.dataset.dictImported, null, { timeout: 20_000 });
const reimported = await page.evaluate(() => localStorage.getItem('zen-tts.dict.v2'));
fs.writeFileSync('/tmp/zen-tts-check-bad.json', '{"format":"other"}');
await page.setInputFiles('#dict-import', '/tmp/zen-tts-check-bad.json');
await page.waitForFunction(() => document.querySelector('#tune-status')?.dataset.kind === 'error', null, { timeout: 10_000 });
const badMsg = await page.textContent('#tune-status');
result.dict = { triedWord, plain, saved, shapeA, shapeB, shapeCpu, shapeCpuAngry, sketch, listed, afterReload, exported: JSON.parse(exportedText), cleared, roundTrip: JSON.stringify(JSON.parse(reimported)) === JSON.stringify(JSON.parse(exportedText)), badMsg };
await browser.close();

const sp = result.speed;
const speedOk = sp.length === 10 && sp.every((s) => !s.error && s.got === s.backend && s.sec > 0.5 && s.peak > 0.05 && s.rms > 0.005);
const tuneOk = result.tune.diff && result.tune.diff.maxPitchChange > 0.5 && result.tune.diff.meanAbsDiff > 0;
const e = result.emotionInPanel;
const panelOk = e.voiceChanged && e.angry > 0.99 && e.pitchMove > 0.3 && e.editsKept && e.remade;
const d = result.dict;
const dictOk = d.triedWord === '田中' && d.plain?.contour?.length === 4 && d.saved?.contour?.length === 4 && d.shapeA && d.shapeB && d.shapeB.rise - d.shapeA.rise > 1
  && d.shapeCpu && Math.abs(d.shapeCpu.rise - d.shapeB.rise) < 0.3 && d.shapeCpuAngry && d.shapeCpuAngry.rise - d.shapeA.rise > 1
  && d.sketch[0]?.intonation === '1' && d.sketch[0]?.points === 4 && d.listed.some((t) => t.includes('田中'))
  && /デンチュー|デンチュウ/.test((d.afterReload ?? '').replace(/[⌝｜]/g, '')) && d.exported.entries[0]?.contour?.length === 4 && d.cleared === 0 && d.roundTrip && /辞書のファイルではありません/.test(d.badMsg ?? '');
result.ok = result.errors.length === 0 && result.requests.length === 0 && speedOk && tuneOk && panelOk && dictOk;
result.parts = { speedOk, tuneOk, panelOk, dictOk };
console.log(JSON.stringify({ tune: result.tune.diff, emotionInPanel: e, dict: { shapeA: d.shapeA, shapeB: d.shapeB, shapeCpu: d.shapeCpu, shapeCpuAngry: d.shapeCpuAngry, sketch: d.sketch, roundTrip: d.roundTrip }, requests: result.requests, errors: result.errors, parts: result.parts }));
if (outFile) fs.writeFileSync(outFile, JSON.stringify(result, null, 1));
console.log(result.ok ? 'PASS' : 'FAIL');
process.exit(result.ok ? 0 : 1);
