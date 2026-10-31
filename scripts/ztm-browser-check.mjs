// Browser check of the release's voices (headless Chromium, muted): for each voice, the demo page loads the voice's .ztm
// from the CDN (ztmUrl) and speaks, on the CPU (WebAssembly) and on WebGPU; the PCM of both backends is reported, and
// the voice files the page fetched (only the .ztm and the G2P). With LEGACY=<URL of the v1.0.0 models/ served by you>,
// the same sentence from the 4-file voice is compared too (same samples expected).
// Usage: pnpm build; pnpm serve (another terminal); node scripts/ztm-browser-check.mjs [base url] [voices=marine,hashiro,kai,kikka]
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://127.0.0.1:8080';
const voices = (process.argv[3] ?? 'marine,hashiro,kai,kikka').split(',');
const legacy = process.env.LEGACY;
const TEXT = 'ご注文の品は、明日の午前中にお届けします。';
const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-gpu', '--autoplay-policy=no-user-gesture-required', '--mute-audio'] });
const out = { base, legacy: legacy ?? null, runs: [], errors: [] };

async function run(backend, voice, fromLegacy) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const fetched = [];
  page.on('request', (r) => { const u = new URL(r.url()); if (/\.ztm$|\/g2p\/|\/(tts|decoder)\//.test(u.pathname)) fetched.push(`${u.host}${u.pathname}`); });
  page.on('pageerror', (e) => out.errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') out.errors.push(`console ${m.type()}: ${m.text()}`); });
  await page.goto(`${base}/demo/`);
  const r = await page.evaluate(async ({ backend, voice, legacy, text }) => {
    const { ZenTts, ztmUrl } = await import('../dist/index.js');
    const t0 = performance.now();
    const t = await ZenTts.load({ modelsUrl: legacy ?? '../models', cpuWasmUrl: '../cpu/tts-cpu.wasm', backend, ...(legacy ? { voice } : { model: ztmUrl(voice) }) });
    const loadMs = Math.round(performance.now() - t0);
    const s = await t.speak(text, { emotion: { happy: 60, neutral: 40 } });
    const d = await crypto.subtle.digest('SHA-256', s.audio.buffer.slice(s.audio.byteOffset, s.audio.byteOffset + s.audio.byteLength));
    const hex = [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, '0')).join('').slice(0, 16);
    let peak = 0; for (const v of s.audio) peak = Math.max(peak, Math.abs(v));
    const res = { backend: t.backend, voice: t.voice, loadMs, samples: s.audio.length, sha: hex, peak: +peak.toFixed(3) };
    t.destroy();
    return res;
  }, { backend, voice, legacy: fromLegacy ? legacy : null, text: TEXT });
  out.runs.push({ want: backend, legacy: fromLegacy, ...r, fetched: [...new Set(fetched)] });
  await ctx.close();
}

for (const v of voices) for (const b of ['cpu', 'webgpu']) { await run(b, v, false); if (legacy) await run(b, v, true); }
// the demo page's own 「話す」 (its engine loads the radio's voice from the CDN)
{
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  page.on('pageerror', (e) => out.errors.push(String(e)));
  await page.goto(`${base}/demo/`);
  await page.check('input[name=voice][value=kikka]');
  await page.click('#speak');
  await page.waitForFunction(() => window.zenTtsDemo || document.querySelector('#status')?.dataset.kind === 'error', null, { timeout: 120_000 });
  out.demo = await page.evaluate(() => ({ r: window.zenTtsDemo && { voice: window.zenTtsDemo.voice, sec: window.zenTtsDemo.sec, backend: window.zenTtsDemo.backend }, status: document.querySelector('#status')?.textContent }));
  await page.click('#stop').catch(() => {});
  await ctx.close();
}
await browser.close();
let ok = out.errors.length === 0 && out.runs.every((x) => x.samples > 0 && x.backend === x.want && x.peak > 0.05);
if (legacy) for (const v of voices) for (const b of ['cpu', 'webgpu']) {
  const [z, l] = [out.runs.find((x) => x.voice === v && x.want === b && !x.legacy), out.runs.find((x) => x.voice === v && x.want === b && x.legacy)];
  const same = !!(z && l && z.sha === l.sha && z.samples === l.samples);
  out[`${v}/${b}Same`] = same; ok &&= same;
}
out.ok = ok && out.demo?.r?.voice === 'kikka';
console.log(JSON.stringify(out, null, 1));
process.exit(out.ok ? 0 : 1);
