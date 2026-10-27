// IEEE half floats: HALF[bits] = the float value, toHalf(v) = round to nearest even half bits. DOM-free.
export const HALF = ((): Float32Array => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

/** float -> half bits (round to nearest even; packLinear's fp16 weights). */
export function toHalf(v: number): number {
  const f = new Float32Array([v]);
  const x = new Uint32Array(f.buffer)[0]!;
  const sign = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const s = 14 - e;
    return sign | ((m + (1 << (s - 1))) >>> s);
  }
  if (e >= 31) return sign | 0x7c00;
  const h = sign | (e << 10) | (m >>> 13);
  return (m & 0x1000) ? h + 1 : h;
}
