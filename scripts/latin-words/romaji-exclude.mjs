import fs from 'node:fs';
const { romajiToHiragana } = await import(new URL('../../dist/runtime/latin-reading.js', import.meta.url));
const kata = (h)=>[...h].map(c=>c>='ぁ'&&c<='ゖ'?String.fromCharCode(c.charCodeAt(0)+0x60):c).join('');
const cmu=new Map(); for (const l of fs.readFileSync('cmudict.dict','utf8').split('\n')) { const [w0,...ph]=l.split(' '); const w=w0.replace(/\(\d+\)$/,''); if (/^[a-z]+$/.test(w)) { if(!cmu.has(w)) cmu.set(w,[]); cmu.get(w).push(ph); } }
const by = new Map();
for (const l of fs.readFileSync('latin.tsv','utf8').split('\n')) { const p=l.split('\t'); if (p.length<2||!/^[A-Za-z]+$/.test(p[0])) continue; const k=p[0].toLowerCase(); if(!by.has(k)) by.set(k,new Set()); by.get(k).add(p[1]); }
const VMAP={AA:'a',AE:'a',AH:'*',AO:'o',AW:'a',AY:'a',EH:'e',ER:'a',EY:'e',IH:'i',IY:'i',OW:'o',OY:'o',UH:'u',UW:'u'};
const kanaV=(h)=>{ // vowels of a hiragana string, small ya/yu/yo replace the i
  const V={あ:'a',い:'i',う:'u',え:'e',お:'o'}; const out=[];
  const row={a:'かがさざただなはばぱまやらわぁゃ',i:'きぎしじちぢにひびぴみりぃ',u:'くぐすずつづぬふぶぷむゆるぅゅ',e:'けげせぜてでねへべぺめれぇ',o:'こごそぞとどのほぼぽもよろをぉょ'};
  for (const c of h) { if (V[c]) { out.push(V[c]); continue; } if ('ゃゅょぁぃぅぇぉ'.includes(c)) { out.pop(); } for (const [v,s] of Object.entries(row)) if (s.includes(c)) out.push(v); }
  return out;
};
const match=(a,b)=>a.length===b.length && a.every((x,i)=>x===b[i]||x==='*');
// CMU vowels -> every romaji vowel sequence it can be written as (EY = e or ei, OW = o or ou / oo, AY = ai, ER = a / aa)
const ALT={EY:[['e'],['e','i']],OW:[['o'],['o','u'],['o','o']],AY:[['a','i']],AW:[['a','u']],OY:[['o','i']],ER:[['a'],['a','a']],IY:[['i'],['i','i']],UW:[['u'],['u','u']]};
const expand=(ph)=>{let seqs=[[]]; for (const p of ph){const b=p.replace(/\d/,''); const alts=ALT[b]||[[VMAP[b]]]; const n=[]; for(const s of seqs) for(const a of alts) n.push([...s,...a]); seqs=n.slice(0,64);} return seqs;};
const block=[]; let allow=0;
for (const [w,prons] of cmu) {
  const r=romajiToHiragana(w); if (!r) continue;
  const jp = [...(by.get(w)||[])].some(x=>x===kata(r) || x.replace(/ー/g,'').replace(/ウ/g,'')===kata(r).replace(/ウ/g,''));
  const rv=kanaV(r);
  const like = prons.some(ph=>expand(ph.filter(p=>/\d/.test(p))).some(seq=>match(seq, rv)));
  if (jp || like) { allow++; continue; }
  block.push(w);
}
console.log({block:block.length, allow});
for (const w of ['sensei','geisha','sake','sayonara','kobe','nose','again','abandon','tanaka','adachi','sushi','tokyo','kobe','osaka','made','game','same','take','time','pine','kimono','karate','tsunami','ramen','arigatou','sake','sumo','hello','mike','ride']) console.log(w, block.includes(w)?'BLOCK':'allow');
fs.writeFileSync('romaji_block.txt', block.sort().join('\n')+'\n');
