// "acceptable" = near (ー/ッ/ヴ) or one small substitution (edit distance <= 1 on kana, or the same after mapping ァ-ォ etc)
import fs from 'node:fs';
import { arpabetToKatakana } from './arpabet-katakana.mjs';
const cmu=new Map(); for (const l of fs.readFileSync(process.argv[2],'utf8').split('\n')){const [w0,...ph]=l.split(' '); if(!w0)continue; const w=w0.replace(/\(\d+\)$/,''); if(!cmu.has(w))cmu.set(w,[]); cmu.get(w).push(ph.filter(p=>p&&!p.startsWith('#')));}
const merged=new Map();
for (const f of ['neologd.tsv','curated.tsv']) for (const l of fs.readFileSync(new URL('../../data/latin-words/'+f, import.meta.url),'utf8').split('\n')){ if(!l||l[0]==='#')continue; const [w,k]=l.split('\t'); if(/^[a-z]+$/.test(w)&&cmu.has(w)) merged.set(w,k);}
const loose=(s)=>s.replace(/[ーッ]/g,'');
const ed=(a,b)=>{a=[...a];b=[...b];const d=Array.from({length:a.length+1},(_,i)=>[i,...Array(b.length).fill(0)]);for(let j=1;j<=b.length;j++)d[0][j]=j;for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++)d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]===b[j-1]?0:1));return d[a.length][b.length];};
let ex=0,near=0,one=0,bad=0; const badL=[];
for (const [w,k] of merged){ const outs=cmu.get(w).map(p=>arpabetToKatakana(p,w)); if(outs.includes(k)){ex++;continue;} if(outs.some(o=>loose(o)===loose(k))){near++;continue;} if(outs.some(o=>ed(loose(o),loose(k))<=1)){one++;continue;} bad++; badL.push([w,k,outs[0]]); }
const N=merged.size; console.log({N,exact:ex,near,oneEdit:one,bad,pct:{exact:+(100*ex/N).toFixed(1),acceptable:+(100*(ex+near+one)/N).toFixed(1)}});
const step=Math.max(1,Math.floor(badL.length/(+process.argv[3]||30))); for(let i=0;i<badL.length;i+=step) console.log(badL[i].join('\t'));
