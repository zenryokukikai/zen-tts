#!/bin/sh
# Builds a release .ztm of one voice from the previous release's .ztm, a new decoder export and this repository's
# models/g2p (the G2P the runtime ships; the .ztm records its SHA-256):
#   sh scripts/build-release-ztm.sh <old.ztm> <decoder dir: nsf.json + nsf.bin> <out.ztm> [created ISO]
# nsf.json keeps only the runtime's keys. The acoustic section is copied unchanged (checked); voice and styles come from the old .ztm (unpack writes them
# to ztm.json), voice.model is this release's version of the voice (src/version.ts VOICES), the decoder and the G2P SHA-256 are the new ones. Then the new .ztm is unpacked again and checked to give
# back the same decoder and acoustic bytes.
set -eu
OLD=$1; DEC=$2; OUT=$3; CREATED=${4:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
node "$ROOT/scripts/pack-ztm.mjs" unpack "$OLD" -o "$W/u" --g2p "$ROOT/models/g2p" >/dev/null 2>"$W/warn" || { cat "$W/warn"; exit 1; }
cp "$DEC/nsf.bin" "$W/u/decoder/nsf.bin"
# nsf.json: only the keys the runtime reads (a training export may carry internal notes such as checkpoint paths)
node -e '
const fs = require("fs"), j = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const keep = ["dtype", "params", "logF0Mean", "logF0Std", "outputGain", "noiseBranch", "noiseMaxMag", "tensors"];
const dropped = Object.keys(j).filter((k) => !keep.includes(k));
if (dropped.length) console.error("nsf.json: dropped " + dropped.join(", "));
fs.writeFileSync(process.argv[2], JSON.stringify(Object.fromEntries(keep.filter((k) => k in j).map((k) => [k, j[k]]))));' "$DEC/nsf.json" "$W/u/decoder/nsf.json"
# voice.model: the voice package version of this release, from src/version.ts (VOICES[id].model, e.g. marine-1.1)
MODEL=$(node --experimental-strip-types --no-warnings -e '
import("'"$ROOT"'/src/version.ts").then((m) => {
  const id = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).voice.id;
  if (!Object.prototype.hasOwnProperty.call(m.VOICES, id)) { console.error("no VOICES entry for " + id); process.exit(1); }
  console.log(m.VOICES[id].model);
});' "$W/u/ztm.json")
node -e '
const fs = require("fs"), f = process.argv[1], c = process.argv[2], model = process.argv[3];
const z = JSON.parse(fs.readFileSync(f, "utf8"));
delete z.source;                       // pack records this build (packer, from, the files SHA-256)
if (c) z.created = c;
z.voice = { ...z.voice, model };
fs.writeFileSync(f, JSON.stringify(z, null, 1));' "$W/u/ztm.json" "$CREATED" "$MODEL"
node "$ROOT/scripts/pack-ztm.mjs" pack "$W/u" -o "$OUT"
node "$ROOT/scripts/pack-ztm.mjs" unpack "$OUT" -o "$W/v" --g2p "$ROOT/models/g2p" >/dev/null
cmp "$W/v/decoder/nsf.bin" "$DEC/nsf.bin"
cmp "$W/v/tts/own.bin" "$W/u/tts/own.bin"
node "$ROOT/scripts/pack-ztm.mjs" info "$OUT" | grep -q "\"model\": \"$MODEL\"" || { echo "voice.model is not $MODEL in $OUT"; exit 1; }
echo "ok $OUT $MODEL $(shasum -a 256 "$OUT" | cut -c1-16)"
