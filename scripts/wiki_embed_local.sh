#!/bin/sh
# Embed the extracted wiki passages with the C engine, one process per core (sandbox helper).
# usage: sh scripts/wiki_embed_local.sh <ext_dir> [n_parts]
set -e
EXT="$1"; N="${2:-2}"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$EXT/emb"
i=0
while [ $i -lt $N ]; do
  P=$(printf "%02d" $i)
  KDR_THREADS=1 nice -n 5 "$ROOT/release/kdr-brain-lite" "$ROOT/release/brain.kdr" embed 160 < "$EXT/embed_part_$P" > "$EXT/emb/part_$P.f32" 2> "$EXT/emb/part_$P.log" &
  i=$((i+1))
done
wait
echo ALL_DONE
