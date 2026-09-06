#!/bin/sh
# Embed the extracted wiki passages with the C engine, one single-threaded process per part, all in parallel.
#   sh scripts/wiki_embed_local.sh <ext_dir> <n_parts> [first_part] [count]
# Parts are the files embed_part_00.. written by wiki_prepare.py; output emb/part_XX.f32 (384 float32 per line).
# Resumable: an existing part_XX.f32 is truncated to whole rows and only the remaining lines are embedded.
# [first_part]/[count] run a sub-range (CI splits the 16 parts over 4 machines: first=4*k, count=4).
set -e
EXT="$1"; N="${2:-2}"; FIRST="${3:-0}"; COUNT="${4:-$N}"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$EXT/emb"
i=$FIRST; LAST=$((FIRST + COUNT))
while [ $i -lt $LAST ] && [ $i -lt $N ]; do
  P=$(printf "%02d" $i); IN="$EXT/embed_part_$P"; OUT="$EXT/emb/part_$P.f32"
  DONE=0
  if [ -s "$OUT" ]; then
    DONE=$(( $(stat -c %s "$OUT") / 1536 ))
    truncate -s $((DONE * 1536)) "$OUT"
  fi
  TOTAL=$(wc -l < "$IN")
  if [ "$DONE" -ge "$TOTAL" ]; then echo "part $P: already complete ($TOTAL rows)"; i=$((i+1)); continue; fi
  echo "part $P: embedding rows $DONE..$TOTAL"
  tail -n +$((DONE + 1)) "$IN" | KDR_THREADS=1 nice -n 5 "$ROOT/release/kdr-brain-lite" "$ROOT/release/brain.kdr" embed 160 >> "$OUT" 2> "$EXT/emb/part_$P.log" &
  i=$((i+1))
done
wait
echo ALL_DONE
