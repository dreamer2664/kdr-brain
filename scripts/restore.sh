#!/bin/sh
# Fetch the built artifacts from the latest GitHub Release (no token needed for a public repository).
#   KDR_REPO=yourname/kdr-brain sh scripts/restore.sh            # brain.kdr + kdr-brain + composer.gguf (+ wiki.kdw when published), ~545 MB
#   KDR_REPO=yourname/kdr-brain KDR_LITE=1 sh scripts/restore.sh # brain.kdr + kdr-brain-lite only (extractive answers, 60 MB)
set -eu
REPO="${KDR_REPO:?set KDR_REPO=owner/repo, e.g. KDR_REPO=alice/kdr-brain}"
cd "$(dirname "$0")/.."
mkdir -p release
if [ "${KDR_LITE:-0}" = "1" ]; then FILES="brain.kdr kdr-brain-lite"; else FILES="brain.kdr kdr-brain composer.gguf"; fi
for f in $FILES; do
  echo "downloading $f ..."
  curl -fL --retry 3 -o "release/$f" "https://github.com/$REPO/releases/latest/download/$f"
done
# general-knowledge pack (optional: present once the wiki-pack workflow has run; KDR_NO_WIKI=1 skips it)
if [ "${KDR_NO_WIKI:-0}" != "1" ]; then
  echo "downloading wiki.kdw (optional) ..."
  curl -fL --retry 3 -o release/wiki.kdw "https://github.com/$REPO/releases/latest/download/wiki.kdw" \
    || curl -fL --retry 3 -o release/wiki.kdw "https://github.com/$REPO/releases/download/wiki/wiki.kdw" \
    || { rm -f release/wiki.kdw; echo "no wiki pack published yet - kingdom knowledge only"; }
fi
WIKI=""; [ -s release/wiki.kdw ] && WIKI="--wiki release/wiki.kdw"
chmod +x release/kdr-brain release/kdr-brain-lite 2>/dev/null || true
ls -la release
if [ "${KDR_LITE:-0}" = "1" ]; then
  ./release/kdr-brain-lite release/brain.kdr $WIKI ask "Who is the king of Dutch Robloxia?"
else
  ./release/kdr-brain release/brain.kdr --chat release/composer.gguf $WIKI chat "Who is the king of Dutch Robloxia?"
fi
