#!/bin/sh
# Fetch the built artifacts from the latest GitHub Release (no token needed for a public repository).
#   KDR_REPO=yourname/kdr-brain sh scripts/restore.sh            # brain.kdr + kdr-brain + composer.gguf (chat, ~465 MB)
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
chmod +x release/kdr-brain release/kdr-brain-lite 2>/dev/null || true
ls -la release
if [ "${KDR_LITE:-0}" = "1" ]; then
  ./release/kdr-brain-lite release/brain.kdr ask "Who is the king of Dutch Robloxia?"
else
  ./release/kdr-brain release/brain.kdr --chat release/composer.gguf chat "Who is the king of Dutch Robloxia?"
fi
