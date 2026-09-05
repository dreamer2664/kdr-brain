#!/bin/sh
# Fetch the built artifacts (release/brain.kdr + release/kdr-brain) from the latest GitHub Release.
# No token needed for a public repository.
#   KDR_REPO=yourname/kdr-brain sh scripts/restore.sh
set -eu
REPO="${KDR_REPO:?set KDR_REPO=owner/repo, e.g. KDR_REPO=alice/kdr-brain}"
cd "$(dirname "$0")/.."
mkdir -p release
for f in brain.kdr kdr-brain; do
  echo "downloading $f ..."
  curl -fL --retry 3 -o "release/$f" "https://github.com/$REPO/releases/latest/download/$f"
done
chmod +x release/kdr-brain
ls -la release
./release/kdr-brain release/brain.kdr ask "Who is the king of Dutch Robloxia?"
