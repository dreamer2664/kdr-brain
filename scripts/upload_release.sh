#!/bin/sh
# Upload the LOCAL build outputs (release/brain.kdr, kdr-brain, kdr-brain-lite, composer.gguf) straight to a GitHub Release,
# without waiting for GitHub Actions. Same token as push_to_github.sh (Contents: read/write).
#   GH_OWNER=yourname GH_TOKEN=github_pat_xxx sh scripts/upload_release.sh
set -eu
# token file written once during setup (git-ignored); env vars given on the command line win
[ -f "$(dirname "$0")/../.kdr-secrets/github.env" ] && { set -a; . "$(dirname "$0")/../.kdr-secrets/github.env"; set +a; }
: "${GH_OWNER:?set GH_OWNER}"; : "${GH_TOKEN:?set GH_TOKEN}"
GH_REPO="${GH_REPO:-kdr-brain}"; TAG="${TAG:-latest}"
API="https://api.github.com/repos/$GH_OWNER/$GH_REPO"
cd "$(dirname "$0")/.."
auth() { curl -sS -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" "$@"; }
jget() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(eval(sys.argv[1], {"d": d}))' "$1"; }

for f in brain.kdr kdr-brain kdr-brain-lite composer.gguf; do [ -s "release/$f" ] || { echo "release/$f missing - run make / pack.py first"; exit 1; }; done
code=$(auth -o /dev/null -w '%{http_code}' "$API")
[ "$code" = 200 ] || { echo "ERROR $code: token invalid/expired (401) or repo $GH_OWNER/$GH_REPO not accessible (404)"; exit 1; }

rel=$(auth "$API/releases/tags/$TAG")
id=$(echo "$rel" | jget 'd.get("id","")')
if [ -z "$id" ]; then
  echo "creating release '$TAG' ..."
  rel=$(auth -X POST "$API/releases" -d "{\"tag_name\":\"$TAG\",\"name\":\"kdr-brain $TAG\",\"body\":\"Uploaded from the sandbox on $(date -u +%F). Assets: brain.kdr (retriever+reader+knowledge), kdr-brain (static Linux x86-64 binary with chat), kdr-brain-lite (extractive only), composer.gguf (Qwen2.5-0.5B-Instruct Q4_K_M).\"}")
  id=$(echo "$rel" | jget 'd["id"]')
fi
for f in brain.kdr kdr-brain kdr-brain-lite composer.gguf; do
  aid=$(echo "$rel" | jget "next((a['id'] for a in d.get('assets',[]) if a['name']=='$f'), '')")
  [ -n "$aid" ] && auth -X DELETE "$API/releases/assets/$aid"
  echo "uploading $f ($(stat -L -c %s release/$f) bytes) ..."
  auth -H "Content-Type: application/octet-stream" --data-binary "@release/$f" \
       "https://uploads.github.com/repos/$GH_OWNER/$GH_REPO/releases/$id/assets?name=$f" \
    | jget '"  ok: %s  %d bytes  %s" % (d["name"], d["size"], d["browser_download_url"])'
done
