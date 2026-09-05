#!/bin/sh
# Push this project to GitHub with a fine-grained personal access token.
# The token is only read from the environment; it is never written to disk.
#
#   GH_OWNER=yourname GH_TOKEN=github_pat_xxx sh scripts/push_to_github.sh
#   optional: GH_REPO=kdr-brain (default)  GIT_NAME / GIT_EMAIL for the commit author
set -eu
: "${GH_OWNER:?set GH_OWNER=<your GitHub username>}"
: "${GH_TOKEN:?set GH_TOKEN=<fine-grained token with Contents + Workflows read/write>}"
GH_REPO="${GH_REPO:-kdr-brain}"
cd "$(dirname "$0")/.."

# 1) sanity-check the token against the repo before touching git
code=$(curl -s -o /tmp/gh_repo.json -w '%{http_code}' -H "Authorization: Bearer $GH_TOKEN" \
       -H "X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/$GH_OWNER/$GH_REPO")
case "$code" in
  200) echo "token OK: repo $GH_OWNER/$GH_REPO found ($(grep -o '"private": *[a-z]*' /tmp/gh_repo.json | head -1))" ;;
  401) echo "ERROR 401: token invalid or expired"; exit 1 ;;
  404) echo "ERROR 404: repo $GH_OWNER/$GH_REPO not found, or the token was not given access to it"; exit 1 ;;
  *)   echo "ERROR $code from api.github.com"; cat /tmp/gh_repo.json; exit 1 ;;
esac
rm -f /tmp/gh_repo.json

# 2) local repo settings (.git/config is not kept between sandbox sessions, so set them every time)
[ -d .git ] || git init -q -b main
git config core.fileMode false
git config user.name  "${GIT_NAME:-kdr-brain}"
git config user.email "${GIT_EMAIL:-kdr-brain@users.noreply.github.com}"
git add -A
git diff --cached --quiet || git commit -q -m "${COMMIT_MSG:-update $(date -u +%Y-%m-%dT%H:%MZ)}"
git branch -M main

# 3) push (token passed inline, remote is not stored)
git push -u "https://x-access-token:${GH_TOKEN}@github.com/${GH_OWNER}/${GH_REPO}.git" main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${GH_OWNER}/${GH_REPO}.git"   # token-free URL for reading

echo
echo "pushed. repo:     https://github.com/${GH_OWNER}/${GH_REPO}"
echo "        CI build: https://github.com/${GH_OWNER}/${GH_REPO}/actions"
echo "        release:  https://github.com/${GH_OWNER}/${GH_REPO}/releases/latest  (ready in ~5 min)"
