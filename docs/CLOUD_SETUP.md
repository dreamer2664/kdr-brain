# Free permanent storage for kdr-brain — setup tutorial

> **Status: ✅ done on 2026-09-05.** Repo <https://github.com/dreamer2664/kdr-brain> (public) ·
> artifacts <https://github.com/dreamer2664/kdr-brain/releases/latest> · CI <https://github.com/dreamer2664/kdr-brain/actions>.
> First CI build passed all gates (59,995,660 B total, 75/75 exact answers) and produced a byte-identical `brain.kdr`.
> The token lives only in the git-ignored file `.kdr-secrets/github.env`; the scripts load it automatically, so
> from now on a rebuild + publish is just `sh scripts/push_to_github.sh`. Token expires 2027 → redo Step 3 only.

## Which service fits best: **GitHub** (free plan)

| Need | GitHub Free | Why the alternatives lose |
|---|---|---|
| Keep the 1.5 MB source (C engine, scripts, crawled wiki data) safe, with history | unlimited public **and** private repos | Google Drive / Dropbox have no `git`, no history, and their APIs need OAuth browser dances |
| Store the built `brain.kdr` (58 MB) + binary (8 MB) + composer model (398 MB) | **Release assets: up to 2 GB per file**, unlimited count, direct download links, no token needed to fetch from a public repo | Git itself blocks files > 100 MB and Git LFS is capped at 1 GB/month bandwidth — releases avoid both limits |
| Let the assistant rebuild everything by itself | **GitHub Actions: unlimited free minutes on public repos** (2,000 min/month if private). One push = compile + quantize + test + publish, ~5 min | Hugging Face Hub is great for the model file, but has no free compute for the C build + test gate |
| Zero cost, forever | yes — no card, no trial, no expiry | S3/GCS free tiers expire after 12 months or need a credit card |

Runner-up: **Hugging Face Hub** (free, unlimited public model repos, files up to 50 GB). Fine if you only
want to park `brain.kdr` somewhere; it can't run the build. Everything below uses GitHub only.

### The one thing only you can do
Creating an account means agreeing to Terms of Service, verifying an e-mail address and passing a
CAPTCHA — that must be a human, so I can't do it for you. It is a **one-time, ~5 minute** job.
After that you hand me a single token that is limited to the one repository, and everything else
(commits, pushes, releases, rebuilds, restores after a sandbox wipe) is done by me with no clicks from you.

---

## Part 1 — you: create the account, the repo and the token (≈ 5 min)

### Step 1 · GitHub account (skip if you have one)
1. Open <https://github.com/signup>.
2. Enter e-mail, password, username, solve the puzzle, enter the code from the e-mail.
3. Choose the **Free** plan when asked (it's the default; never enter a card).

### Step 2 · Create an empty repository
1. Go to <https://github.com/new>.
2. **Repository name:** `kdr-brain` (keep this exact name, the scripts default to it).
3. **Public** (recommended — unlimited free Actions minutes, and the release files can be downloaded
   by anyone with a plain URL, which is what makes restores trivial). Private also works: you'd get
   2,000 build-minutes/month — about 400 rebuilds — which is still plenty.
4. Leave **all** "Initialize this repository with…" boxes **unchecked** (no README, no .gitignore,
   no license). The repo must be empty so the first push goes through cleanly.
5. Click **Create repository**. You land on a page that says "Quick setup" — nothing to do there.

### Step 3 · Create a fine-grained token that can *only* touch this repo
1. Click your avatar (top right) → **Settings** → scroll the left menu to the bottom →
   **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
   Direct link: <https://github.com/settings/personal-access-tokens/new>
2. Fill in:
   * **Token name:** `kdr-brain sandbox`
   * **Expiration:** `Custom` → one year from today (the maximum GitHub allows is 366 days; you'll just
     issue a new one next year — I'll tell you when a push fails with 401).
   * **Resource owner:** your own username.
   * **Repository access:** **Only select repositories** → pick `kdr-brain`.
   * **Permissions → Repository permissions** — set exactly these two, leave everything else "No access":
     * **Contents** → **Read and write** (push code, create releases, upload assets)
     * **Workflows** → **Read and write** (needed once, to push the `.github/workflows/build.yml` file)
   * Under **Account permissions** change nothing.
3. Click **Generate token**, then **copy the token** (it starts with `github_pat_`). It is shown only once.

Why this is safe: the token can't see any other repo, can't read your e-mail, can't change settings,
can't delete the repo, expires by itself, and you can kill it any time at
<https://github.com/settings/personal-access-tokens> → *Revoke*.

### Step 4 · Hand it over
Paste this in the chat (fill in the two values):

```
GH_OWNER=<your github username>
GH_TOKEN=github_pat_....................
```

That's the end of your part.

---

## Part 2 — me: what happens next, fully automated

Everything is already prepared in this workspace and dry-run tested; the only missing input is the token.

1. **Push the source** (already committed locally, 1.5 MB, 24 files):
   ```sh
   GH_OWNER=… GH_TOKEN=… sh scripts/push_to_github.sh
   ```
   The script first checks the token against the API (clear message on 401/404), then pushes with a
   one-shot credential helper, so the token is **never written to disk** — not in `.git/config`, not in
   the remote URL, not in shell history.
2. **Upload today's build immediately** (so nothing depends on CI working):
   ```sh
   GH_OWNER=… GH_TOKEN=… sh scripts/upload_release.sh
   ```
   → Release `latest` with three assets: `brain.kdr` (58 MB), `kdr-brain` (8 MB static binary) and
   `composer.gguf` (398 MB, Qwen2.5-0.5B-Instruct Q4_K_M — the sentence-forming chat model).
3. **GitHub Actions takes over** — the pushed `.github/workflows/build.yml` runs on every future push:
   download the two public fp32 checkpoints + the composer GGUF from Hugging Face (all cached after the
   first run) → `scripts/build_llama.sh` (llama.cpp static libs, cached) → `make` (one static binary) →
   `scripts/pack.py` (int8 quantize + embed 537 passages) → **size gate** (knowledge core < 80 MB, everything
   < 500 MB) → **quality gates** (`scripts/eval_c.py --strict` ≥ 72/75 exact answers, then
   `scripts/score.py --strict` = every block of 10 questions ≥ 9/10, composer included) → re-publish `latest`.
   You can watch it at `https://github.com/<you>/kdr-brain/actions`; a red ✗ means the release is *not*
   overwritten, so the last good build always stays downloadable.
4. **Restoring after a sandbox wipe** needs no token at all (public repo):
   ```sh
   git clone https://github.com/<you>/kdr-brain && cd kdr-brain
   KDR_REPO=<you>/kdr-brain sh scripts/restore.sh      # downloads the three assets, runs a smoke question
   ./release/kdr-brain release/brain.kdr --chat release/composer.gguf serve 8080
   ```

### Day-to-day from then on
* You ask for a change → I edit, rebuild, test locally, `git commit`, push. ~1 minute of my time, 0 of yours.
* Re-crawl when the wikis change: `python3 scripts/crawl.py` → re-extract → push → CI rebuilds the brain.
* Token expired (401 on push) → repeat only Step 3 (2 minutes).

### Limits worth knowing (GitHub Free, public repo)
* 100 MB hard limit per file *inside* git — which is why the 58 MB brain and the 398 MB composer are release
  assets, not committed files (they would also bloat history on every rebuild). Repos should stay under ~1 GB; ours is 1.6 MB.
* Release assets: 2 GB per file, no total cap published. Actions: unlimited minutes, 6 h per job, 10 GB cache.
* If the repo is private instead: 2,000 Actions minutes and 500 MB of artifact/package storage per month.

### Alternatives, in case you'd rather not use GitHub
| Service | Free tier | Fit |
|---|---|---|
| Hugging Face Hub | public model repos, 50 GB/file, token via *Settings → Access Tokens* (write scope, one repo) | best "second copy" of `brain.kdr`; no compute for rebuilds |
| GitLab.com | 5 GB storage, 400 CI min/month, releases up to 5 GB | works, same setup pattern, but 400 min ≈ 80 rebuilds/month |
| Cloudflare R2 | 10 GB, no egress fees, no expiry | needs a credit card on file even at $0 |
| Google Drive | 15 GB | OAuth consent screen + refresh tokens — much more setup for you, no versioning |
