# KDR Brain — a small neural brain that knows (and chats about) the Kingdom of Dutch Robloxia

[![build](https://github.com/dreamer2664/kdr-brain/actions/workflows/build.yml/badge.svg)](https://github.com/dreamer2664/kdr-brain/actions) · source: https://github.com/dreamer2664/kdr-brain · built artifacts: https://github.com/dreamer2664/kdr-brain/releases/latest

A fully self-contained, offline QA + chat system whose kingdom knowledge comes from exactly three sites:

| Source | What was crawled |
|---|---|
| https://the-kingdom-of-dutch-robloxia.fandom.com/ | all 34 articles via the MediaWiki API (wikitext + infoboxes + rank tables) |
| https://kdrwiki.netlify.app/ | the single-page "Official Wiki" (cards, fact strips, laws, songbook, rank table, bios) |
| https://dutchbloxia.netlify.app/ | the single-page "Royal Wiki" (cards, diplomacy table, laws, songbook, games) |

Crawled on 2026-09-05 → **537 knowledge passages** (`data/passages.json`, incl. ~15 curated summary passages that state in one plain sentence what the wikis only imply).

## Size budget

```
release/brain.kdr        58.4 MB   retriever + reader (2 BERT models, int8) + vocab + passage index + passages + BM25
release/kdr-brain         8.0 MB   ONE static Linux x86-64 binary: tokenizer, BERT engine, llama.cpp, HTTP server, web UI
                        -------
   knowledge core        66.4 MB   (< 80 MB; works alone = extractive answers with evidence, no chat)
release/composer.gguf   397.8 MB   Qwen2.5-0.5B-Instruct, Q4_K_M — turns facts into sentences, small talk, arithmetic
                        -------
   everything           464 MB     no Python, no libraries, nothing else needed at runtime

release/kdr-brain-lite    1.5 MB   same engine without llama.cpp (make lite) — brain.kdr + this = the original 60 MB brain
```

Why this composer: the scored candidates were Qwen2.5-0.5B (Q4_K_M 398 MB, IQ4_XS 349 MB, Q4_0 353 MB),
LFM2-350M (229 MB) and Qwen3-0.6B (397 MB). Q4_K_M Qwen2.5 scored best on both sets; the 50 MB-smaller
quantizations lost 3–5 answers each, LFM2 was the fastest but broke the chat rules (bare "(not from the wikis)" replies).
Q4_K_M keeps the 145 MB token-embedding table at Q8 — shrinking that is the next size lever if ever needed.

## Architecture

```
message ─► router (small talk? named kingdom entity? live-data question?)
        ├─ kingdom question ─► WordPiece tokenizer (exact HF BERT-uncased, in C)
        │     ─► Retriever  MiniLM-L6-H384 int8: cosine vs 537 passage vectors + BM25 → fused top-8
        │     ─► Reader     MiniLM-L12-H384 (SQuAD2) int8: best answer span + no-answer logit + confidence
        │     ─► Composer   Qwen2.5-0.5B-Instruct Q4 (llama.cpp): FACTS + short answer + chat history → 1–3 natural sentences
        └─ small talk / arithmetic / general knowledge ─► Composer alone; general answers get "(not from the wikis)" appended by the server
```

* Follow-ups work: "and his wife?" borrows the entities of the last two turns for retrieval; the last 6 turns are
  sent back to the composer.
* The system prompt (persona + cheat-sheet + rules) is pre-computed once and kept in the KV cache.
* Live-data questions (weather, time, news) are answered by a fixed friendly sentence — the small model cannot be
  trusted not to invent a forecast.
* Everything is deterministic (greedy decoding, repetition penalty 1.12, sentence de-duplication).

## Scores

`tests/kingdom.txt` (70 deep questions in 7 blocks of 10) and `tests/chat.txt` (40 small-talk / arithmetic /
instruction / general-knowledge items). A reply scores when it contains an accepted phrase; `!norefuse` fails refusals.

```
python3 scripts/score.py            # prints X/10 per block, failures, average latency
python3 scripts/score.py --strict   # CI gate: every block must be >= 9/10
python3 scripts/eval_c.py --strict  # the older extractive test: 75 questions, reader-only (>= 72 exact)
```

Latest run (2 vCPU sandbox): **kingdom 70/70, chat 39/40 — every block ≥ 9/10**, avg 2.8 s per reply
(details, model comparison and what moved the numbers: `docs/SCORES.md`). Add your own questions to the
`tests/*.txt` files — one line per question, `question | accepted 1 ; accepted 2`.

## Run it

```bash
./release/kdr-brain release/brain.kdr --chat release/composer.gguf serve 8080   # chat UI + API on http://0.0.0.0:8080
./release/kdr-brain release/brain.kdr --chat release/composer.gguf chat "What roles does Leonard II have?"
./release/kdr-brain release/brain.kdr serve 8080                                # without composer: extractive answers only
./release/kdr-brain release/brain.kdr ask "Who is the king?"                    # raw retrieval + reader JSON
curl -X POST localhost:8080/api/chat -d '{"q":"and his wife?","history":[{"role":"user","text":"who is the king?"},{"role":"assistant","text":"King Nicholas Goudhof V."}]}'
```

API: `POST /api/chat {q, history[], stream}` (SSE stream of `{"t":"token"}` events, then the full JSON), `GET /api/ask?q=`,
`GET /api/stats`. `KDR_THREADS=N` sets threads; `KDR_CHAT_MODEL=path` is equivalent to `--chat`.

## Rebuild from scratch

```bash
python3 scripts/crawl.py           # the 3 wikis -> data/raw/  (only network step; data/passages.json is the checked-in extraction)
python3 scripts/pack.py            # quantize + embed -> release/brain.kdr  (needs the two HF checkpoints in /tmp/build, see workflow)
sh scripts/build_llama.sh          # llama.cpp v0.4.0 static libs -> ~/.cache/kdr/llama  (PORTABLE=1 for x86-64-v3)
make                               # -> release/kdr-brain (one static binary)
curl -L -o release/composer.gguf https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
python3 scripts/score.py --strict  # quality gate
```

Only `src/`, `data/`, `scripts/`, `tests/`, `docs/` are versioned (1.5 MB). The build outputs are rebuilt by GitHub
Actions on every push (size gate + both quality gates) and published as a GitHub Release;
`KDR_REPO=owner/kdr-brain sh scripts/restore.sh` downloads them back (`KDR_LITE=1` for the 60 MB chat-less pair).
Setup: `docs/CLOUD_SETUP.md`. Debugging: `KDR_DEBUG_PROMPT=1` prints every prompt sent to the composer (and grounding retries), `KDR_REPEAT=1.05`
overrides the repetition penalty, `KDR_NO_VERIFY=1` disables the grounding retry, `KDR_THREADS=n` sets CPU threads.

## Honest limitations

* The composer is a 0.5B-parameter model: fluent and fact-faithful on retrieved facts, but its own general
  knowledge is shallow and occasionally wrong (hence the "not from the wikis" flag), and it can be pushed into
  nonsense by adversarial prompts.
* Knowledge is a snapshot; re-run the crawl + `pack.py` when the wikis change.
* Some wiki facts contradict each other (e.g. Bjorn is Minister of Defence on Fandom, Defence is vacant on
  dutchbloxia.netlify.app; the Pope's birth year is 1977 vs 1978). The brain answers with whichever passage matches
  best and shows its source under the reply.
* ~2–6 s per reply on 2 vCPUs (the composer generates ~34 tokens/s); streaming makes it feel faster.
