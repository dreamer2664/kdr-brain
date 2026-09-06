# KDR Brain — a small neural brain that knows (and chats about) the Kingdom of Dutch Robloxia — plus general knowledge from Wikipedia

[![build](https://github.com/dreamer2664/kdr-brain/actions/workflows/build.yml/badge.svg)](https://github.com/dreamer2664/kdr-brain/actions) · source: https://github.com/dreamer2664/kdr-brain · built artifacts: https://github.com/dreamer2664/kdr-brain/releases/latest

A fully self-contained, offline QA + chat system whose kingdom knowledge comes from exactly three sites:

| Source | What was crawled |
|---|---|
| https://the-kingdom-of-dutch-robloxia.fandom.com/ | all 34 articles via the MediaWiki API (wikitext + infoboxes + rank tables) |
| https://kdrwiki.netlify.app/ | the single-page "Official Wiki" (cards, fact strips, laws, songbook, rank table, bios) |
| https://dutchbloxia.netlify.app/ | the single-page "Royal Wiki" (cards, diplomacy table, laws, songbook, games) |

Crawled on 2026-09-05 → **537 knowledge passages** (`data/passages.json`, incl. ~15 curated summary passages that state in one plain sentence what the wikis only imply).

General knowledge comes from a second, optional pack, `release/wiki.kdw`, built from the Kiwix dump
`wikipedia_en_top_mini` (the ~50,000 most-read English Wikipedia articles, introductions only, 316 MB ZIM) →
**215,206 passages** compressed, embedded and indexed (~80 MB). Answers from it are marked "(from Wikipedia)" and show
the article as source; questions about the kingdom still go to the kingdom index (see Routing).

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
release/wiki.kdw        ~80 MB    OPTIONAL general-knowledge pack: 215k Wikipedia passages (zstd, 27 MB) + 128-d int8 PCA
                                  embeddings (28 MB) + varint BM25 postings (20 MB) + titles/tables (5 MB)
```

The wiki pack is deliberately lean: passage vectors are PCA-reduced from 384 to 128 dims (85% of the variance;
192 and 384 dims scored the same on the test set), quantised to int8 per row, the text is zstd-19 with a trained
dictionary (ratio 0.32), and BM25 postings are delta-varint with words appearing in > 5% of passages dropped.

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
        ├─ no kingdom entity ─► kingdom reader first (wins only with a clear margin), else
        │     ─► Wikipedia pack: dense (int8 PCA-128) + BM25 → top 8 passages (≤ 2 per article)
        │     ─► Reader on each passage separately ("Title: text"), scored by margin + retrieval + question-term coverage
        │        + agreement voting (+ a sentence-level re-read when no passage answers) → "(from Wikipedia)" + article link
        └─ small talk / arithmetic / anything neither index answers ─► Composer alone; flagged "(not from the wikis)"
```

Routing between the two indexes: a question with a kingdom entity word (a title word of the 537 passages that is not an
ordinary English word, or a "life" word like king/army/laws without a real-world cue) goes to the kingdom index. Without
one, the kingdom reader runs first and wins only when it clearly prefers a span (margin > 8, or its passage title is in the
question); otherwise the wiki pack answers (confidence ≥ 0.45 and span ≥ no-answer, or a definitional match such as
"Mercury" from [Mercury (element)]). So "Who is the king?" → Nicholas Goudhof V, "Who was the king of Spain in 1900?" → Alfonso XIII.

* Follow-ups work: "and his wife?" borrows the entities of the last two turns for retrieval; the last 6 turns are
  sent back to the composer.
* The system prompt (persona + cheat-sheet + rules) is pre-computed once and kept in the KV cache.
* Live-data questions (weather, time, news) are answered by a fixed friendly sentence — the small model cannot be
  trusted not to invent a forecast.
* Everything is deterministic (greedy decoding, repetition penalty 1.12, sentence de-duplication).

## Scores

`tests/kingdom.txt` (70 deep questions in 7 blocks of 10), `tests/chat.txt` (40 small-talk / arithmetic /
instruction / general-knowledge items) and `tests/general.txt` (50 general-knowledge questions in 5 blocks: geography,
science, history, people & culture, everyday knowledge — scored only when `release/wiki.kdw` exists). A reply scores when
it contains an accepted phrase; `!norefuse` fails refusals.

```
python3 scripts/score.py            # prints X/10 per block, failures, average latency
python3 scripts/score.py --strict   # CI gate: kingdom/chat blocks >= 9/10, general-knowledge blocks >= 7/10
python3 scripts/eval_c.py --strict  # the older extractive test: 75 questions, reader-only (>= 72 exact)
python3 scripts/eval_wiki.py tests/general.txt --wiki release/wiki.kdw   # wiki retrieval + reader alone, no composer (~1 min)
```

Latest run (2 vCPU sandbox): **kingdom 70/70, chat 39/40 — every block ≥ 9/10**, avg 2.8 s per reply
(details, model comparison and what moved the numbers: `docs/SCORES.md`). With the wiki pack: see the
"General knowledge" section of `docs/SCORES.md`. Add your own questions to the
`tests/*.txt` files — one line per question, `question | accepted 1 ; accepted 2`.

## Run it

```bash
./release/kdr-brain release/brain.kdr --chat release/composer.gguf --wiki release/wiki.kdw serve 8080   # chat UI + API, kingdom + Wikipedia
./release/kdr-brain release/brain.kdr --chat release/composer.gguf serve 8080   # kingdom only (no wiki pack)
./release/kdr-brain release/brain.kdr --wiki release/wiki.kdw wiki "Who painted the Mona Lisa?"           # raw wiki retrieval + reader JSON
./release/kdr-brain release/brain.kdr --chat release/composer.gguf chat "What roles does Leonard II have?"
./release/kdr-brain release/brain.kdr serve 8080                                # without composer: extractive answers only
./release/kdr-brain release/brain.kdr ask "Who is the king?"                    # raw retrieval + reader JSON
curl -X POST localhost:8080/api/chat -d '{"q":"and his wife?","history":[{"role":"user","text":"who is the king?"},{"role":"assistant","text":"King Nicholas Goudhof V."}]}'
```

API: `POST /api/chat {q, history[], stream}` (SSE stream of `{"t":"token"}` events, then the full JSON), `GET /api/ask?q=`,
`GET /api/stats`. `KDR_THREADS=N` sets threads; `KDR_CHAT_MODEL=path` is equivalent to `--chat`, `KDR_WIKI=path` to `--wiki`.

## Rebuild from scratch

```bash
python3 scripts/crawl.py           # the 3 wikis -> data/raw/  (only network step; data/passages.json is the checked-in extraction)
python3 scripts/pack.py            # quantize + embed -> release/brain.kdr  (needs the two HF checkpoints in /tmp/build, see workflow)
sh scripts/build_llama.sh          # llama.cpp v0.4.0 static libs -> ~/.cache/kdr/llama  (PORTABLE=1 for x86-64-v3)
make                               # -> release/kdr-brain (one static binary)
curl -L -o release/composer.gguf https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
python3 scripts/score.py --strict  # quality gate
```

The Wikipedia pack (`.github/workflows/wiki.yml`, manual trigger, ~1.5 h on a free runner; the same steps work locally):

```bash
pip install libzim zstandard numpy
curl -fL -o dump.zim https://download.kiwix.org/zim/wikipedia/wikipedia_en_top_mini_2026-06.zim   # 316 MB
python3 scripts/wiki_extract.py dump.zim /tmp/wiki/ext      # ZIM -> articles.tsv + passages.tsv (paragraph-sized, 49,999 articles)
python3 scripts/wiki_prepare.py /tmp/wiki/ext 4             # shuffle articles, split into 4 embedding parts
sh scripts/wiki_embed_local.sh /tmp/wiki/ext 4              # `kdr-brain-lite embed` on each part (the slow step: ~13 passages/s per 2 cores)
python3 scripts/wiki_pack.py /tmp/wiki/ext release/wiki.kdw --dims 128   # PCA + int8 + zstd + BM25 -> wiki.kdw (10 s)
```

Any other Kiwix ZIM works too (`wikipedia_en_top_nopic` has the full article texts, 2.1 GB; topic packs like
`wikipedia_en_chemistry_*` exist). More passages = more MB: budget ~0.37 KB per passage.

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
* The Wikipedia pack holds only the *introductions* of the 50k most-read articles: it knows who painted the Mona
  Lisa and the capital of Australia, but not the main ingredient of guacamole (no such article) or facts that live
  deep in an article body. A wrong-but-confident Wikipedia answer is possible when the right article is missing and a
  neighbour reads plausibly ("largest organ" → the Bone article's "femur").
* Wikipedia answers take ~1.5 s longer than kingdom ones (6–12 separate reader passes).
* Some wiki facts contradict each other (e.g. Bjorn is Minister of Defence on Fandom, Defence is vacant on
  dutchbloxia.netlify.app; the Pope's birth year is 1977 vs 1978). The brain answers with whichever passage matches
  best and shows its source under the reply.
* ~2–6 s per reply on 2 vCPUs (the composer generates ~34 tokens/s); streaming makes it feel faster.
