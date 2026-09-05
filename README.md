# KDR Brain — a < 80 MB neural question-answering brain for Dutch Robloxia

A fully self-contained neural QA system whose entire knowledge comes from three sites:

| Source | What was crawled |
|---|---|
| https://the-kingdom-of-dutch-robloxia.fandom.com/ | all 34 articles + 2 project pages, via the MediaWiki API (wikitext + infoboxes + rank tables) |
| https://kdrwiki.netlify.app/ | the single-page "Official Wiki" (cards, fact strips, laws, songbook, rank table, bios) |
| https://dutchbloxia.netlify.app/ | the single-page "Royal Wiki" (cards, diplomacy table, laws, songbook, games) |

Crawled on 2026-09-05 → **539 knowledge passages** (`data/passages.json`).

## Size budget (everything included)

```
release/brain.kdr   58.4 MB   weights (2 BERT models, int8) + vocab + passage index + passages + BM25
release/kdr-brain    1.5 MB   static Linux x86-64 binary: tokenizer, inference engine, HTTP server, web UI
                 ------
                 59.9 MB   total — no Python, no libraries, nothing else needed at runtime
```

## Architecture

```
question ──► WordPiece tokenizer (exact HF BERT-uncased, in C)
         ──► Retriever: MiniLM-L6-H384 (multi-qa-MiniLM-L6-cos-v1), 6 layers, int8 → 384-d embedding
               cosine vs. 539 int8 passage vectors   +   BM25 over the same wordpiece ids   → fused top-8
         ──► Reader: MiniLM-L12-H384 fine-tuned on SQuAD 2.0 (deepset/minilm-uncased-squad2), 12 layers, int8
               question + top passages → best answer span, "no-answer" logit
         ──► confidence = reader margin ⊕ dense similarity ⊕ IDF-weighted question-term coverage
```

Both networks are real transformers (≈22M + ≈33M parameters) executed by a dependency-free C engine
(`src/brain.c`, ~450 lines): int8 per-row quantized matmuls, multi-head attention, LayerNorm, GELU, OpenMP.

## Results

* Test set of 75 questions written against the wikis (`scripts/testset.py`): **75/75 answered correctly**,
  retrieval@8 = 75/75, ~190 ms per question on 2 vCPUs.
* Out-of-domain questions ("capital of France", "bake a cake", "2+2") get low confidence and the UI
  says it could not find the answer in the wikis, instead of hallucinating.
* C tokenizer verified byte-exact against HuggingFace `tokenizers` on all 539 passages + edge cases
  (accents, CJK, ligatures, emoji).

## Run it

```bash
./release/kdr-brain release/brain.kdr serve 8080          # web chat UI + JSON API on http://0.0.0.0:8080
./release/kdr-brain release/brain.kdr ask "Who is the king of Dutch Robloxia?"
curl 'http://localhost:8080/api/ask?q=What+is+the+capital'
```

`KDR_THREADS=4` sets the thread count. `release/kdr-brain-portable` is an SSE2-only build for old CPUs.

## Rebuild from scratch

```bash
python3 scripts/crawl.py         # the 3 wikis -> data/raw/   (only network step; data/passages.json is the checked-in extraction of it)
python3 scripts/pack.py          # quantize models + embed passages -> release/brain.kdr  (needs the two HF checkpoints in /tmp/build, see .github/workflows/build.yml)
python3 scripts/gen_unicode.py   # unicode tables for the tokenizer -> src/unicode_tables.h
make                             # -> release/kdr-brain
python3 scripts/eval_c.py -v     # run the 75-question test set against the C engine (--strict = CI gate)
```

Only `src/`, `data/`, `scripts/` are versioned (1.5 MB). The two build outputs (`release/brain.kdr` 58 MB,
`release/kdr-brain` 1.5 MB) are rebuilt by GitHub Actions on every push and published as a GitHub Release;
`KDR_REPO=owner/kdr-brain sh scripts/restore.sh` downloads them back. Setup: `docs/CLOUD_SETUP.md`.

## Honest limitations

* It is **extractive**: answers are spans copied from the wiki text (with the source sentence + link shown),
  not free-form generated prose. It cannot do math, chit-chat, or anything outside the three sites.
* Its knowledge is a snapshot; re-run the crawl + `pack.py` when the wikis change.
* Some wiki facts contradict each other (e.g. Fandom says Bjorn is Minister of Defence, dutchbloxia.netlify.app
  lists Defence as vacant; the Pope's birth year is 1977 on Fandom and 1978 on kdrwiki). The brain answers with
  whichever passage matches best and always shows which source it came from.
