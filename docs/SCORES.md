# Scores

How the brain is graded, what it scores today, and what was tried. All numbers come from
`python3 scripts/score.py` (whole pipeline: routing → retrieval → reader → composer) run in the 2-vCPU sandbox.

## The test sets

| file | items | what it checks |
|---|---|---|
| `tests/kingdom.txt` | 70 (7 blocks of 10) | deep, specific kingdom questions: crown & family, government & ministers, regions & places, Vaderveen & Goudhof Enterprises, military, laws/songs/culture/diplomacy, vehicles/games/misc |
| `tests/chat.txt` | 40 (4 blocks of 10) | greetings & small talk, arithmetic & logic, instructions (bullet lists, "answer with just the name", translate, repeat), general knowledge outside the kingdom (must answer *and* flag, never refuse) |

One line = `question | accepted phrase 1 ; accepted phrase 2 ; !norefuse ; !short`. A reply passes when it contains
any accepted phrase (case-insensitive), `!norefuse` fails refusals, `!short` fails replies over 40 words.
The target is **≥ 9/10 in every block**; `--strict` (the CI gate) exits non-zero otherwise.

## Current result (Qwen2.5-0.5B-Instruct Q4_K_M composer, 537 passages)

```
== kingdom.txt
  10/10  Crown & royal family
  10/10  Government & ministers
  10/10  Regions & places
  10/10  Vaderveen & Goudhof Enterprises
  10/10  Military
  10/10  Laws, songs, culture, diplomacy
  10/10  Vehicles, games, misc
== chat.txt
  10/10  Greetings & small talk
  10/10  Arithmetic & logic
  10/10  Instructions
   9/10  Outside the kingdom      (fails: "how many continents are there?" -> "6" — the model's own belief; it says 5, 6 or 7 depending on phrasing)
TOTAL 109/110   avg 2.8 s per reply (2 vCPU)
```

`scripts/eval_c.py --strict` (reader-only exact-answer test, 75 questions): 75/75.

## Model choice (same prompt, same tests)

| composer | GGUF size | kingdom | chat | notes |
|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct Q4_K_M** | **398 MB** | **70/70** | **39/40** | chosen |
| Qwen2.5-0.5B-Instruct Q4_0 / IQ4_XS | 353 / 349 MB | −3…−5 | leaks kingdom facts into general answers | rejected — 50 MB is not worth it |
| Qwen3-0.6B Q4_K_M | 397 MB | similar | slower (thinking tokens must be suppressed) | no gain over Qwen2.5 |
| LFM2-350M Q4_K_M | 229 MB | good on facts | broke the chat rules (bare "(not from the wikis)" replies, refused arithmetic) | fastest (60 tok/s); candidate if 170 MB ever matters more than chat quality |

## What actually moved the numbers (in order of impact)

1. **Routing before generation** (`src/main.c route()`): small talk / opinions / "I'm bored" never see FACTS;
   greetings only see facts when a kingdom entity is named; arithmetic never hits retrieval; live-data
   questions (weather/time/news) get a fixed friendly sentence — the 0.5B model *will* invent a forecast otherwise.
2. **Reader span in front of the facts** (`(The wiki reader suggests the answer is: Admiral)`): without it the
   model copied table fragments ("Royal Navy: First Lieutenant") instead of answering.
3. **At most two rows of the same table** in FACTS (rank tables drowned the one sentence that held the answer).
4. **Curated summary passages** (~15 of 537) that state an answer in one plain sentence when the wikis only
   imply it (Prince Friso's birth year, how to join, who "runs the place", what Leonard II built).
5. **Deterministic provenance flag**: general-knowledge answers get "(not from the wikis)" appended by the
   server (`needs_ood_note`) instead of trusting the model to add it.
6. **Cheat-sheet in the system prompt** (capital Dunhag ≠ Oysterdam, "not a real country", first-person
   favourites) + the system prefix kept in the KV cache (saves ~1.5 s per reply).
7. **Assistant-turn prefill** for "I'm bored" ("Bored? Let's fix that!") and for opinion questions about the
   kingdom ("do you like the king?" → "Oh, I…") — the only reliable way to steer this model size away from
   "As a language model I have no preferences"; instructions in the prompt were ignored.
8. **Repetition penalty 1.08 instead of 1.12**: the higher value silently *dropped items from lists*
   (Duurn from the seven regions, law VIII from the eight laws) because their words had already appeared in the
   FACTS; 1.05 in turn let it wander into unrelated fact lines; 1.08 does neither.
9. **List questions get 260 new tokens** ("what are the eight laws", "list all …") and the top passage is kept
   whole (600 chars) instead of trimmed to 240.

## Known misses / not fixable at this size

* General knowledge is the model's own: "6 continents", "orange" for blue+yellow. Flagged, not fixed
  (a bigger model or a small general-facts index would be the next step).
* Answers that need every item of a long list are only as good as the single passage that holds the list;
  when a vague "overview" passage outranks it the model paraphrases the overview instead (fixed case by case by
  putting the list into the overview passage).
* Latency 1–5 s on 2 vCPUs; ~2× faster on 4 cores (`KDR_THREADS=4`).
