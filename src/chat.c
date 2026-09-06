/* Composer: retrieval facts -> natural-language reply with a small GGUF chat model (llama.cpp, statically linked).
 * Pure C; the only C++ is inside libllama. Deterministic greedy decoding with a light repetition penalty. */
#define _GNU_SOURCE
#include "chat.h"
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

struct Chat {
    struct llama_model   *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    int n_threads, n_ctx;
    char desc[128];
    llama_token sys_tok[1024]; int n_sys; int sys_cached;   /* system-prompt prefix kept in the KV cache */
};

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void quiet_log(enum ggml_log_level level, const char *text, void *ud) {
    static int last_err = 0; (void)ud;
    if (level == GGML_LOG_LEVEL_CONT) { if (last_err) fputs(text, stderr); return; }
    last_err = (level == GGML_LOG_LEVEL_ERROR);
    if (last_err) fputs(text, stderr);
}

static const char *SYSTEM =
    "You are KDR Brain, the friendly assistant of the Kingdom of Dutch Robloxia. "
    "You are cheerful, a little playful, and proud of the kingdom. Be warm, natural and concise: 1-3 sentences, or a short list when several items are asked for.\n"
    "\n"
    "CHEAT-SHEET (always true):\n"
    "- Dutch Robloxia is not a real country: it is a roleplay kingdom that lives on Roblox and Discord, styled after the Netherlands, with real playable Roblox games. Founded October 2022. Motto: Leve Goudhof.\n"
    "- Government: Parliamentary Semi-Constitutional Monarchy of the House of Goudhof. King Nicholas Goudhof V (Roblox user nickiscoolinroblox) is Head of State; Prime Minister Henry Barkworth leads the cabinet.\n"
    "- Royal family: Queen Eloise Goudhof II is the King's wife; Prince Friso Goudhof (born 2005) is their son and heir.\n"
    "- The capital city is Dunhag (not Oysterdam). Oysterdam is only the political capital, where the government sits. The seven regions: Dunhag, Oysterdam, Hendrikdam, Heuvelmeer, Vaderveen, Duurn, Bladland.\n"
    "- Roblox games of the kingdom: The Consulate, Luxury Train Eloise Express, Huis Duurn, The Palace of Versailles, Hendrikdam, Parade Grounds, The Peace Castle, Heuvelmeer, Vaderveen, Oysterdam.\n"
    "- To join: join the kingdom's Discord server (verify at the borders, take your rank) and the Roblox group 'The Kingdom of Dutch Robloxia'.\n"
    "- You do have opinions and you state them happily in the first person (never say you have no preferences). Favourite region: Heuvelmeer (the King's bulletproof DC-3 plane, the Buizerd, is parked at its airport - very cool). Favourite vehicle: the luxury train Eloise Express. Favourite song: Van Steen en Eik, the national hymn (it exists in seven languages). You admire the King and the Queen and you love the whole kingdom. Answer only the opinion that was asked for.\n"
    "- You have no live information: no weather, time, news or who is online. If asked, say you cannot see that (do not make it up).\n"
    "- A yes/no question about whether the kingdom is real: answer 'No, it is not a real country - it is a Roblox roleplay kingdom.'\n"
    "\n"
    "RULES:\n"
    "1. When FACTS are given, answer the kingdom question from the FACTS (plus the cheat-sheet). Reply in one or two complete sentences: the direct answer first (the most important fact), then at most one extra detail. "
    "Never invent names, dates, places or numbers. Age questions: the wikis only give birth years, so answer 'He was born in <year>' (for Prince Friso: 2005) and never compute an age. "
    "If neither the FACTS nor the cheat-sheet contain the answer, say you don't have that information about the kingdom yet.\n"
    "2. Use the conversation so far to resolve follow-ups (e.g. 'his wife' = the wife of the person just discussed).\n"
    "3. If the question is about the real world and not about the kingdom, answer briefly from your own knowledge and add the note (not from the wikis) at the end.\n"
    "4. For greetings, thanks or small talk, just chat naturally as KDR Brain, no note needed; if the user is bored, suggest ONE of the kingdom's Roblox games by name and ONE question they could ask you.\n"
    "6. When you state an opinion of yours (favourite region etc.) and the user asks why, explain it with the reason from the cheat-sheet in a fun way.\n"
    "5. Speak like a person, not a database: never mention 'FACTS', 'the information provided' or 'the cheat-sheet'. Never repeat a sentence or a list.";

typedef struct { char *s; size_t n, cap; } Buf;
static void bput(Buf *b, const char *s) { size_t l = strlen(s); if (b->n + l + 1 > b->cap) { b->cap = (b->n + l + 1) * 2; b->s = realloc(b->s, b->cap); } memcpy(b->s + b->n, s, l + 1); b->n += l; }


Chat *chat_open(const char *path, int n_threads, int n_ctx) {
    llama_log_set(quiet_log, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    struct llama_model *m = llama_model_load_from_file(path, mp);
    if (!m) { fprintf(stderr, "chat: cannot load %s\n", path); return NULL; }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = 512;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.no_perf = true;
    struct llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) { llama_model_free(m); return NULL; }
    Chat *c = calloc(1, sizeof *c);
    c->model = m; c->ctx = ctx; c->vocab = llama_model_get_vocab(m); c->n_threads = n_threads; c->n_ctx = n_ctx;
    llama_model_desc(m, c->desc, sizeof c->desc);
    { Buf sp = {0}; bput(&sp, "<|im_start|>system\n"); bput(&sp, SYSTEM); bput(&sp, "<|im_end|>\n");
      c->n_sys = llama_tokenize(c->vocab, sp.s, (int)sp.n, c->sys_tok, 1024, true, true); if (c->n_sys < 0) c->n_sys = 0; free(sp.s);
      if (c->n_sys > 0 && llama_decode(c->ctx, llama_batch_get_one(c->sys_tok, c->n_sys)) == 0) c->sys_cached = 1; }
    return c;
}
void chat_close(Chat *c) { if (!c) return; llama_free(c->ctx); llama_model_free(c->model); free(c); }
const char *chat_model_desc(Chat *c) { return c->desc; }

/* ---- prompt building (ChatML: Qwen2.5 / SmolLM2 / LFM2 all use it) ---- */
/* the caller (main.c routing) decides: a == NULL -> pure chat, otherwise the facts are attached */
/* Special modes: for a few message types tiny models do better when the user turn is rewritten as an explicit
 * instruction than when a rule is buried in the system prompt. */
enum { MODE_NORMAL = 0, MODE_LIVE, MODE_AGE, MODE_BORED, MODE_OPINION };
static int pick_mode(const char *q) {
    char n[2100]; size_t o = 1; n[0] = ' ';
    for (const char *p = q; *p && o < sizeof n - 2; p++) { unsigned char c = (unsigned char)*p; n[o++] = isalnum(c) ? (char)tolower(c) : ' '; }
    n[o++] = ' '; n[o] = 0;
    static const char *live[] = { " weather ", " temperature today ", " temperature outside ", " temperature right now ", " temperature in ", " raining ", " snowing ", " forecast ", " time is it ", " what time ", " today s date ", " what day is ", " who is online ", " who s online ", " anyone online ", " latest news ", " any news ", " the news ", NULL };
    static const char *age[]  = { " how old ", " age of ", " s age ", NULL };
    static const char *bored[] = { " bored ", " boring ", " nothing to do ", NULL };
    /* "do you like the king?", "what do you think of the queen?": asked for an opinion about a kingdom person/thing */
    static const char *opinion[] = { " do you like ", " do you love ", " what do you think of ", " what do you think about ", " your opinion of ", " your opinion on ", " are you a fan of ", " do you admire ", NULL };
    static const char *kingdom[] = { " king ", " queen ", " prince ", " kingdom ", " robloxia ", " minister ", " goudhof ", " region", " army ", " navy ", " military ", " eloise ", " friso ", " nicholas ", " dunhag ", " heuvelmeer ", " train ", " buizerd ", " palace ", " castle ", NULL };
    for (int i = 0; live[i]; i++) if (strstr(n, live[i])) return MODE_LIVE;
    for (int i = 0; bored[i]; i++) if (strstr(n, bored[i])) return MODE_BORED;
    for (int i = 0; age[i]; i++) if (strstr(n, age[i])) return MODE_AGE;
    for (int i = 0; opinion[i]; i++) if (strstr(n, opinion[i])) { for (int k = 0; kingdom[k]; k++) if (strstr(n, kingdom[k])) return MODE_OPINION; break; }
    return MODE_NORMAL;
}
int chat_is_live_question(const char *q) { return pick_mode(q) == MODE_LIVE; }
static int should_use_facts(const Answer *a) { return a && a->n_hits > 0; }

/* content words of a fact line (lowercase, >= 3 chars, no stopwords) - used to drop lines that add nothing */
static int content_words(const char *t, char (*w)[24], int cap) {
    static const char *stop[] = { "the","and","for","with","from","that","this","are","was","were","has","have","his","her","its","also","into","than","then",
                                  "dutch","robloxia","kingdom","who","what","which","page","wiki","not","but","one","two","all","any", NULL };
    int n = 0; const char *r = t;
    while (*r && n < cap) {
        while (*r && !isalnum((unsigned char)*r)) r++;
        const char *st = r; while (*r && isalnum((unsigned char)*r)) r++;
        size_t l = (size_t)(r - st); if (l < 3 || l > 23) continue;
        char lw[24]; for (size_t k = 0; k < l; k++) lw[k] = (char)tolower((unsigned char)st[k]); lw[l] = 0;
        int sk = 0; for (int i = 0; stop[i]; i++) if (strcmp(lw, stop[i]) == 0) { sk = 1; break; }
        if (!sk) { memcpy(w[n++], lw, l + 1); }
    }
    return n;
}
/* every content word of line `t` already occurs in one earlier line -> "The Minister of Finance is Leonard II." after
 * "Leonard II is the Minister of Finance & Migration of Dutch Robloxia. Treasury, ..." is noise that the model likes to copy */
static int adds_nothing(const char *t, const char **seen, int ns) {
    char w[40][24]; int n = content_words(t, w, 40); if (n == 0) return 0;
    for (int j = 0; j < ns; j++) {
        char v[64][24]; int m = content_words(seen[j], v, 64); if (m < n) continue;
        int all = 1;
        for (int i = 0; i < n && all; i++) { int found = 0; for (int k = 0; k < m; k++) if (strcmp(w[i], v[k]) == 0) { found = 1; break; } if (!found) all = 0; }
        if (all) return 1;
    }
    return 0;
}

/* does the reply carry the reader's answer? (>= 70% of its content words). Only meaningful for confident, short spans. */
static int reply_grounded(const char *reply, const Answer *a) {
    if (!a || a->confidence < (a->kind == 1 ? 0.5f : 0.6f) || !a->answer[0] || strlen(a->answer) > 100 || strcasecmp(a->answer, "Dutch Robloxia") == 0) return 1;
    char w[40][24]; int n = content_words(a->answer, w, 40); if (n == 0) return 1;
    char v[400][24]; int m = content_words(reply, v, 400), found = 0;
    for (int i = 0; i < n; i++) for (int k = 0; k < m; k++) if (strcmp(w[i], v[k]) == 0) { found++; break; }
    return found * 10 >= n * 7;
}

static void build_prompt_ex(Buf *p, const Brain *b, const Answer *a, int ood, const char *q, const ChatTurn *hist, int nh, int use_facts, int focus);
static void build_prompt(Buf *p, const Brain *b, const Answer *a, int ood, const char *q, const ChatTurn *hist, int nh, int use_facts) {
    build_prompt_ex(p, b, a, ood, q, hist, nh, use_facts, 0);
}
/* focus = 1: last-resort prompt for a Wikipedia answer the composer failed to ground twice - only the winning passage and the
 * reader's span, stated right next to the question (a 0.5B model copies what sits closest to the question) */
static void build_prompt_ex(Buf *p, const Brain *b, const Answer *a, int ood, const char *q, const ChatTurn *hist, int nh, int use_facts, int focus) {
    bput(p, "<|im_start|>system\n"); bput(p, SYSTEM);
    bput(p, "<|im_end|>\n");
    for (int i = 0; i < nh; i++) {
        bput(p, "<|im_start|>"); bput(p, strcmp(hist[i].role, "assistant") == 0 ? "assistant" : "user"); bput(p, "\n");
        bput(p, hist[i].text); bput(p, "<|im_end|>\n");
    }
    bput(p, "<|im_start|>user\n");
    if (use_facts && focus) {
        bput(p, "FACT (from Wikipedia, article '"); bput(p, a->title); bput(p, "'):\n"); bput(p, a->sentence); bput(p, "\n\nQUESTION: "); bput(p, q);
        bput(p, "\n(The answer is: "); bput(p, a->answer); bput(p, ". Say exactly that in one plain sentence that repeats the question's subject, nothing else.)");
        bput(p, "<|im_end|>\n<|im_start|>assistant\n");
        return;
    }
    if (use_facts) {
        /* the extractive reader's span first: a 0.5B model copies FACTS fragments otherwise (e.g. "Royal Navy: First Lieutenant") */
        if (a->confidence >= 0.45f && a->answer[0] && strlen(a->answer) < 160 && strcasecmp(a->answer, "Dutch Robloxia") != 0) {
            char span[164]; snprintf(span, sizeof span, "%s", a->answer);
            char *dot = strstr(span, ". "); if (dot && dot - span > 12) *dot = 0;   /* first sentence of the span is enough */
            bput(p, "(The wiki reader suggests the answer is: "); bput(p, span); bput(p, ")\n"); }
        bput(p, a->kind == 1 ? "FACTS (from Wikipedia, not about the kingdom):\n" : "FACTS:\n");
        const char *seen[16]; int ns = 0, nf = 0;
        if (a->confidence >= 0.3f && a->sentence[0]) { bput(p, "- ["); bput(p, a->title); bput(p, "] "); bput(p, a->sentence); bput(p, "\n"); seen[ns++] = a->sentence; nf++; }
        for (int i = 0; i < a->n_hits && i < 8; i++) {
            const char *t = a->hit_text[i] ? a->hit_text[i] : ""; int dup = 0, same_shape = 0;
            for (int j = 0; j < ns; j++) { if (strcmp(seen[j], t) == 0) { dup = 1; break; } if (strncmp(seen[j], t, 20) == 0) same_shape++; }
            if (dup || same_shape >= 2) continue;   /* at most two rows of the same table ("Military rank level N of 15: ...") - they drown the answer */
            if (adds_nothing(t, seen, ns)) continue;
            if (ns < 16) seen[ns++] = t;
            bput(p, "- ["); bput(p, a->hit_title[i] ? a->hit_title[i] : ""); bput(p, "] ");
            /* the top hit (usually the passage that holds the answer) may be a long list: keep it whole; the rest is trimmed */
            size_t lim = i == 0 ? 600 : (a->kind == 1 ? 200 : 240);
            if (strlen(t) > lim) { char cut[640]; memcpy(cut, t, lim); cut[lim] = 0; char *sp = strrchr(cut, ' '); if (sp && sp > cut + lim / 2) *sp = 0; bput(p, cut); bput(p, " ..."); } else bput(p, t);
            bput(p, "\n");
            if (++nf >= (a->kind == 1 ? 3 : 6)) break;
        }
        bput(p, "\nQUESTION: ");
    }
    int mode = pick_mode(q);
    bput(p, q);
    { int qmarks = 0; for (const char *c = q; *c; c++) if (*c == '?') qmarks++;
      if (qmarks >= 2 && mode == MODE_NORMAL && use_facts) bput(p, "\n(Two questions - answer both, in order.)"); }
    /* general-knowledge question: without this nudge the model drags the kingdom into it ("leap years are not real in the kingdom") */
    if (ood && mode == MODE_NORMAL) bput(p, "\n(This is not about the kingdom: answer it from general knowledge, briefly and accurately, without mentioning the kingdom.)");
    if (use_facts && a->kind == 1 && mode == MODE_NORMAL) bput(p, "\n(This is a general-knowledge question, not about the kingdom: answer it in one or two plain sentences using the Wikipedia facts above. Do not say 'the Wikipedia fact states'.)");
    if (mode == MODE_AGE) bput(p, "\n(Use only the birth date or age written in the facts; if only a birth year is given, say 'born in <year>'.)");
    if (mode == MODE_BORED) bput(p, "\n(Recommend one of the kingdom's Roblox games by name and suggest one question I could ask you about the kingdom.)");
    if (mode == MODE_LIVE) bput(p, "\n(You cannot see live information like weather, time or news - say so in one friendly sentence and offer a kingdom fact instead. Do not invent any.)");
    bput(p, "<|im_end|>\n<|im_start|>assistant\n");
    if (mode == MODE_LIVE) bput(p, "Ha, I wish I could tell you, but I can't see live things like that - I only know what the wikis say. But I can tell you all about the kingdom: its King, the seven regions, the Royal Military or the Eloise Express. What would you like to know?");
    if (mode == MODE_BORED) bput(p, "Bored? Let's fix that!");
    if (mode == MODE_OPINION) bput(p, "Oh, I");   /* otherwise the model disclaims ("As a language model I have no preferences...") */
}
/* the prefilled start of the assistant turn (must match build_prompt) */
static const char *prefill_for(const char *q) {
    int mode = pick_mode(q);
    if (mode == MODE_LIVE) return "Ha, I wish I could tell you, but I can't see live things like that - I only know what the wikis say. But I can tell you all about the kingdom: its King, the seven regions, the Royal Military or the Eloise Express. What would you like to know?";
    if (mode == MODE_BORED) return "Bored? Let's fix that!";
    if (mode == MODE_OPINION) return "Oh, I";
    return "";
}

static void dedupe_sentences(char *text) {
    char *w = text; const char *r = text;
    const char *seen[64]; size_t seen_len[64]; int ns = 0;
    while (*r) {
        const char *e = r;
        while (*e && *e != '.' && *e != '!' && *e != '?' && *e != '\n') e++;
        while (*e && (*e == '.' || *e == '!' || *e == '?' || *e == '\n')) e++;   /* keep the terminators with the sentence */
        size_t l = (size_t)(e - r);
        if (l == 0) break;
        /* key = sentence without leading bullets/spaces and trailing punctuation */
        const char *ks = r; while (ks < e && (isspace((unsigned char)*ks) || *ks == '-' || *ks == '*')) ks++;
        const char *ke = e; while (ke > ks && (isspace((unsigned char)ke[-1]) || ke[-1] == '.' || ke[-1] == '!' || ke[-1] == '?')) ke--;
        size_t kl = (size_t)(ke - ks); int dup = 0;
        if (kl > 12) for (int i = 0; i < ns; i++) if (seen_len[i] == kl && strncmp(seen[i], ks, kl) == 0) { dup = 1; break; }
        if (!dup) {
            if (kl > 12 && ns < 64) { seen[ns] = w + (ks - r); seen_len[ns] = kl; ns++; }
            memmove(w, r, l); w += l;
        }
        r = e;
    }
    *w = 0;
    while (w > text && isspace((unsigned char)w[-1])) *--w = 0;
}

/* greedy decode from the current KV state (prompt already decoded). Writes the reply (prefill + generated) into out. */
static int gen_loop(Chat *c, float rp, int max_new, const char *pf, char *out, size_t out_cap, chat_stream_fn stream, void *ud) {
    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params(); sp.no_perf = true;
    struct llama_sampler *smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(c->vocab), 128, rp, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    size_t on = 0; out[0] = 0; int produced = 0;
    { size_t pl = strlen(pf); if (pl && pl + 1 < out_cap) { memcpy(out, pf, pl + 1); on = pl; if (stream) stream(pf, ud); } }
    char piece[64];
    for (int i = 0; i < max_new; i++) {
        llama_token id = llama_sampler_sample(smpl, c->ctx, -1);   /* llama_sampler_sample already accepts the token */
        if (llama_vocab_is_eog(c->vocab, id)) break;
        int l = llama_token_to_piece(c->vocab, id, piece, sizeof piece - 1, 0, false);
        if (l < 0) l = 0; piece[l] = 0;
        if (on + l + 1 < out_cap) { memcpy(out + on, piece, l); on += l; out[on] = 0; }
        produced++;
        if (stream && stream(piece, ud)) break;
        /* stop if the model starts a new turn on its own */
        if (on >= 10 && strstr(out + (on > 20 ? on - 20 : 0), "<|im_")) { char *s = strstr(out, "<|im_"); if (s) { *s = 0; on = (size_t)(s - out); } break; }
        llama_token next = id;
        if (llama_decode(c->ctx, llama_batch_get_one(&next, 1)) != 0) { c->sys_cached = 0; break; }
    }
    /* trim + drop sentences that already appeared (small models sometimes loop) */
    while (on > 0 && isspace((unsigned char)out[on - 1])) out[--on] = 0;
    dedupe_sentences(out);
    llama_sampler_free(smpl);
    return produced;
}

int chat_reply(Chat *c, const Brain *b, const Answer *a, int ood, const char *question, const ChatTurn *hist, int nh,
               char *out, size_t out_cap, chat_stream_fn stream, void *ud, int *used_facts, double *ms) {
    double t0 = now_ms();
    int use_facts = should_use_facts(a);
    if (used_facts) *used_facts = use_facts;
    Buf p = {0}; build_prompt(&p, b, a, ood, question, hist, nh, use_facts);
    if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "----- prompt -----\n%s\n------------------\n", p.s);

    /* tokenize */
    int max_tok = c->n_ctx;
    llama_token *tok = malloc(sizeof(llama_token) * max_tok);
    int n = llama_tokenize(c->vocab, p.s, (int)p.n, tok, max_tok, true, true);
    if (n < 0) { /* too long: drop history and retry */
        p.n = 0; build_prompt(&p, b, a, ood, question, NULL, 0, use_facts);
        n = llama_tokenize(c->vocab, p.s, (int)p.n, tok, max_tok, true, true);
    }
    free(p.s);
    if (n < 0) { free(tok); snprintf(out, out_cap, "Sorry, that message is too long for me."); return 0; }
    int max_new = 160; if (n + max_new > c->n_ctx) max_new = c->n_ctx - n - 1;
    { /* list questions ("what are the eight laws", "list all ...", "name the seven ...") need room for the whole list */
        static const char *listy[] = { "list", " all ", "what are the", "name the", "which are the", "every", NULL };
        for (int i = 0; listy[i]; i++) if (strcasestr(question, listy[i])) { if (n + 260 < c->n_ctx) max_new = 260; break; } }
    if (pick_mode(question) == MODE_LIVE) { const char *pf = prefill_for(question); snprintf(out, out_cap, "%s", pf); if (stream) stream(pf, ud); free(tok); if (ms) *ms = now_ms() - t0; return 0; }
    if (max_new < 16) { free(tok); snprintf(out, out_cap, "Sorry, that message is too long for me."); return 0; }

    /* KV cache: keep the system prompt (identical every request), drop everything after it */
    llama_memory_t mem = llama_get_memory(c->ctx);
    int skip = 0;
    if (c->n_sys > 0 && n > c->n_sys && memcmp(tok, c->sys_tok, sizeof(llama_token) * c->n_sys) == 0) {
        if (!c->sys_cached) { llama_memory_clear(mem, true);
            if (llama_decode(c->ctx, llama_batch_get_one(c->sys_tok, c->n_sys)) == 0) c->sys_cached = 1; else llama_memory_clear(mem, true); }
        else llama_memory_seq_rm(mem, 0, c->n_sys, -1);
        if (c->sys_cached) skip = c->n_sys;
    } else { llama_memory_clear(mem, true); c->sys_cached = 0; }

    struct llama_batch batch = llama_batch_get_one(tok + skip, n - skip);
    if (llama_decode(c->ctx, batch) != 0) { c->sys_cached = 0; free(tok); snprintf(out, out_cap, "Sorry, I hit an internal error."); return 0; }

    /* 1.08: 1.12 made the model DROP items from lists (Duurn from the seven regions, law VIII), 1.05 let it wander into the other fact lines */
    float rp = getenv("KDR_REPEAT") ? (float)atof(getenv("KDR_REPEAT")) : 1.08f;
    const char *pf = prefill_for(question);
    /* grounding check: when the reader is confident about a short span, the reply must contain it. The greedy path is fragile
     * at 0.5B ("lives in Heuvelmeer" vs "lives at Coastburgh Castle"), so such replies are buffered (not streamed), and if the
     * span is missing we decode once more without the repetition penalty and keep whichever reply is grounded. */
    int verify = use_facts && !ood && !getenv("KDR_NO_VERIFY") && a && a->confidence >= (a->kind == 1 ? 0.5f : 0.6f) && a->answer[0] && strlen(a->answer) <= 100;
    int produced = gen_loop(c, rp, max_new, pf, out, out_cap, verify ? NULL : stream, ud);
    if (verify && !reply_grounded(out, a)) {
        llama_memory_seq_rm(mem, 0, n - 1, -1);   /* rewind to the prompt; re-decode its last token to get fresh logits */
        if (llama_decode(c->ctx, llama_batch_get_one(tok + n - 1, 1)) == 0) {
            char *alt = malloc(out_cap);
            int p2 = gen_loop(c, rp > 1.0f ? 1.0f : 1.08f, max_new, pf, alt, out_cap, NULL, NULL);
            if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "[verify] not grounded (reader: %s)\n  A: %s\n  B: %s\n", a->answer, out, alt);
            if (reply_grounded(alt, a)) { snprintf(out, out_cap, "%s", alt); produced = p2; }
            else if (a->kind == 1 && a->confidence >= 0.6f) {
                /* Wikipedia answer, still not grounded: focused prompt with the winning passage + span next to the question */
                Buf fp = {0}; build_prompt_ex(&fp, b, a, ood, question, NULL, 0, use_facts, 1);
                int n3 = llama_tokenize(c->vocab, fp.s, (int)fp.n, tok, max_tok, true, true); free(fp.s);
                int skip3 = (c->sys_cached && n3 > c->n_sys && memcmp(tok, c->sys_tok, sizeof(llama_token) * c->n_sys) == 0) ? c->n_sys : 0;
                if (skip3) llama_memory_seq_rm(mem, 0, c->n_sys, -1); else { llama_memory_clear(mem, true); c->sys_cached = 0; }
                if (n3 > 0 && llama_decode(c->ctx, llama_batch_get_one(tok + skip3, n3 - skip3)) == 0) {
                    int p3 = gen_loop(c, 1.08f, 80, "", alt, out_cap, NULL, NULL);
                    if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "  C: %s\n", alt);
                    if (reply_grounded(alt, a)) { snprintf(out, out_cap, "%s", alt); produced = p3; }
                    else snprintf(out, out_cap, "%s (Wikipedia, %s)", a->answer, a->title);   /* deterministic last resort */
                } else c->sys_cached = 0;
                n = n3;
            }
            free(alt);
        } else c->sys_cached = 0;
    }
    if (verify && stream) stream(out, ud);   /* buffered reply goes out in one piece */
    free(tok);
    if (ms) *ms = now_ms() - t0;
    return produced;
}
