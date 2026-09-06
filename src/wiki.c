/* General-knowledge pack: Wikipedia passages (zstd blocks) + PCA-reduced int8 embeddings + BM25 postings.
 * Query path: brain_embed_text (same MiniLM as the kingdom index) -> PCA -> int8 dot products over all N passages,
 * fused with BM25 (title tokens are part of each passage's bag), article-diverse top-k, then the SQuAD2 reader. */
#define _GNU_SOURCE
#include "wiki.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <zstd.h>

typedef struct { char name[48]; uint64_t off, size; } TocEntry;
#define NCACHE 16

struct Wiki {
    uint8_t *map; size_t map_size; TocEntry *toc; int n_toc;
    int N, n_art, D, n_blocks, n_terms, n_post, blk; float avgdl;
    const float *pca_mean, *pca_proj; const int8_t *emb_q; const float *emb_s;
    const uint8_t *tdict; size_t tdict_len; const uint8_t *blocks; const uint32_t *blk_off, *pass_blk, *pass_off;
    const int32_t *pass_art, *art_first; const char **art_title, **art_path;
    const int32_t *bm_terms, *bm_df; const uint32_t *bm_offsets; const uint8_t *bm_post; const uint16_t *bm_doclen;
    ZSTD_DCtx *dctx; ZSTD_DDict *ddict;
    uint32_t *title_hash; int title_hash_size;   /* open addressing set of fnv(lowercased title) for wiki_has_title */
    struct { int blk; char *data; size_t len; unsigned age; } cache[NCACHE]; unsigned tick;
    char url[600]; float *dense; float *lex;
};

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static const void *wblob(Wiki *w, const char *name, size_t *size) {
    for (int i = 0; i < w->n_toc; i++)
        if (strncmp(w->toc[i].name, name, 48) == 0) { if (size) *size = w->toc[i].size; return w->map + w->toc[i].off; }
    fprintf(stderr, "wiki: missing blob %s\n", name); exit(2);
}

Wiki *wiki_open(const char *path, Brain *b) {
    (void)b;
    int fd = open(path, O_RDONLY); if (fd < 0) { perror(path); return NULL; }
    struct stat st; fstat(fd, &st);
    Wiki *w = calloc(1, sizeof *w);
    w->map_size = st.st_size; w->map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (w->map == MAP_FAILED || memcmp(w->map, "KDRW", 4) != 0) { fprintf(stderr, "wiki: %s is not a KDRW file\n", path); free(w); return NULL; }
    w->n_toc = *(uint32_t *)(w->map + 8); w->toc = (TocEntry *)(w->map + 16);
    const int32_t *pr = wblob(w, "params", NULL);
    w->N = pr[0]; w->n_art = pr[1]; w->D = pr[2]; w->n_blocks = pr[3]; w->n_terms = pr[4]; w->n_post = pr[5]; w->avgdl = pr[6] / 1000.0f; w->blk = pr[7];
    w->pca_mean = wblob(w, "pca.mean", NULL); w->pca_proj = wblob(w, "pca.proj", NULL);
    w->emb_q = wblob(w, "emb.q", NULL); w->emb_s = wblob(w, "emb.s", NULL);
    w->tdict = wblob(w, "text.dict", &w->tdict_len); w->blocks = wblob(w, "text.blocks", NULL);
    w->blk_off = wblob(w, "text.blk_off", NULL); w->pass_blk = wblob(w, "text.pass_blk", NULL); w->pass_off = wblob(w, "text.pass_off", NULL);
    w->pass_art = wblob(w, "pass.art", NULL); w->art_first = wblob(w, "art.first", NULL);
    w->art_title = malloc(sizeof(char *) * w->n_art); w->art_path = malloc(sizeof(char *) * w->n_art);
    { const char *p = wblob(w, "art.title", NULL); for (int i = 0; i < w->n_art; i++) { w->art_title[i] = p; p += strlen(p) + 1; }
      p = wblob(w, "art.path", NULL); for (int i = 0; i < w->n_art; i++) { w->art_path[i] = p; p += strlen(p) + 1; } }
    w->bm_terms = wblob(w, "bm25.terms", NULL); w->bm_offsets = wblob(w, "bm25.offsets", NULL); w->bm_df = wblob(w, "bm25.df", NULL);
    w->bm_post = wblob(w, "bm25.postings", NULL); w->bm_doclen = wblob(w, "bm25.doclen", NULL);
    w->dctx = ZSTD_createDCtx(); w->ddict = w->tdict_len ? ZSTD_createDDict(w->tdict, w->tdict_len) : NULL;
    for (int i = 0; i < NCACHE; i++) { w->cache[i].blk = -1; w->cache[i].data = malloc(w->blk + 4096); }
    w->dense = malloc(sizeof(float) * w->N); w->lex = malloc(sizeof(float) * w->N);
    /* title set (single-word and multi-word titles alike; lowercased) */
    int hs = 1; while (hs < w->n_art * 4) hs <<= 1; w->title_hash_size = hs; w->title_hash = malloc(sizeof(uint32_t) * hs); memset(w->title_hash, 0, sizeof(uint32_t) * hs);
    for (int i = 0; i < w->n_art; i++) {
        char low[256]; size_t l = 0; for (const char *c = w->art_title[i]; *c && l < sizeof low - 1; c++) low[l++] = (char)tolower((unsigned char)*c); low[l] = 0;
        uint32_t h = 2166136261u; for (size_t k = 0; k < l; k++) { h ^= (uint8_t)low[k]; h *= 16777619u; } if (!h) h = 1;
        uint32_t pos = h & (hs - 1); while (w->title_hash[pos] && w->title_hash[pos] != h) pos = (pos + 1) & (hs - 1); w->title_hash[pos] = h;
    }
    return w;
}
int wiki_has_title(const Wiki *w, const char *word) {
    char low[256]; size_t l = 0; for (const char *c = word; *c && l < sizeof low - 1; c++) low[l++] = (char)tolower((unsigned char)*c); low[l] = 0;
    uint32_t h = 2166136261u; for (size_t k = 0; k < l; k++) { h ^= (uint8_t)low[k]; h *= 16777619u; } if (!h) h = 1;
    uint32_t pos = h & (w->title_hash_size - 1); while (w->title_hash[pos]) { if (w->title_hash[pos] == h) return 1; pos = (pos + 1) & (w->title_hash_size - 1); }
    return 0;
}
void wiki_close(Wiki *w) {
    if (!w) return;
    for (int i = 0; i < NCACHE; i++) free(w->cache[i].data);
    ZSTD_freeDCtx(w->dctx); if (w->ddict) ZSTD_freeDDict(w->ddict);
    free(w->art_title); free(w->art_path); free(w->dense); free(w->lex); free(w->title_hash); munmap(w->map, w->map_size); free(w);
}
int wiki_n_passages(const Wiki *w) { return w->N; }
int wiki_n_articles(const Wiki *w) { return w->n_art; }

static const char *block_data(Wiki *w, int blk) {
    int slot = -1, oldest = 0;
    for (int i = 0; i < NCACHE; i++) { if (w->cache[i].blk == blk) { slot = i; break; } if (w->cache[i].age < w->cache[oldest].age) oldest = i; }
    if (slot < 0) {
        slot = oldest;
        const uint8_t *src = w->blocks + w->blk_off[blk]; size_t clen = w->blk_off[blk + 1] - w->blk_off[blk];
        size_t n = w->ddict ? ZSTD_decompress_usingDDict(w->dctx, w->cache[slot].data, w->blk + 4096, src, clen, w->ddict)
                            : ZSTD_decompressDCtx(w->dctx, w->cache[slot].data, w->blk + 4096, src, clen);
        if (ZSTD_isError(n)) { fprintf(stderr, "wiki: zstd error %s\n", ZSTD_getErrorName(n)); n = 0; w->cache[slot].data[0] = 0; }
        w->cache[slot].blk = blk; w->cache[slot].len = n;
    }
    w->cache[slot].age = ++w->tick;
    return w->cache[slot].data;
}
const char *wiki_passage_text(Wiki *w, int i) { if (i < 0 || i >= w->N) return ""; return block_data(w, w->pass_blk[i]) + w->pass_off[i]; }
const char *wiki_passage_title(const Wiki *w, int i) { return i >= 0 && i < w->N ? w->art_title[w->pass_art[i]] : ""; }
const char *wiki_passage_url(Wiki *w, int i) { snprintf(w->url, sizeof w->url, "https://en.wikipedia.org/wiki/%s", i >= 0 && i < w->N ? w->art_path[w->pass_art[i]] : ""); return w->url; }

static int term_index(const Wiki *w, int id) { int lo = 0, hi = w->n_terms - 1; while (lo <= hi) { int m = (lo + hi) / 2; if (w->bm_terms[m] == id) return m; if (w->bm_terms[m] < id) lo = m + 1; else hi = m - 1; } return -1; }

/* retrieval: k hits, at most `per_art` passages per article */
static int wiki_retrieve(Wiki *w, Brain *b, const char *question, Hit *hits, int k, int per_art) {
    const int N = w->N, D = w->D;
    float qv[384], qz[384];
    brain_embed_text(b, question, 126, qv);
    for (int i = 0; i < 384; i++) qv[i] -= w->pca_mean[i];
    float nz = 0;
    for (int d = 0; d < D; d++) { float s = 0; for (int i = 0; i < 384; i++) s += qv[i] * w->pca_proj[(size_t)i * D + d]; qz[d] = s; nz += s * s; }
    nz = sqrtf(nz) + 1e-9f; for (int d = 0; d < D; d++) qz[d] /= nz;
    float *dense = w->dense, *lex = w->lex;
    /* int8 x int8 dot products: the query is quantized to int8 too (per-vector scale), so the inner loop is pure integer
     * math the compiler vectorizes (pmaddwd / vpdpbusd). Error of the extra quantization is ~0.5% of the score. */
    int8_t q8[384]; float qmax = 0; for (int d = 0; d < D; d++) if (fabsf(qz[d]) > qmax) qmax = fabsf(qz[d]);
    float qs = qmax > 0 ? qmax / 127.0f : 1.0f; for (int d = 0; d < D; d++) { int v = (int)lrintf(qz[d] / qs); q8[d] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const int8_t *e = w->emb_q + (size_t)i * D; int32_t acc = 0;
        for (int d = 0; d < D; d++) acc += (int32_t)e[d] * (int32_t)q8[d];
        dense[i] = (float)acc * qs * w->emb_s[i];
    }
    memset(lex, 0, sizeof(float) * N);
    int qids[128]; int nq = brain_tokenize(b, question, qids, 128);
    const float k1 = 1.2f, bb = 0.75f;
    for (int a = 0; a < nq; a++) {
        int dup = 0; for (int c = 0; c < a; c++) if (qids[c] == qids[a]) dup = 1;
        if (dup || brain_is_stop(b, qids[a])) continue;
        const char *tok = brain_vocab_str(b, qids[a]); if (tok[0] == '#' && tok[1] == '#') continue;
        int ti = term_index(w, qids[a]); if (ti < 0) continue;
        int df = w->bm_df[ti];
        float idf = logf(1.0f + (N - df + 0.5f) / (df + 0.5f));
        const uint8_t *p = w->bm_post + w->bm_offsets[ti], *end = w->bm_post + w->bm_offsets[ti + 1]; uint32_t doc = 0;
        while (p < end) {
            uint32_t delta = 0, tf = 0; int sh = 0;
            do { delta |= (uint32_t)(*p & 0x7F) << sh; sh += 7; } while (*p++ & 0x80);
            sh = 0; do { tf |= (uint32_t)(*p & 0x7F) << sh; sh += 7; } while (*p++ & 0x80);
            doc += delta;
            lex[doc] += idf * tf * (k1 + 1) / (tf + k1 * (1 - bb + bb * w->bm_doclen[doc] / w->avgdl));
        }
    }
    float lmax = 0; for (int i = 0; i < N; i++) if (lex[i] > lmax) lmax = lex[i];
    /* fuse; collect a generous candidate list, then enforce article diversity. Dense-heavier than the kingdom index:
     * with 200k passages a rare question word ("tallest") makes BM25 pull in "List of tallest buildings" over [Giraffe]. */
    float wd = getenv("KDR_WIKI_WD") ? (float)atof(getenv("KDR_WIKI_WD")) : 0.75f;
    enum { CAND = 64 }; Hit cand[CAND]; int nc = 0;
    for (int i = 0; i < N; i++) {
        float s = wd * dense[i] + (1.0f - wd) * (lmax > 0 ? lex[i] / lmax : 0);
        if (nc < CAND || s > cand[nc - 1].score) {
            int pos = nc < CAND ? nc++ : nc - 1;
            while (pos > 0 && cand[pos - 1].score < s) { cand[pos] = cand[pos - 1]; pos--; }
            cand[pos].passage = i; cand[pos].score = s; cand[pos].dense = dense[i]; cand[pos].lexical = lex[i];
        }
    }
    int n = 0;
    for (int c = 0; c < nc && n < k; c++) {
        int art = w->pass_art[cand[c].passage], cnt = 0;
        for (int j = 0; j < n; j++) if (w->pass_art[hits[j].passage] == art) cnt++;
        if (cnt >= per_art) continue;
        hits[n++] = cand[c];
    }
    return n;
}

/* phrase match: longest run of consecutive question wordpieces that also occurs in the text, as a fraction of the question's
 * length (leading wh-word/auxiliary and the final '?' excluded). "the longest river in the world" is found whole in [Nile]
 * (1.0) but only as "river in the world" in [Tributary] (0.67) or "in the world" in [Amazon River] (0.5). */
static float phrase_match(Brain *b, const int *qids, int nq, const char *text) {
    int tids[512]; int nt = brain_tokenize(b, text, tids, 512);
    int q0 = 0, q1 = nq;
    if (q1 > 0 && strcmp(brain_vocab_str(b, qids[q1 - 1]), "?") == 0) q1--;
    static const char *lead[] = { "what", "which", "who", "where", "when", "how", "why", "is", "are", "was", "were", "does", "do", "did", "many", "much", "the", "s", "'", NULL };
    while (q0 < q1) { const char *t = brain_vocab_str(b, qids[q0]); int hit = 0; for (int i = 0; lead[i]; i++) if (strcmp(t, lead[i]) == 0) { hit = 1; break; } if (!hit) break; q0++; }
    int n = q1 - q0; if (n <= 0) return 0;
    int best = 0;
    for (int i = 0; i < nt; i++) {
        for (int j = q0; j < q1; j++) {
            if (tids[i] != qids[j]) continue;
            int l = 1; while (i + l < nt && j + l < q1 && tids[i + l] == qids[j + l]) l++;
            if (l > best) best = l;
        }
    }
    return (float)best / (float)n;
}

void wiki_retrieve_debug(Wiki *w, Brain *b, const char *question, int k) {
    Hit h[64]; if (k > 64) k = 64; int n = wiki_retrieve(w, b, question, h, k, 2);
    for (int i = 0; i < n; i++) printf("%2d %.3f d%.3f l%5.1f [%s] %.80s\n", i, h[i].score, h[i].dense, h[i].lexical, wiki_passage_title(w, h[i].passage), wiki_passage_text(w, h[i].passage));
}

/* is the extracted answer the article's own subject? ("Giraffe" from [Giraffe], "Mercury" from [Mercury (element)]) */
static int answer_is_title(const char *answer, const char *title) {
    char tn[160], an[160]; size_t o = 0;
    for (const char *ti = title; *ti && *ti != '(' && o < sizeof tn - 1; ti++) { unsigned char ch = (unsigned char)*ti; if (isalnum(ch) || ch == ' ' || ch >= 0x80) tn[o++] = (char)tolower(ch); }
    while (o && tn[o - 1] == ' ') o--; tn[o] = 0;
    const char *r = answer; if (!strncasecmp(r, "the ", 4)) r += 4; o = 0;
    for (; *r && o < sizeof an - 1; r++) { unsigned char ch = (unsigned char)*r; if (isalnum(ch) || ch == ' ' || ch >= 0x80) an[o++] = (char)tolower(ch); }
    while (o && an[o - 1] == ' ') o--; an[o] = 0;
    size_t la = strlen(an), lt = strlen(tn);
    return la >= 3 && lt >= 3 && (strcmp(an, tn) == 0 || (la >= lt && la - lt <= 1 && strncmp(an, tn, lt) == 0) || (lt >= la && lt - la <= 1 && strncmp(an, tn, la) == 0));
}
int wiki_answer_is_title(const Answer *a) { return a->answer[0] && a->title[0] && answer_is_title(a->answer, a->title); }
/* a bare pronoun extracted from an article's own text refers to the article's subject: replace it by the title (no parenthetical) */
static void resolve_pronoun(Answer *a, const char *title) {
    static const char *pron[] = { "it", "they", "he", "she", "this", "these", "the city", "the country", "the species", NULL };
    char an[64]; size_t o = 0; for (const char *r = a->answer; *r && o < sizeof an - 1; r++) an[o++] = (char)tolower((unsigned char)*r); an[o] = 0;
    while (o && (an[o - 1] == ' ' || an[o - 1] == '.')) an[--o] = 0;
    for (int i = 0; pron[i]; i++) if (strcmp(an, pron[i]) == 0) {
        size_t l = 0; while (title[l] && title[l] != '(') l++; while (l && title[l - 1] == ' ') l--;
        if (l >= 2 && l < sizeof a->answer) { memcpy(a->answer, title, l); a->answer[l] = 0; }
        return;
    }
}

/* crude sentence splitter: ". " / "? " / "! " followed by an upper-case letter, digit, quote or bracket; skips initials/abbreviations */
static int split_sentences(const char *text, int *starts, int *ends, int max) {
    int n = 0; const char *p = text, *s = text;
    while (*p && n < max) {
        if ((*p == '.' || *p == '?' || *p == '!') && p[1] == ' ' && (isupper((unsigned char)p[2]) || isdigit((unsigned char)p[2]) || p[2] == '"' || p[2] == '(')) {
            const char *q = p; while (q > s && q[-1] != ' ') q--;
            if (*p == '.' && p - q <= 3 && isupper((unsigned char)*q)) { p++; continue; }   /* "J. R. R.", "Mr.", "Dr." */
            starts[n] = (int)(s - text); ends[n] = (int)(p + 1 - text); n++;
            p++; while (*p == ' ') p++; s = p; continue;
        }
        p++;
    }
    if (*s && n < max) { starts[n] = (int)(s - text); ends[n] = (int)strlen(text); n++; }
    return n;
}

void wiki_answer(Wiki *w, Brain *b, const char *question, Answer *out) {
    memset(out, 0, sizeof *out); out->kind = 1;
    double t0 = now_ms();
    /* a wide candidate list (24, <= 2 per article) is re-ranked by question-term coverage before reading: at 215k passages a
     * global superlative ("longest river in the world") drowns in "longest river in Spain/Asia/..." passages and the right
     * article sits at rank 9-20; coverage is a tokenizer-only check, so it is cheap where reading (~150 ms each) is not */
    int npre = getenv("KDR_WIKI_NPRE") ? atoi(getenv("KDR_WIKI_NPRE")) : 24; if (npre > 32) npre = 32; if (npre < 8) npre = 8;
    float wpre = getenv("KDR_WIKI_WPRE") ? (float)atof(getenv("KDR_WIKI_WPRE")) : 0.15f;
    Hit pre[32]; int np = wiki_retrieve(w, b, question, pre, npre, 2);
    out->ms_retrieve = now_ms() - t0; t0 = now_ms();
    {
        int qids0[64]; int nq0 = brain_tokenize(b, question, qids0, 62); float ps[32]; char pb[2048];
        float wphr = getenv("KDR_WIKI_WPHR") ? (float)atof(getenv("KDR_WIKI_WPHR")) : 0.3f;
        for (int i = 0; i < np; i++) {
            snprintf(pb, sizeof pb, "%s: %s", wiki_passage_title(w, pre[i].passage), wiki_passage_text(w, pre[i].passage));
            ps[i] = pre[i].score + wpre * brain_term_coverage_ext(b, qids0, nq0, pb, w->bm_terms, w->bm_df, w->n_terms, w->N)
                  + wphr * phrase_match(b, qids0, nq0, pb);
        }
        /* stable selection sort of the top 8 by ps */
        out->n_hits = 0;
        for (int k = 0; k < 8 && k < np; k++) {
            int bi = -1; for (int i = 0; i < np; i++) if (ps[i] > -1e29f && (bi < 0 || ps[i] > ps[bi])) bi = i;
            if (bi < 0) break;
            out->hits[out->n_hits] = pre[bi]; out->hits[out->n_hits].score = ps[bi]; out->n_hits++; ps[bi] = -1e30f;
        }
    }
    /* Passage texts come from the block cache (16 slots >= 8 hits, so all pointers stay valid during the read). */
    for (int h = 0; h < out->n_hits; h++) {
        int p = out->hits[h].passage;
        out->hit_text[h] = wiki_passage_text(w, p); out->hit_title[h] = wiki_passage_title(w, p);
        out->hit_url[h] = "https://en.wikipedia.org/wiki/"; out->hit_source[h] = "wikipedia";
    }
    /* Each of the top-K passages is read on its own, prefixed with its article title ("Blue whale: The blue whale is ...");
     * a single concatenated read lets the first passage dominate ("largest mammal" -> "rodents, bats" from [Mammal] while
     * [Blue whale] sits at rank 2). The winner is the passage with the best reader margin, nudged by retrieval score. */
    int K = getenv("KDR_WIKI_READK") ? atoi(getenv("KDR_WIKI_READK")) : 8; if (K > out->n_hits) K = out->n_hits; if (K > 8) K = 8;
    float wret = getenv("KDR_WIKI_WRET") ? (float)atof(getenv("KDR_WIKI_WRET")) : 12.0f;
    float wcov = getenv("KDR_WIKI_WCOV") ? (float)atof(getenv("KDR_WIKI_WCOV")) : 15.0f;
    float wvote = getenv("KDR_WIKI_WVOTE") ? (float)atof(getenv("KDR_WIKI_WVOTE")) : 0.5f;
    float wtitle = getenv("KDR_WIKI_WTITLE") ? (float)atof(getenv("KDR_WIKI_WTITLE")) : 3.0f;
    float wphr2 = getenv("KDR_WIKI_WPHR2") ? (float)atof(getenv("KDR_WIKI_WPHR2")) : 0.0f;
    Answer cand[8]; float sc[8], margin[8], covs[8]; char norm[8][160]; int nc = 0, cand_h[8];
    char buf[2048]; int qids[64]; int nq = brain_tokenize(b, question, qids, 62);
    /* KDR_WIKI_WIN: 0 = read the whole passage; 1 = read only the sentence with the best question-term coverage;
     * 2 = that sentence + the one before; 3 = before + best + after. KDR_WIKI_BOTH=1: read whole passage too, keep the better margin. */
    int win = getenv("KDR_WIKI_WIN") ? atoi(getenv("KDR_WIKI_WIN")) : 0, both = getenv("KDR_WIKI_BOTH") ? atoi(getenv("KDR_WIKI_BOTH")) : 0;
    for (int h = 0; h < K; h++) {
        snprintf(buf, sizeof buf, "%s: %s", out->hit_title[h], out->hit_text[h]);
        const char *t = buf; Answer *a = &cand[nc]; memset(a, 0, sizeof *a); a->n_hits = 1; a->hits[0] = out->hits[h];
        int seg = -1;
        if (win == 0 || both) seg = brain_read(b, question, &t, 1, a);
        if (win > 0) {
            int st[64], en[64]; int ns = split_sentences(out->hit_text[h], st, en, 64), bs = 0; float bc = -1;
            char sb[1024];
            for (int i = 0; i < ns; i++) {
                int len = en[i] - st[i]; if (len > (int)sizeof sb - 1) len = sizeof sb - 1;
                memcpy(sb, out->hit_text[h] + st[i], len); sb[len] = 0;
                float c = brain_term_coverage_ext(b, qids, nq, sb, w->bm_terms, w->bm_df, w->n_terms, w->N);
                if (c > bc) { bc = c; bs = i; }
            }
            if (ns > 0) {
                int i0 = bs, i1 = bs;
                if (win >= 2 && bs > 0) i0 = bs - 1;
                if (win == 3 && bs + 1 < ns) i1 = bs + 1;
                char wb[2048]; int len = en[i1] - st[i0]; if (len > (int)sizeof wb - 200) len = sizeof wb - 200;
                snprintf(wb, sizeof wb, "%s: %.*s", out->hit_title[h], len, out->hit_text[h] + st[i0]);
                Answer a2; memset(&a2, 0, sizeof a2); a2.n_hits = 1; a2.hits[0] = out->hits[h]; const char *t2 = wb;
                int seg2 = brain_read(b, question, &t2, 1, &a2);
                if (seg2 >= 0 && a2.answer[0] && (seg < 0 || !a->answer[0] || a2.span_score - a2.null_score > a->span_score - a->null_score)) { *a = a2; seg = seg2; }
            }
        }
        if (seg < 0 || !a->answer[0]) continue;
        resolve_pronoun(a, out->hit_title[h]);
        /* the span may start on the "Title: " prefix we added: drop it */
        { size_t tl = strlen(out->hit_title[h]); if (strncmp(a->answer, out->hit_title[h], tl) == 0 && a->answer[tl] == ':') { memmove(a->answer, a->answer + tl + 1, strlen(a->answer + tl + 1) + 1); while (a->answer[0] == ' ') memmove(a->answer, a->answer + 1, strlen(a->answer)); } }
        if (!a->answer[0]) continue;
        /* a passage that does not even mention the question's key words ("largest", "mammal") rarely holds the answer,
         * however confident the reader is. Coverage is idf-weighted against the wiki's own postings. */
        margin[nc] = a->span_score - a->null_score;
        float cov = brain_term_coverage_ext(b, qids, nq, buf, w->bm_terms, w->bm_df, w->n_terms, w->N);
        sc[nc] = margin[nc] + wret * (out->hits[h].score - out->hits[0].score) + wcov * (cov - 1.0f) + wphr2 * (phrase_match(b, qids, nq, buf) - 1.0f); covs[nc] = cov;
        /* definitional match: the extracted answer is the article's own subject ("Giraffe" from [Giraffe], "Mercury" from
         * [Mercury (element)]) - lead sentences define their subject, which is exactly what "what/which is the ..." asks for */
        if (wtitle > 0 && answer_is_title(a->answer, out->hit_title[h])) sc[nc] += wtitle;
        /* normalised answer for voting: lowercase, no leading article, no trailing punctuation */
        { const char *r = a->answer; if (!strncasecmp(r, "the ", 4)) r += 4; else if (!strncasecmp(r, "an ", 3)) r += 3; else if (!strncasecmp(r, "a ", 2)) r += 2;
          size_t o = 0; for (; *r && o < sizeof norm[nc] - 1; r++) { unsigned char ch = (unsigned char)*r; if (isalnum(ch) || ch == ' ' || ch >= 0x80) norm[nc][o++] = (char)tolower(ch); }
          while (o && norm[nc][o - 1] == ' ') o--; norm[nc][o] = 0; }
        cand_h[nc] = h; nc++;
    }
    /* answer voting: passages that extracted the same answer support each other ("blue whale" x2 beats one confident
     * misread "rodents, bats, and eulipotyphlans"; "Alexander Graham Bell" x3 beats "Martin Cooper") */
    Answer best; int best_h = -1; float best_score = -1e30f; memset(&best, 0, sizeof best);
    for (int i = 0; i < nc; i++) {
        float agg = sc[i];
        for (int j = 0; j < nc; j++) if (j != i && norm[i][0] && norm[j][0] && (strcmp(norm[i], norm[j]) == 0 ||
                (strlen(norm[i]) >= 4 && strlen(norm[j]) >= 4 && (strstr(norm[i], norm[j]) || strstr(norm[j], norm[i])))))
            agg += wvote * (margin[j] > 0 ? margin[j] : 0);
        if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "[wiki-read] #%d %-28.28s margin %6.2f ret %.3f cov %.2f base %6.2f voted %6.2f '%s'\n", cand_h[i], out->hit_title[cand_h[i]], margin[i], out->hits[cand_h[i]].score, covs[i], sc[i], agg, cand[i].answer);
        if (agg > best_score) { best_score = agg; best = cand[i]; best_h = cand_h[i]; }
    }
    /* nobody answered convincingly (all margins <= 0): the lead sentence alone often reads better than the whole passage
     * ("Giraffe: They are the tallest living terrestrial animals" -> 'Giraffe', margin +6.7 vs -6.3 on the full passage) */
    if (best_h >= 0 && best.span_score - best.null_score <= 0 && win == 0 && !both) {
        float bs2 = -1e30f; Answer best2; int best2_h = -1; memset(&best2, 0, sizeof best2);
        for (int i = 0; i < nc; i++) {
            int h = cand_h[i];
            int st[64], en[64]; int ns = split_sentences(out->hit_text[h], st, en, 64), bsn = 0; float bc = -1;
            char sb[1024];
            for (int k = 0; k < ns; k++) {
                int len = en[k] - st[k]; if (len > (int)sizeof sb - 1) len = sizeof sb - 1;
                memcpy(sb, out->hit_text[h] + st[k], len); sb[len] = 0;
                float c = brain_term_coverage_ext(b, qids, nq, sb, w->bm_terms, w->bm_df, w->n_terms, w->N);
                if (c > bc) { bc = c; bsn = k; }
            }
            if (ns == 0) continue;
            char wb[2048]; snprintf(wb, sizeof wb, "%s: %.*s", out->hit_title[h], en[bsn] - st[bsn], out->hit_text[h] + st[bsn]);
            Answer a2; memset(&a2, 0, sizeof a2); a2.n_hits = 1; a2.hits[0] = out->hits[h]; const char *t2 = wb;
            if (brain_read(b, question, &t2, 1, &a2) < 0 || !a2.answer[0]) continue;
            resolve_pronoun(&a2, out->hit_title[h]);
            float m2 = a2.span_score - a2.null_score;
            float s2 = m2 + wret * (out->hits[h].score - out->hits[0].score) + wcov * (bc - 1.0f) + (wtitle > 0 && answer_is_title(a2.answer, out->hit_title[h]) ? wtitle : 0);
            if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "[wiki-read2] #%d %-28.28s margin %6.2f cov %.2f score %6.2f '%s'\n", h, out->hit_title[h], m2, bc, s2, a2.answer);
            if (m2 > 0 && s2 > bs2) { bs2 = s2; best2 = a2; best2_h = h; }
        }
        if (best2_h >= 0 && bs2 > best_score) { best = best2; best_h = best2_h; }
    }
    if (best_h >= 0) {
        int p = out->hits[best_h].passage;
        snprintf(out->answer, sizeof out->answer, "%s", best.answer);
        snprintf(out->sentence, sizeof out->sentence, "%s", out->hit_text[best_h]);
        out->span_score = best.span_score; out->null_score = best.null_score; out->confidence = best.confidence;
        snprintf(out->title, sizeof out->title, "%s", wiki_passage_title(w, p));
        snprintf(out->url, sizeof out->url, "%s", wiki_passage_url(w, p));
        snprintf(out->source, sizeof out->source, "wikipedia");
        /* move the winning passage to the front so the composer sees it first */
        if (best_h > 0) { Hit hh = out->hits[best_h]; const char *tt = out->hit_text[best_h], *ti = out->hit_title[best_h];
            for (int i = best_h; i > 0; i--) { out->hits[i] = out->hits[i - 1]; out->hit_text[i] = out->hit_text[i - 1]; out->hit_title[i] = out->hit_title[i - 1]; }
            out->hits[0] = hh; out->hit_text[0] = tt; out->hit_title[0] = ti; }
    }
    out->ms_read = now_ms() - t0;
}
