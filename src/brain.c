/* KDR Brain — dependency-free BERT retriever + reader inference engine (int8 weights).
 *
 * Pipeline for a question:
 *   1. WordPiece tokenize (exact HF BERT uncased behaviour)
 *   2. MiniLM-L6 (6-layer BERT, 384d) mean-pooled embedding  -> cosine vs. int8 passage index
 *   3. BM25 lexical scores over the same wordpiece ids        -> fused with dense score
 *   4. Top passages concatenated -> MiniLM-L12 SQuAD2 reader   -> best answer span
 */
#define _GNU_SOURCE
#include "brain.h"
#include "unicode_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ file format */
typedef struct { char name[48]; uint64_t off, size; } TocEntry;

typedef struct { const int8_t *q; const float *s; int rows, cols; } QMat;   /* int8 per-row quantized  */
typedef struct {
    QMat q, k, v, o, f1, f2;
    const float *qb, *kb, *vb, *ob, *f1b, *f2b, *ln1g, *ln1b, *ln2g, *ln2b;
} Layer;
typedef struct {
    int n_layers;
    QMat wte; const float *wpe, *tte, *emb_ln_g, *emb_ln_b;
    Layer L[12];
} BertModel;

struct Brain {
    uint8_t *map; size_t map_size;
    TocEntry *toc; int n_toc;
    int hidden, heads, inter, vocab, max_pos, n_pass, n_terms, n_post, n_stop;
    float avgdl;
    const int32_t *stop_ids;
    /* vocab */
    const char **vocab_str; /* pointers into map */
    /* hash table for wordpiece lookup */
    int32_t *vhash; int vhash_size;
    BertModel ret, rdr;
    const float *qa_w, *qa_b;
    const int8_t *emb_q; const float *emb_s;
    const char **p_text, **p_title, **p_url, **p_source;
    const int32_t *bm_terms, *bm_offsets, *bm_post, *bm_doclen;
    int n_threads;
    /* scratch */
    float *x, *qkv, *att, *ctx, *ff, *tmp;
};

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static const void *blob(Brain *b, const char *name, size_t *size) {
    for (int i = 0; i < b->n_toc; i++)
        if (strncmp(b->toc[i].name, name, 48) == 0) { if (size) *size = b->toc[i].size; return b->map + b->toc[i].off; }
    fprintf(stderr, "brain: missing blob %s\n", name); exit(2);
}
static QMat qmat(Brain *b, const char *pfx, const char *name, int rows, int cols) {
    char nm[64]; QMat m; size_t sz;
    snprintf(nm, 64, "%s.%s.q", pfx, name); m.q = blob(b, nm, &sz);
    if (sz != (size_t)rows * cols) { fprintf(stderr, "brain: bad size %s\n", nm); exit(2); }
    snprintf(nm, 64, "%s.%s.s", pfx, name); m.s = blob(b, nm, NULL);
    m.rows = rows; m.cols = cols; return m;
}
static const float *fvec(Brain *b, const char *pfx, const char *name) {
    char nm[64]; snprintf(nm, 64, "%s.%s", pfx, name); return blob(b, nm, NULL);
}
static void load_bert(Brain *b, BertModel *m, const char *pfx, int n_layers) {
    int H = b->hidden, I = b->inter;
    m->n_layers = n_layers;
    m->wte = qmat(b, pfx, "wte", b->vocab, H);
    m->wpe = fvec(b, pfx, "wpe"); m->tte = fvec(b, pfx, "tte");
    m->emb_ln_g = fvec(b, pfx, "emb_ln_g"); m->emb_ln_b = fvec(b, pfx, "emb_ln_b");
    for (int l = 0; l < n_layers; l++) {
        char n[32]; Layer *L = &m->L[l];
#define Q(field, nm, r, c) snprintf(n, 32, "l%d.%s", l, nm); L->field = qmat(b, pfx, n, r, c);
#define F(field, nm) snprintf(n, 32, "l%d.%s", l, nm); L->field = fvec(b, pfx, n);
        Q(q, "q", H, H) Q(k, "k", H, H) Q(v, "v", H, H) Q(o, "o", H, H) Q(f1, "f1", I, H) Q(f2, "f2", H, I)
        F(qb, "q.b") F(kb, "k.b") F(vb, "v.b") F(ob, "o.b") F(f1b, "f1.b") F(f2b, "f2.b")
        F(ln1g, "ln1_g") F(ln1b, "ln1_b") F(ln2g, "ln2_g") F(ln2b, "ln2_b")
#undef Q
#undef F
    }
}

/* ------------------------------------------------------------------ vocab hash */
static uint32_t fnv(const char *s, size_t n) { uint32_t h = 2166136261u; for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; } return h; }
static void build_vhash(Brain *b) {
    b->vhash_size = 1 << 16;
    b->vhash = malloc(sizeof(int32_t) * b->vhash_size);
    for (int i = 0; i < b->vhash_size; i++) b->vhash[i] = -1;
    for (int i = 0; i < b->vocab; i++) {
        uint32_t h = fnv(b->vocab_str[i], strlen(b->vocab_str[i])) & (b->vhash_size - 1);
        while (b->vhash[h] != -1) h = (h + 1) & (b->vhash_size - 1);
        b->vhash[h] = i;
    }
}
static int vocab_lookup(Brain *b, const char *s, size_t n) {
    uint32_t h = fnv(s, n) & (b->vhash_size - 1);
    while (b->vhash[h] != -1) {
        const char *v = b->vocab_str[b->vhash[h]];
        if (strlen(v) == n && memcmp(v, s, n) == 0) return b->vhash[h];
        h = (h + 1) & (b->vhash_size - 1);
    }
    return -1;
}

/* ------------------------------------------------------------------ tokenizer */
static int in_ranges(const uint32_t r[][2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (cp < r[mid][0]) hi = mid - 1; else if (cp > r[mid][1]) lo = mid + 1; else return 1; }
    return 0;
}
static int is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}
static const char *lower_map(uint32_t cp) {
    int lo = 0, hi = UNI_LOWER_MAP_N - 1;
    while (lo <= hi) { int mid = (lo + hi) / 2; if (uni_lower_map[mid].cp == cp) return uni_lower_map[mid].rep; if (uni_lower_map[mid].cp < cp) lo = mid + 1; else hi = mid - 1; }
    return NULL;
}
static int utf8_decode(const unsigned char *s, uint32_t *cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { *cp = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) { *cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
    if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) { *cp = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4; }
    *cp = 0xFFFD; return 1;
}
static int utf8_encode(uint32_t cp, char *o) {
    if (cp < 0x80) { o[0] = cp; return 1; }
    if (cp < 0x800) { o[0] = 0xC0 | (cp >> 6); o[1] = 0x80 | (cp & 0x3F); return 2; }
    if (cp < 0x10000) { o[0] = 0xE0 | (cp >> 12); o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F); return 3; }
    o[0] = 0xF0 | (cp >> 18); o[1] = 0x80 | ((cp >> 12) & 0x3F); o[2] = 0x80 | ((cp >> 6) & 0x3F); o[3] = 0x80 | (cp & 0x3F); return 4;
}

/* tokenize text -> ids + byte offsets (start,end) into the original text of each token's word */
typedef struct { int *ids; int *off_s; int *off_e; int n, cap; } TokOut;
static void tok_push(TokOut *t, int id, int s, int e) { if (t->n < t->cap) { t->ids[t->n] = id; if (t->off_s) { t->off_s[t->n] = s; t->off_e[t->n] = e; } t->n++; } }

static void wordpiece(Brain *b, const char *w, size_t n, TokOut *t, int s, int e) {
    int unk = vocab_lookup(b, "[UNK]", 5);
    /* count chars */
    size_t nchars = 0; for (size_t i = 0; i < n; ) { uint32_t cp; i += utf8_decode((const unsigned char *)w + i, &cp); nchars++; }
    if (nchars > 100) { tok_push(t, unk, s, e); return; }
    int tmp_ids[256]; int nt = 0;
    size_t start = 0; char buf[512];
    while (start < n) {
        size_t end = n; int cur = -1; size_t cur_end = 0;
        while (start < end) {
            size_t len = end - start; const char *piece = w + start; size_t plen = len;
            if (start > 0) { buf[0] = '#'; buf[1] = '#'; memcpy(buf + 2, piece, len); piece = buf; plen = len + 2; }
            int id = vocab_lookup(b, piece, plen);
            if (id >= 0) { cur = id; cur_end = end; break; }
            /* step back one utf8 char */
            end--; while (end > start && ((unsigned char)w[end] & 0xC0) == 0x80) end--;
        }
        if (cur < 0) { tok_push(t, unk, s, e); return; }
        if (nt < 256) tmp_ids[nt++] = cur;
        start = cur_end;
    }
    for (int i = 0; i < nt; i++) tok_push(t, tmp_ids[i], s, e);
}

static void tokenize(Brain *b, const char *text, TokOut *t) {
    char word[2048]; size_t wl = 0; int ws = -1;
    size_t i = 0, n = strlen(text);
#define FLUSH(endpos) do { if (wl) { wordpiece(b, word, wl, t, ws, (int)(endpos)); wl = 0; ws = -1; } } while (0)
    while (i < n) {
        uint32_t cp; int len = utf8_decode((const unsigned char *)text + i, &cp);
        int wstart = (int)i; i += len;
        if (cp == 0 || cp == 0xFFFD || in_ranges(uni_ctrl, UNI_CTRL_N, cp)) continue;
        if (in_ranges(uni_ws, UNI_WS_N, cp)) { FLUSH(wstart); continue; }
        if (is_cjk(cp) || in_ranges(uni_punct, UNI_PUNCT_N, cp)) {
            FLUSH(wstart);
            char o[8]; int ol = utf8_encode(cp, o); wordpiece(b, o, ol, t, wstart, (int)i);
            continue;
        }
        /* lowercase + strip accents */
        char o[16]; int ol;
        if (cp < 0x80) { o[0] = (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp; ol = 1; }
        else { const char *m = lower_map(cp); if (m) { ol = strlen(m); memcpy(o, m, ol); } else ol = utf8_encode(cp, o); }
        if (ol == 0) { if (ws < 0) ws = wstart; continue; }          /* pure combining mark: dropped */
        if (wl + ol < sizeof(word)) { if (ws < 0) ws = wstart; memcpy(word + wl, o, ol); wl += ol; }
    }
    FLUSH(n);
#undef FLUSH
}

int brain_tokenize(Brain *b, const char *text, int *ids, int max_ids) {
    TokOut t = { ids, NULL, NULL, 0, max_ids }; tokenize(b, text, &t); return t.n;
}

/* ------------------------------------------------------------------ math kernels */
/* y[rows] = W[rows x cols] (int8, per-row scale) @ x[cols] + bias ; T tokens at once: X is T x cols */
static void qmatmul(const QMat *W, const float *X, int T, const float *bias, float *Y, int n_threads) {
    const int cols = W->cols, rows = W->rows;
    (void)n_threads;
    #pragma omp parallel for schedule(static) num_threads(n_threads) if (rows * T > 4096)
    for (int r = 0; r < rows; r++) {
        const int8_t *w = W->q + (size_t)r * cols;
        /* dequantize row once into a small stack buffer (cols <= 1536) */
        float wf[1536];
        for (int c = 0; c < cols; c++) wf[c] = (float)w[c];
        const float s = W->s[r], bb = bias ? bias[r] : 0.f;
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * cols;
            float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
            int c = 0;
            for (; c + 8 <= cols; c += 8) {
                acc0 += wf[c] * x[c]; acc1 += wf[c + 1] * x[c + 1]; acc2 += wf[c + 2] * x[c + 2]; acc3 += wf[c + 3] * x[c + 3];
                acc4 += wf[c + 4] * x[c + 4]; acc5 += wf[c + 5] * x[c + 5]; acc6 += wf[c + 6] * x[c + 6]; acc7 += wf[c + 7] * x[c + 7];
            }
            float acc = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
            for (; c < cols; c++) acc += wf[c] * x[c];
            Y[(size_t)t * rows + r] = acc * s + bb;
        }
    }
}
static void layernorm(float *x, const float *g, const float *bta, int H) {
    float m = 0; for (int i = 0; i < H; i++) m += x[i]; m /= H;
    float v = 0; for (int i = 0; i < H; i++) { float d = x[i] - m; v += d * d; } v /= H;
    float inv = 1.0f / sqrtf(v + 1e-12f);
    for (int i = 0; i < H; i++) x[i] = (x[i] - m) * inv * g[i] + bta[i];
}
static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); }

/* run BERT; returns hidden states in b->x (T x H) */
static void bert_forward(Brain *b, BertModel *m, const int *ids, const int *tt, int T) {
    const int H = b->hidden, NH = b->heads, DH = H / NH, I = b->inter;
    float *x = b->x, *qkv = b->qkv, *att = b->att, *ctx = b->ctx, *ff = b->ff, *tmp = b->tmp;
    /* embeddings */
    for (int t = 0; t < T; t++) {
        const int8_t *w = m->wte.q + (size_t)ids[t] * H; float s = m->wte.s[ids[t]];
        for (int i = 0; i < H; i++) x[t * H + i] = w[i] * s + m->wpe[t * H + i] + m->tte[(tt ? tt[t] : 0) * H + i];
        layernorm(x + t * H, m->emb_ln_g, m->emb_ln_b, H);
    }
    const float scale = 1.0f / sqrtf((float)DH);
    for (int l = 0; l < m->n_layers; l++) {
        Layer *L = &m->L[l];
        float *Q = qkv, *K = qkv + (size_t)T * H, *V = qkv + (size_t)2 * T * H;
        qmatmul(&L->q, x, T, L->qb, Q, b->n_threads);
        qmatmul(&L->k, x, T, L->kb, K, b->n_threads);
        qmatmul(&L->v, x, T, L->vb, V, b->n_threads);
        /* attention */
        #pragma omp parallel for schedule(dynamic) num_threads(b->n_threads) if (T > 32)
        for (int ht = 0; ht < NH * T; ht++) {
            int h = ht / T, i = ht % T;
            float *sc = att + (size_t)ht * T;
            const float *q = Q + (size_t)i * H + h * DH;
            float mx = -1e30f;
            for (int j = 0; j < T; j++) {
                const float *k = K + (size_t)j * H + h * DH; float d = 0;
                for (int e = 0; e < DH; e++) d += q[e] * k[e];
                sc[j] = d * scale; if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0; for (int j = 0; j < T; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            float inv = 1.0f / sum;
            float *o = ctx + (size_t)i * H + h * DH; for (int e = 0; e < DH; e++) o[e] = 0;
            for (int j = 0; j < T; j++) { const float *v = V + (size_t)j * H + h * DH; float p = sc[j] * inv; for (int e = 0; e < DH; e++) o[e] += p * v[e]; }
        }
        qmatmul(&L->o, ctx, T, L->ob, tmp, b->n_threads);
        for (int t = 0; t < T; t++) { for (int i = 0; i < H; i++) x[t * H + i] += tmp[t * H + i]; layernorm(x + t * H, L->ln1g, L->ln1b, H); }
        qmatmul(&L->f1, x, T, L->f1b, ff, b->n_threads);
        for (size_t i = 0; i < (size_t)T * I; i++) ff[i] = gelu(ff[i]);
        qmatmul(&L->f2, ff, T, L->f2b, tmp, b->n_threads);
        for (int t = 0; t < T; t++) { for (int i = 0; i < H; i++) x[t * H + i] += tmp[t * H + i]; layernorm(x + t * H, L->ln2g, L->ln2b, H); }
    }
}

/* ------------------------------------------------------------------ open / close */
Brain *brain_open(const char *path, int n_threads) {
    int fd = open(path, O_RDONLY); if (fd < 0) { perror(path); return NULL; }
    struct stat st; fstat(fd, &st);
    Brain *b = calloc(1, sizeof(Brain));
    b->map_size = st.st_size;
    b->map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (b->map == MAP_FAILED) { perror("mmap"); free(b); return NULL; }
    if (memcmp(b->map, "KDRB", 4) != 0) { fprintf(stderr, "not a KDRB file\n"); return NULL; }
    b->n_toc = *(uint32_t *)(b->map + 8);
    b->toc = (TocEntry *)(b->map + 16);
    const int32_t *pr = blob(b, "params", NULL);
    b->hidden = pr[0]; b->heads = pr[1]; b->inter = pr[2]; b->vocab = pr[3]; b->max_pos = pr[4];
    int rl = pr[5], dl = pr[6]; b->n_pass = pr[7]; b->n_terms = pr[8]; b->n_post = pr[9]; b->avgdl = pr[10] / 1000.0f; b->n_stop = pr[11];
    b->stop_ids = blob(b, "stop_ids", NULL);
    b->n_threads = n_threads > 0 ? n_threads : 1;
    /* vocab */
    size_t vs; const char *v = blob(b, "vocab", &vs);
    b->vocab_str = malloc(sizeof(char *) * b->vocab);
    { const char *p = v; for (int i = 0; i < b->vocab; i++) { b->vocab_str[i] = p; p += strlen(p) + 1; } }
    build_vhash(b);
    load_bert(b, &b->ret, "ret", rl); load_bert(b, &b->rdr, "rdr", dl);
    b->qa_w = blob(b, "rdr.qa_w", NULL); b->qa_b = blob(b, "rdr.qa_b", NULL);
    b->emb_q = blob(b, "emb.q", NULL); b->emb_s = blob(b, "emb.s", NULL);
    const char *fields[4] = { "pass.text", "pass.title", "pass.url", "pass.source" }; const char ***dst[4] = { &b->p_text, &b->p_title, &b->p_url, &b->p_source };
    for (int f = 0; f < 4; f++) { const char *p = blob(b, fields[f], NULL); *dst[f] = malloc(sizeof(char *) * b->n_pass); for (int i = 0; i < b->n_pass; i++) { (*dst[f])[i] = p; p += strlen(p) + 1; } }
    b->bm_terms = blob(b, "bm25.terms", NULL); b->bm_offsets = blob(b, "bm25.offsets", NULL); b->bm_post = blob(b, "bm25.postings", NULL); b->bm_doclen = blob(b, "bm25.doclen", NULL);
    /* scratch for T <= 512 */
    size_t T = 512, H = b->hidden;
    b->x = malloc(sizeof(float) * T * H); b->qkv = malloc(sizeof(float) * 3 * T * H); b->att = malloc(sizeof(float) * b->heads * T * T);
    b->ctx = malloc(sizeof(float) * T * H); b->ff = malloc(sizeof(float) * T * b->inter); b->tmp = malloc(sizeof(float) * T * H);
    return b;
}
void brain_close(Brain *b) {
    if (!b) return;
    munmap(b->map, b->map_size); free(b->vocab_str); free(b->vhash); free((void *)b->p_text); free((void *)b->p_title); free((void *)b->p_url); free((void *)b->p_source);
    free(b->x); free(b->qkv); free(b->att); free(b->ctx); free(b->ff); free(b->tmp); free(b);
}
int brain_n_passages(const Brain *b) { return b->n_pass; }
const char *brain_passage_text(const Brain *b, int i) { return b->p_text[i]; }
const char *brain_passage_title(const Brain *b, int i) { return b->p_title[i]; }
const char *brain_passage_url(const Brain *b, int i) { return b->p_url[i]; }
const char *brain_passage_source(const Brain *b, int i) { return b->p_source[i]; }

/* ------------------------------------------------------------------ retrieval */
static void embed_question(Brain *b, const char *q, float *out) {
    int ids[130]; TokOut t = { ids + 1, NULL, NULL, 0, 126 }; tokenize(b, q, &t);
    int T = t.n + 2; ids[0] = vocab_lookup(b, "[CLS]", 5); ids[T - 1] = vocab_lookup(b, "[SEP]", 5);
    bert_forward(b, &b->ret, ids, NULL, T);
    const int H = b->hidden;
    for (int i = 0; i < H; i++) out[i] = 0;
    for (int t = 0; t < T; t++) for (int i = 0; i < H; i++) out[i] += b->x[t * H + i];
    float n = 0; for (int i = 0; i < H; i++) { out[i] /= T; n += out[i] * out[i]; }
    n = sqrtf(n) + 1e-9f; for (int i = 0; i < H; i++) out[i] /= n;
}
/* mean-pooled, L2-normalised embedding of any text (used to build passage indexes with the same quantized model) */
void brain_embed_text(Brain *b, const char *text, int max_tokens, float *out) {
    if (max_tokens > 510) max_tokens = 510; if (max_tokens < 8) max_tokens = 8;
    int ids[512]; TokOut t = { ids + 1, NULL, NULL, 0, max_tokens }; tokenize(b, text, &t);
    int T = t.n + 2; ids[0] = vocab_lookup(b, "[CLS]", 5); ids[T - 1] = vocab_lookup(b, "[SEP]", 5);
    bert_forward(b, &b->ret, ids, NULL, T);
    const int H = b->hidden;
    for (int i = 0; i < H; i++) out[i] = 0;
    for (int k = 0; k < T; k++) for (int i = 0; i < H; i++) out[i] += b->x[k * H + i];
    float n = 0; for (int i = 0; i < H; i++) { out[i] /= T; n += out[i] * out[i]; }
    n = sqrtf(n) + 1e-9f; for (int i = 0; i < H; i++) out[i] /= n;
}
/* crude stemming: two wordpieces "match" if one is a prefix of the other (>=4 chars) and they differ by <= 3 chars */
static int tok_related(Brain *b, int a, int c) {
    if (a == c) return 1;
    const char *x = b->vocab_str[a], *y = b->vocab_str[c];
    if (x[0] == '#' || y[0] == '#') return 0;
    size_t lx = strlen(x), ly = strlen(y), m = lx < ly ? lx : ly;
    if (m < 4 || (lx > ly ? lx - ly : ly - lx) > 3) return 0;
    return strncmp(x, y, m) == 0;
}
static int is_stop(Brain *b, int id) { for (int i = 0; i < b->n_stop; i++) if (b->stop_ids[i] == id) return 1; return 0; }
static int term_index(Brain *b, int id) { int lo = 0, hi = b->n_terms - 1; while (lo <= hi) { int m = (lo + hi) / 2; if (b->bm_terms[m] == id) return m; if (b->bm_terms[m] < id) lo = m + 1; else hi = m - 1; } return -1; }

int brain_retrieve(Brain *b, const char *question, Hit *hits, int k) {
    const int H = b->hidden, N = b->n_pass;
    float qv[384]; embed_question(b, question, qv);
    float *dense = malloc(sizeof(float) * N), *lex = calloc(N, sizeof(float));
    for (int i = 0; i < N; i++) {
        const int8_t *e = b->emb_q + (size_t)i * H; float d = 0;
        for (int j = 0; j < H; j++) d += e[j] * qv[j];
        dense[i] = d * b->emb_s[i];
    }
    /* bm25 */
    int qids[128]; int nq = brain_tokenize(b, question, qids, 128);
    const float k1 = 1.2f, bb = 0.75f;
    for (int a = 0; a < nq; a++) {
        int dup = 0; for (int c = 0; c < a; c++) if (qids[c] == qids[a]) dup = 1;
        if (dup || is_stop(b, qids[a])) continue;
        for (int ti = 0; ti < b->n_terms; ti++) {
            int term = b->bm_terms[ti]; float w = term == qids[a] ? 1.0f : (strlen(b->vocab_str[qids[a]]) >= 4 && tok_related(b, qids[a], term)) ? 0.5f : 0.0f;
            if (w == 0.0f) continue;
            int df = b->bm_offsets[ti + 1] - b->bm_offsets[ti];
            float idf = logf(1.0f + (N - df + 0.5f) / (df + 0.5f));
            for (int p = b->bm_offsets[ti]; p < b->bm_offsets[ti + 1]; p++) {
                int doc = b->bm_post[2 * p], tf = b->bm_post[2 * p + 1];
                lex[doc] += w * idf * tf * (k1 + 1) / (tf + k1 * (1 - bb + bb * b->bm_doclen[doc] / b->avgdl));
            }
        }
    }
    float lmax = 0; for (int i = 0; i < N; i++) if (lex[i] > lmax) lmax = lex[i];
    /* fuse + top-k */
    int n = 0;
    for (int i = 0; i < N; i++) {
        float s = 0.6f * dense[i] + 0.4f * (lmax > 0 ? lex[i] / lmax : 0);
        if (n < k || s > hits[n - 1].score) {
            int pos = n < k ? n++ : n - 1;
            while (pos > 0 && hits[pos - 1].score < s) { hits[pos] = hits[pos - 1]; pos--; }
            hits[pos].passage = i; hits[pos].score = s; hits[pos].dense = dense[i]; hits[pos].lexical = lex[i];
        }
    }
    free(dense); free(lex);
    return n;
}

/* ------------------------------------------------------------------ reader */
/* IDF-weighted share of the question's content terms that occur in `text` (0..1).
   Terms that appear nowhere in the whole knowledge base get the maximum weight, so a
   question about something the wikis never mention scores low. */
static float term_coverage(Brain *b, const int *qids, int nq, const char *text) {
    int tids[512]; int nt = brain_tokenize(b, text, tids, 512);
    float tot = 0, hit = 0; const int N = b->n_pass;
    for (int a = 0; a < nq; a++) {
        int dup = 0; for (int c = 0; c < a; c++) if (qids[c] == qids[a]) dup = 1;
        if (dup || is_stop(b, qids[a])) continue;
        const char *tok = b->vocab_str[qids[a]];
        if (tok[0] == '#' && tok[1] == '#') continue;                /* sub-word continuation */
        int ti = term_index(b, qids[a]);
        int df = ti < 0 ? 0 : b->bm_offsets[ti + 1] - b->bm_offsets[ti];
        float idf = logf(1.0f + (N - df + 0.5f) / (df + 0.5f));
        tot += idf;
        for (int i = 0; i < nt; i++) if (tids[i] == qids[a] || tok_related(b, qids[a], tids[i])) { hit += idf; break; }
    }
    return tot > 0 ? hit / tot : 1.0f;
}

/* same as term_coverage but with an external (e.g. Wikipedia) postings table for the idf weights */
float brain_term_coverage_ext(Brain *b, const int *qids, int nq, const char *text, const int32_t *terms, const int32_t *dfs, int n_terms, int N) {
    int tids[512]; int nt = brain_tokenize(b, text, tids, 512);
    float tot = 0, hit = 0;
    for (int a = 0; a < nq; a++) {
        int dup = 0; for (int c = 0; c < a; c++) if (qids[c] == qids[a]) dup = 1;
        if (dup || is_stop(b, qids[a])) continue;
        const char *tok = b->vocab_str[qids[a]];
        if (tok[0] == '#' && tok[1] == '#') continue;
        int lo = 0, hi = n_terms - 1, ti = -1; while (lo <= hi) { int m = (lo + hi) / 2; if (terms[m] == qids[a]) { ti = m; break; } if (terms[m] < qids[a]) lo = m + 1; else hi = m - 1; }
        int df = ti < 0 ? 0 : dfs[ti];
        float idf = logf(1.0f + (N - df + 0.5f) / (df + 0.5f));
        if (ti < 0 && strlen(tok) <= 3) continue;                    /* very common short words fell out of the df-capped table */
        tot += idf;
        for (int i = 0; i < nt; i++) if (tids[i] == qids[a] || tok_related(b, qids[a], tids[i])) { hit += idf; break; }
    }
    return tot > 0 ? hit / tot : 1.0f;
}

int brain_read(Brain *b, const char *question, const char **texts, int n_texts, Answer *out) {
    double t0 = now_ms();
    /* context = unique passages concatenated; remember segment boundaries */
    char ctx[8192]; int clen = 0; int seg_start[8], seg_end[8], seg_idx[8], nseg = 0;
    for (int h = 0; h < n_texts && nseg < 6; h++) {
        const char *t = texts[h]; size_t tl = strlen(t);
        int dup = 0;
        for (int s = 0; s < nseg; s++) { const char *u = texts[seg_idx[s]]; if (strcasestr(u, t) || strcasestr(t, u)) { dup = 1; break; } }
        if (dup) continue;
        if (clen + (int)tl + 2 >= (int)sizeof(ctx)) break;
        if (clen) ctx[clen++] = ' ';
        seg_start[nseg] = clen; memcpy(ctx + clen, t, tl); clen += tl; seg_end[nseg] = clen; seg_idx[nseg] = h; nseg++;
    }
    ctx[clen] = 0;
    /* question + context -> reader */
    const int MAXT = 384;
    int ids[512], tt[512], os_[512], oe[512];
    int qids[64]; int nq = brain_tokenize(b, question, qids, 62);
    int T = 0; ids[T] = vocab_lookup(b, "[CLS]", 5); tt[T] = 0; os_[T] = oe[T] = -1; T++;
    for (int i = 0; i < nq; i++) { ids[T] = qids[i]; tt[T] = 0; os_[T] = oe[T] = -1; T++; }
    ids[T] = vocab_lookup(b, "[SEP]", 5); tt[T] = 0; os_[T] = oe[T] = -1; T++;
    int cstart = T;
    TokOut ct = { ids + T, os_ + T, oe + T, 0, MAXT - T - 1 }; tokenize(b, ctx, &ct);
    for (int i = 0; i < ct.n; i++) tt[T + i] = 1;
    T += ct.n; ids[T] = vocab_lookup(b, "[SEP]", 5); tt[T] = 1; os_[T] = oe[T] = -1; T++;
    bert_forward(b, &b->rdr, ids, tt, T);
    const int H = b->hidden;
    float st[512], en[512];
    for (int t = 0; t < T; t++) {
        float s = b->qa_b[0], e = b->qa_b[1];
        for (int i = 0; i < H; i++) { s += b->qa_w[i] * b->x[t * H + i]; e += b->qa_w[H + i] * b->x[t * H + i]; }
        st[t] = s; en[t] = e;
    }
    out->null_score = st[0] + en[0];
    /* a span made only of question words ("the speed of light" for "what is the speed of light") is an echo, not an answer */
    static unsigned char inq[512];
    for (int t = cstart; t < T; t++) { inq[t] = 0; for (int i = 0; i < nq; i++) if (qids[i] == ids[t]) { inq[t] = 1; break; } }
    float best = -1e30f; int bi = -1, bj = -1;
    for (int i = cstart; i < T - 1; i++) {
        if (os_[i] < 0) continue;
        int echo = 1;
        for (int j = i; j < T - 1 && j < i + 24; j++) {
            if (!inq[j]) echo = 0;
            if (oe[j] < 0 || echo) continue;
            float sc = st[i] + en[j];
            if (sc > best) { best = sc; bi = i; bj = j; }
        }
    }
    out->span_score = best;
    float coverage = 0; int seg_ret = -1;
    out->answer[0] = 0; out->sentence[0] = 0;
    if (bi >= 0) {
        int a = os_[bi], z = oe[bj];
        int seg = 0; for (int s = 0; s < nseg; s++) if (a >= seg_start[s] && a < seg_end[s]) seg = s;
        if (z > seg_end[seg]) z = seg_end[seg];                      /* never cross into the next passage */
        int len = z - a; if (len > 500) len = 500; if (len < 0) len = 0;
        memcpy(out->answer, ctx + a, len); out->answer[len] = 0;
        while (len > 0 && (out->answer[len - 1] == ' ' || out->answer[len - 1] == '.' || out->answer[len - 1] == ',' || out->answer[len - 1] == ';')) out->answer[--len] = 0;
        snprintf(out->sentence, sizeof(out->sentence), "%s", texts[seg_idx[seg]]);
        seg_ret = seg_idx[seg];
        coverage = term_coverage(b, qids, nq, ctx);                  /* against everything the reader saw */
    }
    /* confidence: reader margin (does it prefer an answer over "no answer"?), dense similarity, term coverage */
    float margin = out->span_score - out->null_score;
    float c = 1.0f / (1.0f + expf(-0.35f * margin));
    float ev = out->n_hits ? (out->hits[0].dense - 0.25f) * 2.5f : 0; if (ev < 0) ev = 0; if (ev > 1) ev = 1;
    out->confidence = 0.5f * c + 0.2f * ev + 0.3f * coverage;
    /* an "answer" that is basically the whole passage is usually the reader giving up -> damp */
    { size_t al = strlen(out->answer), sl = strlen(out->sentence);
      if (sl && (al > 0.6 * sl || al > 140)) out->confidence *= 0.6f; }
    out->ms_read = now_ms() - t0;
    return seg_ret;
}

void brain_answer(Brain *b, const char *question, Answer *out) {
    memset(out, 0, sizeof(*out));
    double t0 = now_ms();
    out->n_hits = brain_retrieve(b, question, out->hits, 8);
    out->ms_retrieve = now_ms() - t0;
    const char *texts[8];
    for (int h = 0; h < out->n_hits; h++) {
        int p = out->hits[h].passage;
        texts[h] = b->p_text[p]; out->hit_text[h] = b->p_text[p]; out->hit_title[h] = b->p_title[p]; out->hit_url[h] = b->p_url[p]; out->hit_source[h] = b->p_source[p];
    }
    int seg = brain_read(b, question, texts, out->n_hits, out);
    if (seg >= 0) {
        int p = out->hits[seg].passage;
        snprintf(out->title, sizeof(out->title), "%s", b->p_title[p]);
        snprintf(out->url, sizeof(out->url), "%s", b->p_url[p]);
        snprintf(out->source, sizeof(out->source), "%s", b->p_source[p]);
    }
}
const char *brain_vocab_str(const Brain *b, int id) { return id >= 0 && id < b->vocab ? b->vocab_str[id] : ""; }
int brain_is_stop(const Brain *b, int id) { for (int i = 0; i < b->n_stop; i++) if (b->stop_ids[i] == id) return 1; return 0; }
