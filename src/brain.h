#ifndef KDR_BRAIN_H
#define KDR_BRAIN_H
#include <stdint.h>
#include <stddef.h>

typedef struct Brain Brain;

typedef struct {
    int   passage;      /* index */
    float score;        /* fused retrieval score */
    float dense;        /* cosine similarity */
    float lexical;      /* bm25 */
} Hit;

typedef struct {
    char  answer[512];      /* extracted short answer ("" if none) */
    char  sentence[1024];   /* the supporting passage text */
    char  title[256];
    char  url[512];
    char  source[64];
    float span_score;       /* reader: start+end logits of the best span */
    float null_score;       /* reader: logit for "no answer" */
    float confidence;       /* 0..1 heuristic */
    int   n_hits;
    Hit   hits[8];
    /* the retrieved passages themselves (valid until the next answer call), so callers do not need to know which index a hit lives in */
    const char *hit_text[8], *hit_title[8], *hit_url[8], *hit_source[8];
    int   kind;             /* 0 = kingdom wikis, 1 = wikipedia pack */
    double ms_retrieve, ms_read;
} Answer;

Brain *brain_open(const char *path, int n_threads);
void   brain_close(Brain *b);
int    brain_n_passages(const Brain *b);
const char *brain_passage_text(const Brain *b, int i);
const char *brain_passage_title(const Brain *b, int i);
const char *brain_passage_url(const Brain *b, int i);
const char *brain_passage_source(const Brain *b, int i);

/* run the full pipeline */
void brain_answer(Brain *b, const char *question, Answer *out);
/* tokenizer exposed for tests: returns number of ids written */
int  brain_tokenize(Brain *b, const char *text, int *ids, int max_ids);
/* embed arbitrary text with the retriever (mean pooling, unit length); out has `hidden` (384) floats */
void brain_embed_text(Brain *b, const char *text, int max_tokens, float *out);
/* run the extractive reader over the given passage texts (out->hits/n_hits should already be filled, they feed the
 * confidence). Fills answer, sentence, span/null scores, confidence, ms_read. Returns the index (into texts) of the
 * passage holding the answer, or -1. */
int  brain_read(Brain *b, const char *question, const char **texts, int n_texts, Answer *out);
const char *brain_vocab_str(const Brain *b, int id);
float brain_term_coverage_ext(Brain *b, const int *qids, int nq, const char *text, const int32_t *terms, const int32_t *dfs, int n_terms, int N);
int  brain_is_stop(const Brain *b, int id);
/* retrieval only */
int  brain_retrieve(Brain *b, const char *question, Hit *hits, int k);

#endif
