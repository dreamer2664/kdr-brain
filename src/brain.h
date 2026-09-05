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
/* retrieval only */
int  brain_retrieve(Brain *b, const char *question, Hit *hits, int k);

#endif
