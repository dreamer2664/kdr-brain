#ifndef KDR_WIKI_H
#define KDR_WIKI_H
#include "brain.h"

/* General-knowledge pack (release/wiki.kdw): compressed Wikipedia passages + PCA-reduced int8 embeddings + BM25.
 * Retrieval reuses the brain's retriever (same wordpiece vocab / weights) and its extractive reader. */
typedef struct Wiki Wiki;

Wiki *wiki_open(const char *path, Brain *b);
void  wiki_close(Wiki *w);
int   wiki_n_passages(const Wiki *w);
int   wiki_n_articles(const Wiki *w);

/* Answers `question` from the wiki pack. Fills `out` like brain_answer (answer span, sentence, title, url, source="wikipedia",
 * confidence, hits). Hit.passage indexes are wiki passage ids: use wiki_passage_text/title to read them. */
void  wiki_answer(Wiki *w, Brain *b, const char *question, Answer *out);
/* passage text is decompressed on demand into an internal cache; the pointer stays valid until the next call */
const char *wiki_passage_text(Wiki *w, int i);
const char *wiki_passage_title(const Wiki *w, int i);
const char *wiki_passage_url(Wiki *w, int i);   /* https://en.wikipedia.org/wiki/<path>, internal buffer */
int   wiki_has_title(const Wiki *w, const char *word);   /* case-insensitive exact match against article titles */
int   wiki_answer_is_title(const Answer *a);
void  wiki_retrieve_debug(Wiki *w, Brain *b, const char *question, int k);   /* prints the top-k fused hits */              /* the extracted answer is the article's own subject ("Giraffe" from [Giraffe]) */

#endif
