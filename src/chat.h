#ifndef KDR_CHAT_H
#define KDR_CHAT_H
#include <stddef.h>
#include "brain.h"

/* Composer: a small instruction-tuned LLM (GGUF, via llama.cpp) that turns retrieved facts into a natural reply. */
typedef struct Chat Chat;

typedef struct {
    const char *role;   /* "user" | "assistant" */
    const char *text;
} ChatTurn;

/* callback receives each generated piece; return 0 to keep going, non-zero to stop */
typedef int (*chat_stream_fn)(const char *piece, void *ud);

Chat *chat_open(const char *gguf_path, int n_threads, int n_ctx);
void  chat_close(Chat *c);
const char *chat_model_desc(Chat *c);
int chat_is_live_question(const char *q);   /* weather/time/news: never attach facts */

/* Compose a reply for `question` using retrieval result `a` (NULL for pure chat or general-knowledge questions).
 * ood = 1 marks a real question the wikis do not cover (answered from the model's own knowledge).
 * history: previous turns (oldest first), may be NULL/0.
 * Returns number of generated tokens; writes UTF-8 reply into out (NUL terminated). */
int chat_reply(Chat *c, const Brain *b, const Answer *a, int ood, const char *question,
               const ChatTurn *history, int n_history,
               char *out, size_t out_cap, chat_stream_fn stream, void *ud,
               int *used_facts, double *ms);

#endif
