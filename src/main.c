/* kdr-brain: CLI + tiny HTTP server for the Dutch Robloxia neural brain.
 *   kdr-brain brain.kdr ask "question"
 *   kdr-brain brain.kdr serve 8080
 *   kdr-brain brain.kdr tokens "text"          (debug)
 *   kdr-brain brain.kdr bench                  (debug)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include "brain.h"
#include "wiki.h"
#ifndef KDR_NO_CHAT
#include "chat.h"
#else
typedef struct Chat Chat; typedef struct { const char *role; const char *text; } ChatTurn; typedef int (*chat_stream_fn)(const char *, void *);
static Chat *chat_open(const char *p, int t, int n) { (void)p; (void)t; (void)n; return NULL; }
static void chat_close(Chat *c) { (void)c; }
static const char *chat_model_desc(Chat *c) { (void)c; return ""; }
static int chat_is_live_question(const char *q) { (void)q; return 0; }
static int chat_reply(Chat *c, const Brain *b, const Answer *a, int ood, const char *q, const ChatTurn *h, int nh, char *out, size_t cap, chat_stream_fn s, void *ud, int *uf, double *ms)
{ (void)c; (void)b; (void)a; (void)ood; (void)q; (void)h; (void)nh; (void)s; (void)ud; (void)uf; (void)ms; snprintf(out, cap, "chat disabled"); return 0; }
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <time.h>

static Chat *g_chat = NULL;   /* optional composer (GGUF chat model) */
static Wiki *g_wiki = NULL;   /* optional general-knowledge pack (wiki.kdw) */
static int g_history_max = 6;  /* previous turns fed back to the composer */
extern const char kdr_index_html[];   /* web UI, embedded at build time (webui.c) */
extern const unsigned int kdr_index_html_len;

static void json_escape(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = *s;
        if (c == '"') fputs("\\\"", f); else if (c == '\\') fputs("\\\\", f); else if (c == '\n') fputs("\\n", f);
        else if (c == '\r') fputs("\\r", f); else if (c == '\t') fputs("\\t", f); else if (c < 0x20) fprintf(f, "\\u%04x", c); else fputc(c, f);
    }
    fputc('"', f);
}
static void write_answer_json(FILE *f, Brain *b, const Answer *a, const char *q) {
    fputs("{\"question\":", f); json_escape(f, q);
    fputs(",\"answer\":", f); json_escape(f, a->answer);
    fputs(",\"sentence\":", f); json_escape(f, a->sentence);
    fputs(",\"title\":", f); json_escape(f, a->title);
    fputs(",\"url\":", f); json_escape(f, a->url);
    fputs(",\"source\":", f); json_escape(f, a->source);
    fprintf(f, ",\"confidence\":%.3f,\"span_score\":%.2f,\"null_score\":%.2f,\"ms_retrieve\":%.1f,\"ms_read\":%.1f,\"hits\":[",
            a->confidence, a->span_score, a->null_score, a->ms_retrieve, a->ms_read);
    (void)b;
    for (int i = 0; i < a->n_hits && i < 8; i++) {
        fprintf(f, "%s{\"score\":%.3f,\"dense\":%.3f,\"lexical\":%.2f,\"title\":", i ? "," : "", a->hits[i].score, a->hits[i].dense, a->hits[i].lexical);
        json_escape(f, a->hit_title[i] ? a->hit_title[i] : ""); fputs(",\"text\":", f); json_escape(f, a->hit_text[i] ? a->hit_text[i] : "");
        fputs(",\"url\":", f); json_escape(f, a->hit_url[i] ? a->hit_url[i] : ""); fputs(",\"source\":", f); json_escape(f, a->hit_source[i] ? a->hit_source[i] : ""); fputc('}', f);
    }
    fprintf(f, "],\"kind\":\"%s\"}", a->kind == 1 ? "wikipedia" : "kingdom");
}


/* ---------------------------------------------------------------- chat (composer) */
/* minimal JSON string extractor: finds "key":"value" (handles \" \\ \n \uXXXX) into out */
static int json_get_str(const char *json, const char *key, char *out, size_t cap) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *k = strstr(json, pat); if (!k) return 0;
    const char *p = strchr(k + strlen(pat), ':'); if (!p) return 0; p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return 0; p++;
    size_t o = 0;
    while (*p && *p != '"') {
        unsigned c = (unsigned char)*p;
        if (c == '\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break; case 'r': c = '\r'; break;
                case 'u': { unsigned v = 0; if (sscanf(p + 1, "%4x", &v) == 1) { p += 4;
                    if (v < 0x80) c = v; else if (v < 0x800) { if (o + 2 < cap) { out[o++] = 0xC0 | (v >> 6); out[o++] = 0x80 | (v & 63); } p++; continue; }
                    else { if (o + 3 < cap) { out[o++] = 0xE0 | (v >> 12); out[o++] = 0x80 | ((v >> 6) & 63); out[o++] = 0x80 | (v & 63); } p++; continue; } } break; }
                default: c = (unsigned char)*p;
            }
        }
        if (o + 1 < cap) out[o++] = (char)c;
        p++;
    }
    out[o] = 0; return 1;
}
/* parse "history":[{"role":"user","text":"..."},...] into turns (bounded) */
static int json_get_history(const char *json, ChatTurn *turns, char *store, size_t store_cap, int max_turns) {
    const char *h = strstr(json, "\"history\""); if (!h) return 0;
    const char *p = strchr(h, '['); if (!p) return 0;
    int n = 0; size_t used = 0;
    while (n < max_turns) {
        const char *obj = strchr(p, '{'); if (!obj) break;
        const char *end = strchr(obj, '}'); if (!end) break;
        char tmp[4096]; size_t l = (size_t)(end - obj + 1); if (l >= sizeof tmp) l = sizeof tmp - 1; memcpy(tmp, obj, l); tmp[l] = 0;
        char role[16] = "user", text[2048] = "";
        json_get_str(tmp, "role", role, sizeof role); json_get_str(tmp, "text", text, sizeof text);
        size_t tl = strlen(text) + 1;
        if (text[0] && used + tl < store_cap) { memcpy(store + used, text, tl); turns[n].role = strcmp(role, "assistant") == 0 ? "assistant" : "user"; turns[n].text = store + used; used += tl; n++; }
        p = end + 1; if (*p == ']') break;
    }
    return n;
}

/* Follow-up questions ("and his wife?") carry no entity: borrow proper nouns from the last turns for retrieval. */
static int needs_context(const char *q) {
    static const char *pron[] = { "his","her","hers","their","theirs","he","she","they","them","him","it","its","that","this","there","those","these","one","else","also","too","about", NULL };
    int words = 0; char w[64]; int wl = 0; int hit = 0;
    for (const char *p = q;; p++) {
        if (*p && (isalnum((unsigned char)*p) || *p == '\'')) { if (wl < 63) w[wl++] = (char)tolower((unsigned char)*p); }
        else if (wl) { w[wl] = 0; words++; for (int i = 0; pron[i]; i++) if (strcmp(w, pron[i]) == 0) hit = 1; wl = 0; }
        if (!*p) break;
    }
    if (hit) return 1;
    /* a short question that names a capitalised thing of its own ("What's the population of Tokyo?") is complete;
     * only really bare ones ("and the queen?", "capital?") borrow the previous turn's entities */
    if (words <= 5) {
        int caps = 0, first = 1;
        for (const char *p = q; *p; p++) {
            if (isalnum((unsigned char)*p)) { if (!first && isupper((unsigned char)*p) && (p == q || !isalnum((unsigned char)p[-1]))) caps++; first = 0; }
            else if (*p == ' ') first = 0;
        }
        return caps == 0;
    }
    return 0;
}
/* set of words that occur in passage titles (lowercased): the kingdom's entity vocabulary */
static char **g_ent = NULL; static int g_n_ent = 0;
static int ent_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static void build_entities(const Brain *b) {
    int cap = 4096; g_ent = malloc(sizeof(char *) * cap);
    for (int i = 0; i < brain_n_passages(b); i++) {
        const char *t = brain_passage_title(b, i);
        while (*t) {
            while (*t && !isalnum((unsigned char)*t)) t++;
            const char *st = t; while (*t && (isalnum((unsigned char)*t) || *t == '\'')) t++;
            size_t l = (size_t)(t - st); if (l < 3 || l > 40) continue;
            char w[48]; for (size_t k = 0; k < l; k++) w[k] = (char)tolower((unsigned char)st[k]); w[l] = 0;
            static const char *skip[] = { "the","and","for","von","van","der","und","his","every","page","wiki","site","footer","image","caption","how","works","notable","official","poster","propaganda",
                                          "france","germany","siberia","netherlands","london",   /* real-world names in game titles are not kingdom entities */
                                          /* ordinary English words that happen to occur in passage titles: they must not drag general questions
                                           * ("which metal is liquid at room temperature", "smallest country") into the kingdom index */
                                          "country","island","government","culture","science","education","finance","housing","defence","defense","military","army","navy","air","police",
                                          "laws","law","games","game","train","station","palace","castle","home","bank","royal","national","history","economy","politics","migration","affairs",
                                          "social","foreign","health","healthcare","ministry","minister","ministers","cabinet","parliament","constitution","citizens","citizen","region","regions",
                                          "city","capital","kingdom","king","queen","prince","princess","emperor","empire","republic","state","states","nation","people","language","anthem","flag","motto",
                                          "standard","battle","war","wars","peace","luxury","forest","express","lines","enterprises","company","companies","group","history","timeline","overview","list","map","guide","rules","ranks","commands", NULL };
            int sk = 0; for (int j = 0; skip[j]; j++) if (strcmp(w, skip[j]) == 0) { sk = 1; break; }
            if (sk) continue;
            char *key = w; if (bsearch(&key, g_ent, g_n_ent, sizeof(char *), ent_cmp)) continue;
            if (g_n_ent < cap) { g_ent[g_n_ent++] = strdup(w); qsort(g_ent, g_n_ent, sizeof(char *), ent_cmp); }
        }
    }
}
static int is_entity_word(const char *w) {
    char lw[48]; size_t l = strlen(w); if (l >= sizeof lw) return 0;
    for (size_t k = 0; k <= l; k++) lw[k] = (char)tolower((unsigned char)w[k]);
    char *e = strstr(lw, "'s"); if (e && !e[2]) *e = 0;
    char *key = lw; return g_ent && bsearch(&key, g_ent, g_n_ent, sizeof(char *), ent_cmp) != NULL;
}
static void add_proper_nouns(const char *text, char *out, size_t cap, int max_words) {
    static const char *stop[] = { "The","A","An","I","It","If","You","Yes","No","In","On","At","He","She","They","We","This","That","There","Hi","Hello","Hey","How","What","Who","Where","When","Why","Which","Sure","Of","And","Or","But","So","Is","Are","Was","Were","Do","Does","Did","Can","Could","Would","Should","My","Your","His","Her","Its","Our","Their","Not","To","For","With","As","By","From","About", NULL };
    int added = 0; const char *p = text;
    while (*p && added < max_words) {
        while (*p && !isalnum((unsigned char)*p)) p++;
        const char *st = p; while (*p && (isalnum((unsigned char)*p) || *p == '\'' || *p == '&')) p++;
        size_t l = (size_t)(p - st); if (!l) continue;
        if (!isupper((unsigned char)*st)) continue;
        char w[64]; if (l >= sizeof w) l = sizeof w - 1; memcpy(w, st, l); w[l] = 0;
        int skip = 0; for (int i = 0; stop[i]; i++) if (strcmp(w, stop[i]) == 0) { skip = 1; break; }
        if (skip || !is_entity_word(w)) continue;
        char pat[70]; snprintf(pat, sizeof pat, " %s ", w); if (strstr(out, pat)) continue;   /* dedupe */
        size_t ol = strlen(out); if (ol + l + 2 >= cap) break;
        out[ol] = ' '; memcpy(out + ol + 1, w, l); out[ol + 1 + l] = ' '; out[ol + 2 + l] = 0; added++;
    }
}
static void build_retrieval_query(const char *q, const ChatTurn *turns, int nh, char *out, size_t cap) {
    snprintf(out, cap, "%s", q);
    if (nh == 0 || !needs_context(q)) return;
    char extra[512] = " ";
    for (int i = nh - 1; i >= 0 && i >= nh - 2; i--) add_proper_nouns(turns[i].text, extra, sizeof extra, 8);   /* last assistant + last user turn */
    if (strlen(extra) > 1) { size_t l = strlen(out); snprintf(out + l, cap - l, " (%s)", extra + 1); }
}


/* ---- routing: does this message need the knowledge base, or is it small talk? ---- */
/* "the king", "our navy", "the eight laws": kingdom-life words that are ordinary English too. They count as an entity unless the
 * question clearly points at the real world ("the king of Spain", "the world's largest army"). */
static int world_cue(const char *q) {
    static const char *cue[] = { " world ", " earth ", " europe ", " asia ", " africa ", " america ", " country ", " countries ", " planet ", " history ", " real life ", " real world ",
        " france ", " spain ", " england ", " britain ", " uk ", " usa ", " united states ", " germany ", " italy ", " japan ", " china ", " india ", " russia ", " netherlands ", " belgium ",
        " roman ", " british ", " french ", " german ", " dutch republic ", " ancient ", " medieval ", " century ", " bc ", " ad ", NULL };
    char n[2100]; size_t o = 1; n[0] = ' ';
    for (const char *p = q; *p && o < sizeof n - 2; p++) { unsigned char c = (unsigned char)*p; n[o++] = isalnum(c) ? (char)tolower(c) : ' '; }
    n[o++] = ' '; n[o] = 0;
    for (int i = 0; cue[i]; i++) if (strstr(n, cue[i])) return 1;
    return 0;
}
/* a capitalised word (not sentence-initial) that is not a kingdom entity but is a Wikipedia article title ("Australia", "Spain",
 * "Beethoven") marks a real-world question */
static int foreign_proper_noun(const char *q) {
    if (!g_wiki) return 0;
    const char *p = q; int first = 1;
    while (*p) {
        while (*p && !isalnum((unsigned char)*p)) { if (*p == '.' || *p == '?' || *p == '!') first = 1; p++; }
        const char *st = p; while (*p && (isalnum((unsigned char)*p) || *p == '\'' || *p == '-')) p++;
        size_t l = (size_t)(p - st); if (!l) continue;
        int cap = isupper((unsigned char)*st) && l >= 3; int was_first = first; first = 0;
        if (!cap || was_first) continue;
        char w[64]; if (l >= sizeof w) continue; memcpy(w, st, l); w[l] = 0;
        char *ap = strstr(w, "'s"); if (ap && !ap[2]) *ap = 0;
        if (is_entity_word(w)) continue;
        if (wiki_has_title(g_wiki, w)) return 1;
    }
    return 0;
}
static int has_entity(const char *q) {
    /* a few kingdom-life words that do not occur in passage titles but clearly ask about the kingdom ("what can I play?") */
    static const char *domain[] = { "play", "game", "games", "rank", "ranks", "citizen", "citizens", "server", NULL };
    static const char *life[] = { "king", "queen", "prince", "princess", "kingdom", "minister", "ministers", "ministry", "cabinet", "parliament", "army", "navy", "airforce", "military", "police",
        "laws", "law", "anthem", "hymn", "motto", "flag", "capital", "region", "regions", "train", "palace", "castle", "consulate", "royal", NULL };
    char w[64]; int wl = 0; int wc = -1;
    for (const char *p = q;; p++) {
        if (*p && (isalnum((unsigned char)*p) || *p == '\'')) { if (wl < 63) w[wl++] = *p; }
        else if (wl) { w[wl] = 0; wl = 0; if (strlen(w) >= 3 && is_entity_word(w)) return 1;
                       for (int i = 0; domain[i]; i++) if (strcasecmp(w, domain[i]) == 0) return 1;
                       for (int i = 0; life[i]; i++) if (strcasecmp(w, life[i]) == 0) { if (wc < 0) wc = world_cue(q) || foreign_proper_noun(q); if (!wc) return 1; } }
        if (!*p) break;
    }
    return 0;
}
/* 1 = opinion/persona/small talk that never needs facts; 2 = greeting/thanks/bye (needs facts only if an entity is named); 0 = neither */
static int smalltalk_tier(const char *q) {
    char norm[2100]; size_t o = 1; norm[0] = ' ';
    for (const char *p = q; *p && o < sizeof norm - 2; p++) { unsigned char c = (unsigned char)*p; norm[o++] = isalnum(c) || c == '\'' ? (char)tolower(c) : ' '; }
    norm[o++] = ' '; norm[o] = 0;
    static const char *tierA[] = { " how are you ", " how r u ", " who are you ", " what are you ", " what can you do ", " what do you do ", " favourite ", " favorite ", " do you like ", " do you love ",
        " what do you think ", " your opinion ", " joke ", " bored ", " boring ", " i m bored ", " tell me about yourself ", " are you a bot ", " are you human ", " are you real ", " what s your name ", " your name ", NULL };
    static const char *tierB[] = { " hello ", " hi ", " hey ", " hiya ", " yo ", " sup ", " what s up ", " whats up ", " good morning ", " good evening ", " good night ", " good afternoon ", " thank ", " thanks ", " thx ", " ty ",
        " bye ", " goodbye ", " see you ", " see ya ", " cya ", " lol ", " haha ", " nice ", " cool ", " ok ", " okay ", " great ", " awesome ", " wow ", " love you ", " you re great ", " you are great ", " good bot ", " well done ", NULL };
    for (int i = 0; tierA[i]; i++) if (strstr(norm, tierA[i])) return 1;
    /* tier B only counts when it makes up (almost) the whole message: short and no question word */
    int words = 0; for (size_t i = 1; i < o; i++) if (norm[i] == ' ' && norm[i - 1] != ' ') words++;
    static const char *qw[] = { " who ", " what ", " when ", " where ", " which ", " why ", " how ", " tell ", " list ", " name ", " explain ", " give ", " show ", " is ", " are ", " does ", " do ", " can ", " did ", " was ", NULL };
    int hasq = 0; for (int i = 0; qw[i]; i++) if (strstr(norm, qw[i])) { hasq = 1; break; }
    if (words <= 6 && !hasq) for (int i = 0; tierB[i]; i++) if (strstr(norm, tierB[i])) return 2;
    return 0;
}
/* "what is 15% of 200", "is 17 a prime number", "which is bigger, 0.9 or 0.11": numbers + an arithmetic word/operator -> never retrieval
 * (the entity vocabulary contains words like "prime", and the reader would hand the composer nonsense) */
static int is_arithmetic(const char *q) {
    int digits = 0; for (const char *p = q; *p; p++) if (isdigit((unsigned char)*p)) digits++;
    if (!digits) return 0;
    char norm[2100]; size_t o = 1; norm[0] = ' ';
    for (const char *p = q; *p && o < sizeof norm - 2; p++) { unsigned char c = (unsigned char)*p; norm[o++] = isalnum(c) || c == '.' ? (char)tolower(c) : ' '; }
    norm[o++] = ' '; norm[o] = 0;
    static const char *ops[] = { " plus ", " minus ", " times ", " divided ", " multiplied ", " percent ", " square root ", " squared ", " cubed ", " prime number ",
        " bigger ", " larger ", " smaller ", " greater ", " less than ", " more than ", " how many are left ", " sum of ", " product of ", " difference between ", " average of ", " half of ", " twice ", " equals ", NULL };
    for (int i = 0; ops[i]; i++) if (strstr(norm, ops[i])) return 1;
    if (strchr(q, '%')) return 1;
    for (const char *p = q + 1; *p; p++) if (strchr("+-*/x^", *p)) {   /* digit <op> digit, spaces allowed */
        const char *l = p - 1; while (l > q && *l == ' ') l--; const char *r = p + 1; while (*r == ' ') r++;
        if (isdigit((unsigned char)*l) && isdigit((unsigned char)*r)) return 1;
    }
    return 0;
}
enum { ROUTE_CHAT = 0, ROUTE_FACTS = 1, ROUTE_OOD = 2, ROUTE_WIKI = 3 };
/* the Wikipedia pack answers a general question when the reader found a span it prefers over "no answer" with decent confidence */
static int wiki_lookup(Brain *b, const char *q, Answer *ans) {
    if (!g_wiki) return 0;
    Answer wa; wiki_answer(g_wiki, b, q, &wa);
    float thr = getenv("KDR_WIKI_THR") ? (float)atof(getenv("KDR_WIKI_THR")) : 0.45f;
    /* accepted when the reader prefers the span over "no answer" - or, for a definitional match (the answer is the article's own
     * subject: "Mercury" from [Mercury (element)] for "which metal is liquid at room temperature"), with a small negative margin */
    if (wa.n_hits > 0 && wa.answer[0] && wa.confidence >= thr && (wa.span_score >= wa.null_score || (wiki_answer_is_title(&wa) && wa.span_score >= wa.null_score - 6.0f))) { *ans = wa; return 1; }
    if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "[wiki] rejected: conf %.2f span %.1f null %.1f answer '%s' (%s)\n", wa.confidence, wa.span_score, wa.null_score, wa.answer, wa.title);
    return 0;
}
/* runs retrieval when useful. ROUTE_FACTS: attach ans; ROUTE_CHAT: small talk, no facts; ROUTE_OOD: a real question the wikis
 * do not cover -> the composer answers from its own knowledge and the reply gets flagged "(not from the wikis)" */
static int route(Brain *b, const char *q, const ChatTurn *turns, int nh, Answer *ans, char *rq, size_t rq_cap) {
    memset(ans, 0, sizeof *ans);
    build_retrieval_query(q, turns, nh, rq, rq_cap);
    int ent = has_entity(rq), tier = smalltalk_tier(q);
    if (tier == 1 || (tier == 2 && !ent) || chat_is_live_question(q)) return ROUTE_CHAT;
    /* a short follow-up ("why that one?", "really?") right after an opinion/small-talk turn stays in chat mode */
    /* ...but only for reactions ("why?", "really?", "how come", "and yours?"), not for a fresh question that happens to be
     * short ("What's the population of Tokyo?" after an opinion turn must still reach the indexes) */
    if (nh >= 2 && !has_entity(q) && strlen(q) < 40 && smalltalk_tier(turns[nh - 2].text) == 1) {
        static const char *react[] = { "why", "really", "how come", "and you", "and yours", "same", "me too", "cool", "nice", "ok", "okay", "lol", "haha", "seriously", "sure", "no way", "wow", "what about you", NULL };
        char lq[64]; size_t i = 0; for (; q[i] && i < sizeof lq - 1; i++) lq[i] = (char)tolower((unsigned char)q[i]); lq[i] = 0;
        for (int k = 0; react[k]; k++) if (strncmp(lq, react[k], strlen(react[k])) == 0) return ROUTE_CHAT;
        if (strlen(q) < 12) return ROUTE_CHAT;
    }
    if (is_arithmetic(q)) return ROUTE_OOD;
    /* No kingdom entity word in the question: both indexes are consulted. The kingdom wins when its reader is confident
     * and prefers a span over "no answer" (Firewood Forest, Koninklijke Marechaussee - proper names the title list misses);
     * otherwise the Wikipedia pack answers (capital of France -> the kingdom reader says "Dunhag" but with span < null). */
    if (!ent) {
        brain_answer(b, rq, ans);
        /* "sure": the reader clearly prefers a span (margin > 8: the kingdom's own facts score 10-15, real-world questions
         * -5..+7 there), whatever the length-damped confidence says ("Imperial Lines (shipping) and Silver Wings Air" is long) */
        int kingdom_sure = ans->span_score > ans->null_score + 8.0f && (ans->confidence >= 0.6f || strlen(ans->answer) > 25) && strcasecmp(ans->answer, "Dutch Robloxia") != 0;
        /* a kingdom passage whose TITLE is in the question ("What is the Royal Bank?" -> passage [Royal Bank]) with a decent
         * margin also counts: the wiki pack has a "Royal Bank of Scotland" article that would otherwise take over */
        if (!kingdom_sure && ans->title[0] && ans->span_score > ans->null_score + 4.0f && strlen(ans->title) >= 6 && strcasestr(rq, ans->title)
            && strcasecmp(ans->title, "Dutch Robloxia") != 0) kingdom_sure = 1;
        if (!kingdom_sure && ans->n_hits > 0 && ans->hit_title[0] && strlen(ans->hit_title[0]) >= 6 && strcasestr(rq, ans->hit_title[0])
            && strcasecmp(ans->hit_title[0], "Dutch Robloxia") != 0 && ans->span_score > ans->null_score + 4.0f) kingdom_sure = 1;
        if (!kingdom_sure) { Answer wa; if (wiki_lookup(b, rq, &wa)) { *ans = wa; return ROUTE_WIKI; } }
        else if (getenv("KDR_DEBUG_PROMPT")) fprintf(stderr, "[route] kingdom index sure (conf %.2f, span %.1f > null %.1f): %s\n", ans->confidence, ans->span_score, ans->null_score, ans->answer);
    } else brain_answer(b, rq, ans);
    /* "what's the capital? and who rules there?" - retrieve the second question too and interleave its hits */
    { const char *qm = strchr(q, '?');
      if (qm && qm[1]) {
        const char *p2 = qm + 1; while (*p2 && (isspace((unsigned char)*p2) || *p2 == ',')) p2++;
        if (strncasecmp(p2, "and ", 4) == 0) p2 += 4;
        if (strlen(p2) >= 6) {
            char q2[2600]; snprintf(q2, sizeof q2, "%s", p2);
            char extra[512] = " "; add_proper_nouns(q, extra, sizeof extra, 6);
            if (strlen(extra) > 1) { size_t l = strlen(q2); snprintf(q2 + l, sizeof q2 - l, " (%s)", extra + 1); }
            Answer a2; brain_answer(b, q2, &a2);
            Hit merged[8]; int nm = 0;
            for (int i = 0; i < 8 && nm < 8; i++) {
                if (i < ans->n_hits) merged[nm++] = ans->hits[i];
                if (i < a2.n_hits && nm < 8) { int dup = 0; for (int k = 0; k < nm; k++) if (merged[k].passage == a2.hits[i].passage) dup = 1; if (!dup) merged[nm++] = a2.hits[i]; }
            }
            memcpy(ans->hits, merged, sizeof(Hit) * nm); ans->n_hits = nm;
            if (a2.confidence > ans->confidence + 0.2f) ans->confidence = a2.confidence;   /* facts are worth attaching if either part is answerable */
        }
      } }
    /* without a named entity we also need the reader to prefer a span over "no answer" (capital of France -> Versailles otherwise) */
    if (ent || (ans->confidence >= 0.30f && ans->span_score >= ans->null_score)) return ROUTE_FACTS;
    return ROUTE_OOD;
}
/* the reply names something only the kingdom has -> it was answered from the cheat-sheet, not general knowledge */
static int mentions_kingdom(const char *reply) {
    static const char *names[] = { "robloxia", "goudhof", "dunhag", "oysterdam", "hendrikdam", "heuvelmeer", "vaderveen", "duurn", "bladland", "barkworth",
        "eloise", "friso", "buizerd", "mosselman", "theerots", "houtman", "silverrail", "marechaussee", "nickiscoolinroblox", "kdrwiki", "dutchbloxia", NULL };
    char low[8192]; size_t i = 0; for (; reply[i] && i < sizeof low - 1; i++) low[i] = (char)tolower((unsigned char)reply[i]); low[i] = 0;
    for (int k = 0; names[k]; k++) if (strstr(low, names[k])) return 1;
    return 0;
}
static const char *OOD_NOTE = " (not from the wikis)";
static const char *WIKI_NOTE = " (from Wikipedia)";
/* deterministic provenance flag: the 0.5B composer cannot be trusted to add it itself */
static int needs_ood_note(int rt, const char *reply) {
    if (rt != ROUTE_OOD || !reply[0]) return 0;
    if (strcasestr(reply, "not from the wiki") || strcasestr(reply, "wikis say")) return 0;
    return !mentions_kingdom(reply);
}
static const char *note_for(int rt, const char *reply) {
    if (rt == ROUTE_WIKI && reply[0] && !strcasestr(reply, "wikipedia")) return WIKI_NOTE;
    return needs_ood_note(rt, reply) ? OOD_NOTE : NULL;
}

typedef struct { int fd; int ok; } StreamCtx;
static int stream_piece(const char *piece, void *ud) {
    StreamCtx *sc = ud; if (!sc->ok) return 1;
    /* SSE event with JSON-escaped token */
    char buf[512]; size_t o = 0; const char *pre = "data: {\"t\":\""; memcpy(buf, pre, strlen(pre)); o = strlen(pre);
    for (const unsigned char *s = (const unsigned char *)piece; *s && o < sizeof buf - 12; s++) {
        if (*s == '"') { buf[o++] = '\\'; buf[o++] = '"'; } else if (*s == '\\') { buf[o++] = '\\'; buf[o++] = '\\'; }
        else if (*s == '\n') { buf[o++] = '\\'; buf[o++] = 'n'; } else if (*s < 0x20) { o += snprintf(buf + o, 8, "\\u%04x", *s); } else buf[o++] = *s;
    }
    memcpy(buf + o, "\"}\n\n", 4); o += 4;
    ssize_t w = send(sc->fd, buf, o, MSG_NOSIGNAL); if (w <= 0) { sc->ok = 0; return 1; }
    return 0;
}
static void write_chat_json(FILE *f, Brain *b, const Answer *a, const char *q, const char *reply, int used_facts, double ms_chat) {
    fputs("{\"reply\":", f); json_escape(f, reply);
    fprintf(f, ",\"used_facts\":%s,\"ms_chat\":%.0f,\"retrieval\":", used_facts ? "true" : "false", ms_chat);
    write_answer_json(f, b, a, q); fputs("}", f);
}

/* ---------------------------------------------------------------- HTTP */
static int url_decode(char *s) {
    char *o = s; for (; *s; s++) {
        if (*s == '+') *o++ = ' ';
        else if (*s == '%' && s[1] && s[2]) { char h[3] = { s[1], s[2], 0 }; *o++ = (char)strtol(h, NULL, 16); s += 2; }
        else *o++ = *s;
    }
    *o = 0; return 0;
}
static void send_all(int fd, const char *buf, size_t n) { while (n) { ssize_t w = send(fd, buf, n, MSG_NOSIGNAL); if (w <= 0) return; buf += w; n -= w; } }
static void http_reply(int fd, int code, const char *ctype, const char *body, size_t n) {
    char h[512]; int hl = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                                   code, code == 200 ? "OK" : "Not Found", ctype, n);
    send_all(fd, h, hl); send_all(fd, body, n);
}
static void serve(Brain *b, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = htonl(INADDR_ANY) };
    if (bind(s, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); exit(1); }
    listen(s, 16);
    fprintf(stderr, "kdr-brain listening on http://0.0.0.0:%d  (%d passages)\n", port, brain_n_passages(b));
    for (;;) {
        int c = accept(s, NULL, NULL); if (c < 0) continue;
        char req[8192]; ssize_t n = recv(c, req, sizeof req - 1, 0); if (n <= 0) { close(c); continue; }
        req[n] = 0;
        char method[8] = {0}, path[4096] = {0}; sscanf(req, "%7s %4095s", method, path);
        if (strncmp(path, "/api/chat", 9) == 0 && g_chat) {
            /* POST {"q":"...","history":[{"role":"user","text":".."},...],"stream":true} */
            char *body = strstr(req, "\r\n\r\n"); if (body) body += 4; else body = "";
            /* if the body did not fully arrive in the first recv, read the rest (Content-Length) */
            char *cl = strcasestr(req, "content-length:"); size_t want = cl ? strtoul(cl + 15, NULL, 10) : 0; size_t have = strlen(body);
            char *full = NULL;
            if (want > have) { full = malloc(want + 1); memcpy(full, body, have); size_t got = have; while (got < want) { ssize_t r = recv(c, full + got, want - got, 0); if (r <= 0) break; got += r; } full[got] = 0; body = full; }
            static char q[2048], store[16384]; ChatTurn turns[12]; q[0] = 0;
            json_get_str(body, "q", q, sizeof q);
            int nh = json_get_history(body, turns, store, sizeof store, g_history_max);
            int do_stream = strstr(body, "\"stream\":true") != NULL;
            if (!q[0]) { http_reply(c, 400, "text/plain", "missing q", 9); free(full); close(c); continue; }
            char rq[2600]; Answer ans; int facts = route(b, q, turns, nh, &ans, rq, sizeof rq);
            static char reply[8192]; int used = 0; double ms = 0;
            if (do_stream) {
                const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                send_all(c, hdr, strlen(hdr));
                StreamCtx sc = { c, 1 };
                chat_reply(g_chat, b, facts == ROUTE_FACTS || facts == ROUTE_WIKI ? &ans : NULL, facts == ROUTE_OOD, q, turns, nh, reply, sizeof reply, stream_piece, &sc, &used, &ms);
                { const char *note = note_for(facts, reply); if (note) { strncat(reply, note, sizeof reply - strlen(reply) - 1); stream_piece(note, &sc); } }
                char *buf = NULL; size_t bl = 0; FILE *f = open_memstream(&buf, &bl); fputs("data: ", f); write_chat_json(f, b, &ans, q, reply, used, ms); fputs("\n\ndata: [DONE]\n\n", f); fclose(f);
                send_all(c, buf, bl); free(buf);
            } else {
                chat_reply(g_chat, b, facts == ROUTE_FACTS || facts == ROUTE_WIKI ? &ans : NULL, facts == ROUTE_OOD, q, turns, nh, reply, sizeof reply, NULL, NULL, &used, &ms);
                { const char *note = note_for(facts, reply); if (note) strncat(reply, note, sizeof reply - strlen(reply) - 1); }
                char *buf = NULL; size_t bl = 0; FILE *f = open_memstream(&buf, &bl); write_chat_json(f, b, &ans, q, reply, used, ms); fclose(f);
                http_reply(c, 200, "application/json; charset=utf-8", buf, bl); free(buf);
            }
            fprintf(stderr, "[chat] %-45.45s -> %-50.50s (%s conf %.2f, %.0f ms%s)\n", q, reply, facts == ROUTE_FACTS ? "facts" : facts == ROUTE_WIKI ? "wiki " : facts == ROUTE_OOD ? "ood  " : "chat ", ans.confidence, ms, strcmp(rq, q) ? " +ctx" : "");
            free(full);
        } else if (strncmp(path, "/api/ask", 8) == 0) {
            char q[2048] = {0};
            char *qs = strstr(path, "q=");
            if (qs) { snprintf(q, sizeof q, "%s", qs + 2); char *amp = strchr(q, '&'); if (amp) *amp = 0; url_decode(q); }
            else if (strcmp(method, "POST") == 0) { char *body = strstr(req, "\r\n\r\n"); if (body) { body += 4; char *qq = strstr(body, "\"q\":\""); if (qq) { qq += 5; char *e = strchr(qq, '"'); if (e) { size_t l = e - qq; if (l >= sizeof q) l = sizeof q - 1; memcpy(q, qq, l); q[l] = 0; } } } }
            Answer ans; if (!(g_wiki && !has_entity(q) && wiki_lookup(b, q, &ans))) brain_answer(b, q, &ans);
            char *buf = NULL; size_t bl = 0; FILE *f = open_memstream(&buf, &bl); write_answer_json(f, b, &ans, q); fclose(f);
            http_reply(c, 200, "application/json; charset=utf-8", buf, bl); free(buf);
            fprintf(stderr, "[ask] %-50.50s -> %-40.40s (%.0f+%.0f ms, conf %.2f)\n", q, ans.answer, ans.ms_retrieve, ans.ms_read, ans.confidence);
        } else if (strncmp(path, "/api/stats", 10) == 0) {
            char buf[512]; int l = snprintf(buf, sizeof buf, "{\"passages\":%d,\"ok\":true,\"chat\":%s,\"model\":\"%s\",\"wiki_passages\":%d,\"wiki_articles\":%d}", brain_n_passages(b), g_chat ? "true" : "false", g_chat ? chat_model_desc(g_chat) : "", g_wiki ? wiki_n_passages(g_wiki) : 0, g_wiki ? wiki_n_articles(g_wiki) : 0);
            http_reply(c, 200, "application/json", buf, l);
        } else if (strcmp(path, "/") == 0 || strncmp(path, "/index.html", 11) == 0 || strncmp(path, "/?", 2) == 0) {
            http_reply(c, 200, "text/html; charset=utf-8", kdr_index_html, kdr_index_html_len);
        } else {
            http_reply(c, 404, "text/plain", "not found", 9);
        }
        close(c);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s brain.kdr [--chat model.gguf] [--wiki wiki.kdw] ask|chat|wiki \"question\" | serve PORT | tokens TEXT | embed [max_tokens] | bench\n", argv[0]); return 1; }
    int threads = 2; const char *env = getenv("KDR_THREADS"); if (env) threads = atoi(env);
    Brain *b = brain_open(argv[1], threads); if (!b) return 1;
    const char *model = getenv("KDR_CHAT_MODEL");
    const char *wiki = getenv("KDR_WIKI");
    for (int i = 2; i + 1 < argc; i++) if (strcmp(argv[i], "--chat") == 0) { model = argv[i + 1]; for (int j = i; j + 2 < argc; j++) argv[j] = argv[j + 2]; argc -= 2; break; }
    for (int i = 2; i + 1 < argc; i++) if (strcmp(argv[i], "--wiki") == 0) { wiki = argv[i + 1]; for (int j = i; j + 2 < argc; j++) argv[j] = argv[j + 2]; argc -= 2; break; }
    if (model && model[0]) { g_chat = chat_open(model, threads, 2048); if (!g_chat) return 1; fprintf(stderr, "composer: %s\n", chat_model_desc(g_chat)); }
    if (wiki && wiki[0]) { g_wiki = wiki_open(wiki, b); if (!g_wiki) return 1; fprintf(stderr, "wikipedia pack: %d passages, %d articles\n", wiki_n_passages(g_wiki), wiki_n_articles(g_wiki)); }
    build_entities(b);
    if (strcmp(argv[2], "ask") == 0 && argc > 3) {
        Answer a; brain_answer(b, argv[3], &a);
        write_answer_json(stdout, b, &a, argv[3]); printf("\n");
    } else if (strcmp(argv[2], "wiki") == 0 && argc > 3) {
        if (!g_wiki) { fprintf(stderr, "no wiki pack: pass --wiki wiki.kdw or set KDR_WIKI\n"); return 1; }
        Answer a; wiki_answer(g_wiki, b, argv[3], &a);
        write_answer_json(stdout, b, &a, argv[3]); printf("\n");
    } else if (strcmp(argv[2], "chat") == 0 && argc > 3) {
        if (!g_chat) { fprintf(stderr, "no composer: pass --chat model.gguf or set KDR_CHAT_MODEL\n"); return 1; }
        Answer a; char rq[2600]; int facts = route(b, argv[3], NULL, 0, &a, rq, sizeof rq);
        static char reply[8192]; int used = 0; double ms = 0;
        chat_reply(g_chat, b, facts == ROUTE_FACTS || facts == ROUTE_WIKI ? &a : NULL, facts == ROUTE_OOD, argv[3], NULL, 0, reply, sizeof reply, NULL, NULL, &used, &ms);
        { const char *note = note_for(facts, reply); if (note) strncat(reply, note, sizeof reply - strlen(reply) - 1); }
        write_chat_json(stdout, b, &a, argv[3], reply, used, ms); printf("\n");
    } else if (strcmp(argv[2], "tokens") == 0 && argc > 3) {
        int ids[512]; int n = brain_tokenize(b, argv[3], ids, 512);
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    } else if (strcmp(argv[2], "retrieve") == 0 && argc > 3) {
        Hit h[8]; int n = brain_retrieve(b, argv[3], h, 8);
        for (int i = 0; i < n; i++) printf("%d\t%.4f\t%.4f\t%.3f\t%s\n", h[i].passage, h[i].score, h[i].dense, h[i].lexical, brain_passage_text(b, h[i].passage));
    } else if (strcmp(argv[2], "wiki-retrieve") == 0 && argc > 3) {
        if (!g_wiki) { fprintf(stderr, "no wiki pack\n"); return 1; }
        wiki_retrieve_debug(g_wiki, b, argv[3], argc > 4 ? atoi(argv[4]) : 16);
    } else if (strcmp(argv[2], "read") == 0 && argc > 4) {
        /* debug: run the reader on one given passage */
        Answer a; memset(&a, 0, sizeof a); const char *t = argv[4]; brain_read(b, argv[3], &t, 1, &a);
        printf("span %.2f null %.2f margin %.2f conf %.2f -> '%s'\n", a.span_score, a.null_score, a.span_score - a.null_score, a.confidence, a.answer);
    } else if (strcmp(argv[2], "tokenize-lines") == 0) {
        /* bulk tokenizer for index building: one text per stdin line -> space-separated wordpiece ids per line */
        static char line[65536]; static int ids[8192];
        while (fgets(line, sizeof line, stdin)) { size_t l = strlen(line); while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
            int n = brain_tokenize(b, line, ids, 8192); for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]); putchar('\n'); }
    } else if (strcmp(argv[2], "embed") == 0) {
        /* bulk embedding: one text per stdin line -> 384 little-endian floats per line on stdout (index building) */
        int maxt = argc > 3 ? atoi(argv[3]) : 192; static char line[65536]; float v[384]; long n = 0;
        while (fgets(line, sizeof line, stdin)) { size_t l = strlen(line); while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
            brain_embed_text(b, line, maxt, v); fwrite(v, sizeof(float), 384, stdout); if (++n % 1000 == 0) { fprintf(stderr, "\r  embedded %ld", n); fflush(stdout); } }
        fflush(stdout); fprintf(stderr, "\r  embedded %ld\n", n);
    } else if (strcmp(argv[2], "serve") == 0) {
        serve(b, argc > 3 ? atoi(argv[3]) : 8080);
    } else if (strcmp(argv[2], "bench") == 0) {
        const char *qs[] = { "Who is the king of Dutch Robloxia?", "What is the capital?", "When was the Eloise Express built?" };
        for (int i = 0; i < 3; i++) { Answer a; brain_answer(b, qs[i], &a); printf("%s -> %s  [%.0f ms retrieve, %.0f ms read]\n", qs[i], a.answer, a.ms_retrieve, a.ms_read); }
    } else { fprintf(stderr, "bad command\n"); return 1; }
    if (g_chat) chat_close(g_chat);
    if (g_wiki) wiki_close(g_wiki);
    brain_close(b); return 0;
}
