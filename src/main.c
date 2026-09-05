/* kdr-brain: CLI + tiny HTTP server for the Dutch Robloxia neural brain.
 *   kdr-brain brain.kdr ask "question"
 *   kdr-brain brain.kdr serve 8080
 *   kdr-brain brain.kdr tokens "text"          (debug)
 *   kdr-brain brain.kdr bench                  (debug)
 */
#define _GNU_SOURCE
#include "brain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <time.h>

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
    for (int i = 0; i < a->n_hits && i < 5; i++) {
        int p = a->hits[i].passage;
        fprintf(f, "%s{\"score\":%.3f,\"dense\":%.3f,\"lexical\":%.2f,\"title\":", i ? "," : "", a->hits[i].score, a->hits[i].dense, a->hits[i].lexical);
        json_escape(f, brain_passage_title(b, p)); fputs(",\"text\":", f); json_escape(f, brain_passage_text(b, p));
        fputs(",\"url\":", f); json_escape(f, brain_passage_url(b, p)); fputs(",\"source\":", f); json_escape(f, brain_passage_source(b, p)); fputc('}', f);
    }
    fputs("]}", f);
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
        if (strncmp(path, "/api/ask", 8) == 0) {
            char q[2048] = {0};
            char *qs = strstr(path, "q=");
            if (qs) { snprintf(q, sizeof q, "%s", qs + 2); char *amp = strchr(q, '&'); if (amp) *amp = 0; url_decode(q); }
            else if (strcmp(method, "POST") == 0) { char *body = strstr(req, "\r\n\r\n"); if (body) { body += 4; char *qq = strstr(body, "\"q\":\""); if (qq) { qq += 5; char *e = strchr(qq, '"'); if (e) { size_t l = e - qq; if (l >= sizeof q) l = sizeof q - 1; memcpy(q, qq, l); q[l] = 0; } } } }
            Answer ans; brain_answer(b, q, &ans);
            char *buf = NULL; size_t bl = 0; FILE *f = open_memstream(&buf, &bl); write_answer_json(f, b, &ans, q); fclose(f);
            http_reply(c, 200, "application/json; charset=utf-8", buf, bl); free(buf);
            fprintf(stderr, "[ask] %-50.50s -> %-40.40s (%.0f+%.0f ms, conf %.2f)\n", q, ans.answer, ans.ms_retrieve, ans.ms_read, ans.confidence);
        } else if (strncmp(path, "/api/stats", 10) == 0) {
            char buf[256]; int l = snprintf(buf, sizeof buf, "{\"passages\":%d,\"ok\":true}", brain_n_passages(b));
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
    if (argc < 3) { fprintf(stderr, "usage: %s brain.kdr ask \"question\" | serve PORT | tokens TEXT | bench\n", argv[0]); return 1; }
    int threads = 2; const char *env = getenv("KDR_THREADS"); if (env) threads = atoi(env);
    Brain *b = brain_open(argv[1], threads); if (!b) return 1;
    if (strcmp(argv[2], "ask") == 0 && argc > 3) {
        Answer a; brain_answer(b, argv[3], &a);
        write_answer_json(stdout, b, &a, argv[3]); printf("\n");
    } else if (strcmp(argv[2], "tokens") == 0 && argc > 3) {
        int ids[512]; int n = brain_tokenize(b, argv[3], ids, 512);
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    } else if (strcmp(argv[2], "retrieve") == 0 && argc > 3) {
        Hit h[8]; int n = brain_retrieve(b, argv[3], h, 8);
        for (int i = 0; i < n; i++) printf("%d\t%.4f\t%.4f\t%.3f\t%s\n", h[i].passage, h[i].score, h[i].dense, h[i].lexical, brain_passage_text(b, h[i].passage));
    } else if (strcmp(argv[2], "serve") == 0) {
        serve(b, argc > 3 ? atoi(argv[3]) : 8080);
    } else if (strcmp(argv[2], "bench") == 0) {
        const char *qs[] = { "Who is the king of Dutch Robloxia?", "What is the capital?", "When was the Eloise Express built?" };
        for (int i = 0; i < 3; i++) { Answer a; brain_answer(b, qs[i], &a); printf("%s -> %s  [%.0f ms retrieve, %.0f ms read]\n", qs[i], a.answer, a.ms_retrieve, a.ms_read); }
    } else { fprintf(stderr, "bad command\n"); return 1; }
    brain_close(b); return 0;
}
