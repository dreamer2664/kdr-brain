"""Extract clean passages from a Kiwix Wikipedia ZIM (e.g. wikipedia_en_top_mini).

    python3 scripts/wiki_extract.py ~/.cache/kdr/wiki/wikipedia_en_top_mini_2026-06.zim out_dir

Writes out_dir/articles.tsv  (aid \\t path \\t title)
       out_dir/passages.tsv  (aid \\t text)            one passage per line, in article order
Passages are paragraphs, split at sentence boundaries when longer than MAX_CHARS and merged
with the next paragraph when shorter than MIN_CHARS. Title-only stubs are skipped.
"""
import sys, os, re, html as H, time
from libzim.reader import Archive

MAX_CHARS, MIN_CHARS = 600, 60
RE_TAGS = re.compile(r'<[^>]+>')
RE_SCRIPT = re.compile(r'<(script|style|table|math)[^>]*>.*?</\1>', re.S)
RE_P = re.compile(r'<p(?:\s[^>]*)?>(.*?)</p>', re.S)
RE_WS = re.compile(r'\s+')
RE_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')
RE_EMPTY_PAREN = re.compile(r'\(\s*[;,]?\s*\)')

def clean(frag):
    frag = re.sub(r'<sup[^>]*class="[^"]*reference[^"]*"[^>]*>.*?</sup>', '', frag, flags=re.S)
    frag = re.sub(r'<span[^>]*class="[^"]*mw-editsection[^"]*"[^>]*>.*?</span>', '', frag, flags=re.S)
    t = H.unescape(RE_TAGS.sub('', frag))
    t = RE_EMPTY_PAREN.sub('', t)
    t = RE_WS.sub(' ', t).strip()
    t = t.replace('[edit]', '')
    return t

def split_long(t):
    if len(t) <= MAX_CHARS: return [t]
    out, cur = [], ''
    for s in RE_SENT.split(t):
        if cur and len(cur) + 1 + len(s) > MAX_CHARS:
            out.append(cur); cur = s
        else:
            cur = (cur + ' ' + s) if cur else s
    if cur: out.append(cur)
    return out

def passages_from_html(raw):
    m = re.search(r'class="[^"]*mw-parser-output[^"]*"[^>]*>(.*)', raw, re.S)
    body = m.group(1) if m else raw
    i = body.find('class="zim-footer"')
    if i > 0: body = body[:i]
    body = RE_SCRIPT.sub('', body)
    paras = [clean(p) for p in RE_P.findall(body)]
    paras = [p for p in paras if p and len(p) >= 20]
    # merge tiny paragraphs forward
    merged = []
    for p in paras:
        if merged and len(merged[-1]) < MIN_CHARS: merged[-1] = merged[-1] + ' ' + p
        else: merged.append(p)
    out = []
    for p in merged: out.extend(split_long(p))
    return out

def main():
    zim, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    z = Archive(zim)
    t0 = time.time(); aid = 0; npass = 0; nchars = 0; skipped = 0
    with open(os.path.join(outdir, 'articles.tsv'), 'w', encoding='utf-8') as fa, \
         open(os.path.join(outdir, 'passages.tsv'), 'w', encoding='utf-8') as fp:
        for i in range(z.all_entry_count):
            e = z._get_entry_by_id(i)
            if e.is_redirect: continue
            it = e.get_item()
            if not it.mimetype.startswith('text/html'): continue
            raw = bytes(it.content).decode('utf-8', 'replace')
            ps = passages_from_html(raw)
            if not ps: skipped += 1; continue
            title = e.title.replace('\t', ' ').replace('\n', ' ')
            fa.write(f"{aid}\t{e.path}\t{title}\n")
            for p in ps:
                p = p.replace('\t', ' ')
                fp.write(f"{aid}\t{p}\n"); npass += 1; nchars += len(p)
            aid += 1
            if aid % 5000 == 0: print(f"  {aid} articles, {npass} passages, {nchars/1e6:.1f} M chars, {time.time()-t0:.0f}s", flush=True)
    print(f"done: {aid} articles, {npass} passages, {nchars/1e6:.1f} M chars, skipped {skipped} stubs, {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
