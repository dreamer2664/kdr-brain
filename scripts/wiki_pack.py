"""Pack extracted Wikipedia passages + their embeddings into ONE compact file: release/wiki.kdw

    python3 scripts/wiki_pack.py <ext_dir> [out.kdw] [--dims 128] [--max-passages N]

<ext_dir> must contain shuffled_articles.tsv, shuffled_passages.tsv and emb/part_*.f32 (384-float rows,
same order as shuffled_passages.tsv, produced by `kdr-brain embed`).

Layout ("KDRW" magic, same blob table of contents as brain.kdr):
  pca.mean[384] pca.proj[384*D]         float32      PCA projection used for questions at query time
  emb.q[N*D] int8 + emb.s[N] float32                 per-row quantized reduced passage embeddings
  text.blocks (zstd, with dictionary), text.dict, text.blk_off[u32], text.pass_blk[u32], text.pass_off[u32]
  pass.art[N] int32                                  passage -> article id
  art.title / art.path (\\0-joined), art.first[u32]  article -> title, url path, first passage index
  bm25.terms/bm25.offsets/bm25.postings/bm25.doclen  lexical index over wordpiece ids (df-capped)
  params int32[]
"""
import sys, os, struct, time, numpy as np, zstandard as zstd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bertref import WordPiece, quantize_int8_rows

def parse_args():
    a = sys.argv[1:]; ext = a[0]; out = os.path.join(ROOT, "release", "wiki.kdw"); dims = 128; maxp = None
    i = 1
    while i < len(a):
        if a[i] == "--dims": dims = int(a[i + 1]); i += 2
        elif a[i] == "--max-passages": maxp = int(a[i + 1]); i += 2
        else: out = a[i]; i += 1
    return ext, out, dims, maxp

def main():
    ext, out, D, maxp = parse_args()
    t0 = time.time()
    arts = [l.rstrip("\n").split("\t") for l in open(os.path.join(ext, "shuffled_articles.tsv"), encoding="utf-8")]
    pas = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(ext, "shuffled_passages.tsv"), encoding="utf-8")]
    # emb/part_XX.f32 correspond to embed_part_XX (line-aligned). A part may be incomplete (interrupted run): rows are
    # matched to their passage lines part by part, and the pack keeps the longest complete prefix over all parts.
    parts = sorted(f for f in os.listdir(os.path.join(ext, "emb")) if f.endswith(".f32"))
    rows = []; row_idx = []; base = 0
    for f in parts:
        n_lines = sum(1 for _ in open(os.path.join(ext, "embed_part_" + f[5:7]), encoding="utf-8"))
        v = np.fromfile(os.path.join(ext, "emb", f), dtype=np.float32); v = v[: (len(v) // 384) * 384].reshape(-1, 384)
        if len(v) > n_lines: v = v[:n_lines]
        if len(v) < n_lines: print(f"  note: {f} has {len(v)}/{n_lines} rows (partial run)")
        rows.append(v); row_idx.extend(range(base, base + len(v))); base += n_lines
    E = np.concatenate(rows); row_idx = np.array(row_idx)
    if maxp is not None: E = E[:maxp]; row_idx = row_idx[:maxp]
    pas = [pas[i] for i in row_idx]
    N = len(E)
    # drop a trailing partial article (its remaining passages were never embedded)
    if N and N < base:
        last_art = int(pas[-1][0])
        while N > 0 and int(pas[N - 1][0]) == last_art: N -= 1
    E = E[:N]; pas = pas[:N]
    # renumber articles densely (partial runs skip some)
    remap = {}; new_pas = []
    for a, t in pas:
        a = int(a)
        if a not in remap: remap[a] = len(remap)
        new_pas.append((str(remap[a]), t))
    arts = [arts[a] for a in sorted(remap, key=remap.get)]
    pas = new_pas
    art_ids = np.array([int(p[0]) for p in pas], dtype=np.int32)
    n_art = len(arts)
    texts = [p[1] for p in pas]
    print(f"{N} passages, {n_art} articles, {sum(len(t) for t in texts)/1e6:.1f} M chars", flush=True)

    blobs = []
    def blob(name, x): blobs.append((name, x.tobytes() if isinstance(x, np.ndarray) else x))

    # ---- PCA (covariance of a 20k sample, eigen-decomposition: tiny memory) + int8, projected in chunks
    mu = E.mean(0).astype(np.float32)
    idx = np.random.RandomState(0).permutation(N)[:20000]
    fit = (E[idx] - mu).astype(np.float64)
    cov = fit.T @ fit / max(1, len(fit)); del fit
    evals, evecs = np.linalg.eigh(cov); order = np.argsort(evals)[::-1]
    P = np.ascontiguousarray(evecs[:, order[:D]]).astype(np.float32)          # 384 x D
    Zq = np.zeros((N, D), np.int8); Zs = np.zeros(N, np.float32)
    for i in range(0, N, 20000):
        Z = (E[i:i + 20000] - mu) @ P
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        q, sc = quantize_int8_rows(Z); Zq[i:i + 20000] = q; Zs[i:i + 20000] = sc.reshape(-1)
    del E
    blob("pca.mean", mu); blob("pca.proj", P); blob("emb.q", Zq); blob("emb.s", Zs)
    print(f"pca {D} dims done (variance kept {100*evals[order[:D]].sum()/evals.sum():.1f}%) {time.time()-t0:.0f}s", flush=True)

    # ---- text: 64 KB blocks, zstd-19 with a trained dictionary
    BLK = 65536
    def iter_blocks():
        cur, cur_n = [], 0
        for i, t in enumerate(texts):
            b = t.encode("utf-8") + b"\0"
            if cur_n + len(b) > BLK and cur:
                yield b"".join(cur); cur, cur_n = [], 0
            pass_blk[i] = n_blocks[0]; pass_off[i] = cur_n
            cur.append(b); cur_n += len(b)
        if cur: yield b"".join(cur)
    pass_blk = np.zeros(N, np.uint32); pass_off = np.zeros(N, np.uint32); n_blocks = [0]
    # dictionary from a 1-in-k sample of blocks (first pass), then compress block by block (second pass)
    est_blocks = max(1, sum(len(t) + 1 for t in texts) // BLK + 1); step = max(1, est_blocks // 400)
    samples = []
    for k, b in enumerate(iter_blocks()):
        if k % step == 0: samples.append(b)
        n_blocks[0] += 1
    n_blocks[0] = 0
    d = zstd.train_dictionary(112 * 1024, samples) if len(samples) >= 8 else zstd.ZstdCompressionDict(b"")
    del samples
    cctx = zstd.ZstdCompressor(level=19, dict_data=d, write_content_size=True)
    comp = bytearray(); off = [0]
    for b in iter_blocks():
        comp += cctx.compress(b); off.append(len(comp)); n_blocks[0] += 1
    off = np.array(off, np.uint32)
    blob("text.dict", d.as_bytes()); blob("text.blocks", bytes(comp)); blob("text.blk_off", off)
    blob("text.pass_blk", pass_blk); blob("text.pass_off", pass_off)
    print(f"text: {n_blocks[0]} blocks, {len(comp)/1e6:.1f} MB compressed ({time.time()-t0:.0f}s)", flush=True)
    del comp

    # ---- articles
    blob("pass.art", art_ids)
    first = np.full(n_art, -1, np.int32)
    for i in range(N - 1, -1, -1): first[art_ids[i]] = i
    blob("art.first", first)
    blob("art.title", ("\0".join(a[2] for a in arts) + "\0").encode("utf-8"))
    blob("art.path", ("\0".join(a[1] for a in arts) + "\0").encode("utf-8"))

    # ---- BM25 postings (wordpiece ids, df-capped at 5% of passages); title tokens count as part of the passage.
    # Tokenised in the C engine (identical to query time) via `kdr-brain tokens` batch mode: much faster than Python.
    import subprocess
    binp = os.path.join(ROOT, "release", "kdr-brain-lite") if os.path.exists(os.path.join(ROOT, "release", "kdr-brain-lite")) else os.path.join(ROOT, "release", "kdr-brain")
    post = {}; doclen = np.zeros(N, np.int32)
    inp = "\n".join((arts[art_ids[i]][2] + " " + texts[i]).replace("\n", " ") for i in range(N)) + "\n"
    proc = subprocess.run([binp, os.path.join(ROOT, "release", "brain.kdr"), "tokenize-lines"], input=inp.encode("utf-8"), capture_output=True)
    lines = proc.stdout.decode("utf-8").split("\n")
    assert len(lines) >= N, (len(lines), N, proc.stderr[-500:])
    tl = []; dl = []; cl = []
    for i in range(N):
        ids = np.fromstring(lines[i], dtype=np.int32, sep=" ") if lines[i] else np.zeros(0, np.int32)
        doclen[i] = len(ids)
        u, c = np.unique(ids, return_counts=True)
        tl.append(u); dl.append(np.full(len(u), i, np.int32)); cl.append(c.astype(np.int32))
        if i % 50000 == 0: print(f"  bm25 {i}/{N}", flush=True)
    del lines, inp, proc
    T = np.concatenate(tl); Dd = np.concatenate(dl); C = np.concatenate(cl); del tl, dl, cl
    o = np.argsort(T, kind="stable"); T = T[o]; Dd = Dd[o]; C = C[o]
    uniq, starts, counts = np.unique(T, return_index=True, return_counts=True)
    # every term keeps its document frequency (the reader's coverage check needs real idf for common words like "world"),
    # but terms in more than 5% of the passages get no postings: they cost megabytes and add nothing to the ranking
    cap = max(1000, int(0.05 * N))
    post = {int(t): (Dd[s:s + n], C[s:s + n]) if n <= cap else (Dd[:0], C[:0]) for t, s, n in zip(uniq, starts, counts)}
    dfall = {int(t): int(n) for t, n in zip(uniq, counts)}
    terms = sorted(post)
    # postings as varint bytes: per term, a run of (doc_id delta, tf) LEB128 pairs; bm25.offsets = byte offsets, bm25.df = counts
    def varint(v, out):
        while True:
            b = v & 0x7F; v >>= 7
            if v: out.append(b | 0x80)
            else: out.append(b); return
    buf = bytearray(); offsets = [0]; dfs = []; npost = 0
    for t in terms:
        docs, cs = post[t]; prev = 0
        for doc, c in zip(docs.tolist(), cs.tolist()):
            varint(doc - prev, buf); varint(c, buf); prev = doc
        offsets.append(len(buf)); dfs.append(dfall[t]); npost += len(docs)
    blob("bm25.terms", np.array(terms, dtype=np.int32)); blob("bm25.offsets", np.array(offsets, dtype=np.uint32))
    blob("bm25.df", np.array(dfs, dtype=np.int32)); blob("bm25.postings", bytes(buf)); blob("bm25.doclen", doclen.astype(np.uint16))
    avgdl = float(doclen.mean()) if N else 1.0
    print(f"bm25: {len(terms)} terms, {npost/1e6:.1f} M postings, {len(buf)/1e6:.1f} MB varint ({time.time()-t0:.0f}s)", flush=True)

    params = np.array([N, n_art, D, n_blocks[0], len(terms), npost, int(round(avgdl * 1000)), BLK], dtype=np.int32)
    blob("params", params)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = len(blobs); toc = 16 + n * 64; body = bytearray(); layout = []
    for name, b in blobs:
        assert len(name) < 48
        body += b"\0" * ((-(toc + len(body))) % 64)
        layout.append((name, toc + len(body), len(b))); body += b
    with open(out, "wb") as f:
        f.write(b"KDRW"); f.write(struct.pack("<III", 1, n, 0))
        for name, o, s in layout: f.write(name.encode().ljust(48, b"\0")); f.write(struct.pack("<QQ", o, s))
        f.write(body)
    print("wrote", out, "%.1f MB" % (os.path.getsize(out) / 1e6), "in %.0fs" % (time.time() - t0))
    for name, b in blobs:
        if len(b) > 1e6: print("   %-16s %6.1f MB" % (name, len(b) / 1e6))

if __name__ == "__main__":
    main()
