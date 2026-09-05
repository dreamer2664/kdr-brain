"""Pack everything the C engine needs into ONE binary file: release/brain.kdr

Layout: "KDRB" magic, u32 version, u32 blob count, then a table of contents of
(48-byte name, u64 offset, u64 size) entries, then the 64-byte aligned blobs:
  vocab, int8 per-row quantized weights of both BERT models, int8 passage embeddings,
  passage texts/titles/urls/sources, BM25 postings, parameters.
"""
import sys, os, json, struct, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bertref import load_safetensors, WordPiece, Bert, quantize_int8_rows, dequant_int8_rows

MODELS = os.environ.get("KDR_MODELS", "/tmp/build")
RET = os.path.join(MODELS, "model.safetensors"); RDR = os.path.join(MODELS, "reader", "model.safetensors")
VOCAB = os.path.join(ROOT, "data", "vocab.txt")
OUT = os.path.join(ROOT, "release", "brain.kdr")

tok = WordPiece(VOCAB)
P = json.load(open(os.path.join(ROOT, "data", "passages.json")))

blobs = []
def blob(name, arr_or_bytes):
    b = arr_or_bytes.tobytes() if isinstance(arr_or_bytes, np.ndarray) else arr_or_bytes
    blobs.append((name, b)); return name

def pack_bert(W, prefix, n_layers, tag):
    def q8(name, key):
        w = W[prefix + key].astype(np.float32)
        q, s = quantize_int8_rows(w)
        blob(f"{tag}.{name}.q", q); blob(f"{tag}.{name}.s", s.reshape(-1))
    def f32(name, key):
        blob(f"{tag}.{name}", W[prefix + key].astype(np.float32))
    q8("wte", "embeddings.word_embeddings.weight")
    f32("wpe", "embeddings.position_embeddings.weight")
    f32("tte", "embeddings.token_type_embeddings.weight")
    f32("emb_ln_g", "embeddings.LayerNorm.weight"); f32("emb_ln_b", "embeddings.LayerNorm.bias")
    for l in range(n_layers):
        pre = f"encoder.layer.{l}."
        for nm, key in [("q", "attention.self.query"), ("k", "attention.self.key"), ("v", "attention.self.value"),
                        ("o", "attention.output.dense"), ("f1", "intermediate.dense"), ("f2", "output.dense")]:
            q8(f"l{l}.{nm}", pre + key + ".weight"); f32(f"l{l}.{nm}.b", pre + key + ".bias")
        f32(f"l{l}.ln1_g", pre + "attention.output.LayerNorm.weight"); f32(f"l{l}.ln1_b", pre + "attention.output.LayerNorm.bias")
        f32(f"l{l}.ln2_g", pre + "output.LayerNorm.weight"); f32(f"l{l}.ln2_b", pre + "output.LayerNorm.bias")

Wr = load_safetensors(RET); Wq = load_safetensors(RDR)
hidden, heads, inter, vocab, max_pos, ret_layers, rdr_layers = 384, 12, 1536, 30522, 512, 6, 12

vocab_lines = [l.rstrip("\n") for l in open(VOCAB, encoding="utf-8")]
assert len(vocab_lines) == vocab
blob("vocab", ("\0".join(vocab_lines) + "\0").encode("utf-8"))

pack_bert(Wr, "", ret_layers, "ret")
pack_bert(Wq, "bert.", rdr_layers, "rdr")
blob("rdr.qa_w", Wq["qa_outputs.weight"].astype(np.float32)); blob("rdr.qa_b", Wq["qa_outputs.bias"].astype(np.float32))

# passage embeddings computed with the *quantized* retriever so the index matches the engine exactly
Wr_q = {}
for k, v in Wr.items():
    if v.ndim == 2 and v.shape[0] >= 128 and v.shape[1] >= 128:
        q, s = quantize_int8_rows(v); Wr_q[k] = dequant_int8_rows(q, s)
    else:
        Wr_q[k] = v
retriever = Bert(Wr_q, ret_layers)
t = time.time(); E = []
for i, p in enumerate(P):
    ids = tok.encode_single(p["text"], 128); h = retriever.forward(ids); v = h.mean(0); E.append(v / np.linalg.norm(v))
    if i % 100 == 0: print(f"  embedded {i}/{len(P)}", flush=True)
E = np.stack(E); print("embedded %d passages in %.1fs" % (len(P), time.time() - t))
Eq, Es = quantize_int8_rows(E)
blob("emb.q", Eq); blob("emb.s", Es.reshape(-1).astype(np.float32))

def strs(field): return ("\0".join(p[field] for p in P) + "\0").encode("utf-8")
blob("pass.text", strs("text")); blob("pass.title", strs("title")); blob("pass.url", strs("url")); blob("pass.source", strs("source"))

# BM25 postings over wordpiece ids
docs = [tok.encode_words(p["text"])[0] for p in P]
N = len(docs); avgdl = sum(len(d) for d in docs) / N
post = {}
for i, d in enumerate(docs):
    for tid in set(d): post.setdefault(tid, []).append((i, d.count(tid)))
terms = sorted(post)
offsets = [0]; plist = []
for tid in terms:
    plist.extend(post[tid]); offsets.append(len(plist))
blob("bm25.terms", np.array(terms, dtype=np.int32))
blob("bm25.offsets", np.array(offsets, dtype=np.int32))
blob("bm25.postings", np.array(plist, dtype=np.int32).reshape(-1, 2))
blob("bm25.doclen", np.array([len(d) for d in docs], dtype=np.int32))
STOP = "the a an of is are was were in on at to for and or what who when where which how does do did it its this that be by with as from about tell me name his her their s ' ? , ."
stop_ids = sorted(set(tok.encode_words(STOP)[0]))

params = np.array([hidden, heads, inter, vocab, max_pos, ret_layers, rdr_layers, len(P), len(terms), len(plist),
                   int(round(avgdl * 1000)), len(stop_ids)], dtype=np.int32)
blob("params", params); blob("stop_ids", np.array(stop_ids, dtype=np.int32))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
n = len(blobs); toc_size = 16 + n * 64
body = bytearray(); layout = []
for name, b in blobs:
    assert len(name) < 48, name
    pad = (-(toc_size + len(body))) % 64
    body += b"\0" * pad
    layout.append((name, toc_size + len(body), len(b)))
    body += b
with open(OUT, "wb") as f:
    f.write(b"KDRB"); f.write(struct.pack("<III", 2, n, 0))
    for name, off, size in layout:
        f.write(name.encode().ljust(48, b"\0")); f.write(struct.pack("<QQ", off, size))
    f.write(body)
print("wrote", OUT, "%.1f MB" % (os.path.getsize(OUT) / 1e6), "blobs", n)
