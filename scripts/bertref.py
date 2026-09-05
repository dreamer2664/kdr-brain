"""Minimal numpy BERT reference + safetensors loader + WordPiece tokenizer.
Used to (1) verify the C engine, (2) evaluate quantization impact."""
import json, struct, unicodedata, numpy as np

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        f.seek(0); data = f.read()
    for k, v in hdr.items():
        if k == "__metadata__": continue
        s, e = v["data_offsets"]
        dt = {"F32": np.float32, "F16": np.float16, "I64": np.int64}[v["dtype"]]
        out[k] = np.frombuffer(data[base+s:base+e], dtype=dt).reshape(v["shape"]).astype(np.float32 if dt != np.int64 else np.int64)
    return out

class WordPiece:
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.inv = {v: k for k, v in self.vocab.items()}
        self.unk = self.vocab["[UNK]"]; self.cls = self.vocab["[CLS]"]; self.sep = self.vocab["[SEP]"]

    @staticmethod
    def _is_punct(ch):
        cp = ord(ch)
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126): return True
        return unicodedata.category(ch).startswith("P")

    @staticmethod
    def _is_cjk(cp):
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
                or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)

    def basic(self, text):
        """returns list of (word, start_char, end_char) using original char offsets"""
        words = []; cur = []; cur_start = None
        def flush(end):
            nonlocal cur, cur_start
            if cur: words.append(("".join(cur), cur_start, end)); cur = []; cur_start = None
        for i, ch in enumerate(text):
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or unicodedata.category(ch).startswith("C") and ch not in "\t\n\r":
                continue
            if ch.isspace():
                flush(i); continue
            if self._is_cjk(cp) or self._is_punct(ch):
                flush(i); words.append((ch, i, i+1)); continue
            lower = ch.lower()
            for c in unicodedata.normalize("NFD", lower):
                if unicodedata.category(c) == "Mn": continue
                if cur_start is None: cur_start = i
                cur.append(c)
            if cur_start is None: cur_start = i  # in case all combining
        flush(len(text))
        return words

    def encode_words(self, text):
        """returns ids, offsets [(start,end)]"""
        ids = []; offs = []
        for w, s, e in self.basic(text):
            if len(w) > 100:
                ids.append(self.unk); offs.append((s, e)); continue
            start = 0; sub = []; bad = False
            while start < len(w):
                end = len(w); cur = None
                while start < end:
                    piece = w[start:end]
                    if start > 0: piece = "##" + piece
                    if piece in self.vocab: cur = self.vocab[piece]; break
                    end -= 1
                if cur is None: bad = True; break
                sub.append(cur); start = end
            if bad: ids.append(self.unk); offs.append((s, e))
            else:
                ids.extend(sub); offs.extend([(s, e)] * len(sub))
        return ids, offs

    def encode_pair(self, q, ctx, max_len=384):
        qi, _ = self.encode_words(q)
        ci, co = self.encode_words(ctx)
        room = max_len - 3 - len(qi)
        ci = ci[:room]; co = co[:room]
        ids = [self.cls] + qi + [self.sep] + ci + [self.sep]
        tt = [0] * (len(qi) + 2) + [1] * (len(ci) + 1)
        offs = [None] * (len(qi) + 2) + co + [None]
        return ids, tt, offs

    def encode_single(self, text, max_len=256):
        ti, _ = self.encode_words(text)
        ti = ti[:max_len - 2]
        return [self.cls] + ti + [self.sep]

def gelu(x):
    from math import sqrt
    return 0.5 * x * (1.0 + erf_np(x / sqrt(2.0)))

def erf_np(x):
    # numerically fine vectorized erf via math.erf
    import math
    return np.vectorize(math.erf, otypes=[np.float32])(x)

def layernorm(x, g, b, eps=1e-12):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

class Bert:
    def __init__(self, W, n_layers, prefix=""):
        self.W = W; self.L = n_layers; self.p = prefix
        self.H = W[prefix + "embeddings.word_embeddings.weight"].shape[1]
        self.nh = 12; self.dh = self.H // self.nh

    def g(self, name): return self.W[self.p + name]

    def forward(self, ids, token_type=None):
        W = self.g
        ids = np.asarray(ids); T = len(ids)
        tt = np.zeros(T, dtype=np.int64) if token_type is None else np.asarray(token_type)
        x = W("embeddings.word_embeddings.weight")[ids] + W("embeddings.position_embeddings.weight")[:T] + W("embeddings.token_type_embeddings.weight")[tt]
        x = layernorm(x, W("embeddings.LayerNorm.weight"), W("embeddings.LayerNorm.bias"))
        for l in range(self.L):
            pre = f"encoder.layer.{l}."
            q = x @ W(pre + "attention.self.query.weight").T + W(pre + "attention.self.query.bias")
            k = x @ W(pre + "attention.self.key.weight").T + W(pre + "attention.self.key.bias")
            v = x @ W(pre + "attention.self.value.weight").T + W(pre + "attention.self.value.bias")
            q = q.reshape(T, self.nh, self.dh).transpose(1, 0, 2)
            k = k.reshape(T, self.nh, self.dh).transpose(1, 0, 2)
            v = v.reshape(T, self.nh, self.dh).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(self.dh)
            s = s - s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
            a = (p @ v).transpose(1, 0, 2).reshape(T, self.H)
            a = a @ W(pre + "attention.output.dense.weight").T + W(pre + "attention.output.dense.bias")
            x = layernorm(x + a, W(pre + "attention.output.LayerNorm.weight"), W(pre + "attention.output.LayerNorm.bias"))
            h = gelu(x @ W(pre + "intermediate.dense.weight").T + W(pre + "intermediate.dense.bias"))
            h = h @ W(pre + "output.dense.weight").T + W(pre + "output.dense.bias")
            x = layernorm(x + h, W(pre + "output.LayerNorm.weight"), W(pre + "output.LayerNorm.bias"))
        return x

def quantize_int8_rows(w):
    """symmetric per-row int8; returns (q, scale)"""
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1e-8
    q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)

def dequant_int8_rows(q, s): return q.astype(np.float32) * s

def quantize_int4_groups(w, group=32):
    """symmetric per-group int4 (levels -7..7) with fp16 scales; returns dequantized sim"""
    r, c = w.shape
    wg = w.reshape(r, c // group, group)
    s = np.abs(wg).max(axis=2, keepdims=True) / 7.0
    s[s == 0] = 1e-8
    q = np.clip(np.round(wg / s), -7, 7)
    return (q * s.astype(np.float16).astype(np.float32)).reshape(r, c)
