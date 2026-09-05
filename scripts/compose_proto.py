"""Prototype: retrieval (C engine) + small LLM (llama.cpp) -> fluent answer.  Used to pick the model and tune the prompt
before porting the logic into the C server.

  python3 scripts/compose_proto.py MODEL.gguf "question"
  python3 scripts/compose_proto.py MODEL.gguf --score tests/kingdom.txt tests/chat.txt
"""
import subprocess, json, sys, os, re, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "release", "kdr-brain"); BRAIN = os.path.join(ROOT, "release", "brain.kdr")
LLAMA = os.environ.get("LLAMA_BIN", os.path.expanduser("~/.cache/kdr/llama/build-native/bin/llama-completion"))
REFUSAL = "I can only answer questions about"

SYSTEM = ("You are KDR Brain, the friendly assistant of the Kingdom of Dutch Robloxia (a Roblox roleplay kingdom ruled by King Nicholas Goudhof V). "
          "Be warm, natural and concise: 1-3 sentences, or a short list when several items are asked for.\n"
          "Rules:\n"
          "1. If FACTS are given and the question is about the kingdom, answer only from the FACTS. Start with the direct answer. Never invent names, dates, places or numbers. "
          "If the FACTS do not contain the answer, say you don't have that information about the kingdom yet.\n"
          "2. If the question is about the real world and not about the kingdom, ignore the FACTS, answer briefly from your own knowledge, and add the note (not from the wikis) at the end.\n"
          "3. For greetings or small talk, just chat naturally as KDR Brain; no note needed.")

def retrieve(q):
    return json.loads(subprocess.run([BIN, BRAIN, "ask", q], capture_output=True, text=True).stdout)

def build_prompt(q, r, template):
    facts, seen = [], set()
    if r["confidence"] >= 0.3 and r["sentence"]:
        seen.add(r["sentence"].strip()); facts.append(f"- [{r['title']}] {r['sentence'].strip()}")
    for h in r["hits"][:8]:
        t = h["text"].strip()
        if t in seen: continue
        seen.add(t); facts.append(f"- [{h['title']}] {t}")
    use_facts = r["confidence"] >= 0.30 or (r["hits"] and r["hits"][0]["score"] >= 0.55)
    user = q
    if use_facts:
        user = "FACTS:\n" + "\n".join(facts) + "\n\nQUESTION: " + q
    if template == "chatml":
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", use_facts
    if template == "qwen3":   # empty think block = non-thinking mode
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", use_facts
    if template == "lfm2":
        return f"<|startoftext|><|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", use_facts
    raise ValueError(template)

def generate(model, prompt, n=120, extra=()):
    cmd = [LLAMA, "-m", model, "-t", "2", "-c", "2048", "-n", str(n), "--temp", "0", "-no-cnv", "--no-display-prompt",
           "--no-warmup", "--repeat-penalty", "1.1", "-p", prompt] + list(extra)
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    out = out.replace("[end of text]", "").strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()   # qwen3
    return out

def answer(model, q, template, extra=()):
    r = retrieve(q)
    prompt, used = build_prompt(q, r, template)
    t0 = time.time(); out = generate(model, prompt, extra=extra); dt = time.time() - t0
    return out, r, used, dt

def load_tests(path):
    tests = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        q, _, acc = line.partition("|")
        tests.append((q.strip(), [a.strip() for a in acc.split(";") if a.strip()]))
    return tests

def grade(reply, accepted):
    low = reply.lower()
    flags = [a for a in accepted if a.startswith("!")]
    phrases = [a for a in accepted if not a.startswith("!")]
    ok = any(p.lower() in low for p in phrases) if phrases else True
    if "!norefuse" in flags and REFUSAL.lower() in low: ok = False
    if "!short" in flags and len(reply.split()) > 40: ok = False
    return ok

if __name__ == "__main__":
    model = sys.argv[1]
    template = "lfm2" if "lfm2" in model else ("qwen3" if "qwen3" in model else "chatml")
    extra = []
    if sys.argv[2] == "--score":
        total_ok = total_n = 0; total_t = 0
        for path in sys.argv[3:]:
            tests = load_tests(path); ok_n = 0; fails = []
            for q, acc in tests:
                out, r, used, dt = answer(model, q, template, extra)
                total_t += dt
                g = grade(out, acc); ok_n += g
                if not g: fails.append((q, out[:160], r["confidence"]))
            print(f"{os.path.basename(path)}: {ok_n}/{len(tests)}  ({100*ok_n/len(tests):.0f}%)")
            for q, out, c in fails: print(f"   FAIL [{c:.2f}] {q}\n        -> {out}")
            total_ok += ok_n; total_n += len(tests)
        print(f"TOTAL {total_ok}/{total_n}   avg {total_t/total_n:.1f}s per reply")
    else:
        q = " ".join(sys.argv[2:])
        out, r, used, dt = answer(model, q, template, extra)
        print(f"[facts={used} conf={r['confidence']:.2f} {dt:.1f}s]\n{out}")
