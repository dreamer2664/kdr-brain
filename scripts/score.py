"""Score the shipped binary (retrieval + composer) on the question sets in tests/.
   python3 scripts/score.py                       # all sets, summary per block of 10 + failures
   python3 scripts/score.py tests/kingdom.txt -v  # one set, print every reply
   KDR_MODEL=path/to/model.gguf overrides the composer (default release/composer.gguf)
   --strict : exit 1 unless every kingdom/chat block scores >= 9/10 and every general-knowledge block >= 7/10 (CI gate)
Format of a test line:  question | accepted 1 ; accepted 2 ; !norefuse ; !short
"""
import subprocess, json, sys, os, re, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "release", "kdr-brain"); BRAIN = os.path.join(ROOT, "release", "brain.kdr")
MODEL = os.environ.get("KDR_MODEL", os.path.join(ROOT, "release", "composer.gguf"))
WIKI = os.environ.get("KDR_WIKI", os.path.join(ROOT, "release", "wiki.kdw"))   # general-knowledge pack, used when present
REFUSAL = ["i can only answer questions about", "i don't have that information", "i cannot answer", "i can't answer"]

def load_tests(path):
    blocks, cur, name = [], [], "block 1"
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("## "):
            if cur: blocks.append((name, cur)); cur = []
            name = line[3:].strip(); continue
        line = line.strip()
        if not line or line.startswith("#"): continue
        q, _, acc = line.partition("|")
        cur.append((q.strip(), [a.strip() for a in acc.split(";") if a.strip()]))
    if cur: blocks.append((name, cur))
    return blocks

def grade(reply, accepted):
    low = reply.lower()
    flags = [a for a in accepted if a.startswith("!")]
    phrases = [a for a in accepted if not a.startswith("!")]
    ok = any(p.lower() in low for p in phrases) if phrases else True
    if "!norefuse" in flags and any(r in low for r in REFUSAL): ok = False
    if "!short" in flags and len(reply.split()) > 40: ok = False
    return ok

import urllib.request, socket, atexit
PORT = int(os.environ.get("KDR_SCORE_PORT", "8099")); SERVER = None
def ensure_server():
    global SERVER
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/stats", timeout=2); return
    except Exception: pass
    cmd = [BIN, BRAIN, "--chat", MODEL] + (["--wiki", WIKI] if os.path.exists(WIKI) else []) + ["serve", str(PORT)]
    SERVER = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(SERVER.terminate)
    for _ in range(300):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/stats", timeout=1); return
        except Exception: time.sleep(0.2)
    raise SystemExit("server did not start")
def ask(q):
    ensure_server()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/chat", data=json.dumps({"q": q}).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv; strict = "--strict" in sys.argv
    paths = args or [os.path.join(ROOT, "tests", "kingdom.txt"), os.path.join(ROOT, "tests", "chat.txt")] + \
        ([os.path.join(ROOT, "tests", "general.txt")] if os.path.exists(WIKI) else [])
    worst = 10; worst_gen = 10; grand_ok = grand_n = 0; t_all = 0
    for path in paths:
        print(f"== {os.path.basename(path)}")
        general = os.path.basename(path) == "general.txt"   # Wikipedia-mini coverage: gate at 7/10 instead of 9/10
        for name, tests in load_tests(path):
            ok_n = 0; fails = []
            for q, acc in tests:
                j = ask(q); reply = j["reply"]; t_all += j["ms_chat"] + j["retrieval"]["ms_retrieve"] + j["retrieval"]["ms_read"]
                g = grade(reply, acc); ok_n += g
                if verbose: print(f"   {'ok  ' if g else 'FAIL'} {q}\n        -> {reply[:200]}")
                elif not g: fails.append((q, reply[:200], j["retrieval"]["confidence"], j["retrieval"].get("kind", "")))
            score10 = round(10 * ok_n / len(tests), 1)
            if general: worst_gen = min(worst_gen, score10)
            else: worst = min(worst, score10)
            grand_ok += ok_n; grand_n += len(tests)
            print(f"  {score10:>4}/10  {name}  ({ok_n}/{len(tests)})")
            for q, r, c, k in fails: print(f"        FAIL [{c:.2f} {k}] {q}\n             -> {r}")
    print(f"\nTOTAL {grand_ok}/{grand_n} ({100*grand_ok/grand_n:.0f}%)   worst block {min(worst, worst_gen)}/10   avg {t_all/grand_n/1000:.1f}s per reply")
    if strict and (worst < 9 or worst_gen < 7): print("FAIL: a kingdom/chat block scored below 9/10 or a general-knowledge block below 7/10"); sys.exit(1)
