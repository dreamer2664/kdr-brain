"""Evaluate the compiled C engine on the test set.
   python3 scripts/eval_c.py            summary
   python3 scripts/eval_c.py -v         print every question
   python3 scripts/eval_c.py --strict   exit 1 unless >= MIN_OK exact answers (CI quality gate)
"""
import subprocess, json, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from testset import TESTS, OOD
BIN = os.path.join(ROOT, "release", "kdr-brain"); BRAIN = os.path.join(ROOT, "release", "brain.kdr")
MIN_OK = int(os.environ.get("KDR_MIN_OK", "72"))

def ask(q):
    out = subprocess.run([BIN, BRAIN, "ask", q], capture_output=True, text=True).stdout
    return json.loads(out)

ok_r = ok_a = 0; total_ms = 0; verbose = "-v" in sys.argv
rows = []
for q, golds in TESTS:
    j = ask(q)
    top = " ".join(h["text"].lower() for h in j["hits"])
    r = any(g in top for g in golds); a = any(g in j["answer"].lower() for g in golds)
    a2 = a or any(g in j["sentence"].lower() for g in golds)
    ok_r += r; ok_a += a; total_ms += j["ms_retrieve"] + j["ms_read"]
    rows.append((q, j["answer"], j["confidence"], a, a2))
    if verbose or not a:
        print(f"{'R' if r else '-'}{'A' if a else ('e' if a2 else '-')} {j['confidence']:.2f} | {q}\n        -> '{j['answer']}'  | {j['sentence'][:120]}")
print(f"\nC engine: retrieval@8 {ok_r}/{len(TESTS)}   exact-answer {ok_a}/{len(TESTS)}   "
      f"answer-or-evidence {sum(1 for r in rows if r[4])}/{len(TESTS)}   avg {total_ms/len(TESTS):.0f} ms/question")
print("\nOut-of-domain (should be low confidence):")
for q in OOD:
    j = ask(q); print(f"  {j['confidence']:.2f} | {q} -> '{j['answer']}'")
in_conf = sorted(r[2] for r in rows)
print(f"\nin-domain confidence: min {in_conf[0]:.2f}  p10 {in_conf[len(in_conf)//10]:.2f}  median {in_conf[len(in_conf)//2]:.2f}")
if "--strict" in sys.argv and ok_a < MIN_OK:
    print(f"\nFAIL: only {ok_a} exact answers, need >= {MIN_OK}"); sys.exit(1)
