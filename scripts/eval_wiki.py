"""Retrieval+reader only check of the Wikipedia pack (no composer): does the extracted span contain an accepted answer?
   python3 scripts/eval_wiki.py [tests/general.txt] [--wiki path] [-v]
Prints per-block scores, the reader's span, confidence and the article it came from. Much faster than score.py:
use it to tune retrieval; score.py is the real gate (it includes the composer)."""
import subprocess, json, sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from score import load_tests, grade
BIN = os.path.join(ROOT, "release", "kdr-brain-lite") if os.path.exists(os.path.join(ROOT, "release", "kdr-brain-lite")) else os.path.join(ROOT, "release", "kdr-brain")
BRAIN = os.path.join(ROOT, "release", "brain.kdr")
args = [a for a in sys.argv[1:] if not a.startswith("-")]
wiki = sys.argv[sys.argv.index("--wiki") + 1] if "--wiki" in sys.argv else os.environ.get("KDR_WIKI", os.path.join(ROOT, "release", "wiki.kdw"))
if "--wiki" in sys.argv: args = [a for a in args if a != wiki]
path = args[0] if args else os.path.join(ROOT, "tests", "general.txt")
verbose = "-v" in sys.argv
tot = ok_all = 0; t0 = time.time(); worst = 10
for name, tests in load_tests(path):
    ok_n = 0
    for q, acc in tests:
        j = json.loads(subprocess.run([BIN, BRAIN, "--wiki", wiki, "wiki", q], capture_output=True, text=True).stdout)
        hit_titles = [h["title"] for h in j["hits"][:3]]
        g = grade(j["answer"] + " | " + j["sentence"], acc); ok_n += g
        if verbose or not g:
            print(f"   {'ok  ' if g else 'FAIL'} {q}\n        -> '{j['answer']}'  conf {j['confidence']:.2f} span {j['span_score']:.1f} null {j['null_score']:.1f}  [{j['title']}]  top: {hit_titles}")
    tot += len(tests); ok_all += ok_n; worst = min(worst, 10 * ok_n / len(tests))
    print(f"  {10*ok_n/len(tests):>4.1f}/10  {name}  ({ok_n}/{len(tests)})")
print(f"TOTAL {ok_all}/{tot}  worst block {worst:.0f}/10  ({(time.time()-t0)/tot:.1f}s per question incl. process start)")
