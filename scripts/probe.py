"""Ask a batch of questions against a throw-away server (port 8097) and print reply + latency.
   python3 scripts/probe.py questions.txt        (one question per line)
   KDR_REPEAT=1.0 KDR_FREQ=0.3 python3 scripts/probe.py q.txt   (env vars are passed to the server)"""
import subprocess, json, sys, os, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PORT = int(os.environ.get("KDR_PROBE_PORT", "8097"))
srv = subprocess.Popen([os.path.join(ROOT, "release/kdr-brain"), os.path.join(ROOT, "release/brain.kdr"), "--chat", os.path.join(ROOT, "release/composer.gguf"), "serve", str(PORT)],
                       stdout=subprocess.DEVNULL, stderr=open("/tmp/probe-server.log", "a"))
for _ in range(600):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/stats", timeout=1); break
    except Exception: time.sleep(0.2)
try:
    for q in [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]:
        q = q.split("|")[0].strip()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/chat", data=json.dumps({"q": q}).encode(), headers={"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(req, timeout=180))
        print(f"### {q}\n  -> {j['reply'][:300]}   [{j['ms_chat']} ms]", flush=True)
finally:
    srv.terminate(); srv.wait()
