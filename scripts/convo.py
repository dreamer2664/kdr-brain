"""Multi-turn conversation smoke test against a running server (default http://127.0.0.1:8080).
   python3 scripts/convo.py                       # runs the built-in 12-turn script
   python3 scripts/convo.py "hi" "who is the king?" "and his wife?"
   KDR_URL=http://127.0.0.1:8099 python3 scripts/convo.py
Prints reply, latency, route (facts/chat) and reader confidence per turn; history = last 6 turns like the web UI."""
import json, urllib.request, sys, os
URL = os.environ.get("KDR_URL", "http://127.0.0.1:8080")
DEFAULT = ["hey there!", "who runs the place?", "and what about his wife?", "how old is the prince?",
           "do you have a favourite region?", "why that one?", "is this a real country?", "what's the weather like in Dunhag?",
           "I'm bored", "who is the minister of finance?", "what does he actually do?", "how do I join? thanks!"]
hist = []
def say(q):
    req = urllib.request.Request(URL + "/api/chat", data=json.dumps({"q": q, "history": hist[-6:]}).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180)); r = d["reply"]
    hist.append({"role": "user", "text": q}); hist.append({"role": "assistant", "text": r.replace(" (not from the wikis)", "")})
    print(f"YOU: {q}\nKDR: {r}   [{d['ms_chat']/1000:.1f}s, facts={d['used_facts']}, conf={d['retrieval']['confidence']:.2f}]\n")
for q in (sys.argv[1:] or DEFAULT): say(q)
