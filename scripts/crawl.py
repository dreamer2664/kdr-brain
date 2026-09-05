"""Crawl the three Dutch Robloxia wikis into data/raw/ (the only network step of the pipeline).

  fandom_pages.json            all main-namespace articles + the two project pages (MediaWiki API)
  kdrwiki.netlify.app.html     single-page site, with embedded base64 media stripped
  dutchbloxia.netlify.app.html single-page site, with embedded base64 media stripped
"""
import json, os, re, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw"); os.makedirs(RAW, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (KDR-Brain crawler; +https://github.com)"}
API = "https://the-kingdom-of-dutch-robloxia.fandom.com/api.php"

def get_json(params):
    params["format"] = "json"; params["formatversion"] = "2"
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=60))

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")

# 1) Fandom
titles = []
cont = {}
while True:
    r = get_json({"action": "query", "list": "allpages", "aplimit": "500", "apnamespace": "0", **cont})
    titles += [p["title"] for p in r["query"]["allpages"]]
    if "continue" not in r: break
    cont = r["continue"]
titles += ["The Kingdom of Dutch Robloxia Wiki:Wiki rules", "The Kingdom of Dutch Robloxia Wiki:Policies"]
pages = {}
for t in titles:
    d = get_json({"action": "parse", "page": t, "prop": "wikitext|text|categories|sections", "disabletoc": "1"})["parse"]
    pages[t] = {"wikitext": d["wikitext"], "html": d["text"], "categories": [c["category"] for c in d.get("categories", [])],
                "sections": [s["line"] for s in d.get("sections", [])]}
    time.sleep(0.15)
json.dump(pages, open(os.path.join(RAW, "fandom_pages.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"fandom: {len(pages)} pages, {sum(len(p['wikitext']) for p in pages.values())} wikitext chars")

# 2) + 3) the two Netlify single-page wikis
for url, name in [("https://kdrwiki.netlify.app/", "kdrwiki.netlify.app.html"), ("https://dutchbloxia.netlify.app/", "dutchbloxia.netlify.app.html")]:
    s = fetch(url)
    s = re.sub(r"data:[a-z/+.-]+;base64,[A-Za-z0-9+/=\s]+", "data:removed", s)   # drop embedded images/audio
    s = re.sub(r"<style[^>]*>.*?</style>", "", s, flags=re.S)
    open(os.path.join(RAW, name), "w", encoding="utf-8").write(s)
    print(f"{name}: {len(s)} chars")

open(os.path.join(RAW, "CRAWLED_AT.txt"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
