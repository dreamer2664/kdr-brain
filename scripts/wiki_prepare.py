"""Shuffle the extracted articles (so partial embedding runs still cover every topic) and split the passage list
into N parts for parallel embedding with `kdr-brain embed`.

    python3 scripts/wiki_prepare.py <ext_dir> [n_parts]

Reads  <ext_dir>/articles.tsv, passages.tsv   (from wiki_extract.py)
Writes <ext_dir>/shuffled_articles.tsv, shuffled_passages.tsv, embed_part_00..NN
The embedding input line is "Title: passage" - the title is part of what gets embedded (and of the BM25 bag),
because most paragraphs never repeat the subject's name ("He was born in ...").
"""
import sys, os, random, collections

def main():
    ext = sys.argv[1]; n_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    arts = {}
    for l in open(os.path.join(ext, "articles.tsv"), encoding="utf-8"):
        aid, path, title = l.rstrip("\n").split("\t"); arts[int(aid)] = (path, title)
    by = collections.defaultdict(list)
    for l in open(os.path.join(ext, "passages.tsv"), encoding="utf-8"):
        aid, text = l.rstrip("\n").split("\t", 1); by[int(aid)].append(text)
    order = sorted(by); random.Random(20260906).shuffle(order)
    lines = []
    with open(os.path.join(ext, "shuffled_articles.tsv"), "w", encoding="utf-8") as fa, \
         open(os.path.join(ext, "shuffled_passages.tsv"), "w", encoding="utf-8") as fp:
        for new_id, aid in enumerate(order):
            path, title = arts[aid]
            fa.write(f"{new_id}\t{path}\t{title}\n")
            for t in by[aid]:
                fp.write(f"{new_id}\t{t}\n"); lines.append(f"{title}: {t}\n")
    per = (len(lines) + n_parts - 1) // n_parts
    for i in range(n_parts):
        with open(os.path.join(ext, f"embed_part_{i:02d}"), "w", encoding="utf-8") as f:
            f.writelines(lines[i * per:(i + 1) * per])
    print(f"{len(order)} articles, {len(lines)} passages -> {n_parts} parts of <= {per} lines")

if __name__ == "__main__":
    main()
