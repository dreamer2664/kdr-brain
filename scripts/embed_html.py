data = open("src/index.html", "rb").read()
with open("src/webui.c", "w") as f:
    f.write("/* generated from src/index.html */\n#include <stddef.h>\nconst char kdr_index_html[] = {")
    f.write(",".join(str(b) for b in data) + ",0};\n")
    f.write("const unsigned int kdr_index_html_len = %d;\n" % len(data))
