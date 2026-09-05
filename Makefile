CC      ?= gcc
CFLAGS  ?= -O3 -march=x86-64-v3 -ffast-math -fopenmp -std=gnu11 -Wall -Wno-unused-function
LDFLAGS ?= -static -fopenmp -lm

all: release/kdr-brain

src/webui.c: src/index.html scripts/embed_html.py
	python3 scripts/embed_html.py

release/kdr-brain: src/brain.c src/main.c src/webui.c src/brain.h src/unicode_tables.h
	mkdir -p release
	$(CC) $(CFLAGS) -o $@ src/brain.c src/main.c src/webui.c $(LDFLAGS)
	strip $@
	@ls -la $@

# portable build (no AVX requirement)
portable: src/brain.c src/main.c src/webui.c
	$(CC) -O3 -ffast-math -fopenmp -std=gnu11 -o release/kdr-brain-portable src/brain.c src/main.c src/webui.c -static -fopenmp -lm && strip release/kdr-brain-portable

clean:
	rm -f release/kdr-brain src/webui.c
