CC      ?= gcc
CXX     ?= g++
CFLAGS  ?= -O3 -march=x86-64-v3 -ffast-math -fopenmp -std=gnu11 -Wall -Wno-unused-function
LDFLAGS ?= -static -fopenmp -lzstd -lm

# llama.cpp (composer). Built once by scripts/build_llama.sh into $(LLAMA)/build-native
LLAMA   ?= $(HOME)/.cache/kdr/llama
LLAMA_INC = -I$(LLAMA)/include -I$(LLAMA)/ggml/include
LLAMA_LIBS = $(LLAMA)/build-native/src/libllama.a $(LLAMA)/build-native/ggml/src/libggml.a \
             $(LLAMA)/build-native/ggml/src/libggml-cpu.a $(LLAMA)/build-native/ggml/src/libggml-base.a

all: release/kdr-brain

src/webui.c: src/index.html scripts/embed_html.py
	python3 scripts/embed_html.py

# retrieval + reader + composer, one static binary (C++ runtime only because libllama is C++)
release/kdr-brain: src/brain.c src/main.c src/chat.c src/wiki.c src/webui.c src/brain.h src/chat.h src/wiki.h src/unicode_tables.h $(LLAMA_LIBS)
	mkdir -p release
	$(CC) $(CFLAGS) $(LLAMA_INC) -c src/brain.c -o /tmp/kdr_brain.o
	$(CC) $(CFLAGS) $(LLAMA_INC) -c src/main.c  -o /tmp/kdr_main.o
	$(CC) $(CFLAGS) $(LLAMA_INC) -c src/chat.c  -o /tmp/kdr_chat.o
	$(CC) $(CFLAGS) -c src/wiki.c -o /tmp/kdr_wiki.o
	$(CC) $(CFLAGS) -c src/webui.c -o /tmp/kdr_webui.o
	$(CXX) -O3 -fopenmp -o $@ /tmp/kdr_brain.o /tmp/kdr_main.o /tmp/kdr_chat.o /tmp/kdr_wiki.o /tmp/kdr_webui.o $(LLAMA_LIBS) -static -fopenmp -lzstd -lm -lpthread
	strip $@
	@ls -la $@

# retrieval-only build (no llama.cpp needed, no chat)
lite: src/brain.c src/main.c src/wiki.c src/webui.c
	mkdir -p release
	$(CC) $(CFLAGS) -DKDR_NO_CHAT -o release/kdr-brain-lite src/brain.c src/main.c src/wiki.c src/webui.c $(LDFLAGS) && strip release/kdr-brain-lite

clean:
	rm -f release/kdr-brain release/kdr-brain-lite src/webui.c
