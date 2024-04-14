all: libsafetensors.so example

libsafetensors.so: safetensors.h safetensors.c
	clang -std=c99 -shared -o $@ safetensors.c

example: libsafetensors.so tests/example.c
	clang -std=c99 tests/example.c -o $@ libsafetensors.so

test:
	pytest

format:
	clang-format -i safetensors.h safetensors.c

clean:
	rm -rf libsafetensors.so example
