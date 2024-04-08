all: safetensors.so

safetensors.so: safetensors.h safetensors.c
	clang -std=c99 -shared -o $@ safetensors.c

example: safetensors.so tests/example.c
	clang -std=c99 tests/example.c -o example safetensors.so

test:
	pytest

format:
	clang-format -i safetensors.h safetensors.c

clean:
	rm -rf safetensors.so
