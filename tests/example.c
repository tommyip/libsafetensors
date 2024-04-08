#include <stdio.h>
#include "../safetensors.h"

int main() {
    st_ctx* ctx = st_open("tests/arr.safetensors");
    if (ctx == NULL) {
        fprintf(stderr, "Failed to open safetensors file\n");
        return -1;
    }
    st_close(ctx);

    return 0;
}
