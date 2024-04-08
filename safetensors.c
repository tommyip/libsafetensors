#include "safetensors.h"
#include <ctype.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef enum {
    TOK_BRACE_OPEN,
    TOK_BRACE_CLOSE,
    TOK_BRACKET_OPEN,
    TOK_BRACKET_CLOSE,
    TOK_COLON,
    TOK_COMMA,
    TOK_INTEGER,
    TOK_STRING,
    TOK_EOH,  // End of header
} token_kind;

typedef struct {
    token_kind kind;
    union {
        uint64_t integer;
        st_string string;
    };
} token;

typedef struct tensor_node tensor_node;
struct tensor_node {
    struct tensor_node* next;
    st_tensor tensor;
};

typedef struct metadata_node metadata_node;
struct metadata_node {
    struct metadata_node* next;
    st_metadata metadata;
};

typedef enum {
    TF_DTYPE = (1 << 0),
    TF_SHAPE = (1 << 1),
    TF_DATA_OFFSETS = (1 << 2)
} tensor_info;

typedef struct {
    char* tok;   // JSON tokenizer position
    char* data;  // Start of the data section
    // `tensor_node` is not allocated until we parse the `shape` field since
    // it has variable length. The parsed `dtype` and `data_offsets` fields are
    // stored here until we flush them to tensor_node.
    unsigned char tensor_fields;
    tensor_node* tensor_node;
    st_dtype tensor_dtype;
    uint8_t* tensor_data;
    size_t tensor_data_len;
} parse_ctx;

static bool is_ws(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

/* JSON tokenizer at the end of header? */
static bool eoh(parse_ctx* pcx) {
    return pcx->tok >= pcx->data;
}

static bool tokenize_integer(parse_ctx* pcx, uint64_t* x) {
    *x = 0;
    char c = *pcx->tok;
    if (c == '0') {
        ++pcx->tok;
    } else {
        while (isdigit(c)) {
            *x = *x * 10 + (c - '0');
            c = *++pcx->tok;
        }
    }
    return true;
}

static bool tokenize_hex(parse_ctx* pcx, int* x) {
    int c = *pcx->tok++;
    if (c >= '0' && c <= '9') {
        *x = c - '0';
    } else if (c >= 'A' && c <= 'F') {
        *x = c - 'A' + 10;
    } else if (c >= 'a' && c <= 'f') {
        *x = c - 'a' + 10;
    } else {
        return false;
    }
    return true;
}

bool tokenize_string(parse_ctx* pcx, st_string* string) {
    ++pcx->tok;
    string->string = pcx->tok;
    while (true) {
        if (eoh(pcx)) return false;
        char c = *pcx->tok++;
        if (c == '"') {
            break;
        } else if (iscntrl(c)) {
            return false;
        } else if (c == '\\') {
            c = *pcx->tok++;
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
                c == 'n' || c == 'r' || c == 't') {
                continue;
            } else if (c == 'u') {
                int x;
                for (int i = 0; i < 4; ++i)
                    if (!tokenize_hex(pcx, &x)) return false;
            } else {
                return false;
            }
        }
    }
    string->len = pcx->tok - string->string - 1;
    return true;
}

bool next_token(parse_ctx* pcx, token* tok) {
    if (eoh(pcx)) {
        tok->kind = TOK_EOH;
    } else {
        char c;
        while (is_ws(c = *pcx->tok++)) {
            if (eoh(pcx)) {
                tok->kind = TOK_EOH;
                break;
            }
        }
        if (c == '{') tok->kind = TOK_BRACE_OPEN;
        else if (c == '}') tok->kind = TOK_BRACE_CLOSE;
        else if (c == '[') tok->kind = TOK_BRACKET_OPEN;
        else if (c == ']') tok->kind = TOK_BRACKET_CLOSE;
        else if (c == ':') tok->kind = TOK_COLON;
        else if (c == ',') tok->kind = TOK_COMMA;
        else if (isdigit(c)) {
            tok->kind = TOK_INTEGER;
            --pcx->tok;
            return tokenize_integer(pcx, &tok->integer);
        } else if (c == '"') {
            tok->kind = TOK_STRING;
            --pcx->tok;
            return tokenize_string(pcx, &tok->string);
        } else {
            return false;
        }
    }
    return true;
}

#define NEXT_TOKEN() \
    if (!next_token(pcx, &tok)) return false;
#define CONSUME(token_kind) \
    if (!next_token(pcx, &tok) || tok.kind != token_kind) return false;

static bool str_eq(st_string actual, const char* expected) {
    size_t n = strlen(expected);
    return actual.len == n && strncmp(expected, actual.string, n) == 0;
}

static bool parse_dtype(parse_ctx* pcx, st_dtype* dtype) {
    token tok;
    CONSUME(TOK_STRING);
    if (str_eq(tok.string, "BOOL")) *dtype = ST_DTYPE_BOOL;
    else if (str_eq(tok.string, "U8")) *dtype = ST_DTYPE_U8;
    else if (str_eq(tok.string, "I8")) *dtype = ST_DTYPE_I8;
    else if (str_eq(tok.string, "F8_E5M2")) *dtype = ST_DTYPE_F8_E5M2;
    else if (str_eq(tok.string, "F8_E4M3")) *dtype = ST_DTYPE_F8_E4M3;
    else if (str_eq(tok.string, "I16")) *dtype = ST_DTYPE_I16;
    else if (str_eq(tok.string, "U16")) *dtype = ST_DTYPE_U16;
    else if (str_eq(tok.string, "F16")) *dtype = ST_DTYPE_F16;
    else if (str_eq(tok.string, "BF16")) *dtype = ST_DTYPE_BF16;
    else if (str_eq(tok.string, "I32")) *dtype = ST_DTYPE_I32;
    else if (str_eq(tok.string, "U32")) *dtype = ST_DTYPE_U32;
    else if (str_eq(tok.string, "F32")) *dtype = ST_DTYPE_F32;
    else if (str_eq(tok.string, "F64")) *dtype = ST_DTYPE_F64;
    else if (str_eq(tok.string, "I64")) *dtype = ST_DTYPE_I64;
    else if (str_eq(tok.string, "U64")) *dtype = ST_DTYPE_U64;
    else return false;

    return true;
}

/* Check how many elements there are for memory allocation. */
bool array_length(parse_ctx* pcx, size_t* len) {
    char* start = pcx->tok;
    token tok;
    CONSUME(TOK_BRACKET_OPEN);
    while (next_token(pcx, &tok)) {
        if (tok.kind == TOK_BRACKET_CLOSE) {
            pcx->tok = start;
            return true;
        } else if (tok.kind == TOK_INTEGER) {
            if (!next_token(pcx, &tok)) break;
            switch (tok.kind) {
                case TOK_COMMA: *len += 1; break;
                case TOK_BRACKET_CLOSE:
                    pcx->tok = start;
                    *len += 1;
                    return true;
                default: return false;
            }
        } else {
            break;
        }
    }
    return false;
}

bool parse_array(parse_ctx* pcx, uint64_t* arr, int len) {
    token tok;
    CONSUME(TOK_BRACKET_OPEN);
    for (int i = 0; i < len; ++i) {
        CONSUME(TOK_INTEGER);
        arr[i] = tok.integer;
        if (i < len - 1) CONSUME(TOK_COMMA);
    }
    CONSUME(TOK_BRACKET_CLOSE);
    return true;
}

bool parse_object(st_ctx* ctx,
                  parse_ctx* pcx,
                  bool (*value_parser)(st_ctx*, parse_ctx*, st_string)) {
    token tok;
    CONSUME(TOK_BRACE_OPEN);
    while (true) {
        NEXT_TOKEN();
        if (tok.kind == TOK_STRING) {
            st_string name = tok.string;
            CONSUME(TOK_COLON);
            if (!value_parser(ctx, pcx, name)) return false;
            NEXT_TOKEN();
            if (tok.kind == TOK_COMMA) continue;
            else if (tok.kind == TOK_BRACE_CLOSE) break;
            else return false;
        } else if (tok.kind == TOK_BRACE_CLOSE) {
            break;
        } else {
            return false;
        }
    }
    return true;
}

static bool parse_tensor_kv(st_ctx* ctx, parse_ctx* pcx, st_string name) {
    if (str_eq(name, "dtype")) {
        if (pcx->tensor_fields & TF_DTYPE ||
            !parse_dtype(pcx, &pcx->tensor_dtype))
            return false;
        pcx->tensor_fields |= TF_DTYPE;
    } else if (str_eq(name, "shape")) {
        size_t len = 0;
        if (pcx->tensor_fields & TF_SHAPE || !array_length(pcx, &len))
            return false;
        pcx->tensor_node =
            (tensor_node*)malloc(sizeof(tensor_node) + sizeof(uint64_t) * len);
        pcx->tensor_node->next = NULL;
        pcx->tensor_node->tensor.shape_len = len;
        if (!parse_array(pcx, pcx->tensor_node->tensor.shape, len))
            return false;
        pcx->tensor_fields |= TF_SHAPE;
    } else if (str_eq(name, "data_offsets")) {
        uint64_t data_offsets[2];
        if (pcx->tensor_fields & TF_DATA_OFFSETS ||
            !parse_array(pcx, data_offsets, 2) ||
            data_offsets[0] > data_offsets[1])
            return false;
        pcx->tensor_data = ctx->data + data_offsets[0];
        pcx->tensor_data_len = data_offsets[1] - data_offsets[0];
        pcx->tensor_fields |= TF_DATA_OFFSETS;
    } else {
        return false;
    }
    return true;
}

bool parse_tensor(st_ctx* ctx, parse_ctx* pcx, st_string name) {
    pcx->tensor_fields = 0;
    if (!parse_object(ctx, pcx, parse_tensor_kv)) return false;
    if (pcx->tensor_fields != (TF_DTYPE | TF_SHAPE | TF_DATA_OFFSETS))
        return false;
    pcx->tensor_node->tensor.name = name;
    pcx->tensor_node->tensor.dtype = pcx->tensor_dtype;
    pcx->tensor_node->tensor.data_len = pcx->tensor_data_len;
    pcx->tensor_node->tensor.data = pcx->tensor_data;
    if (ctx->tensors) {
        ((tensor_node*)ctx->cur_tensor)->next = pcx->tensor_node;
        ctx->cur_tensor = (void*)pcx->tensor_node;
    } else {
        ctx->tensors = ctx->cur_tensor = (void*)pcx->tensor_node;
    }
    return true;
}

bool parse_metadata(st_ctx* ctx, parse_ctx* pcx, st_string name) {
    token tok;
    CONSUME(TOK_STRING);
    metadata_node* node = (metadata_node*)malloc(sizeof(metadata_node));
    node->next = NULL;
    node->metadata.name = name;
    node->metadata.value = tok.string;
    if (ctx->metadatas) {
        ((metadata_node*)ctx->cur_metadata)->next = node;
        ctx->cur_metadata = (void*)node;
    } else {
        ctx->metadatas = ctx->cur_metadata = (void*)node;
    }
    return true;
}

bool parse_header_value(st_ctx* ctx, parse_ctx* pcx, st_string name) {
    if (str_eq(name, "__metadata__")) {
        return parse_object(ctx, pcx, parse_metadata);
    } else {
        return parse_tensor(ctx, pcx, name);
    }
}

static bool mmap_file(st_ctx* ctx, int fd) {
    struct stat sb;
    if (fstat(fd, &sb) == -1) return false;
    void* buf = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) return false;

    ctx->buf = buf;
    ctx->buf_len = sb.st_size;
    return true;
}

st_ctx* st_open(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) return NULL;

    st_ctx* ctx = calloc(1, sizeof(st_ctx));
    if (!mmap_file(ctx, fd)) {
        close(fd);
        free(ctx);
        return NULL;
    }
    close(fd);

    // First 8 bytes of the file is the size of the header
    // as an unsigned little-endian 64-bit integer
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i)
        header_len += (unsigned char)ctx->buf[i] << i * 8;
    ctx->data = ctx->buf + 8 + header_len;

    // Sanity check
    if (ctx->data > ctx->buf + ctx->buf_len) {
        st_close(ctx);
        return NULL;
    }

    parse_ctx pcx;
    pcx.tok = (char*)&ctx->buf[8];
    pcx.data = (char*)ctx->data;

    if (!parse_object(ctx, &pcx, parse_header_value)) {
        st_close(ctx);
        return NULL;
    }
    st_rewind_tensor(ctx);
    st_rewind_metadata(ctx);

    return ctx;
}

void st_close(st_ctx* ctx) {
    munmap(ctx->buf, ctx->buf_len);
    free(ctx);
}

st_tensor* st_next_tensor(st_ctx* ctx) {
    tensor_node* cur = ctx->cur_tensor;
    if (!cur) return NULL;
    ctx->cur_tensor = ((tensor_node*)ctx->cur_tensor)->next;
    return &cur->tensor;
}

st_metadata* st_next_metadata(st_ctx* ctx) {
    metadata_node* cur = ctx->cur_metadata;
    if (!cur) return NULL;
    ctx->cur_metadata = ((metadata_node*)ctx->cur_metadata)->next;
    return &cur->metadata;
}

void st_rewind_tensor(st_ctx* ctx) {
    ctx->cur_tensor = ctx->tensors;
}

void st_rewind_metadata(st_ctx* ctx) {
    ctx->cur_metadata = ctx->metadatas;
}
