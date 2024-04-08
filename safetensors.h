/*
 * libsafetensors - A zero-dependency, zero-copy, single source file
 * C/C++ safetensors loader.
 *
 * Copyright (c) 2024 Thomas Ip <thomas@ipthomas.com>
 *
 * This work is licensed under the terms of the MIT license.
 * For a copy, see LICENSE.
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ST_DTYPE_BOOL,
    ST_DTYPE_U8,
    ST_DTYPE_I8,
    ST_DTYPE_F8_E5M2,
    ST_DTYPE_F8_E4M3,
    ST_DTYPE_I16,
    ST_DTYPE_U16,
    ST_DTYPE_F16,
    ST_DTYPE_BF16,
    ST_DTYPE_I32,
    ST_DTYPE_U32,
    ST_DTYPE_F32,
    ST_DTYPE_F64,
    ST_DTYPE_I64,
    ST_DTYPE_U64
} st_dtype;

typedef struct {
    uint8_t* buf;        // Pointer to the mmapped .safetensor file
    size_t buf_len;      // File size in bytes
    uint8_t* data;       // Start of the rest of the file (byte-buffer)
    void* tensors;       // Linked list of parsed tensors
    void* cur_tensor;    // Current position in the tensor list
    void* metadatas;     // Linked list of parsed metadata
    void* cur_metadata;  // Current position in the metadata list
} st_ctx;

typedef struct {
    char* string;
    size_t len;
} st_string;

typedef struct {
    st_string name;
    st_dtype dtype;    // Tensor element type
    size_t data_len;   // Tensor data size in bytes
    uint8_t* data;     // Pointer to the mmaped tensor data
    size_t shape_len;  // Number of dimensions of the tensor
    uint64_t shape[];  // Dimensions (slowest-moving dimension
                       // first)
} st_tensor;

typedef struct {
    st_string name;
    st_string value;
} st_metadata;

// Parse and memory-map a .safetensors file. Returns NULL if any step failed,
// otherwise return a context used by the other functions.
st_ctx* st_open(const char* filename);

// Close all allocated resources. All pointers returned by the library will be
// freed here; **freeing them manually is undefined behaviour.**
void st_close(st_ctx* ctx);

// Get the next tensor in the file. Tensors are returned in the order they
// appear in the header section. The buffer pointed by the `data` field is valid
// until `st_close` is called.
st_tensor* st_next_tensor(st_ctx* ctx);

// Get the next metadata in the file. Metadata are returned in the order they
// appear in the header section.
st_metadata* st_next_metadata(st_ctx* ctx);

// Resets internal state such that the next call to `st_next_tensor` will return
// the first tensor.
void st_rewind_tensor(st_ctx* ctx);

// Resets internal state such that the next call to `st_next_metadata` will
// return the first metadata.
void st_rewind_metadata(st_ctx* ctx);

#ifdef __cplusplus
}
#endif
#endif  // SAFETENSORS_H
