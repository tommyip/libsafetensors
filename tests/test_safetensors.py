import struct
from contextlib import contextmanager
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_size_t,
    c_uint8,
    c_uint64,
    cast,
)
from tempfile import NamedTemporaryFile
from typing import Dict, Generator, Tuple

import pytest
import safetensors.torch
import torch

from .conftest import TEST_DIR
from .types import (
    PARSE_CTX,
    ST_CTX,
    ST_STRING,
    TOKEN,
    Tensor,
    parse_ctx_p,
    pointer_offset,
    st_ctx_p,
)


@contextmanager
def safetensor_file(
    tensors: Dict[str, torch.Tensor], metadata: Dict[str, str]
) -> Generator[bytes, None, None]:
    with NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        print(tmp.name)
        safetensors.torch.save_file(tensors, tmp.name, metadata)
        yield bytes(tmp.name, "ascii")


def parsing_ctx(string: bytes) -> Tuple[st_ctx_p, parse_ctx_p]:
    header_len = struct.pack("<Q", len(string))
    buf = header_len + string
    buf_p = c_char_p(buf)
    data_p = pointer_offset(buf_p, c_char_p, len(buf) + 8)
    tok_p = pointer_offset(buf_p, c_char_p, 8)
    ctx = ST_CTX(
        buf=cast(buf_p, POINTER(c_uint8)),
        buf_len=len(buf),
        data=cast(data_p, POINTER(c_uint8)),
    )
    pcx = PARSE_CTX(tok=tok_p, data=data_p)
    return st_ctx_p(ctx), parse_ctx_p(pcx)


@pytest.mark.parametrize(
    "tensors,metadata",
    [
        ({}, {}),
        (
            {
                "embedding": torch.randn((5000,)),
                "attention": torch.randn((100, 256, 3)),
            },
            {},
        ),
        ({}, {"n_layer": "20", "d_model": "2048"}),
        (
            {
                "embedding": torch.randn((5000,)),
                "attention": torch.randn((100, 256, 3)),
            },
            {"n_layer": "20", "d_model": "2048"},
        ),
        (
            {
                "many.dimensions.f16": torch.zeros(
                    (10, 2, 256, 1, 4, 30), dtype=torch.bfloat16
                ),
                "ints": torch.ones((10,), dtype=torch.int8),
            },
            {"n_layer": "20"},
        ),
    ],
)
def test_everything(lib, tensors, metadata):
    with safetensor_file(tensors, metadata) as file:
        st_ctx = lib.st_open(file)
        assert st_ctx
        lib.st_close(st_ctx)


def test_open_non_existent_file(lib):
    assert not lib.st_open(bytes(TEST_DIR / "non_existent"))


@pytest.mark.parametrize(
    "content",
    [
        # Missing leading '{'
        b'\x98\x00\x00\x00\x00\x00\x00\x00"attention":{"dtype":"F32","shape":[100'
        b',256,3],"data_offsets":[0,307200]},"embedding":{"dtype":"F32","shape":'
        b'[5000],"data_offsets":[307200,327200]}}',
        # Header size value larger than the file size
        b'\x10\x27\x00\x00\x00\x00\x00\x00{"attention":{"dtype":"F32","shape":[100'
        b',256,3],"data_offsets":[0,307200]},"embedding":{"dtype":"F32","shape":'
        b'[5000],"data_offsets":[307200,327200]}}',
    ],
)
def test_open_malformed_file(lib, content):
    with NamedTemporaryFile() as f:
        f.write(content)
        f.seek(0)
        assert not lib.st_open(bytes(f.name, "ascii"))


@pytest.mark.parametrize(
    "string",
    [
        b"",
        b"embedding",
        b"embedding.weight",
        rb"\"",
        rb"\\",
        rb"\/",
        rb"\b",
        rb"\f",
        rb"\n",
        rb"\r",
        rb"\t" rb"\u0000",
        rb"\u01aD",
        rb"\u01234",
        bytes("👪", "utf-8"),
        bytes("世界", "utf-8"),
    ],
)
def test_tokenize_string(lib, string):
    _, pcx = parsing_ctx(b'"' + string + b'"')
    st_string = ST_STRING()
    assert lib.tokenize_string(pcx, byref(st_string))
    assert st_string.to_str() == string


@pytest.mark.parametrize(
    "string",
    [
        b"\\",
        b"\r",
        b"\n",
        rb"\u0",
        rb"\u01",
        rb"\uAAA",
        rb"\uHaha",
    ],
)
def test_tokenize_string_invalid(lib, string):
    _, pcx = parsing_ctx(b'"' + string + b'"')
    st_string = ST_STRING()
    assert not lib.tokenize_string(pcx, byref(st_string))


@pytest.mark.parametrize(
    "string,expected",
    [
        (b"0", 0),
        (b"1", 1),
        (b"42", 42),
        (b"9876543210", 9876543210),
        (b"0x1234", 0),  # Parse the 0, the next parse call will fail.
    ],
)
def test_tokenize_integer(lib, string, expected):
    _, pcx = parsing_ctx(string)
    actual = TOKEN()
    assert lib.next_token(pcx, byref(actual))
    actual_py = actual.deserialize()
    assert isinstance(actual_py, int) and actual_py == expected


@pytest.mark.parametrize(
    "string",
    [b"a", b"+42", b"-42"],
)
def test_tokenize_integer_invalid(lib, string):
    _, pcx = parsing_ctx(string)
    assert not lib.next_token(pcx, byref(TOKEN()))


@pytest.mark.parametrize(
    "string,expected",
    [(b"[]", 0), (b"[1]", 1), (b"[1, 42, 1337]", 3)],
)
def test_array_length(lib, string, expected):
    actual = c_size_t(0)
    _, pcx = parsing_ctx(string)
    assert lib.array_length(pcx, byref(actual))
    assert actual.value == expected


@pytest.mark.parametrize("string", [b"[}", b"[0x1]", b"[1,,2]"])
def test_array_length_invalid(lib, string):
    _, pcx = parsing_ctx(string)
    assert not lib.array_length(pcx, byref(c_size_t()))


@pytest.mark.parametrize(
    "string,expected",
    [
        (b"[]", []),
        (b"[0]", [0]),
        (b"[1, 42, 1337]", [1, 42, 1337]),
        (b"[42,10000, 0 , 999, 133337]", [42, 10000, 0, 999, 133337]),
    ],
)
def test_parse_array(lib, string, expected):
    _, pcx = parsing_ctx(string)
    arr = (c_uint64 * len(expected))()
    assert lib.parse_array(pcx, byref(arr), len(expected))
    assert list(arr) == expected


@pytest.mark.parametrize(
    "string",
    [
        b'["embedding"]',
        b"[42}",
        b"[1,, 1337]",
        b"{42, 42]",
    ],
)
def test_parse_array_invalid(lib, string):
    _, pcx = parsing_ctx(string)
    arr = (c_uint64 * 5)()
    assert not lib.parse_array(pcx, byref(arr), 5)


@pytest.mark.parametrize(
    "string,expected",
    [
        (
            b'{"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [0, 8192]}',
            Tensor(b"", 7, (0, 8192), [1, 16, 256]),
        ),
        (
            b'{"data_offsets":[1024,1024],"shape":[],"dtype": "F8_E4M3"}',
            Tensor(b"", 4, (1024, 1024), []),
        ),
        (
            b'{"shape":[1000],"dtype":"BOOL","data_offsets":[12345,13345]}',
            Tensor(b"", 0, (12345, 13345), [1000]),
        ),
    ],
)
def test_parse_tensor(lib, string, expected):
    name = ST_STRING(b"", 0)
    ctx, pcx = parsing_ctx(string)
    assert lib.parse_tensor(ctx, pcx, name)
    lib.st_rewind_tensor(ctx)
    tensor_p = lib.st_next_tensor(ctx)
    assert tensor_p
    assert tensor_p.contents.deserialize(ctx.contents.data) == expected


@pytest.mark.parametrize(
    "string",
    [
        b"{}",  # Missing everything
        b'{"shape": [1, 16, 256], "data_offsets": [0, 8192]}',  # Missing dtype
        b'{"dtype": "F16", "data_offsets": [0, 8192]}',  # Missing shape
        b'{"dtype": "F16", "shape": [1, 16, 256]}',  # Missing data offsets
        b'{"dtype": "F16", "dtype": "U16", "shape": [1, 16, 256], "data_offsets": [0, 8192]}',  # Repeated dtype
        b'{"dtype": "F16", "shape": [1, 16, 256], "shape": [256], "data_offsets": [0, 8192]}',  # Repeated shape
        b'{"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [0, 8192], "data_offsets": [8192, 16384]}',  # Repeated data_offsets
        b'{"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [0, 8192], "order": "C"}',  # Unknown key
        b'{"dtype": "I4", "shape": [1, 16, 256], "data_offsets": [0, 8192]}',  # Unknown dtype
    ],
)
def test_parse_tensor_invalid(lib, string):
    ctx, pcx = parsing_ctx(string)
    assert not lib.parse_tensor(ctx, pcx, ST_STRING(b"", 0))


@pytest.mark.parametrize(
    "string,expected",
    [
        (b"{}", []),
        (b'{"n_layer": "256"}', [(b"n_layer", b"256")]),
        (
            b'{"n_layer": "256", "d_model": "10000"}',
            [(b"n_layer", b"256"), (b"d_model", b"10000")],
        ),
    ],
)
def test_parse_metadatas(lib, string, expected):
    ctx, pcx = parsing_ctx(string)
    assert lib.parse_object(ctx, pcx, lib.parse_metadata)
    lib.st_rewind_metadata(ctx)
    for metadata_expected in expected:
        actual = lib.st_next_metadata(ctx)
        assert actual
        assert actual.contents.deserialize() == metadata_expected
    assert not lib.st_next_metadata(ctx)


@pytest.mark.parametrize(
    "string",
    [
        b"{",
        b'{"n_layer": 256}',
        b'{"n_layer": "256", d_model: "10000"}',
    ],
)
def test_parse_metadatas_invalid(lib, string):
    ctx, pcx = parsing_ctx(string)
    assert not lib.parse_object(ctx, pcx, lib.parse_metadata)
