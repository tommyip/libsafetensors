import subprocess
from ctypes import CDLL, POINTER, c_char_p, c_size_t
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper

import pytest

from .types import ST_METADATA, ST_STRING, ST_TENSOR, TOKEN, parse_ctx_p, st_ctx_p

TEST_DIR = Path(__file__).resolve().parent
SOURCE_PATH = TEST_DIR.parent / "safetensors.c"


class CompilationError(Exception):
    pass


def compile(source: Path) -> _TemporaryFileWrapper:
    binary = NamedTemporaryFile(suffix=".so")
    result = subprocess.run(
        ["clang", "-g", "-O0", "-std=c99", "-shared", "-o", binary.name, source]
    )

    if result.returncode != 0:
        binary.close()
        raise CompilationError(result.stdout.decode())

    return binary


@pytest.fixture(scope="session")
def lib():
    with compile(SOURCE_PATH) as binary_file:
        binary = CDLL(binary_file.name)
        binary.st_open.restype = st_ctx_p
        binary.st_open.argtypes = [c_char_p]
        binary.next_token.argtypes = [parse_ctx_p, POINTER(TOKEN)]
        binary.array_length.argtypes = [parse_ctx_p, POINTER(c_size_t)]
        binary.parse_tensor.argtypes = [st_ctx_p, parse_ctx_p, ST_STRING]
        binary.st_next_tensor.argtypes = [st_ctx_p]
        binary.st_next_tensor.restype = POINTER(ST_TENSOR)
        binary.st_next_metadata.argtypes = [st_ctx_p]
        binary.st_next_metadata.restype = POINTER(ST_METADATA)
        yield binary
