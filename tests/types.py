from ctypes import (
    POINTER,
    Structure,
    addressof,
    c_char_p,
    c_int,
    c_size_t,
    c_ubyte,
    c_uint8,
    c_uint64,
    c_void_p,
    cast,
    string_at,
)
from ctypes import (
    Union as CUnion,
)
from typing import List, Literal, NamedTuple, Optional, Tuple, Union


def pointer_offset(ptr, ptr_t, offset):
    ptr_addr = addressof(cast(ptr, POINTER(c_ubyte)).contents)
    return cast(c_void_p(ptr_addr + offset), ptr_t)


def pointer_dist(ptr1, ptr2) -> int:
    ptr1_addr = addressof(cast(ptr1, POINTER(c_ubyte)).contents)
    ptr2_addr = addressof(cast(ptr2, POINTER(c_ubyte)).contents)
    return ptr2_addr - ptr1_addr


class ST_STRING(Structure):
    _fields_ = [("string", c_char_p), ("len", c_size_t)]

    def to_str(self):
        return string_at(self.string, size=self.len)


Token = Optional[Union[Literal["{", "}", "[", "]", ":", ","], int, str]]


class TOKEN_VALUE(CUnion):
    _fields_ = [("integer", c_uint64), ("string", POINTER(ST_STRING))]


class TOKEN(Structure):
    _fields_ = [("kind", c_int), ("value", TOKEN_VALUE)]

    def deserialize(self) -> Token:
        match self.kind:
            case 0:
                return "{"
            case 1:
                return "}"
            case 2:
                return "["
            case 3:
                return "]"
            case 4:
                return ":"
            case 5:
                return ","
            case 6:
                return self.value.integer
            case 7:
                return self.value.string.to_str()
            case 8:
                return None


class Tensor(NamedTuple):
    name: bytes
    dtype: int
    data_offsets: Tuple[int, int]
    shape: List[int]


class ST_TENSOR(Structure):
    _fields_ = [
        ("name", ST_STRING),
        ("dtype", c_int),
        ("data_len", c_size_t),
        ("data", POINTER(c_uint8)),
        ("shape_len", c_size_t),
        ("shape", (c_uint64 * 0)),
    ]

    def deserialize(self, data_p: POINTER(c_uint8)):
        name = self.name.to_str()
        data_offset = pointer_dist(data_p, self.data)
        data_offsets = (data_offset, data_offset + self.data_len)
        shape_t = c_uint64 * self.shape_len
        shape = [
            int(x)
            for x in shape_t.from_address(addressof(self) + ST_TENSOR.shape.offset)
        ]

        return Tensor(name, self.dtype, data_offsets, shape)


class TENSOR_NODE(Structure):
    pass


TENSOR_NODE._fields_ = [("next", POINTER(TENSOR_NODE)), ("tensor", ST_TENSOR)]


class ST_METADATA(Structure):
    _fields_ = [("name", ST_STRING), ("value", ST_STRING)]

    def deserialize(self) -> Tuple[str, str]:
        return self.name.to_str(), self.value.to_str()


class METADATA_NODE(Structure):
    pass


METADATA_NODE._fields_ = [("next", POINTER(METADATA_NODE)), ("metadata", ST_METADATA)]


class ST_CTX(Structure):
    _fields_ = [
        ("buf", POINTER(c_uint8)),
        ("buf_len", c_size_t),
        ("data", POINTER(c_uint8)),
        ("tensors", c_void_p),
        ("cur_tensors", c_void_p),
        ("metadatas", c_void_p),
        ("cur_metadatas", c_void_p),
    ]


class PARSE_CTX(Structure):
    _fields_ = [
        ("tok", c_char_p),
        ("data", c_char_p),
        ("tensor_fields", c_ubyte),
        ("tensor_node", POINTER(TENSOR_NODE)),
        ("tensor_dtype", c_int),
        ("tensor_data", POINTER(c_uint8)),
        ("tensor_data_len", c_size_t),
    ]


st_ctx_p = POINTER(ST_CTX)
parse_ctx_p = POINTER(PARSE_CTX)
