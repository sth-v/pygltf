from __future__ import annotations
import json
import struct
from collections import namedtuple
from typing import Generator,NamedTuple

import numpy as np

class BinaryGltfChunkType(NamedTuple):
    chunk_type:int
    ascii:str
    description:str



    def tobytes(self)->bytes:
        return struct.pack('I',self.chunk_type)
    def __int__(self):
        return self.chunk_type

    def __str__(self):
        return self.ascii
    def __bytes__(self):
        return self.tobytes()
    def __repr__(self):
        return f'GLTF ChunkType ({self.chunk_type}, "{self.ascii}", "{self.description})"'

gltf_chunk_types={
    1313821514:BinaryGltfChunkType(
        1313821514,
        'JSON',
        "Structured JSON content"
    ),
    5130562:BinaryGltfChunkType(
        5130562,
        'BIN',
        "Binary buffer"
    )
}
JSON=gltf_chunk_types[1313821514]
BIN=gltf_chunk_types[5130562]
uint32 = np.dtype('uint32')
header_t = np.dtype('3uint32')
chunk_info_t = np.dtype('2uint32')

BinaryGltfHeaderChunk = namedtuple('BinaryGltfHeaderChunk', ['magic', 'version', 'length'])
BinaryGltfJsonChunk = namedtuple('BinaryGltfJsonChunk', ['chunkLength', 'chunkType', 'chunkData'
                                                         ])
BinaryGltfChunk = namedtuple('BinaryGltfJsonChunk', ['chunkLength', 'chunkType', 'chunkData'])
UnpackChunkResult = namedtuple('UnpackResult', ['chunk', 'ptr']
                               )


def unpack_header_chunk(bts: bytes) -> UnpackChunkResult:
    header = np.frombuffer(bts[:header_t.itemsize], dtype=uint32)

    return UnpackChunkResult(BinaryGltfHeaderChunk(struct.pack('I', header[0]).decode(), *header[1:]),
                             header_t.itemsize)


def unpack_json_chunk(bts: bytes) -> UnpackChunkResult:
    chunk_info = np.frombuffer(bts[: chunk_info_t.itemsize], uint32)
    chunk_type = int(chunk_info[1])
    chunk_type_obj=gltf_chunk_types.get(chunk_type, chunk_type)
    if chunk_type_obj is not JSON:
        raise ValueError(f'JSON (1313821514) chunk type is expected, but {chunk_type} exists:')
    return UnpackChunkResult(
        BinaryGltfJsonChunk(
            chunk_info[0], chunk_type_obj,
            json.loads(bts[chunk_info_t.itemsize:][: int(chunk_info[0])].decode())
        ),
        chunk_info[0] + chunk_info_t.itemsize
    )


def unpack_binary_chunk(bts: bytes) -> UnpackChunkResult:
    chunk_info = np.frombuffer(bts[: chunk_info_t.itemsize], uint32)
    chunk_type=int(chunk_info[1])

    return UnpackChunkResult(
        BinaryGltfChunk(
            chunk_info[0],
            gltf_chunk_types.get(chunk_type,chunk_type),
            bts[chunk_info_t.itemsize:][: int(chunk_info[0])]
        ),
        chunk_info[0] + chunk_info_t.itemsize
    )


def unpack_all(bts: bytes) -> Generator[BinaryGltfHeaderChunk | BinaryGltfJsonChunk | BinaryGltfChunk, None, None]:
    header_result = unpack_header_chunk(bts)
    yield header_result.chunk
    length = header_result.chunk.length
    ptr = header_result.ptr
    json_result = unpack_json_chunk(bts[ptr:])
    yield json_result.chunk
    ptr += json_result.ptr

    while length > ptr:
        binary_result = unpack_binary_chunk(bts[ptr:])
        ptr += binary_result.ptr
        yield binary_result.chunk



if __name__ == '__main__':
    with open('/Users/andrewastakhov/dev/mmcore-clean/mmcore/examples/scene.glb', 'rb') as f:
        data = f.read()
    chunks = list(unpack_all(data))
    # result1=unpack_header_chunk(data)
    # result2=unpack_json_chunk(data[result1.ptr:])
