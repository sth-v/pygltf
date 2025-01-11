#!/usr/bin/env python3
"""
Example of packing MeshGeometry objects into a single .glb file,
with correct BufferView targets for vertex and index data.
"""


from __future__ import annotations
import sys
sys.setrecursionlimit(100000)
import struct
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Dict, List, Optional, Any, TypedDict
import numpy as np
from dataclasses_json import dataclass_json,Exclude,config,Undefined
from numpy.typing import NDArray

from gltf.components import COMPONENT_DTYPES, NUM_COMPONENTS, COMPONENT_FMT

###############################################################################
#                               gltf.definitions                              #
###############################################################################

#
# Utility type: GLTFId is just an integer >= 0
#
GLTFId = int

@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class GLTFProperty:
    """
    glTF Property
    """
    extensions: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class GLTFChildOfRootProperty(GLTFProperty):
    """
    glTF Child of Root Property
    """
    name: Optional[str] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AccessorSparseIndices(GLTFProperty):
    bufferView: GLTFId
    byteOffset: int = 0
    componentType: int = 0


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AccessorSparseValues(GLTFProperty):
    bufferView: GLTFId
    byteOffset: int = 0


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AccessorSparse(GLTFProperty):
    count: int
    indices: AccessorSparseIndices
    values: AccessorSparseValues


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Accessor(GLTFChildOfRootProperty):
    componentType: int
    count: int
    type: str
    bufferView: Optional[GLTFId] = None
    byteOffset: int = 0
    normalized: bool = False
    max: Optional[List[float]] = None
    min: Optional[List[float]] = None
    sparse: Optional[AccessorSparse] = None
 


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class TextureInfo(GLTFProperty):
    index: GLTFId
    texCoord: int = 0


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MaterialNormalTextureInfo(TextureInfo):
    scale: float = 1.0


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MaterialOcclusionTextureInfo(TextureInfo):
    strength: float = 1.0


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MaterialPBRMetallicRoughness(GLTFProperty):
    baseColorFactor: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    baseColorTexture: Optional[TextureInfo] = None
    metallicFactor: float = 1.0
    roughnessFactor: float = 1.0
    metallicRoughnessTexture: Optional[TextureInfo] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Material(GLTFChildOfRootProperty):
    pbrMetallicRoughness: Optional[MaterialPBRMetallicRoughness] = None
    normalTexture: Optional[MaterialNormalTextureInfo] = None
    occlusionTexture: Optional[MaterialOcclusionTextureInfo] = None
    emissiveTexture: Optional[TextureInfo] = None
    emissiveFactor: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    alphaMode: str = "OPAQUE"
    alphaCutoff: float = 0.5
    doubleSided: bool = False


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Sampler(GLTFChildOfRootProperty):
    magFilter: Optional[int] = None
    minFilter: Optional[int] = None
    wrapS: int = 10497
    wrapT: int = 10497


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MeshPrimitive(GLTFProperty):
    attributes: Dict[str, GLTFId]
    mode: int = 4
    indices: Optional[GLTFId] = None
    material: Optional[GLTFId] = None
    targets: Optional[List[Dict[str, GLTFId]]] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Mesh(GLTFChildOfRootProperty):
    primitives: List[MeshPrimitive]
    weights: Optional[List[float]] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class CameraOrthographic(GLTFProperty):
    xmag: float
    ymag: float
    zfar: float
    znear: float


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class CameraPerspective(GLTFProperty):
    yfov: float
    znear: float
    aspectRatio: Optional[float] = None
    zfar: Optional[float] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Camera(GLTFChildOfRootProperty):
    type: str
    orthographic: Optional[CameraOrthographic] = None
    perspective: Optional[CameraPerspective] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Node(GLTFChildOfRootProperty):
    camera: Optional[GLTFId] = None
    children: Optional[List[GLTFId]] = None
    skin: Optional[GLTFId] = None
    matrix: Optional[List[float]] = None
    mesh: Optional[GLTFId] = None
    rotation: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    translation: Optional[List[float]] = None
    weights: Optional[List[float]] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Skin(GLTFChildOfRootProperty):
    joints: List[GLTFId] = field(default_factory=list)
    inverseBindMatrices: Optional[GLTFId] = None
    skeleton: Optional[GLTFId] = None

@dataclass_json
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class BufferView(GLTFChildOfRootProperty):
    buffer: GLTFId
    byteLength: int
    byteOffset: int = 0
    byteStride: Optional[int] = None
    target: Optional[int]=None



@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Buffer(GLTFChildOfRootProperty):
    byteLength: int
    uri: Optional[str] = None



@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Image(GLTFChildOfRootProperty):
    uri: Optional[str] = None
    mimeType: Optional[str] = None
    bufferView: Optional[GLTFId] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Texture(GLTFChildOfRootProperty):
    sampler: Optional[GLTFId] = None
    source: Optional[GLTFId] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Scene(GLTFChildOfRootProperty):
    nodes: Optional[List[GLTFId]] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AnimationChannelTarget(GLTFProperty):
    path: str
    node: Optional[GLTFId] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AnimationChannel(GLTFProperty):
    sampler: GLTFId
    target: AnimationChannelTarget


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AnimationSampler(GLTFProperty):
    input: GLTFId
    output: GLTFId
    interpolation: str = "LINEAR"


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Animation(GLTFChildOfRootProperty):
    channels: List[AnimationChannel]
    samplers: List[AnimationSampler]


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Asset(GLTFProperty):
    version: str
    copyright: Optional[str] = None
    generator: Optional[str] = None
    minVersion: Optional[str] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass()
class GLTF(GLTFProperty):
    asset: Asset
    extensionsUsed: Optional[List[str]] = None
    extensionsRequired: Optional[List[str]] = None
    accessors: Optional[List[Accessor]] = None
    animations: Optional[List[Animation]] = None
    buffers: Optional[List[Buffer]] = None
    bufferViews: Optional[List[BufferView]] = None
    cameras: Optional[List[Camera]] = None
    images: Optional[List[Image]] = None
    materials: Optional[List[Material]] = None
    meshes: Optional[List[Mesh]] = None
    nodes: Optional[List[Node]] = None
    samplers: Optional[List[Sampler]] = None
    scene: Optional[GLTFId] = None
    scenes: Optional[List[Scene]] = None
    skins: Optional[List[Skin]] = None
    textures: Optional[List[Texture]] = None

###############################################################################
#                               gltf.geometry                                 #
###############################################################################

@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class MeshGeometry:
    """
    A simple container for geometry data in NumPy arrays.
    - vertices: shape (N,3)
    - colors: shape (N,3) or (N,4)
    - normals: shape (N,3)
    - uv: shape (N,2)
    - faces: shape (M,) or (M,x) e.g. (M,3) for triangles
    - extras: optional dictionary
    """
    vertices: NDArray[np.float32]
    colors: NDArray[np.float32] | None = None
    normals: NDArray[np.float32] | None = None
    uv: NDArray[np.float32] | None = None
    faces: NDArray[np.uint32] | None = None
    extras: Dict[str, Any] | None = None

###############################################################################
#                      gltf packing (pack_header, pack_json, etc.)            #
###############################################################################

def _pad_to_4bytes(data: bytes, pad_byte=b"\x00" ) -> bytes:
    padding = (4 - (len(data) % 4)) % 4
    return data + (pad_byte * padding)


def pack_header_chunk(total_length: int, version: int = 2) -> bytes:
    """
    GLB Header: magic (4 bytes), version (4), total_length (4).
    magic=0x46546C67 = 'glTF' in ASCII
    """
    magic = 0x46546C67
    return struct.pack("<III", magic, version, total_length)


JSON_CHUNK_TYPE = 0x4E4F534A  # 'JSON'
def pack_json_chunk(json_obj: dict) -> bytes:
    json_str = json.dumps(json_obj, separators=(',', ':'))
    json_bytes = json_str.encode("utf-8")
    json_bytes = _pad_to_4bytes(json_bytes,b"\x20")
    chunk_header = struct.pack("<II", len(json_bytes), JSON_CHUNK_TYPE)
    return chunk_header + json_bytes


BIN_CHUNK_TYPE = 0x004E4942  # 'BIN'
def pack_binary_chunk(bin_data: bytes) -> bytes:
    bin_data = _pad_to_4bytes(bin_data)
    chunk_header = struct.pack("<II", len(bin_data), BIN_CHUNK_TYPE)
    return chunk_header + bin_data


def pack_all(gltf_obj: GLTF, bin_data: bytes) -> bytes:
    """
    1. Convert gltf_obj to dict
    2. Create JSON chunk
    3. Create BIN chunk
    4. Create header
    5. Return all .glb bytes
    """
    gltf_dict = _gltf_to_dict(gltf_obj)
    json_chunk = pack_json_chunk(gltf_dict)
    bin_chunk = pack_binary_chunk(bin_data)
    total_length = 12 + len(json_chunk) + len(bin_chunk)  # 12 = size of header
    header = pack_header_chunk(total_length, version=2)
    return header + json_chunk + bin_chunk


import json
from typing import Any, Union


def remove_empty(data: Any) -> Any:
    """
    Recursively remove keys from dictionaries where the value is None, empty dict, or empty list.
    Also cleans lists by removing such elements.

    Args:
        data (Any): The JSON-like data (dict, list, etc.)

    Returns:
        Any: The cleaned data with specified empty values removed.
    """
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            cleaned_value = remove_empty(value)
            # Check if the cleaned_value should be kept
            if cleaned_value is not None:
                if isinstance(cleaned_value, dict) and not cleaned_value:
                    continue  # Skip empty dict
                if isinstance(cleaned_value, list) and not cleaned_value:
                    continue  # Skip empty list
                cleaned_dict[key] = cleaned_value
        return cleaned_dict
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_item = remove_empty(item)
            if cleaned_item is not None:
                if isinstance(cleaned_item, dict) and not cleaned_item:
                    continue  # Skip empty dict
                if isinstance(cleaned_item, list) and not cleaned_item:
                    continue  # Skip empty list
                cleaned_list.append(cleaned_item)
        return cleaned_list
    else:
        # For other data types, return as is unless it's None
        return data if data is not None else None


def _gltf_to_dict(gltf_obj: GLTF) -> dict:
    """
    Convert the GLTF dataclass to a plain dict, removing None/empty fields.
    """
    raw = gltf_obj.to_dict()

    return remove_empty(raw)

###############################################################################
#                     Creating GLTF from MeshGeometry objects                 #
###############################################################################

# WebGL / glTF constants
COMPONENT_FLOAT = 5126        # gl.FLOAT
COMPONENT_UNSIGNED_INT = 5125 # gl.UNSIGNED_INT
TARGET_ARRAY_BUFFER = 34962   # ARRAY_BUFFER
TARGET_ELEMENT_ARRAY_BUFFER = 34963  # ELEMENT_ARRAY_BUFFER

BUFFER_TYPE=defaultdict(lambda : TARGET_ARRAY_BUFFER)


class AccessorData(TypedDict):
    view: BufferView
    name: str
    dtype:int
    type:str
    data: np.ndarray





class AccessorNode:
    def __init__(self, data: AccessorData):
        self.data = data
        self.next = None
        self.size = self.count * byte_stride(self.data['type'], self.data['dtype'])
        self._buffofset = None
        self._end = None

    @property
    def byteOffset(self):
        if self.next is None:
            return 0

        else:
            if self._buffofset is None:
                self._buffofset = self.next.byteOffset + self.next.size
            return self._buffofset

    @property
    def min(self):
        return np.min(self.buffer_data, axis=0).tolist()

    @property
    def max(self):
        return np.max(self.buffer_data, axis=0).tolist()

    @property
    def view(self):
        return self.data['view']

    @property
    def count(self):
        return len(self.buffer_data)

    @property
    def prev_count(self):
        if self.next is None:
            return 0
        else:
            return self.next.count

    @property
    def start(self):
        return self.prev_count

    @property
    def end(self):
        if self.next is None:
            return self.count
        elif self._end is None:
            self._end = self.next.end + self.count
        return self._end

    @property
    def buffer_data(self):
        return self.data['data']

    @buffer_data.setter
    def buffer_data(self, v):
        if len(v) != len(self.data['data']):

            self.data['data'] = np.array(v, dtype=COMPONENT_DTYPES[self.data['dtype']])
            self._end = None
            self._buffofset = None
        else:
            self.data['data'] = np.array(v, dtype=COMPONENT_DTYPES[self.data['dtype']])

    def togltf(self, buffer_views):
        res = {

            "componentType": self.data['dtype'],
            "count": self.count,
            "max": self.max,
            "min": self.min,
            "type": self.data['type']
        }
        if self.byteOffset > 0:
            res['byteOffset'] = self.byteOffset


        return Accessor(
            bufferView=buffer_views.index(self.view),
            **res

        )

    def todict(self, doc:GLTF):

        res = {
            "bufferView": doc.bufferViews.index(self.view),

            "componentType": self.data['dtype'],
            "count": self.count,
            "max": self.max,
            "min": self.min,
            "type": self.data['type']
        }
        if self.byteOffset > 0:
            res['byteOffset'] = self.byteOffset
        return res


    def deps(self):
        return dict(bufferViews=[self.view])




# Create a LinkedList class


class AccessorList:
    def __init__(self):
        self.head = None

    # Method to add a node at begin of LL
    def insertAtBegin(self, data):
        new_node = AccessorNode(data)
        if self.head is None:
            self.head = new_node
            return new_node
        else:
            new_node.next = self.head
            self.head = new_node
        return new_node

    # Method to add a node at any index
    # Indexing starts from 0.
    def insertAtIndex(self, data: AccessorData, index: int):
        new_node = AccessorNode(data)
        current_node = self.head
        position = 0
        if position == index:
            self.insertAtBegin(data)
        else:
            while (current_node != None and position + 1 != index):
                position = position + 1
                current_node = current_node.next

            if current_node != None:
                new_node.next = current_node.next
                current_node.next = new_node
            else:
                print("Index not present")
        return new_node

    # Method to add a node at the end of LL

    def insertAtEnd(self, data: AccessorData):
        new_node = AccessorNode(data)
        if self.head is None:
            self.head = new_node
            return new_node

        current_node = self.head
        while (current_node.next):
            current_node = current_node.next

        current_node.next = new_node
        return new_node

    # Update node of a linked list
    # at given position
    def updateNode(self, val, index):
        current_node = self.head
        position = 0
        if position == index:
            current_node.data = val
        else:
            while (current_node != None and position != index):
                position = position + 1
                current_node = current_node.next

            if current_node != None:
                current_node.data = val
            else:
                print("Index not present")

    # Method to remove first node of linked list

    def remove_first_node(self):
        if (self.head == None):
            return

        self.head = self.head.next

    # Method to remove last node of linked list
    def remove_last_node(self):

        if self.head is None:
            return

        current_node = self.head
        while (current_node.next.next):
            current_node = current_node.next

        current_node.next = None

    # Method to remove at given index
    def remove_at_index(self, index):
        if self.head == None:
            return

        current_node = self.head
        position = 0
        if position == index:
            self.remove_first_node()
        else:
            while (current_node != None and position + 1 != index):
                position = position + 1
                current_node = current_node.next

            if current_node != None:
                current_node.next = current_node.next.next
            else:
                print("Index not present")

    # Method to remove a node from linked list
    def remove_node(self, node):
        current_node = self.head

        while (current_node != None and current_node.next.global_index != node.global_index):
            current_node = current_node.next

        if current_node == None:
            return
        else:
            current_node.next = current_node.next.next

    def index(self, node):
        current_node = self.head
        i = 0
        while (current_node != None and current_node.next.global_index != node.global_index):
            current_node = current_node.next
            i += 1
        if current_node == None:
            return
        else:
            return i

    def get(self, ixs, __default=None):
        current_node = self.head
        i = 0
        while True:
            if i == ixs:
                break
            elif current_node.next is None:
                break
            current_node = current_node.next
            i += 1
        if current_node == None:
            return __default
        else:
            return current_node

    # Print the size of linked list
    def sizeOfLL(self):
        size = 0
        if (self.head):
            current_node = self.head
            while (current_node):
                size = size + 1
                current_node = current_node.next
            return size
        else:
            return 0

    # print method for the linked list
    def printLL(self):
        current_node = self.head
        while (current_node):
            print(current_node.data)
            current_node = current_node.next

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, item, v):

        return self.insertAtIndex(item, v)

    def __contains__(self, item: AccessorNode):
        return self.index(item) is not None

    def __iter__(self):
        return AccessorListIterator(self)

class AccessorListIterator:
    def __init__(self, ll):
        self.curnode = ll.head

    def __iter__(self):
        return self

    def __next__(self):
        if self.curnode is not None:
            node = self.curnode
            self.curnode = self.curnode.next
            return node
        else:
            raise StopIteration()

def _guess_gltf_type_from_array(array: np.ndarray) -> str:
    """
    Return 'SCALAR', 'VEC2', 'VEC3', or 'VEC4' depending on number of columns.
    """
    if array.ndim != 2:
        raise ValueError("Array must be 2D for glTF Accessor.")
    cols = array.shape[1]
    if cols == 1:
        return "SCALAR"
    elif cols == 2:
        return "VEC2"
    elif cols == 3:
        return "VEC3"
    elif cols == 4:
        return "VEC4"
    raise ValueError(f"Unsupported component count {cols}.")


def _compute_min_max(array: np.ndarray) -> (List[float], List[float]):
    """
    Compute per-component min/max for the 2D array.
    """
    minimum = array.min(axis=0).tolist()
    maximum = array.max(axis=0).tolist()
    return (minimum, maximum)


def create_gltf_from_mesh_geometries(meshes: List[MeshGeometry]) -> (GLTF, bytes):
    """
    Build a single-Buffer GLTF from a list of MeshGeometry objects.
    Return (gltf_obj, bin_data).
    """
    # 1) top-level GLTF with required fields
    gltf_obj = GLTF(
        asset=Asset(version="2.0"),
        buffers=[],
        bufferViews=[],
        accessors=[],
        meshes=[],
        nodes=[],
        scenes=[],
        scene=0
    )

    bin_data_list = bytearray()
    current_offset = 0  # track offset in the buffer

    # We'll have a single Scene referencing multiple Node(s).
    scene_nodes = []

    buffer_views = []
    for mesh_idx, geometry in enumerate(meshes):


        # ========== Build attributes & indices ==========

        attributes_dict: Dict[str, int] = {}
        indices_accessor_id: Optional[int] = None

        # Positions
        if geometry.vertices is not None and geometry.vertices.size > 0:
            if geometry.vertices.ndim != 2 or geometry.vertices.shape[1] != 3:
                raise ValueError("Positions must be (N,3).")

            if len(gltf_obj.bufferViews)==0:
                    gltf_obj.bufferViews.append(BufferView(**add_buffer_view(geometry.vertices, bin_data_list,"VEC3",COMPONENT_FLOAT,current_offset)))

            else:
                gltf_obj.bufferViews.append(BufferView(**append_buffer_view(geometry.vertices,bin_data_list,"VEC3",COMPONENT_FLOAT,None,True,False)))



    
            bv_id = len(gltf_obj.bufferViews) - 1

            accessor=Accessor(bufferView=  bv_id,
                     byteOffset=0,
                     componentType=COMPONENT_FLOAT,
                     count=len(geometry.vertices),
                     type="VEC3",
                     min=geometry.vertices.min(axis=0).tolist(),
                     max=geometry.vertices.max(axis=0).tolist()
                   )
            accessor_id=len(gltf_obj.accessors)
            gltf_obj.accessors.append(accessor)


            attributes_dict["POSITION"] = accessor_id

        # Normals
        if geometry.normals is not None and geometry.normals.size > 0:
            if geometry.normals.ndim != 2 or geometry.normals.shape[1] != 3:
                raise ValueError("Normals must be (N,3).")

            if len(gltf_obj.bufferViews) == 0:
                gltf_obj.bufferViews.append(BufferView(
                    **add_buffer_view(geometry.normals, bin_data_list, "VEC3", COMPONENT_FLOAT, current_offset),
                ))
            else:
                gltf_obj.bufferViews.append(BufferView(
                    **append_buffer_view(geometry.normals, bin_data_list, "VEC3", COMPONENT_FLOAT, None, True, False)))
            bv_id = len(gltf_obj.bufferViews) - 1

            accessor = Accessor(bufferView=bv_id,
                                byteOffset=0,
                                componentType=COMPONENT_FLOAT,
                                count=len(geometry.normals),
                                type="VEC3",
                                min=geometry.normals.min(axis=0).tolist(),
                                max=geometry.normals.max(axis=0).tolist())
            accessor_id = len(gltf_obj.accessors)
            gltf_obj.accessors.append(accessor)

            attributes_dict["NORMAL"] = accessor_id

        # UV
        if geometry.uv is not None and geometry.uv.size > 0:
            if geometry.uv.ndim != 2 or geometry.uv.shape[1] != 2:
                raise ValueError("UV must be (N,2).")
            if len(gltf_obj.bufferViews) == 0:
                gltf_obj.bufferViews.append(BufferView(
                    **add_buffer_view(geometry.uv, bin_data_list, "VEC2", COMPONENT_FLOAT, current_offset)))
            else:
                gltf_obj.bufferViews.append(BufferView(
                    **append_buffer_view(geometry.uv, bin_data_list, "VEC2", COMPONENT_FLOAT, None, True, False),))
            bv_id = len(gltf_obj.bufferViews) - 1

            accessor = Accessor(bufferView=bv_id,
                                byteOffset=0,
                                componentType=COMPONENT_FLOAT,
                                count=len(geometry.uv.normals),
                                type="VEC2",
                                min=geometry.uv.min(axis=0).tolist(),
                                max=geometry.uv.max(axis=0).tolist())
            accessor_id = len(gltf_obj.accessors)
            gltf_obj.accessors.append(accessor)
            attributes_dict["TEXCOORD_0"] = accessor_id

        # Colors
        if geometry.colors is not None and geometry.colors.size > 0:
            if geometry.colors.ndim != 2 or (geometry.colors.shape[1] not in [3, 4]):
                raise ValueError("Colors must be (N,3) or (N,4).")
            if len(gltf_obj.bufferViews) == 0:
                gltf_obj.bufferViews.append(BufferView(
                    **add_buffer_view(geometry.colors, bin_data_list, "VEC3", COMPONENT_FLOAT, current_offset)))
            else:
                gltf_obj.bufferViews.append(BufferView(
                    **append_buffer_view(geometry.colors, bin_data_list, "VEC3", COMPONENT_FLOAT, None, True, False))
                    )
            bv_id = len(gltf_obj.bufferViews) - 1

            accessor = Accessor(bufferView=bv_id,
                                byteOffset=0,
                                componentType=COMPONENT_FLOAT,
                                count=len(geometry.colors),
                                type="VEC3",
                                min=geometry.colors.min(axis=0).tolist(),
                                max=geometry.colors.max(axis=0).tolist())
            accessor_id = len(gltf_obj.accessors)
            gltf_obj.accessors.append(accessor)
            attributes_dict["COLOR_0"] = accessor_id

        # Indices
        if geometry.faces is not None and geometry.faces.size > 0:
            indices_array = geometry.faces.flatten()
            if len(gltf_obj.bufferViews) == 0:
                gltf_obj.bufferViews.append(BufferView(
                    **
                    add_buffer_view(indices_array, bin_data_list, "SCALAR", COMPONENT_UNSIGNED_INT, current_offset)))
            else:
                gltf_obj.bufferViews.append(BufferView(
                    **append_buffer_view(indices_array, bin_data_list, "SCALAR", COMPONENT_UNSIGNED_INT, None, False, True)))
            bv_id = len(gltf_obj.bufferViews) - 1
            accessor = Accessor(bufferView=bv_id,
                                byteOffset=0,
                                componentType=COMPONENT_UNSIGNED_INT,
                                count=len(indices_array),
                                type="SCALAR",
                                min=[int(min(indices_array))],
                                max=[int(max(indices_array))])
            indices_accessor_id = len(gltf_obj.accessors)
            gltf_obj.accessors.append(accessor)


        # ========== Create the MeshPrimitive ==========
        mesh_primitive = MeshPrimitive(
            attributes=attributes_dict,
            mode=4,  # TRIANGLES
            indices=indices_accessor_id
        )

        # ========== Create the Mesh ==========
        new_mesh = Mesh(
            name=f"Mesh_{mesh_idx}",
            primitives=[mesh_primitive]
        )
        mesh_id = len(gltf_obj.meshes)
        gltf_obj.meshes.append(new_mesh)

        # ========== Create a Node referencing the Mesh ==========
        new_node = Node(
            name=f"Node_{mesh_idx}",
            mesh=mesh_id
        )
        node_id = len(gltf_obj.nodes)
        gltf_obj.nodes.append(new_node)
        scene_nodes.append(node_id)

    # Create a single Scene referencing all node IDs
    new_scene = Scene(
        name="Scene_0",
        nodes=scene_nodes
    )
    gltf_obj.scenes.append(new_scene)

    # Now create the Buffer with the final length
    bin_data = bytes(bin_data_list)
    total_buffer_length = len(bin_data)
    new_buffer = Buffer(
        name="buffer0",
        byteLength=total_buffer_length,
        uri=None
    )
    gltf_obj.buffers.append(new_buffer)

    return gltf_obj, bin_data


def _write_array_to_buffer_views_and_accessors(
    array: np.ndarray,
    component_type: int,
    gltf_obj: GLTF,
    bin_data_list: bytearray,
    current_offset_ref: list,
    semantic: str,
    is_index: bool
) -> int:
    """
    Appends array data to bin_data_list with 4-byte alignment.
    Creates BufferView (with correct target) and Accessor referencing it.
    Returns the new accessor ID.
    """
    get_offset = current_offset_ref[0]
    set_offset = current_offset_ref[1]

    # Ensure 2D
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    # 4-byte align
    offset_before = get_offset()
    #padding = (4 - (offset_before % 4)) % 4

    #if padding:
    #    bin_data_list.extend(b"\x00" * padding)
    #    offset_before += padding
    #    set_offset(offset_before)

    raw_bytes = np.array(array, dtype=COMPONENT_DTYPES[component_type]).tobytes()
    bin_data_list.extend(raw_bytes)

    offset_after = offset_before + len(raw_bytes)
    set_offset(offset_after)

    # Choose the target based on is_index
    buffer_view_target =  TARGET_ELEMENT_ARRAY_BUFFER if is_index else TARGET_ARRAY_BUFFER

    # Create BufferView
    buffer_view_id = len(gltf_obj.bufferViews)
    new_buffer_view = BufferView(
        name=f"BV_{semantic}_{buffer_view_id}",
        buffer=0,  # single buffer
        byteOffset=offset_before,
        byteLength=len(raw_bytes),
        target=buffer_view_target
    )
    gltf_obj.bufferViews.append(new_buffer_view)

    # Create Accessor
    accessor_id = len(gltf_obj.accessors)
    gltf_type = _guess_gltf_type_from_array(array)
    min_vals, max_vals = _compute_min_max(array)

    new_accessor = Accessor(
        name=f"A_{semantic}_{accessor_id}",
        bufferView=buffer_view_id,
        byteOffset=0,
        componentType=component_type,
        count=array.shape[0],
        type=gltf_type,
        min=min_vals,
        max=max_vals
    )
    gltf_obj.accessors.append(new_accessor)

    return accessor_id


###############################################################################
#                         High-level convenience function                     #
###############################################################################

def save_glb(meshes: List[MeshGeometry], output_path: str):
    """
    Convert a list of MeshGeometry into a GLTF object and write a .glb file.
    """
    gltf_obj, bin_data = create_gltf_from_mesh_geometries(meshes)
    glb_bytes = pack_all(gltf_obj, bin_data)
    with open(output_path, "wb") as f:
        f.write(glb_bytes)

@lru_cache(maxsize=None)
def byte_stride(type: str, componentType: int):
    return COMPONENT_DTYPES[componentType].itemsize * NUM_COMPONENTS[type]


def struct_fmt(data, dtype:  int):

    return f"{data.shape[0]}{COMPONENT_FMT[dtype]}"




def pack(data, dtype=5126):
    res = np.ascontiguousarray(np.array(data, dtype=COMPONENT_DTYPES[dtype]).flatten())

    return res.tobytes()


def add_buffer_view(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, offset=0, name=None,is_index=False):
    flatview = np.ascontiguousarray(np.array(arr, dtype=COMPONENT_DTYPES[dtype]).flatten())

    buffer.extend(flatview.tobytes())


    return {
        "buffer": 0,
        "byteLength": len(flatview) * int(COMPONENT_DTYPES[dtype].itemsize),
        "byteOffset": offset,
        "byteStride": byte_stride(gltf_type, dtype),
        "name": name,
        "target": TARGET_ELEMENT_ARRAY_BUFFER if is_index else TARGET_ARRAY_BUFFER
    }


def append_buffer_view(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, name=None, use_stride=False,is_index=False):
    flatview = np.ascontiguousarray(np.array(arr, dtype=COMPONENT_DTYPES[dtype]).flatten() )

    res = {
        "buffer": 0,
        "byteLength": len(flatview)* COMPONENT_DTYPES[dtype].itemsize,
        "byteOffset": len(buffer),
        "name": name,
        "target": TARGET_ELEMENT_ARRAY_BUFFER if is_index else TARGET_ARRAY_BUFFER
    }
    if all([use_stride, (gltf_type != 'SCALAR')]):
        res["byteStride"] = byte_stride(gltf_type, dtype)
    buffer.extend(flatview.tobytes())
    return res



###############################################################################
#                                Example Usage                                #
###############################################################################

if __name__ == "__main__":
    # Example: create 2 geometry objects (triangle + quad) and save them as a single .glb
    tri_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    tri_indices = np.array([0, 1, 2], dtype=np.uint32)

    mesh1 = MeshGeometry(
        vertices=tri_vertices,
        faces=tri_indices
    )

    quad_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    quad_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    quad_colors = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 1]
    ], dtype=np.float32)

    mesh2 = MeshGeometry(
        vertices=quad_vertices,
        colors=quad_colors,
        faces=quad_faces
    )

    save_glb([mesh1, mesh2], "scene-2w.glb")
    print("Wrote scene-2w.glb successfully. Validate it with your favorite glTF validator!")
