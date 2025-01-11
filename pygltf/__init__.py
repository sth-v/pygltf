from __future__ import annotations

import json
import struct
from collections import namedtuple
from functools import lru_cache
from io import FileIO, BytesIO
from pathlib import Path
from typing import Any,Dict, List, Optional,BinaryIO,Generator, NamedTuple

from dataclasses import dataclass,field, asdict
from dataclasses_json import dataclass_json, Undefined

import numpy as np
from numpy.typing import NDArray




COMPONENTS = [
    {
        "const": 5120,
        "name": "BYTE",
        "fmt": "b",
        "size": struct.calcsize("b"),
        "three": "Int8",
        "numpy": np.dtype('int8'),
        "py": int
    },
    {
        "const": 5121,
        "name": "UNSIGNED_BYTE",
        "fmt": "B",
        "size": struct.calcsize("B"),
        "three": "UInt8",
        "numpy": np.dtype('uint8'),
        "py": int,

    }, {
        "const": 5122,
        "name": "SHORT",
        "three": "Int16",
        "fmt": "h",
        "size": struct.calcsize("h"),
        "numpy": np.dtype('int16'),
        "py": int
    }, {
        "const": 5123,
        "name": "UNSIGNED_SHORT",
        "three": "UInt16",
        "fmt": "H",
        "size": struct.calcsize("H"),
        "numpy": np.dtype('uint16'),
        "py": int
    }, {
        "const": 5125,
        "name": "UNSIGNED_INT",
        "three": "UInt32",
        "fmt": "I",
        "size": struct.calcsize("I"),
        "numpy": np.dtype('uint32'),
        "py": int
    }, {
        "const": 5126,
        "name": "FLOAT",
        "three": "Float32",
        "fmt": "f",
        "size": struct.calcsize("f"),
        "numpy": np.dtype('float32'),
        "py": float
    },
]
COMPONENT_FMT = {
    5120: 'b',
    5121: 'B',
    5122: 'h',
    5123: 'H',
    5125: 'I',
    5126: 'f'
}
COMPONENT_DTYPES = {
    5120: np.dtype('int8'),
    5121: np.dtype('uint8'),
    5122: np.dtype('int16'),
    5123: np.dtype('uint16'),
    5125: np.dtype('uint32'),
    5126: np.dtype('float32')
}

NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16
}

# Maps componentType -> np.dtype plus number of bytes
# This is a minimal mapping for demonstration.
# glTF uses WebGL enumerations for data types:
#   5120 = BYTE
#   5121 = UNSIGNED_BYTE
#   5122 = SHORT
#   5123 = UNSIGNED_SHORT
#   5125 = UNSIGNED_INT
#   5126 = FLOAT

# For each accessor "type", how many components?
#   SCALAR -> 1, VEC2 -> 2, VEC3 -> 3, VEC4 -> 4
#   MAT2 -> 4, MAT3 -> 9, MAT4 -> 16
_TYPE_NUMCOMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16
}

#
# Type alias for glTF IDs (integer >= 0)
#
GLTFId = int



@dataclass
class PointsGeometry:
    vertices: NDArray[float]
    colors: NDArray[float] | None = None
    extras: dict[str, Any] | None = None


@dataclass
class LineGeometry:
    vertices: NDArray[float]
    colors: NDArray[float] | None = None
    extras: dict[str, Any] | None = None


@dataclass
class MeshGeometry:
    vertices: NDArray[float]
    colors: NDArray[float] | None = None
    normals: NDArray[float] | None = None
    uv: NDArray[float] | None = None
    faces: NDArray[int] | None = None
    extras: dict[str, Any] | None = None

    def to_tris(self):
        return self.vertices[self.faces.reshape((len(self.faces) // 3, 3))]


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class GLTFProperty:
    """
    glTF Property

    Schema Description:
      A base class for most glTF property objects. It carries `extensions` and `extras`.
    """
    extensions: Dict[str, Any] = field(default_factory=dict)
    """
    JSON object with extension-specific objects (extension schema).
    Maps to an open-ended dictionary of data.
    """

    extras: Dict[str, Any] = field(default_factory=dict)
    """
    Application-specific data (`extras.schema.json`).
    Maps to an open-ended dictionary of data.
    """


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class GLTFChildOfRootProperty(GLTFProperty):
    """
    glTF Child of Root Property

    Schema Description:
      A base class for all objects that are children of the glTF root
      (e.g., Accessor, Texture, Node, etc.), carrying a user-defined `name`.
    """
    name: Optional[str] = None
    """
    The user-defined name of this object (string). Not necessarily unique.
    """


#
#  ACCESSOR SPARSE INDICES
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AccessorSparseIndices(GLTFProperty):
    """
    Animation Sampler Sparse Indices

    Schema Description:
      An object pointing to a buffer view containing the indices of deviating accessor values.
      The number of indices is equal to `accessor.sparse.count`.
      Indices MUST strictly increase.
    """
    bufferView: GLTFId
    """
    The index of the buffer view with sparse indices.
    The referenced buffer view MUST NOT have its `target` or `byteStride` properties defined.
    """

    byteOffset: int = 0
    """
    The offset relative to the start of the buffer view in bytes (default 0).
    Must be >= 0.
    """

    componentType: int
    """
    The indices data type.
    One of 5121 (UNSIGNED_BYTE), 5123 (UNSIGNED_SHORT), or 5125 (UNSIGNED_INT).
    """


#
#  ACCESSOR SPARSE VALUES
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AccessorSparseValues(GLTFProperty):
    """
    Accessor Sparse Values

    Schema Description:
      An object pointing to a buffer view containing the deviating accessor values.
      The number of elements is equal to `accessor.sparse.count` times number of components.
      The elements have the same component type as the base accessor.
      The elements are tightly packed.
    """
    bufferView: GLTFId
    """
    The index of the bufferView with sparse values.
    The referenced buffer view MUST NOT have its `target` or `byteStride` properties defined.
    """

    byteOffset: int = 0
    """
    The offset relative to the start of the bufferView in bytes (default 0).
    Must be >= 0.
    """


#
#  ACCESSOR SPARSE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AccessorSparse(GLTFProperty):
    """
    Accessor Sparse

    Schema Description:
      Sparse storage of accessor values that deviate from their initialization value.
    """
    count: int
    """
    Number of deviating accessor values stored in the sparse array (must be >= 1).
    """

    indices: AccessorSparseIndices
    """
    An object pointing to a buffer view containing the indices of deviating accessor values.
    The number of indices is equal to `count`. Indices MUST strictly increase.
    """

    values: AccessorSparseValues
    """
    An object pointing to a buffer view containing the deviating accessor values.
    """


#
#  ACCESSOR
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Accessor(GLTFChildOfRootProperty):
    """
    Accessor

    Schema Description:
      A typed view into a buffer view that contains raw binary data.
      An accessor provides typed data for attributes or indices.
    """
    componentType: int
    """
    The datatype of the accessor's components.
    One of:
      5120 (BYTE), 5121 (UNSIGNED_BYTE), 5122 (SHORT), 5123 (UNSIGNED_SHORT),
      5125 (UNSIGNED_INT), 5126 (FLOAT).
    """

    count: int
    """
    The number of elements referenced by this accessor (must be >= 1).
    """

    type: str
    """
    Specifies if the accessor's elements are scalars, vectors, or matrices.
    One of: 'SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4'.
    """

    bufferView: Optional[GLTFId] = None
    """
    The index of the bufferView. Not defined if the accessor should be initialized with zeros.
    """

    byteOffset: int = 0
    """
    The offset relative to the start of the bufferView in bytes (default 0). Must be multiple
    of the size of the component data type. Must not be defined when bufferView is undefined.
    """

    normalized: bool = False
    """
    Specifies whether integer data values are normalized (true) to [0, 1] or [-1, 1] before usage.
    Must not be true for FLOAT or UNSIGNED_INT component types.
    """

    max: Optional[List[float]] = None
    """
    Maximum value of each component in this accessor (length depends on `type`).
    """

    min: Optional[List[float]] = None
    """
    Minimum value of each component in this accessor (length depends on `type`).
    """

    sparse: Optional[AccessorSparse] = None
    """
    Sparse storage of elements that deviate from their initialization value.
    """


#
#  TEXTURE INFO (base class used for normal/occlusion/etc.)
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class TextureInfo(GLTFProperty):
    """
    Texture Info

    Schema Description:
      Reference to a texture.
    """
    index: GLTFId
    """
    The index of the texture.
    """

    texCoord: int = 0
    """
    The set index of texture's TEXCOORD attribute used for texture coordinate mapping (default 0).
    Must be >= 0.
    """


#
#  MATERIAL NORMAL TEXTURE INFO
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MaterialNormalTextureInfo(TextureInfo):
    """
    Material Normal Texture Info

    Schema Description:
      The tangent space normal texture. Inherits fields from TextureInfo.
    """
    scale: float = 1.0
    """
    The scalar parameter applied to each normal vector of the normal texture (default 1.0).
    """


#
#  MATERIAL OCCLUSION TEXTURE INFO
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MaterialOcclusionTextureInfo(TextureInfo):
    """
    Material Occlusion Texture Info

    Schema Description:
      The occlusion texture. Inherits fields from TextureInfo.
    """
    strength: float = 1.0
    """
    A scalar multiplier controlling the amount of occlusion applied (default 1.0).
    Must be between [0.0, 1.0].
    """


#
#  MATERIAL PBR METALLIC ROUGHNESS
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MaterialPBRMetallicRoughness(GLTFProperty):
    """
    Material PBR Metallic Roughness

    Schema Description:
      A set of parameter values that define the metallic-roughness material model from PBR.
    """
    baseColorFactor: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    """
    The factors for the base color of the material (default [1.0, 1.0, 1.0, 1.0]).
    Must have length = 4, each in [0.0, 1.0].
    """

    baseColorTexture: Optional[TextureInfo] = None
    """
    The base color texture (RGB in sRGB, A is linear alpha if present).
    """

    metallicFactor: float = 1.0
    """
    The factor for the metalness of the material (default 1.0).
    Must be in [0.0, 1.0].
    """

    roughnessFactor: float = 1.0
    """
    The factor for the roughness of the material (default 1.0).
    Must be in [0.0, 1.0].
    """

    metallicRoughnessTexture: Optional[TextureInfo] = None
    """
    The metallic-roughness texture.
    B channel -> metalness, G channel -> roughness, linear transfer function.
    """


#
#  MATERIAL
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Material(GLTFChildOfRootProperty):
    """
    Material

    Schema Description:
      The material appearance of a primitive.
    """
    pbrMetallicRoughness: Optional[MaterialPBRMetallicRoughness] = None
    """
    A set of parameter values for the metallic-roughness material model.
    """

    normalTexture: Optional[MaterialNormalTextureInfo] = None
    """
    The tangent space normal texture.
    """

    occlusionTexture: Optional[MaterialOcclusionTextureInfo] = None
    """
    The occlusion texture (R channel).
    """

    emissiveTexture: Optional[TextureInfo] = None
    """
    The emissive texture (RGB in sRGB).
    """

    emissiveFactor: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """
    The factors for the emissive color of the material (default [0.0, 0.0, 0.0]).
    Each must be in [0.0, 1.0].
    """

    alphaMode: str = "OPAQUE"
    """
    The alpha rendering mode of the material. One of OPAQUE, MASK, BLEND. Default OPAQUE.
    """

    alphaCutoff: float = 0.5
    """
    The alpha cutoff value of the material (default 0.5). Used only in MASK mode.
    """

    doubleSided: bool = False
    """
    Whether the material is double sided (default false).
    """


#
#  SAMPLER
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Sampler(GLTFChildOfRootProperty):
    """
    Sampler

    Schema Description:
      Texture sampler properties for filtering and wrapping modes.
    """
    magFilter: Optional[int] = None
    """
    Magnification filter. One of 9728 (NEAREST), 9729 (LINEAR), or other integer.
    """

    minFilter: Optional[int] = None
    """
    Minification filter. One of 9728 (NEAREST), 9729 (LINEAR), 9984..9987, or other integer.
    """

    wrapS: int = 10497
    """
    S (U) wrapping mode (default 10497, REPEAT).
    One of 33071 (CLAMP_TO_EDGE), 33648 (MIRRORED_REPEAT), 10497 (REPEAT).
    """

    wrapT: int = 10497
    """
    T (V) wrapping mode (default 10497, REPEAT).
    One of 33071 (CLAMP_TO_EDGE), 33648 (MIRRORED_REPEAT), 10497 (REPEAT).
    """


#
#  MESH PRIMITIVE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class MeshPrimitive(GLTFProperty):
    """
    Mesh Primitive

    Schema Description:
      Geometry to be rendered with the given material.
    """
    attributes: Dict[str, GLTFId]
    """
    A plain JSON object, where each key corresponds to a mesh attribute semantic
    and each value is the index of the accessor containing the attribute's data.
    Must have at least 1 property.
    """

    mode: int = 4
    """
    The topology type of primitives to render (default 4, TRIANGLES).
    Possible values: 0..6.
    """

    indices: Optional[GLTFId] = None
    """
    The index of the accessor that contains the vertex indices (SCALAR, unsigned int type).
    If not defined, non-indexed geometry is used.
    """

    material: Optional[GLTFId] = None
    """
    The index of the material to apply to this primitive.
    """

    targets: Optional[List[Dict[str, GLTFId]]] = None
    """
    An array of morph targets, each a dict from attribute semantic to an accessor index.
    Must have at least 1 item if defined.
    """


#
#  MESH
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Mesh(GLTFChildOfRootProperty):
    """
    Mesh

    Schema Description:
      A set of primitives to be rendered. Its global transform is defined by a node that references it.
    """
    primitives: List[MeshPrimitive]
    """
    An array of primitives, each defining geometry to be rendered (minItems=1).
    """

    weights: Optional[List[float]] = None
    """
    Array of weights to be applied to the morph targets (minItems=1).
    """


#
#  CAMERA ORTHOGRAPHIC
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class CameraOrthographic(GLTFProperty):
    """
    Camera Orthographic

    Schema Description:
      An orthographic camera containing properties to create an orthographic projection matrix.
    """
    xmag: float
    """
    The floating-point horizontal magnification of the view. Must not be zero.
    """

    ymag: float
    """
    The floating-point vertical magnification of the view. Must not be zero.
    """

    zfar: float
    """
    The floating-point distance to the far clipping plane. Must not be zero, and > znear.
    """

    znear: float
    """
    The floating-point distance to the near clipping plane. Must be >= 0.0.
    """


#
#  CAMERA PERSPECTIVE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class CameraPerspective(GLTFProperty):
    """
    Camera Perspective

    Schema Description:
      A perspective camera containing properties to create a perspective projection matrix.
    """
    yfov: float
    """
    The floating-point vertical field of view in radians (must be > 0 and < pi).
    """

    znear: float
    """
    The floating-point distance to the near clipping plane (must be > 0).
    """

    aspectRatio: Optional[float] = None
    """
    The floating-point aspect ratio of the field of view. Must be > 0 if defined.
    """

    zfar: Optional[float] = None
    """
    The floating-point distance to the far clipping plane (must be > znear if defined).
    If undefined, an infinite projection matrix is used.
    """


#
#  CAMERA
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Camera(GLTFChildOfRootProperty):
    """
    Camera

    Schema Description:
      A camera's projection. A node may reference a camera to apply a transform to place the camera in the scene.
    """
    type: str
    """
    Specifies if the camera uses a perspective or orthographic projection. Required.
    One of 'perspective', 'orthographic'.
    """

    orthographic: Optional[CameraOrthographic] = None
    """
    An orthographic camera containing properties to create an orthographic projection matrix.
    Must not be defined when `perspective` is defined.
    """

    perspective: Optional[CameraPerspective] = None
    """
    A perspective camera containing properties to create a perspective projection matrix.
    Must not be defined when `orthographic` is defined.
    """


#
#  NODE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Node(GLTFChildOfRootProperty):
    """
    Node

    Schema Description:
      A node in the node hierarchy. When the node contains `skin`, all `mesh.primitives`
      must contain `JOINTS_0` and `WEIGHTS_0` attributes. A node may have either a `matrix`
      or any combination of `translation`/`rotation`/`scale` properties (TRS).
    """
    camera: Optional[GLTFId] = None
    """
    The index of the camera referenced by this node.
    """

    children: Optional[List[GLTFId]] = None
    """
    The indices of this node's children (unique items, minItems=1 if defined).
    """

    skin: Optional[GLTFId] = None
    """
    The index of the skin referenced by this node. Must also define `mesh` if defined.
    """

    matrix: Optional[List[float]] = None
    """
    A floating-point 4x4 transformation matrix in column-major order (length=16).
    Default: identity [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1].
    """

    mesh: Optional[GLTFId] = None
    """
    The index of the mesh in this node.
    """

    rotation: Optional[List[float]] = None
    """
    The node's unit quaternion rotation in the order (x, y, z, w). Default [0,0,0,1].
    """

    scale: Optional[List[float]] = None
    """
    The node's non-uniform scale along the x, y, and z axes. Default [1,1,1].
    """

    translation: Optional[List[float]] = None
    """
    The node's translation along the x, y, and z axes. Default [0,0,0].
    """

    weights: Optional[List[float]] = None
    """
    The weights of the instantiated morph target. Must have at least 1 item if defined.
    Must also define `mesh`.
    """


#
#  SKIN
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Skin(GLTFChildOfRootProperty):
    """
    Skin

    Schema Description:
      Joints and matrices defining a skin.
    """
    joints: List[GLTFId]
    """
    Indices of skeleton nodes, used as joints in this skin (minItems=1).
    """

    inverseBindMatrices: Optional[GLTFId] = None
    """
    The index of the accessor containing the floating-point 4x4 inverse-bind matrices.
    """

    skeleton: Optional[GLTFId] = None
    """
    The index of the node used as a skeleton root.
    """


#
#  BUFFERVIEW
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class BufferView(GLTFChildOfRootProperty):
    """
    Buffer View

    Schema Description:
      A view into a buffer generally representing a subset of the buffer.
    """
    buffer: GLTFId
    """
    The index of the buffer.
    """

    byteLength: int
    """
    The length of the bufferView in bytes (must be >= 1).
    """

    byteOffset: int = 0
    """
    The offset into the buffer in bytes (default 0).
    """

    byteStride: Optional[int] = None
    """
    The stride in bytes (>=4, <=252, multiple of 4). If not defined, data is tightly packed.
    """

    target: Optional[int] = None
    """
    The intended GPU buffer type. One of 34962 (ARRAY_BUFFER), 34963 (ELEMENT_ARRAY_BUFFER).
    """


#
#  BUFFER
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Buffer(GLTFChildOfRootProperty):
    """
    Buffer

    Schema Description:
      A buffer points to binary geometry, animation, or skins.
    """
    byteLength: int
    """
    The length of the buffer in bytes (must be >= 1).
    """

    uri: Optional[str] = None
    """
    The URI (or IRI) of the buffer. May be a 'data:'-URI or a relative/absolute path.
    """


#
#  IMAGE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Image(GLTFChildOfRootProperty):
    """
    Image

    Schema Description:
      Image data used to create a texture. Image may be referenced by a URI or a bufferView index.
    """
    uri: Optional[str] = None
    """
    The URI (or IRI) of the image. Must not be defined if `bufferView` is defined.
    """

    mimeType: Optional[str] = None
    """
    The image's media type. Must be defined when `bufferView` is defined.
    """

    bufferView: Optional[GLTFId] = None
    """
    The index of the bufferView that contains the image. Must not be defined if `uri` is defined.
    """


#
#  TEXTURE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Texture(GLTFChildOfRootProperty):
    """
    Texture

    Schema Description:
      A texture and its sampler.
    """
    sampler: Optional[GLTFId] = None
    """
    The index of the sampler used by this texture.
    """

    source: Optional[GLTFId] = None
    """
    The index of the image used by this texture.
    """


#
#  SCENE
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Scene(GLTFChildOfRootProperty):
    """
    Scene

    Schema Description:
      The root nodes of a scene.
    """
    nodes: Optional[List[GLTFId]] = None
    """
    The indices of each root node (unique items, minItems=1 if defined).
    """


#
#  ANIMATION CHANNEL TARGET
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AnimationChannelTarget(GLTFProperty):
    """
    Animation Channel Target

    Schema Description:
      The descriptor of the animated property.
    """
    path: str
    """
    The name of the node's TRS property to animate or 'weights'.
    One of 'translation', 'rotation', 'scale', 'weights'.
    """

    node: Optional[GLTFId] = None
    """
    The index of the node to animate. When undefined, the animated object may be defined by an extension.
    """


#
#  ANIMATION CHANNEL
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AnimationChannel(GLTFProperty):
    """
    Animation Channel

    Schema Description:
      An animation channel combines an animation sampler with a target property being animated.
    """
    sampler: GLTFId
    """
    The index of a sampler in this animation used to compute the value for the target.
    """

    target: AnimationChannelTarget
    """
    The descriptor of the animated property.
    """


#
#  ANIMATION SAMPLER
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class AnimationSampler(GLTFProperty):
    """
    Animation Sampler

    Schema Description:
      An animation sampler combines timestamps with a sequence of output values
      and defines an interpolation algorithm.
    """
    input: GLTFId
    """
    The index of an accessor containing keyframe timestamps (scalar float).
    Times must be strictly increasing.
    """

    output: GLTFId
    """
    The index of an accessor containing keyframe output values.
    """

    interpolation: str = "LINEAR"
    """
    Interpolation algorithm (default 'LINEAR').
    One of 'LINEAR', 'STEP', 'CUBICSPLINE'.
    """


#
#  ANIMATION
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Animation(GLTFChildOfRootProperty):
    """
    Animation

    Schema Description:
      A keyframe animation.
    """
    channels: List[AnimationChannel]
    """
    An array of animation channels (minItems=1).
    """

    samplers: List[AnimationSampler]
    """
    An array of animation samplers (minItems=1).
    """


#
#  ASSET
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class Asset(GLTFProperty):
    """
    Asset

    Schema Description:
      Metadata about the glTF asset.
    """
    version: str
    """
    The glTF version in the form '<major>.<minor>' that this asset targets. Required.
    """

    copyright: Optional[str] = None
    """
    A copyright message suitable for display to credit the content creator (optional).
    """

    generator: Optional[str] = None
    """
    Tool that generated this glTF model (useful for debugging).
    """

    minVersion: Optional[str] = None
    """
    The minimum glTF version in the form '<major>.<minor>' that this asset targets.
    Must not be greater than `version`.
    """


#
#  GLTF (ROOT)
#
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(kw_only=True)
class GLTF(GLTFProperty):
    """
    glTF

    Schema Description:
      The root object for a glTF asset.
    """
    asset: Asset
    """
    Metadata about the glTF asset (required).
    """

    extensionsUsed: Optional[List[str]] = None
    """
    Names of glTF extensions used in this asset (unique items, minItems=1 if defined).
    """

    extensionsRequired: Optional[List[str]] = None
    """
    Names of glTF extensions required to properly load this asset (unique items, minItems=1 if defined).
    """

    accessors: Optional[List[Accessor]] = None
    """
    An array of accessors (minItems=1 if defined).
    """

    animations: Optional[List[Animation]] = None
    """
    An array of keyframe animations (minItems=1 if defined).
    """

    buffers: Optional[List[Buffer]] = None
    """
    An array of buffers (minItems=1 if defined).
    """

    bufferViews: Optional[List[BufferView]] = None
    """
    An array of bufferViews (minItems=1 if defined).
    """

    cameras: Optional[List[Camera]] = None
    """
    An array of cameras (minItems=1 if defined).
    """

    images: Optional[List[Image]] = None
    """
    An array of images (minItems=1 if defined).
    """

    materials: Optional[List[Material]] = None
    """
    An array of materials (minItems=1 if defined).
    """

    meshes: Optional[List[Mesh]] = None
    """
    An array of meshes (minItems=1 if defined).
    """

    nodes: Optional[List[Node]] = None
    """
    An array of nodes (minItems=1 if defined).
    """

    samplers: Optional[List[Sampler]] = None
    """
    An array of samplers (minItems=1 if defined).
    """

    scene: Optional[GLTFId] = None
    """
    The index of the default scene. Must not be defined if `scenes` is undefined.
    """

    scenes: Optional[List[Scene]] = None
    """
    An array of scenes (minItems=1 if defined).
    """

    skins: Optional[List[Skin]] = None
    """
    An array of skins (minItems=1 if defined).
    """

    textures: Optional[List[Texture]] = None
    """
    An array of textures (minItems=1 if defined).
    """



class BinaryGltfChunkType(NamedTuple):
    chunk_type: int
    ascii: str
    description: str

    def tobytes(self) -> bytes:
        return struct.pack('I', self.chunk_type)

    def __int__(self):
        return self.chunk_type

    def __str__(self):
        return self.ascii

    def __bytes__(self):
        return self.tobytes()

    def __repr__(self):
        return f'GLTF ChunkType ({self.chunk_type}, "{self.ascii}", "{self.description})"'


gltf_chunk_types = {
    1313821514: BinaryGltfChunkType(
        1313821514,
        'JSON',
        "Structured JSON content"
    ),
    5130562: BinaryGltfChunkType(
        5130562,
        'BIN',
        "Binary buffer"
    )
}
JSON = gltf_chunk_types[1313821514]
BIN = gltf_chunk_types[5130562]
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
    chunk_type_obj = gltf_chunk_types.get(chunk_type, chunk_type)
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
    chunk_type = int(chunk_info[1])

    return UnpackChunkResult(
        BinaryGltfChunk(
            chunk_info[0],
            gltf_chunk_types.get(chunk_type, chunk_type),
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


def parse_json_chunk_to_gltf(json_dict: dict) -> GLTF:
    """
    Convert the top-level JSON dictionary into a GLTF dataclass tree.
    """
    return GLTF.from_dict(json_dict)


def decode_accessor(
        accessor: Accessor | AccessorSparse,
        gltf: GLTF,
        bin_data: bytes
) -> np.ndarray:
    """
    Reads the accessor data from the binary chunk
    and returns a NumPy array of the typed data.
    """
    if accessor.bufferView is None:
        # This usually implies the accessor is "all zeros" or a sparse storage
        # with no bufferView. For demonstration, just return a zero array:
        n_components = NUM_COMPONENTS[accessor.type]
        return np.zeros((accessor.count, n_components), dtype=COMPONENT_DTYPES[accessor.componentType])

    # 1) Find the relevant bufferView:
    buffer_view = gltf.bufferViews[accessor.bufferView]
    byte_offset_in_buf = buffer_view.byteOffset + accessor.byteOffset
    # The length of the portion in BIN for this bufferView is buffer_view.byteLength

    # 2) Identify the dtype and number of components
    dtype = COMPONENT_DTYPES[accessor.componentType]
    n_components = NUM_COMPONENTS[accessor.type]

    # 3) Extract the raw data slice from bin_data
    #    We assume single, unified BIN chunk.
    #    (If multiple buffers exist, you'd differentiate by `buffer_view.buffer` index.)
    raw_data_start = byte_offset_in_buf
    # total number of elements
    element_count = accessor.count * n_components

    # If `byteStride` is None, data is tightly packed. If it's not None, we have to carefully interpret.
    stride = buffer_view.byteStride
    if stride is None:
        # Tightly packed
        array_data = bin_data[raw_data_start: raw_data_start + element_count * np.dtype(dtype).itemsize]
        arr = np.frombuffer(array_data, dtype=dtype)
        arr = arr.reshape(accessor.count, n_components)
    else:
        # We have an interleaved array. We'll do a slice with stepping.
        # E.g. the stride is in bytes, so we must carefully extract each element.
        arr = np.zeros((accessor.count, n_components), dtype=dtype)
        item_size = np.dtype(dtype).itemsize * n_components
        for i in range(accessor.count):
            offset = raw_data_start + i * stride
            chunk = bin_data[offset: offset + item_size]
            arr[i] = np.frombuffer(chunk, dtype=dtype)
        # Now arr has shape (count, n_components)

    # 4) If `accessor.sparse` is present, apply sparse updates
    #    (This is a demonstration – real code would handle that carefully.)
    if accessor.sparse is not None:
        # Read the indices
        idx_arr = decode_accessor(accessor.sparse.indices, gltf, bin_data).ravel()
        # Read the updated values
        val_arr = decode_accessor(accessor.sparse.values, gltf, bin_data)
        # Overwrite the relevant rows
        arr[idx_arr] = val_arr

    # 5) If `normalized`, you'd apply normalization (for integral types).
    if accessor.normalized:
        # glTF spec says integer data are mapped to [0,1] or [-1,1].
        # For demonstration, we skip the full detail.
        # Typically you'd check dtype and do the proper scaling.
        pass

    return arr


def extract_mesh_geometries(
        gltf_obj: GLTF,
        bin_data: bytes
) -> list[MeshGeometry]:
    """
    Create a MeshGeometry object for each mesh in the glTF.
    Each mesh may contain multiple primitives, but for demonstration,
    we flatten each primitive into a separate MeshGeometry.
    """
    output_meshes = []

    if not gltf_obj.meshes:
        return output_meshes  # No meshes present

    for mesh in gltf_obj.meshes:
        for prim in mesh.primitives:
            # 1) Gather attributes
            pos_arr = None
            normal_arr = None
            uv_arr = None
            color_arr = None
            faces_arr = None

            # Indices (faces)
            if prim.indices is not None:
                index_accessor = gltf_obj.accessors[prim.indices]
                faces_arr = decode_accessor(index_accessor, gltf_obj, bin_data)
                # For consistency, shape into (N, )
                faces_arr = faces_arr.reshape(-1)

            # Attributes
            for semantic, accessor_index in prim.attributes.items():
                accessor_obj = gltf_obj.accessors[accessor_index]
                arr_data = decode_accessor(accessor_obj, gltf_obj, bin_data)
                if semantic == "POSITION":
                    pos_arr = arr_data
                elif semantic == "NORMAL":
                    normal_arr = arr_data
                elif semantic.startswith("TEXCOORD_"):
                    # We'll only store the first set (TEXCOORD_0) in 'uv' for demo
                    uv_arr = arr_data
                elif semantic.startswith("COLOR_"):
                    color_arr = arr_data
                # Could handle JOINTS_0, WEIGHTS_0, TANGENT, etc. here if needed

            # 2) Create our pygltf.geometry.MeshGeometry
            mesh_geom = MeshGeometry(
                vertices=pos_arr if pos_arr is not None else np.array([], dtype=np.float32),
                colors=color_arr,
                normals=normal_arr,
                uv=uv_arr,
                faces=faces_arr,
                extras=None
            )
            output_meshes.append(mesh_geom)

    return output_meshes


def load_glb_to_mesh_geometries(glb_data: bytes) -> list[MeshGeometry]:
    """
    Reads a .glb file, extracts the JSON chunk into GLTF classes,
    and returns a list of MeshGeometry objects from the BIN chunk.
    """
    data = glb_data

    chunks = list(unpack_all(data))

    # The first chunk is the header, second chunk is the JSON, subsequent chunk(s) are BIN(s)
    header = chunks[0]
    json_chunk = chunks[1]
    bin_chunks = chunks[2:]  # might be only one chunk, or more

    gltf_dict = json_chunk.chunkData  # This is the dictionary we got from JSON
    gltf_obj = parse_json_chunk_to_gltf(gltf_dict)

    # For simplicity, we assume there's only one BIN chunk, typical in glb:
    # If multiple BIN chunks exist, you'd have to handle them more carefully
    # (and match them to the correct 'buffer' in gltf_obj.buffers).
    if len(bin_chunks) == 1:
        bin_data = bin_chunks[0].chunkData if bin_chunks else b""

        # Now extract the geometry

        meshes = extract_mesh_geometries(gltf_obj, bin_data)
        return meshes
    else:
        print("multiple chunks")
        raise ValueError(f"Only one BIN buffer is allowed. {len(bin_chunks)} exists.")


###############################################################################
#                      pygltf packing (pack_header, pack_json, etc.)            #
###############################################################################

JSON_CHUNK_TYPE = 0x4E4F534A  # 'JSON'
BIN_CHUNK_TYPE = 0x004E4942  # 'BIN'
# WebGL / glTF constants
COMPONENT_FLOAT = 5126  # gl.FLOAT
COMPONENT_UNSIGNED_INT = 5125  # gl.UNSIGNED_INT
TARGET_ARRAY_BUFFER = 34962  # ARRAY_BUFFER
TARGET_ELEMENT_ARRAY_BUFFER = 34963  # ELEMENT_ARRAY_BUFFER


def _pad_to_4bytes(data: bytes, pad_byte=b"\x00") -> bytes:
    padding = (4 - (len(data) % 4)) % 4
    return data + (pad_byte * padding)


def pack_header_chunk(total_length: int, version: int = 2) -> bytes:
    """
    GLB Header: magic (4 bytes), version (4), total_length (4).
    magic=0x46546C67 = 'glTF' in ASCII
    """
    magic = 0x46546C67
    return struct.pack("<III", magic, version, total_length)


def pack_json_chunk(json_obj: dict) -> bytes:
    json_str = json.dumps(json_obj, separators=(',', ':'))
    json_bytes = json_str.encode("utf-8")
    json_bytes = _pad_to_4bytes(json_bytes, b"\x20")
    chunk_header = struct.pack("<II", len(json_bytes), JSON_CHUNK_TYPE)
    return chunk_header + json_bytes


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

            if len(gltf_obj.bufferViews) == 0:
                gltf_obj.bufferViews.append(BufferView(
                    **add_buffer_view(geometry.vertices, bin_data_list, "VEC3", COMPONENT_FLOAT, current_offset)))

            else:
                gltf_obj.bufferViews.append(BufferView(
                    **append_buffer_view(geometry.vertices, bin_data_list, "VEC3", COMPONENT_FLOAT, None, True, False)))

            bv_id = len(gltf_obj.bufferViews) - 1

            accessor = Accessor(bufferView=bv_id,
                                byteOffset=0,
                                componentType=COMPONENT_FLOAT,
                                count=len(geometry.vertices),
                                type="VEC3",
                                min=geometry.vertices.min(axis=0).tolist(),
                                max=geometry.vertices.max(axis=0).tolist()
                                )
            accessor_id = len(gltf_obj.accessors)
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
                    **append_buffer_view(geometry.uv, bin_data_list, "VEC2", COMPONENT_FLOAT, None, True, False), ))
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
                    **append_buffer_view(indices_array, bin_data_list, "SCALAR", COMPONENT_UNSIGNED_INT, None, False,
                                         True)))
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


@lru_cache(maxsize=None)
def byte_stride(type: str, componentType: int):
    return COMPONENT_DTYPES[componentType].itemsize * NUM_COMPONENTS[type]


def struct_fmt(data, dtype: int):
    return f"{data.shape[0]}{COMPONENT_FMT[dtype]}"


def pack(data, dtype=5126):
    res = np.ascontiguousarray(np.array(data, dtype=COMPONENT_DTYPES[dtype]).flatten())

    return res.tobytes()


def add_buffer_view(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, offset=0, name=None, is_index=False):
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


def append_buffer_view(arr, buffer: bytearray, gltf_type="VEC3", dtype=5126, name=None, use_stride=False,
                       is_index=False):
    flatview = np.ascontiguousarray(np.array(arr, dtype=COMPONENT_DTYPES[dtype]).flatten())

    res = {
        "buffer": 0,
        "byteLength": len(flatview) * COMPONENT_DTYPES[dtype].itemsize,
        "byteOffset": len(buffer),
        "name": name,
        "target": TARGET_ELEMENT_ARRAY_BUFFER if is_index else TARGET_ARRAY_BUFFER
    }
    if all([use_stride, (gltf_type != 'SCALAR')]):
        res["byteStride"] = byte_stride(gltf_type, dtype)
    buffer.extend(flatview.tobytes())
    return res


###############################################################################
#                         High-level convenience function                     #
###############################################################################

def write_glb(meshes: List[MeshGeometry], output_path: str | Path) -> None:
    """
    Convert a list of MeshGeometry into a GLTF object and write a .glb file.
    """
    gltf_obj, bin_data = create_gltf_from_mesh_geometries(meshes)
    glb_bytes = pack_all(gltf_obj, bin_data)
    with open(output_path, "wb") as f:
        f.write(glb_bytes)


def read_glb(path: str | Path) -> list[MeshGeometry]:
    with open(path, 'rb') as f:
        result = load(f)
    return result


def loads(glb_data: bytes):
    return load_glb_to_mesh_geometries(glb_data)


def load(fl: BytesIO | FileIO | BinaryIO):
    data: bytes = fl.read()
    return loads(data)


def dumps(geometry: list[MeshGeometry]) -> bytes:
    """
        Convert a list of MeshGeometry into a GLB bytes.
        """
    gltf_obj, bin_data = create_gltf_from_mesh_geometries(geometry)
    glb_bytes = pack_all(gltf_obj, bin_data)
    return glb_bytes


def dump(geometry: list[MeshGeometry], fl: BytesIO | FileIO | BinaryIO) -> None:
    """
    Convert a list of MeshGeometry into a GLB bytes.
    """

    glb_bytes = dumps(geometry)
    fl.write(glb_bytes)


