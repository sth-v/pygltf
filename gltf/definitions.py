from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
from dataclasses_json import dataclass_json,Undefined

#
# Type alias for glTF IDs (integer >= 0)
#
GLTFId = int


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

#
# Example usage
#
if __name__ == "__main__":
    # Create a minimal valid GLTF object with the required 'asset.version'
    example_gltf = GLTF(
        asset=Asset(version="2.0")
    )

    # Convert to JSON
    gltf_json = json.dumps(asdict(example_gltf), indent=2)
    print(gltf_json)
