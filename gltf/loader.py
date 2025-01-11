import numpy as np

from gltf.definitions import *
from gltf.geometry import MeshGeometry
from gltf.unpack import unpack_all
from gltf.components import NUM_COMPONENTS,COMPONENT_DTYPES

from pathlib import Path
from io import FileIO

def parse_json_chunk_to_gltf(json_dict: dict) -> GLTF:
    """
    Convert the top-level JSON dictionary into a GLTF dataclass tree.
    """
    return GLTF.from_dict(json_dict)


def decode_accessor(
        accessor: Accessor|AccessorSparse,
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

            # 2) Create our gltf.geometry.MeshGeometry
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



def load_glb_to_mesh_geometries(glb_file: str|FileIO) -> list[MeshGeometry]:
    """
    Reads a .glb file, extracts the JSON chunk into GLTF classes,
    and returns a list of MeshGeometry objects from the BIN chunk.
    """
    if isinstance(glb_file,(str,Path)):
        with Path(glb_file).open( 'rb') as f:

            data = f.read()
    else:
        data=glb_file.read()

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
    if len(bin_chunks)==1:
        bin_data = bin_chunks[0].chunkData if bin_chunks else b""

        # Now extract the geometry

        meshes = extract_mesh_geometries(gltf_obj, bin_data)
        return meshes
    else:
        print("multiple chunks")
        raise ValueError(f"Only one BIN buffer is allowed. {len(bin_chunks)} exists.")

