import struct
import numpy as np
COMPONENTS= [
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
        "numpy":  np.dtype('uint8'),
        "py": int,

    }, {
        "const": 5122,
        "name": "SHORT",
        "three": "Int16",
        "fmt": "h",
        "size":struct.calcsize("h"),
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

NUM_COMPONENTS={
    "SCALAR":1,
    "VEC2":2,
    "VEC3":3,
    "VEC4":4,
    "MAT2":4,
    "MAT3":9,
    "MAT4":16
}


import numpy as np
from gltf.definitions import GLTF, Accessor
from gltf.geometry import MeshGeometry

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
