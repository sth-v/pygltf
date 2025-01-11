from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray
import numpy as np

@dataclass
class PointsGeometry:
    vertices: NDArray[float]
    colors: NDArray[float] | None = None
    extras:dict[str,Any]|None=None

@dataclass
class LineGeometry:
    vertices: NDArray[float]
    colors: NDArray[float] | None = None
    extras:dict[str,Any]|None=None

@dataclass
class MeshGeometry:
    vertices:NDArray[float]
    colors:NDArray[float] | None=None
    normals: NDArray[float] | None = None
    uv:NDArray[float] | None=None
    faces:NDArray[int] | None=None
    extras:dict[str,Any]|None=None

    def to_tris(self):
        return self.vertices[self.faces.reshape((len(self.faces)//3,3))]