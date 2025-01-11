# pygltf
A simple pure Python package to extract and write geometry from pygltf/glb files.

## Goals

`pygltf` tries to make it so that you don’t have to think about pygltf.

Just like you (hopefully) don’t think about .obj, .ply, etc., you simply load them, get the geometry, and can afford to have no idea how these formats are structured. This is exactly what we are trying to achieve with pygltf/glb: to provide a direct path from pygltf/glb to geometric abstractions and vice versa. You do not have to interact with the elements of pygltf representations; you can simply obtain and pass geometric abstractions.

This makes `pygltf` useful in applications related to the analysis, processing, and generation of geometry and associated geometric data.

Enough words; now we will show an example of usage that explains everything by itself:

First, let’s read a couple of glb files:

```python
import pygltf

with open('examples/scene-1-1.glb', 'rb') as fl1:
    geometry1 = pygltf.load(fl1)
with open('examples/scene-1-2.glb', 'rb') as fl1:
    geometry2 = pygltf.load(fl1)
```  
```python  
>>> geometry1[0]  
```  
A single Mesh object looks like:
```text
MeshGeometry(vertices=array([[ 7.705, ..., -2.81 ],
       ...,  
       [ 7.505, ..., -2.81 ]], shape=(8, 3), dtype=float32), colors=array([[0.942549  , ..., 0.93509805],
       ...,  
       [0.942549  , ..., 0.93509805]], shape=(8, 3), dtype=float32), normals=None, uv=None, faces=array([1, ..., 1], shape=(36,), dtype=uint32), extras=None)  
```  
```python  
>>> len(geometry1),len(geometry2)  
(207, 206)  
```  
Let’s get to the point. Suppose we want to merge data from two glb files into one:
```python
geometries = geometry1 + geometry2

with open('examples/merge-scene-1.glb','wb') as f:
    pygltf.dump(geometries, f)
```  
That is all that is needed.

Once again, a complete example of merging the geometric content from two glb files into one:

```python
import pygltf

with open('examples/scene-1-1.glb', 'rb') as fl1:
    geometry1 = pygltf.load(fl1)
with open('examples/scene-1-2.glb', 'rb') as fl1:
    geometry2 = pygltf.load(fl1)

geometries = geometry1 + geometry2

with open('examples/merge-scene-1.glb', 'wb') as f:
    pygltf.dump(geometries, f)
```  


## Not Goals

Despite the fact that some of the points below can quite possibly be implemented with `pygltf`, they are not the main goals of `pygltf`:
- Conversion from pygltf to glb and vice versa.
- Validation of pygltf/glb files.
- Modification of the pygltf/glb files and objects within them. Although this is quite possible, it is not the primary aim of this module.
- Conversion from pygltf to glb and vice versa.
- Support for absolutely all extensions.

