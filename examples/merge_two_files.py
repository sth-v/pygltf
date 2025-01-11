import pygltf

with open('examples/scene-1-1.glb', 'rb') as fl1:
    geometry1 = pygltf.load(fl1)
with open('examples/scene-1-2.glb', 'rb') as fl1:
    geometry2 = pygltf.load(fl1)

geometries = geometry1 + geometry2

with open('examples/merge-scene-1.glb', 'wb') as f:
    pygltf.dump(geometries, f)
