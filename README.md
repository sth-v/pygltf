# pygltf
A simple pure python package to extract and write geometry from pygltf/glb files.

## Goals

`pygltf` пытается сделать так, чтобы вы не думали о pygltf.

Tак-же как вы (надеемся) не думаете об .obj, .ply, и т.д. Вы просто читаете их, получаете геометрию и можете позволить не иметь никакого представления о том как устроены эти форматы. Именно этого мы пытаемся достичь с pygltf/glb
Предоставить прямой путь от pygltf/glb к геометрическим абстракциям и наоборот. Вы не обязаны взаимодействовать с элементами pygltf представлений, вы можете просто получать и передавать геометрические абстракции.

Это делает `pygltf` полезным в приложениях связанных с анализом, обработкой и генерацией геометрии и связанных с ней данных геометрией. 

Довольно слов, сейчас мы покажем пример использования который сам все объяснит:

Для начала прочтем несколько glb файлов:

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
Single Mesh object look like:
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
Перейдем к делу. Допустим мы хотим объединить данные из двух glb файлов в один:
```python
geometries = geometry1 + geometry2

geometries = geometry1 + geometry2
with open('examples/merge-scene-1.glb','wb') as f:
    pygltf.dump(geometries, f)

```
Это все что нужно.

Еще раз, полный пример объединения геометрического содержимого из двух glb файлов в один:

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

Несмотря на то что некоторые пункты ниже вполне возможно реализовать с помощью `pygltf` , это не является основной целью `pygltf`
- Конвертация из pygltf в glb и наоборот.
- Валидация pygltf/glb файлов
- Модификация самих pygltf/glb файлов и объектов внутри них. Несмотря на то что это вполне возможно, это не является основной целью данного модуля.
- Конвертация из pygltf в glb и наоборот.
- Поддержка абсолютно всех расширений.


