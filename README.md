# gltf
A simple pure python module to extract and write geometry from gltf/glb files.

## Goals

`gltf` пытается сделать так, чтобы вы не думали о gltf.

Tак-же как вы (надеемся) не думаете об .obj, .ply, и т.д. Вы просто читаете их, получаете геометрию и можете позволить не иметь никакого представления о том как устроены эти форматы. Именно этого мы пытаемся достичь с gltf/glb
Предоставить прямой путь от gltf/glb к геометрическим абстракциям и наоборот. Вы не обязаны взаимодействовать с элементами gltf представлений, вы можете просто получать и передавать геометрические абстракции.

Это делает `gltf` полезным в приложениях связанных с анализом, обработкой и генерацией геометрии и связанных с ней данных геометрией. Ниже мы приведем несколько примеров использования:

### Examples
1. Объедините несколько glb в один
```python
import gltf

```

### П


## Not Goals
- Валидация gltf/glb файлов
- Модификация самих gltf/glb файлов и объектов внутри них. Несмотря на то что это вполне возможно, это не является основной целью данного модуля.
- Конвертация из gltf в glb и наоборот.
- Поддержка абсолютно всех расширений. 