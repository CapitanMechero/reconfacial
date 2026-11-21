# Módulo de detección y preprocesamiento facial (Integrante 1)

> 📜 **Documento histórico del integrante 1.** La descripción del detector (`recognition/detector.py`, `preprocessing.py`) sigue siendo válida. Los consejos de integración para el resto del equipo (SQLite, cifrado, interfaz) están superados. El estado vigente está en [docs/README.md](../README.md).

Este documento describe los archivos del módulo de detección, qué comprueban sus tests y cómo ejecutarlos.

## Qué hace el módulo

Dada una fotografía, devuelve **la única cara** de la imagen, alineada, recortada y normalizada (escala de grises, `uint8`, 128×128 por defecto), lista para HOG y ORB. Si la imagen no es válida, no tiene caras, la cara es demasiado pequeña o hay **más de una cara**, lanza una excepción controlada con un mensaje que la interfaz puede mostrar directamente.

Flujo interno:

```text
imagen -> load_image (validación) -> MediaPipe (caras + landmarks)
       -> filtro de tamaño mínimo -> exigir exactamente una cara
       -> alineamiento por los ojos + recorte + gris -> (CLAHE opcional) -> cara normalizada
```

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `recognition/__init__.py` | Archivo vacío que convierte `recognition/` en un paquete de Python. Debe existir. |
| `recognition/errors.py` | Excepciones del módulo. Todas heredan de `FaceProcessingError`, así que la interfaz puede capturarlas de una vez: `InvalidImageError`, `NoFaceDetectedError`, `FaceTooSmallError`, `MultipleFacesError` (con el atributo `count`). `ModelNotFoundError` es aparte: indica un problema de configuración, no de la foto. Existe como archivo propio para evitar imports circulares entre `detector.py` y `preprocessing.py`. |
| `recognition/preprocessing.py` | Carga y valida la imagen (ruta, bytes o array), calcula el alineamiento a partir de los ojos, recorta, redimensiona, pasa a gris y aplica CLAHE si se activa. No depende de MediaPipe. Contiene `PreprocessingConfig` con los parámetros (tamaño de salida, posición de los ojos, tamaño mínimo de cara, CLAHE). |
| `recognition/detector.py` | Envuelve MediaPipe Face Landmarker (`FaceDetector`) y contiene el pipeline completo: `process_image` (devuelve la cara con su bounding box y landmarks), `detect_and_align` (contrato del diseño: lista con una cara normalizada) y `draw_detection` (dibuja el recuadro para la interfaz). |
| `scripts/download_models.py` | Descarga el modelo `face_landmarker.task` a `models/`. Esa carpeta está en `.gitignore`, así que cada persona (y el Dockerfile) debe ejecutarlo. |
| `models/face_landmarker.task` | Modelo de MediaPipe. No se sube a git. |
| `tests/test_detector.py` | Tests del módulo (ver más abajo). |
| `conftest.py` | **Archivo vacío en la raíz del repo. No eliminar** (ver sección siguiente). |

## Por qué existe `conftest.py` y no hay que borrarlo

`conftest.py` está vacío a propósito. Su utilidad es que, al estar en la raíz del repositorio, pytest añade esa carpeta al path de Python al arrancar. Gracias a eso, los tests pueden hacer `from recognition.detector import ...` sin instalar el proyecto.

Si se borra, los tests fallarán con `ModuleNotFoundError: No module named 'recognition'` aunque el código esté bien. Se puede sustituir por un `pyproject.toml` con `pythonpath = ["."]`, pero mientras no exista, **`conftest.py` es imprescindible**.

## Qué se testea

`tests/test_detector.py` tiene 19 tests, que se dividen en cuatro grupos.

**Alineamiento y normalización** (no necesitan el modelo)

- `test_alignment_places_eyes_in_canonical_positions` (4 casos): con ojos nivelados, inclinados o en distintas posiciones y tamaños, la transformación deja los ojos exactamente en su posición objetivo.
- `test_eye_centers_are_ordered_left_to_right`: el ojo izquierdo siempre queda a la izquierda de la imagen, sea cual sea el orden de los landmarks.
- `test_align_and_crop_output_shape_and_dtype`: la salida es 128×128, gris y `uint8`.
- `test_clahe_keeps_shape_and_is_optional`: CLAHE está desactivado por defecto y, si se activa, mantiene forma y tipo.

**Carga y validación de imágenes** (no necesitan el modelo)

- Rechaza bytes corruptos, un archivo vacío, un archivo inexistente y una imagen demasiado pequeña (`InvalidImageError`).
- Acepta arrays en gris y con canal alfa (BGRA), y bytes PNG válidos.

**Política de una sola cara** (no necesitan el modelo: usan un detector falso)

- `test_single_face_is_processed`: con una cara válida, se devuelve procesada.
- `test_multiple_faces_raise_error_with_count`: con dos caras válidas se lanza `MultipleFacesError` con `count == 2`.
- `test_small_background_face_is_ignored`: una cara diminuta de fondo no cuenta como segunda cara.
- `test_only_tiny_faces_raise_too_small`: si todas las caras son demasiado pequeñas, se lanza `FaceTooSmallError`.

**Detección real con MediaPipe** (requieren el modelo)

- `test_blank_image_raises_no_face`: una imagen sin caras lanza `NoFaceDetectedError`.
- `test_corrupt_file_raises_controlled_error`: un archivo corrupto lanza `InvalidImageError` y no un error inesperado.

Si falta `models/face_landmarker.task`, estos dos últimos se saltan con el aviso `skipped` y el resto se ejecuta con normalidad (17 pasan, 2 se saltan).

**Pendiente de añadir:** tests con fotos reales en `tests/fixtures/` (una cara, varias caras, cara inclinada), marcados también con `needs_model`.

## Cómo ejecutar los tests

Desde la **raíz del repositorio**:

```powershell
pip install mediapipe numpy pytest
python scripts/download_models.py
python -m pytest tests/test_detector.py
```

Con `-v` se ve el resultado de cada test:

```powershell
python -m pytest tests/test_detector.py -v
```

Notas:

- Se usa `python -m pytest` y no `pytest` porque, en Windows, la carpeta `Scripts` de Python suele no estar en el PATH y el comando `pytest` no se reconoce.
- No ejecutar `python tests/test_detector.py`: eso lo lanza como un script normal, no encuentra el paquete `recognition` y tampoco ejecuta ningún test.
- Resultado esperado con el modelo descargado: `19 passed`. Sin él: `17 passed, 2 skipped`.

## Uso desde otros módulos

```python
from recognition.detector import process_image, detect_and_align, draw_detection
from recognition.errors import FaceProcessingError, MultipleFacesError

try:
    result = process_image(uploaded_bytes)        # ProcessedFace: face, bbox, landmarks
    face = result.face                             # array 128x128, gris, uint8
except MultipleFacesError as e:
    mostrar_error(f"Hay {e.count} caras; sube una foto con una sola persona.")
except FaceProcessingError as e:
    mostrar_error(str(e))
```

`detect_and_align(image)` mantiene el contrato de la sección 12 del diseño y devuelve `[cara]`, una lista de un solo elemento.

## Decisiones de diseño

- **Una sola cara por foto.** Si hay más de una cara válida, se rechaza la imagen. Una cara por debajo del tamaño mínimo (48 px) se ignora y no cuenta como segunda cara.
- **Bounding box calculado a partir de los landmarks**, porque Face Landmarker no lo proporciona directamente.
- **CLAHE desactivado por defecto.** Solo debería activarse si la evaluación demuestra que mejora el reconocimiento sin perjudicar a ORB.
- **Parámetros en `PreprocessingConfig`**: tamaño de salida (128 px), posición de los ojos y tamaño mínimo de cara, para poder ajustarlos tras las pruebas.

## Consejos de implementación para el resto del equipo

### Para todos

- **Contrato de entrada que recibís:** una cara como `np.ndarray` de forma `(128, 128)`, `dtype=uint8`, en escala de grises, ya alineada. No hace falta volver a recortar, alinear ni pasar a gris. No modifiquéis el array original; si necesitáis cambiarlo, trabajad sobre una copia.
- **Usad siempre la misma configuración de preprocesamiento al registrar y al reconocer.** Si una persona se registra con una `PreprocessingConfig` (por ejemplo, CLAHE desactivado) y se consulta con otra, los vectores HOG y los descriptores ORB no serán comparables. Si el equipo cambia algún parámetro, habrá que regenerar todos los templates y reentrenar el SVM.
- **Para probar vuestros módulos sin depender del detector**, podéis usar caras sintéticas (`np.random.randint(0, 255, (128, 128), dtype=np.uint8)`) o guardar unas cuantas caras ya procesadas en `tests/fixtures/`. Así avanzáis en paralelo sin esperar al modelo de MediaPipe.
- **Los errores de imagen ya vienen resueltos.** Vuestros módulos no tienen que validar imágenes corruptas, sin caras o con varias: eso lo hace el detector antes de llegar a vosotros.

### Integrante 2 (HOG y SVM)

- Con `skimage.feature.hog` sobre 128×128, el tamaño del vector depende de `orientations`, `pixels_per_cell` y `cells_per_block`. Fijadlos en `config.py` y no los cambiéis entre entrenamiento y predicción, porque el SVM exige vectores de longitud fija.
- No redimensionéis la cara dentro de `hog.py`; ya llega al tamaño correcto.
- La puntuación de decisión del SVM no es una probabilidad. Si se quiere una confianza interpretable, hay que calibrarla y validarla (sección 5.2 del diseño).
- Con una sola foto por persona el SVM tiene muy poca información. Si hacéis aumentos de datos (pequeñas rotaciones, cambios de brillo), marcad esos ejemplos como sintéticos y no los uséis para calibrar T1.
- Al registrar o eliminar a alguien hay que reentrenar o reconstruir el SVM. Conviene que `train.py` se pueda llamar como función desde el gestor de la base de datos, y no solo como script.

### Integrante 3 (ORB y matching)

- ORB tiene un `edgeThreshold` y un `patchSize` por defecto de 31 px. En una imagen de 128×128, la banda de 31 px de cada borde queda sin puntos clave, así que apenas se detectan puntos. Probad a reescalar la cara (por ejemplo, a 256×256) dentro de `orb.py` o a reducir esos parámetros, y comprobad cuántos puntos clave obtenéis.
- Fijad `nfeatures` en `config.py` y comprobad que `detectAndCompute` puede devolver `None` como descriptores cuando no hay puntos clave. Hay que tratar ese caso como «sin coincidencias» y no dejar que falle.
- Los descriptores son un array `uint8` de forma `(N, 32)`. Para que el integrante 4 pueda cifrarlos y guardarlos, definid una forma sencilla de serializarlos (por ejemplo, bytes más la forma) y acordadla antes de implementar.
- Los puntos clave (`cv2.KeyPoint`) no se pueden serializar directamente. Si necesitáis la posición para una comprobación geométrica, guardad sus coordenadas en un array aparte.

### Integrante 4 (base de datos, cifrado, interfaz y Docker)

- **Interfaz.** El archivo subido en Streamlit se puede pasar tal cual como bytes (`uploaded_file.getvalue()`) a `process_image`. Para dibujar el recuadro sin decodificar dos veces, usad:

  ```python
  bgr = load_image(uploaded_file.getvalue())
  face = process_image(bgr)              # también acepta el array
  preview = draw_detection(bgr, face)    # imagen BGR; convertir a RGB para st.image
  ```

- **Errores.** Capturad `FaceProcessingError` y mostrad `str(e)`; ya es un mensaje pensado para el usuario. `ModelNotFoundError` no hereda de ella a propósito, porque es un fallo de configuración, no de la foto: dejadlo aparte, para mostrar un error general.
- **Registro.** Guardad los templates a partir de `result.face`. Si se decide no conservar las fotos originales, no las escribáis en disco en ningún momento.
- **Docker.** El modelo no está en el repositorio, así que ejecutad `python scripts/download_models.py` durante la construcción de la imagen (necesita Internet) o montad `models/` como volumen. También se puede indicar otra ruta con la variable de entorno `FACE_LANDMARKER_MODEL`.
- **Dependencias del sistema.** En imágenes base ligeras de Python (`slim`), OpenCV suele fallar al importarse por falta de librerías como `libgl1` y `libglib2.0-0`. Si ocurre, instaladlas con `apt-get` en el Dockerfile. Comprobadlo al construir la imagen.
- **Versiones.** MediaPipe no soporta todas las versiones de Python. Fijad en `requirements.txt` versiones de `mediapipe`, `numpy` y `opencv` que funcionen juntas, y verificadlas dentro del contenedor, no solo en vuestro equipo (punto 9 de riesgos del diseño).
- **Rendimiento.** `get_default_detector()` carga el modelo una sola vez y es seguro entre hilos (usa un lock). No creéis un `FaceDetector` nuevo en cada petición de Streamlit.

### Integración final (`recognition.py`)

- La cadena completa es: `process_image` → HOG + SVM → (si no llega a T1) ORB → resultado. Si el detector lanza una excepción, la cascada ni siquiera empieza: dejad que la excepción suba hasta la interfaz.
- Documentad en `docs/interfaces.md` el formato de los resultados (identidad, puntuación, etapa que decidió) antes de que cada uno implemente su parte.
