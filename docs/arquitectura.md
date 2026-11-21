# Arquitectura y contratos entre módulos

Este documento describe la implementación real. Si cambias una firma, actualiza este archivo en el mismo cambio.

## Dependencias entre capas

```text
app.py (Streamlit) · sistema.py (API de la práctica)
  └── recognition.service.FaceRecognitionService      fachada única
        ├── recognition.recognition.RecognitionEngine cascada T1 → T2 → desconocido
        │     ├── FaceProcessor      = recognition.detector.process_image (MediaPipe)
        │     ├── HogSvmClassifier   = SvmClassifier     (hog.py + svm.py)
        │     ├── OrbMatcher         = OrbDatabaseMatcher (orb.py + matching.py)
        │     └── TemplateSource     = database.csv_store.CsvIdentityStore
        ├── database.csv_store.CsvIdentityStore         CSV en claro + HMAC + manifiesto
        └── recognition.model_lifecycle.ModelLifecycle  reentreno y sincronía del SVM
recognition.bootstrap.build_service                   raíz de composición (único sitio que conecta todo)
```

Reglas:

- `recognition.types` y `recognition.config` no dependen de MediaPipe, de la BD ni de Streamlit.
- El motor no importa `database` ni `streamlit`: recibe sus dependencias por constructor.
- No hay dependencias circulares.

## Tipos compartidos (`recognition/types.py`)

| Tipo | Contenido |
|---|---|
| `RecognitionStatus` | `IDENTIFIED`, `UNCERTAIN`, `UNKNOWN` o `ERROR`. Los cuatro son distintos y no se mezclan. |
| `Stage` | Etapa que decidió: `HOG_SVM` (T1), `ORB` (T2) o `NONE`. |
| `ScoreKind` | Tipo de `score`: `SVM_DECISION` (margen de LinearSVC) u `ORB_INLIER_RATIO`. **Ninguno es una probabilidad.** |
| `ErrorCode` + `ProcessingError` | Error técnico tipado: código, mensaje para el usuario y etapa. |
| `HogPrediction` | `identity`, `score` y `runner_up_score`, más la propiedad `margin`. |
| `OrbOutcome` | `best_candidate`, `score`, `second_score`, `margin`, `accepted`, `n_identities`, `scores` e `insufficient_keypoints`. |
| `RecognitionResult` | `status`, `identity`, `stage`, `score`, `score_kind`, `hog`, `orb`, `error`, `thresholds_calibrated` y `notes`. Invariantes: `identity` existe ⇔ `IDENTIFIED`, y `error` existe ⇔ `ERROR`. |
| `Identity` | `identity_id`, `name`, `n_templates`, `created_at` y `updated_at`. No contiene datos biométricos. |
| `RegisteredFace` | Vector HOG (float32, 8100) y `OrbTemplate` de una foto: es el template público. |
| `VerificationResult` | `status` (`ACCEPTED`/`REJECTED`/`ERROR`), `claimed`, `stage`, `svm_score`, `orb_score`, `error` y `notes`. `bool(r) == r.accepted`. |

Los protocolos (inyectables y sustituibles por dobles en los tests) son:

- `FaceProcessor`: `(image) -> ProcessedFace`.
- `HogSvmClassifier`: `predict`, `scores`, `predict_hog` y `scores_hog`. Las dos últimas reciben el vector HOG ya extraído.
- `OrbMatcher`: `match(face, references)` y `match_template(orb, references)`, que reciben la cara o el template ORB.
- `TemplateSource.load_orb_templates() -> {nombre: [OrbTemplate]}`.

## Contrato de la cara normalizada

La cara que circula entre módulos es un `np.ndarray` gris `uint8` de 128×128, alineada por los ojos. La produce `recognition.detector.process_image`. **Registro y reconocimiento usan siempre la misma `PreprocessingConfig`**: si cambia, hay que regenerar los templates y reentrenar el SVM.

## Motor (`recognition/recognition.py`)

`RecognitionEngine.recognize(image)` y `recognize_face(face)` devuelven siempre un `RecognitionResult`. Solo las excepciones **no previstas**, que indican errores de programación, se propagan.

| Situación | Resultado |
|---|---|
| 0 caras / imagen inválida / cara demasiado pequeña | `ERROR` con `NO_FACE`, `INVALID_IMAGE` o `FACE_TOO_SMALL` |
| 2 o más caras válidas | `ERROR` con `MULTIPLE_FACES`. Se exige una sola persona por foto. |
| Falta el modelo de MediaPipe | `ERROR` con `DETECTOR_MODEL_MISSING` |
| Template corrupto o manipulado / clave incorrecta / fallo de BD | `ERROR` con `TEMPLATE_CORRUPT`, `KEY_ERROR` o `STORAGE_FAILURE` |
| BD vacía | `UNKNOWN` con una nota |
| SVM ausente, no verificado o **desactualizado** | Se omite T1 (lo indica una nota) y decide ORB |
| El SVM propone una identidad ya eliminada | Se ignora T1 (lo indica una nota) |
| Sin evidencia ORB (sin puntos, o puntuación 0 contra todas las identidades) | `UNCERTAIN` con una nota (filtro previo a T1) |
| `score_svm ≥ T1` | `IDENTIFIED` por `HOG_SVM` |
| ORB `best ≥ T2` y margen sobre el 2.º `≥ orb_min_margin` | `IDENTIFIED` por `ORB` |
| ORB `best ≥ T2` pero margen insuficiente | `UNCERTAIN` (coincidencia ambigua) |
| Resto | `UNKNOWN` |

`UNCERTAIN` nunca equivale a una identificación: la interfaz lo muestra como "no se puede confirmar".

**Filtro de evidencia ORB:** se aplica antes de T1, en identificación y verificación. Si la consulta no tiene puntos ORB suficientes, o su mejor puntuación ORB contra todas las identidades no supera `orb_min_evidence` (0,0), el resultado es `UNCERTAIN` en identificación y `REJECTED` en verificación, con una nota.

**Verificación** (`verify`, `verify_face` y `verify_template`): solo compara con la identidad reclamada. Si el margen del SVM para esa clase es ≥ T1v, acepta por `HOG_SVM`. Si no, acepta por `ORB` cuando la puntuación contra sus templates es ≥ T2v; en caso contrario, rechaza. Una identidad que no está inscrita se rechaza con una nota.

## API de la práctica (`sistema.py`)

`SistemaBiometrico` y funciones de módulo con los mismos nombres (usan una instancia compartida):

| Método | Entrada | Salida |
|---|---|---|
| `inscribir(id, foto)` / `inscribir_sd(id, foto)` / `inscribir_t(id, template)` | `foto`: ruta, bytes o `np.ndarray`. `template`: base64, bytes o `RegisteredFace` | `Identity`. Solo identidades **nuevas**: un `id` existente lanza `DuplicateIdentityError` |
| `verificar(id, foto)` / `verificar_sd` / `verificar_t` | ídem | `VerificationResult`: `accepted`, `status` (`accepted`/`rejected`/`error`), `svm_score`, `orb_score` y `notes`. `bool(r) == r.accepted` |
| `identificar(foto)` / `identificar_sd` / `identificar_t` | ídem | `RecognitionResult`: `identity` (`None` si no se acepta), `status`, `candidate` (el más parecido aunque no se acepte), `stage` y `score` |
| `extraer_template(foto, detector=True)` | foto | Template público en base64, para usarlo con los métodos `_t` |

Cómo trata cada variante la entrada:

- **Sin sufijo:** usa MediaPipe y exige exactamente una cara.
- **`_sd`:** toma la imagen como una cara ya recortada; solo la convierte a gris y la redimensiona a 128×128, sin alinear.
- **`_t`:** recibe el template ya extraído y valida su formato.

Errores de inscripción:

- `ValueError`: nombre inválido o template malformado.
- `PhotoRejectedError(index)`: sin cara, varias caras o sin textura ORB.
- `DuplicateIdentityError`: el id ya existe (solo se inscriben personas nuevas).
- `DuplicateFaceError`: la cara ya pertenece a otra identidad.

Los errores de identificación y verificación van **dentro** del resultado, con `status == error`.

El `id` se convierte a texto, así que también se aceptan ids numéricos. La práctica no pide eliminar, y `sistema.py` no lo ofrece.

**Línea de comandos:** `python -m sistema <método> [id] <foto|template>` imprime el resultado en JSON (`sistema.to_dict`). Los templates se pasan en base64 o como `@archivo`. Hay además dos órdenes de ayuda: `extraer_template` e `identidades`.

## Formato del template (`recognition/template.py`)

```text
b"FRT" | versión (uint8 = 1) | HOG float32[8100] little-endian | OrbTemplate.to_bytes()
OrbTemplate.to_bytes() = uint32 N | float32[N,5] (x, y, size, angle, response) | uint8[N,32]
```

En el CSV y en los métodos `_t` va codificado en base64. Al leerlo se rechaza si:

- La cabecera o la versión no son las esperadas, o el template está truncado.
- Algún valor HOG no es finito o cae fuera de [0, 1].
- Algún punto ORB cae fuera de la cara de 128×128.
- No tiene ningún punto ORB.

## Servicio (`recognition/service.py`)

| Método | Comportamiento |
|---|---|
| `analyze(foto)` / `identify(foto)` / `identify_without_detector(foto)` / `identify_template(t)` | Identificación 1:N. `analyze` devuelve además la cara detectada |
| `verify(id, foto)` / `verify_without_detector` / `verify_template` | Verificación 1:1 contra la identidad reclamada |
| `enroll(id, fotos, detector=True, allow_duplicate_face=False)` / `enroll_templates(id, templates)` | Todo o nada. Solo crea identidades nuevas (un id existente se rechaza), comprueba si la cara ya pertenece a otra identidad y reentrena |
| `delete(id)` | **Uso interno** (tests y mantenimiento). No está expuesto en la app ni en `sistema.py`. Borra la identidad, reentrena el SVM (o lo descarta si falla) y verifica que no queda nada |
| `model_status()` / `ensure_model_current()` / `retrain()` | Estado del SVM, recuperación tras interrupciones y reentreno |

## Persistencia (`database/csv_store.py`)

`CsvIdentityStore(db_path, key, manifest_path, seed_path)` es un CSV con las columnas `sample_id,identity,template,created_at,hmac`, una fila por muestra.

Al abrir el almacén:

- Si el CSV no existe, se copia la semilla.
- Si no hay manifiesto y ninguna fila está firmada, se firman todas las filas (primer arranque).

Operaciones:

- Consulta: `list_identities`, `get_identity` y `exists`.
- Altas y bajas: `add_identity`, `add_templates` y `delete_identity`.
- Lectura de templates: `load_templates`, `load_orb_templates` y `load_hog_training_set() -> (X, y, generation)`.

Las escrituras se serializan con un bloqueo entre procesos (`database/filelock.py`, `fcntl.flock`), porque la app y `python -m sistema` pueden escribir a la vez. El reentrenamiento del SVM usa otro bloqueo.

Cualquier modificación hecha fuera del sistema lanza `DatabaseIntegrityError`, que hereda de `TemplateCorruptError`. Las garantías están en [seguridad.md](seguridad.md) y sus límites en [vulnerabilidades.md](vulnerabilidades.md).

## Ciclo de vida del SVM (`recognition/model_lifecycle.py`)

`ModelLifecycle.retrain(store, discard_on_failure=False) -> TrainingReport`, con `ModelStatus` igual a `UP_TO_DATE`, `STALE`, `MISSING`, `INVALID` o `NOT_ENOUGH_IDENTITIES`. Las garantías están en la cabecera del módulo y en [decisiones.md](decisiones.md).

## Configuración (`recognition/config.py`)

`load_settings(env=None) -> Settings` aplica esta precedencia: valores por defecto → `data/thresholds.json` (T1, T2, T1v, T2v y `orb_min_evidence`) → variables `FACEREC_*`.

| Variable | Uso |
|---|---|
| `FACEREC_STATE_DIR` | Directorio de la clave HMAC y del manifiesto (separado de la BD) |
| `FACEREC_SEED_DB` | CSV semilla (por defecto `data/famosos.csv`) |
| `FACEREC_DATA_DIR`, `FACEREC_DB_PATH`, `FACEREC_MODELS_DIR` | Rutas (por defecto `var/` en local) |
| `FACEREC_T1`, `FACEREC_T2` | Overrides de umbrales; marcan el sistema como *no calibrado* |
| `FACEREC_LOG_LEVEL` | Nivel de logging |
| `FACE_LANDMARKER_MODEL` | Ruta del modelo de MediaPipe (la lee `detector.py`) |

Los parámetros de HOG, ORB y matching se definen en sus propios módulos y `config.py` los re-exporta, para no duplicarlos.

## Código heredado (`legacy/`)

El sistema no usa nada de `legacy/`, y sus tests no se ejecutan en la CI (`python -m pytest legacy` para lanzarlos a mano). El contenido se describe en [legacy/README.md](../legacy/README.md).
