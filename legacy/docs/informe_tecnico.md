# Informe técnico de integración

> ⚠️ **Superado.** Este informe describe la versión anterior a la adaptación a la práctica (SQLite + AES-GCM, sin verificación ni los 9 métodos). El estado vigente está en [estado_implementacion.md](estado_implementacion.md). Se reescribirá al cerrar la adaptación.

Fecha: 21-11-2026. Los detalles están en los documentos enlazados. Aquí se resume el estado y lo que **no** está terminado.

## 1. Arquitectura final

```text
app.py ─► FaceRecognitionService ─┬─► RecognitionEngine (T1 HOG+SVM → T2 ORB → desconocido)
                                  ├─► IdentityStore (SQLite + AES-256-GCM)
                                  └─► ModelLifecycle (SVM derivado de la BD, publicación atómica)
bootstrap.build_service = único punto que conecta las dependencias reales
```

Las capas y sus contratos se describen en [interfaces.md](interfaces.md). El motor no depende de Streamlit ni de la BD, y no hay ciclos de importación.

## 2. Componentes

**Nuevos:**

- `recognition/`: `types.py`, `config.py`, `recognition.py`, `service.py`, `model_lifecycle.py` y `bootstrap.py`.
- `database/`: `manager.py` y `encryption.py`.
- `app.py`.
- `evaluation/`: `splits.py`, `metrics.py`, `cascade.py`, `data.py` y `run.py`.
- `scripts/rotate_key.py`.
- Infraestructura: `Dockerfile`, `docker-compose.yml`, `.env.example`, `.dockerignore`, `.github/workflows/ci.yml` y `pyproject.toml`.
- Documentación: `interfaces.md`, `decisiones.md`, `seguridad.md`, `evaluacion.md` y este informe.
- 9 archivos de tests nuevos.

**Cambios mínimos en código existente:**

- `errors.py`: excepciones nuevas.
- `svm.py`: `decision_scores`. `predict_identity` mantiene su comportamiento.
- `training/train.py`: ahora es invocable como función con rutas explícitas, y la CLI conserva sus valores por defecto.
- `download_models.py`: verificación SHA-256.
- `requirements.txt`: versiones fijadas.
- Correcciones automáticas de lint en otros archivos: salto de línea final e imports sin usar.

## 3. Decisiones de diseño

Las 11 decisiones están en [decisiones.md](decisiones.md). Las principales:

- SQLite.
- Se exige una cara por foto.
- No se conservan las fotos.
- Duplicados detectados por nombre y por cara (ORB).
- El SVM se reentrena por completo y se publica de forma atómica, con hash y generación.
- Un SVM desactualizado nunca decide.
- Resultado `UNCERTAIN` separado de "desconocido".

## 4. Cascada

La tabla completa de casos está en [interfaces.md](interfaces.md#motor-recognitionrecognitionpy). Los errores técnicos siempre se devuelven como `ERROR` y nunca como "desconocido". Las puntuaciones no se presentan como probabilidades.

## 5. Datos y cifrado

Ver [seguridad.md](seguridad.md):

- AES-256-GCM con nonce aleatorio de 96 bits y AAD ligado a la identidad.
- Verificación de clave al abrir la BD.
- Rotación transaccional.
- Clave cargada desde un *secret* o variable de entorno, sin valor por defecto.

## 6. Identidades y entrenamiento

- Alta y baja son transaccionales.
- El reentrenamiento falla de forma segura: tras un alta se conserva el modelo anterior marcado `STALE`; tras una baja se descarta.
- Al arrancar se recupera el estado tras interrupciones.
- Si un registro tiene una sola muestra por identidad, la aplicación avisa.

## 7. Tests

`pytest`: **135 pasan**, tanto en el entorno de desarrollo como en un entorno limpio instalado solo desde `requirements-dev.txt`. `ruff` y `mypy` pasan sin errores.

| Área | Qué se cubre |
|---|---|
| Unitarios | Tipos y configuración, HOG, SVM, `train.py`, `register.py`, cifrado (manipulación, clave errónea, AAD, formatos de clave), BD (atomicidad, duplicados, borrado, rotación, esquema) y métricas y particiones |
| Integración | Cascada con SVM y ORB reales; equivalencia entre la regla offline y el motor sobre 300 casos |
| Extremo a extremo | Registro → persistencia → reconocimiento con una foto independiente → rechazo de un desconocido → eliminación → deja de reconocerse → sus templates ya no existen. También fallos de reentrenamiento, modelo manipulado y publicación interrumpida |
| Interfaz | `streamlit.testing`: renderizado, error claro sin clave y arranque real con `/_stcore/health` |

Además, `tests/test_real_photos.py` ejecuta el pipeline real (MediaPipe, BD cifrada, SVM y ORB) con fotos de `data/fotos`: registro, consulta con fotos independientes, foto con dos caras y eliminación. Se omite si las fotos no existen, por ejemplo en la CI, porque no se versionan. La subida de archivos desde el navegador **no se ha probado de forma automática**: `streamlit.testing` no lo permite.

## 8. Resultados experimentales

Con 274 fotos reales de 50 identidades (263 válidas), la partición 42 y 20 particiones adicionales (detalle en [evaluacion.md](evaluacion.md)):

- **Rank-1 en conjunto cerrado:** HOG+SVM 46 % y ORB 31 %. Con una foto por persona, 23 % y 10 %.
- **FAR objetivo ≤ 1 % en calibración:** IR media de test del 17 % (HOG), del 13 % (ORB) y del 22 % (cascada). El FAR medio de test está entre el 1,4 % y el 2,6 %, con picos de hasta el 11,3 %.
- Las puntuaciones ORB de genuinos e impostores casi se solapan (medianas de 0,017 y 0,015).

## 9. Umbrales

La calibración se ha ejecutado, pero **sus umbrales no se han adoptado**. Con unas 54 impostoras de calibración, un FAR de 0/54 no garantiza el 1 %, y en test el FAR llegó al 11,3 %. La interfaz sigue mostrando T1 = 0,0 y T2 = 0,05 como provisionales. No se ha ajustado ningún umbral mirando test.

## 10–11. Instalación y Docker

Ver [README](../README.md). Las versiones están fijadas para Python 3.12. El contenedor:

- Se ejecuta sin root, con sistema de archivos de solo lectura y sin capacidades.
- Recibe la clave como *secret*.
- Guarda los datos en un volumen con nombre.
- Tiene healthcheck.
- Publica el puerto solo en 127.0.0.1.

## 12. Limitaciones conocidas

- HOG+SVM discrimina poco con estos datos, y con una sola foto por persona empeora mucho.
- Sin detección de vida y sin autenticación en la interfaz.
- Los nombres y el SVM no están cifrados (está justificado en [seguridad.md](seguridad.md)).
- El borrado no alcanza a las copias de seguridad anteriores.
- Los umbrales calibrados dependen de la población registrada.

## 13. Pendiente

| Pendiente | Bloqueo | Cómo cerrarlo |
|---|---|---|
| Probar Docker desde un entorno limpio | El usuario no tiene permiso sobre `/var/run/docker.sock` (no está en el grupo `docker`) | `docker compose up --build`, o dejar que lo haga el job `docker` de la CI |
| Comprobar que la CI pasa en GitHub | No se ha hecho push | Abrir un PR |
| Umbrales que cumplan el FAR objetivo | Dataset pequeño (≈50 impostoras de calibración) y algoritmos poco discriminativos | Más identidades no registradas para calibrar y más fotos por persona |
| Probar la subida de fotos desde el navegador | No es automatizable con `streamlit.testing` | Prueba manual en <http://localhost:8501> |
| Decidir si `data/dataset.csv` y `models/svm.joblib` siguen versionados | Requiere decisión del equipo (ver abajo) | — |
| Autenticación, plazo de conservación y EIPD | Fuera del alcance actual | [seguridad.md](seguridad.md) |

**Sobre `data/dataset.csv` y `models/svm.joblib`.** El README anterior decía que no se versionaban, pero lo están. Son vectores HOG de personas reales y un modelo derivado de ellos, así que son datos biométricos. También son lo único que permite reproducir la evaluación de HOG. No se han eliminado del repositorio:

- Dejar de versionarlos (`git rm --cached`) borraría esos archivos a los compañeros al hacer pull.
- Eliminarlos del historial requiere reescribirlo.

La aplicación ya no depende de ellos: su SVM se guarda en `svm_runtime.joblib`.
