# Sistema de reconocimiento facial (HOG + SVM, ORB y MediaPipe)

Práctica de Sistemas Biométricos: un sistema de reconocimiento facial que **identifica** (¿quién es?) y **verifica** (¿es quien dice ser?) a partir de fotografías. La base de datos es un CSV precargado con los famosos de la asignatura.

El sistema usa una cascada de dos técnicas clásicas:

1. MediaPipe detecta y alinea la cara.
2. **HOG + SVM** decide primero (umbral T1).
3. **ORB** actúa como segunda opinión (umbral T2).

> **Estado:** 51 personas inscritas (50 del docente y Henry Cavill, elegido por el equipo), con umbrales calibrados y 161 tests.
>
> **Precisión moderada:** con un FAR objetivo del 1 %, la verificación acepta al 55 % de los genuinos (FAR en test del 1,0 %) y la identificación reconoce a un 22–24 %. Ver [docs/evaluacion.md](docs/evaluacion.md).
>
> **Vulnerabilidades conocidas:** están en [docs/vulnerabilidades.md](docs/vulnerabilidades.md).

## Ejecución con Docker (recomendada)

Requisitos: Docker Desktop (Windows o macOS) o Docker Engine con el plugin Compose (Linux).

```bash
docker compose up -d --build
docker compose ps          # espera a que el estado sea «healthy» (~1 min la primera vez)
```

Después, abre <http://localhost:8501>. No hace falta configurar nada:

- En el primer arranque se crea la BD a partir de `data/famosos.csv`.
- Se genera una clave de integridad en un volumen aparte y se firman las filas.
- Se entrena el SVM.

| Acción | Comando |
|---|---|
| Parar, conservando los datos | `docker compose down` |
| Volver al estado inicial (BD de famosos, clave nueva) | `docker compose down -v` |
| Usar la API desde el contenedor | `docker compose exec facerec python -c "import sistema; print(sistema.identificar('/ruta/foto.jpg'))"` |

Notas:

- Los datos viven en dos volúmenes con nombre: `facerec-data` (BD y SVM) y `facerec-state` (clave HMAC y manifiesto). Se usan volúmenes con nombre, y no carpetas del host, para evitar problemas de permisos en Windows y Linux.
- El puerto solo se publica en `127.0.0.1`. **La interfaz no tiene autenticación.**

## La aplicación

| Pestaña | Qué hace |
|---|---|
| **Identificar** | Subes una foto y te dice qué famoso es, o "desconocido". Siempre muestra el candidato más parecido y su puntuación |
| **Verificar** | Eliges un famoso y subes una foto: te dice si es correcto |
| **Inscribir** | Nombre y fotos de una persona **nueva**. Los nombres ya inscritos se rechazan |

Cada pestaña tiene un interruptor para no usar el detector de caras (modo `_sd`). No hay opción de eliminar personas: la práctica no la pide.

## API de la práctica (`sistema.py`)

```python
import sistema

sistema.inscribir("persona_nueva", "foto.jpg")     # con detector; solo ids nuevos
sistema.verificar("adele", "foto.jpg")             # -> VerificationResult (bool(r) == aceptado)
sistema.identificar("foto.jpg")                    # -> RecognitionResult (.identity, .candidate)

sistema.inscribir_sd(id, foto)    # sin detector: la foto ya es la cara recortada
sistema.verificar_sd(id, foto)
sistema.identificar_sd(foto)

t = sistema.extraer_template("foto.jpg")           # template público (base64)
sistema.inscribir_t(id, t)        # sin detector ni extractor
sistema.verificar_t(id, t)
sistema.identificar_t(t)
```

- `foto` puede ser una ruta, bytes de un JPEG o PNG, o un `np.ndarray`. `id` es el nombre de la persona (por ejemplo `adele`); se puede consultar la lista con `python -m sistema identidades`.
- `inscribir*` solo admite personas **nuevas**: un id existente da error.

### Desde la línea de comandos (también dentro de Docker)

Cada método está disponible como orden, con la salida en JSON. En Docker, deja las fotos en la carpeta `entrada/` del repositorio, que se monta en el contenedor como `/entrada`:

```bash
docker compose exec facerec python -m sistema identificar /entrada/foto.jpg
docker compose exec facerec python -m sistema verificar adele /entrada/foto.jpg
docker compose exec facerec python -m sistema inscribir persona_nueva /entrada/foto.jpg
docker compose exec facerec python -m sistema identificar_sd /entrada/cara_recortada.png
docker compose exec facerec python -m sistema extraer_template /entrada/foto.jpg   # imprime el template
docker compose exec facerec python -m sistema verificar_t adele @/entrada/template.txt
docker compose exec facerec python -m sistema identidades
```

Sin Docker es lo mismo con `python -m sistema …`.
- El formato del template, los valores de retorno y los errores están en [docs/arquitectura.md](docs/arquitectura.md#api-de-la-práctica-sistemapy).

## Cómo funciona

```text
Fotografía ─► MediaPipe: exactamente 1 cara ─► alineado, gris, 128×128 ─► template (HOG + ORB)
Identificar: SVM margen ≥ T1 ─► identidad │ si no: ORB best ≥ T2 y margen ─► identidad │ si no: desconocido
Verificar:   SVM margen(reclamada) ≥ T1v ─► aceptado │ si no: ORB(reclamada) ≥ T2v ─► aceptado │ si no: rechazado
```

La base de datos (`data/famosos.csv` como semilla) es un CSV **en claro** con una fila por foto: `sample_id,identity,template,created_at,hmac`. Cada fila está firmada con HMAC, y un manifiesto firmado detecta filas borradas o restauradas (ver [docs/seguridad.md](docs/seguridad.md)).

## Estructura

| Ruta | Responsabilidad |
|---|---|
| `sistema.py` | Los 9 métodos de la práctica |
| `app.py` | Interfaz Streamlit (solo presentación) |
| `recognition/` | Detector, preprocesado, HOG, SVM, ORB, formato de template, motor de la cascada, servicio y ciclo de vida del SVM |
| `database/` | BD CSV con integridad HMAC (`csv_store.py`, `integrity.py`) |
| `data/famosos.csv`, `data/holdout.csv`, `data/thresholds.json` | BD semilla (174 fotos de 51 personas), fotos reservadas para calibrar y probar, y umbrales calibrados |
| `data/fotos/` | Fotos del docente. **No se versionan** |
| `scripts/` | `seed_db.py` (regenera la semilla desde `data/fotos/`), `download_models.py` (modelo de MediaPipe) y `demo_hog.py` (visualiza HOG) |
| `evaluation/` | Evaluación del sistema desplegado (`deployed.py`), ataques (`attacks.py`) y estudio del filtro HOG descartado (`ood_study.py`) |
| `legacy/` | Código y datos anteriores que el sistema ya no usa, a la espera de que el equipo decida ([legacy/README.md](legacy/README.md)) |
| `tests/` | Tests unitarios, de integración, extremo a extremo, de la interfaz, de la API y con fotos reales |
| `docs/` | Documentación: ver el [índice](docs/README.md) |

## Desarrollo local

```bash
uv venv --python 3.12 .venv            # MediaPipe no tiene ruedas para Python 3.14
uv pip install -p .venv/bin/python -r requirements-dev.txt
.venv/bin/python scripts/download_models.py          # MediaPipe, verificado por SHA-256
.venv/bin/streamlit run app.py                       # estado de ejecución en var/ (ignorado por git)
.venv/bin/ruff check . && .venv/bin/mypy && .venv/bin/python -m pytest -q
```

Regenerar la BD y evaluar (requiere `data/fotos/`):

```bash
.venv/bin/python -m scripts.seed_db              # data/famosos.csv + data/holdout.csv
.venv/bin/python -m evaluation.deployed --write-thresholds   # recalibra y adopta los umbrales
.venv/bin/python -m evaluation.attacks           # escenarios de ataque
rm -rf var/                                      # para que la app recargue la nueva semilla
```

## Documentación

| | |
|---|---|
| [Arquitectura y API](docs/arquitectura.md) | Módulos, contratos y formato del template |
| [Decisiones](docs/decisiones.md) | Qué se decidió y por qué |
| [Seguridad](docs/seguridad.md) | Qué protege el sistema y cómo |
| [**Vulnerabilidades y limitaciones**](docs/vulnerabilidades.md) | Lo que **no** resiste. Documento vivo para las pruebas del equipo |
| [Evaluación](docs/evaluacion.md) | Métricas, resultados y ataques |
| [Estado](docs/estado_implementacion.md) | Requisitos de la práctica y tareas pendientes |
