# legacy/: código y datos anteriores que el sistema ya no usa

> **El sistema de la práctica no depende de nada de esta carpeta.** La app (`app.py`), la API (`sistema.py`), la base de datos (`data/famosos.csv`) y la evaluación oficial (`evaluation/`) funcionan sin ella. Tampoco se incluye en la imagen Docker y sus tests no se ejecutan en la CI.
>
> Se conserva provisionalmente, a la espera de que el equipo decida si se mantiene o se elimina. Todo lo que se movió aquí sigue en el historial de git.

## Contenido

### Datos del flujo CSV anterior (integrante 2)

| Archivo | Qué es |
|---|---|
| `dataset.csv` (39 MB) | 262 vectores HOG (8100 valores cada uno) de 49 identidades, una fila por foto. Columnas: `name,hog_0…hog_8099`. Sin ORB ni imágenes. Se generó a partir de una versión anterior de las fotos |
| `svm.joblib` (3 MB) | SVM (`LinearSVC`) entrenado con `dataset.csv`. Es un *pickle* **sin firmar** |

### `integrante2/`: inscripción y entrenamiento sobre CSV

| Archivo | Antes en | Qué hace |
|---|---|---|
| `register.py` | `recognition/register.py` | Añade el vector HOG de una foto a `legacy/dataset.csv` |
| `train.py` | `training/train.py` | Entrena el SVM con `dataset.csv` y lo guarda en `legacy/svm.joblib` |
| `demo_svm.py` | `scripts/demo_svm.py` | Demo: predice la identidad de una foto con `legacy/svm.joblib` |
| `tests/test_register_train.py` | `tests/test_hog_svm_train.py` (parte) | Tests de `register.py` y `train.py`. Los tests de `hog.py` y `svm.py` siguen en `tests/test_hog_svm.py` |

### `integrante3/`: almacén ORB provisional y scripts

| Archivo | Antes en | Qué hace |
|---|---|---|
| `orb_store.py` | `recognition/orb_store.py` | Almacén ORB provisional: un `.npz` por persona y un CSV en base64, ambos sin firmar |
| `register_orb.py`, `demo_orb.py` | `scripts/` | Inscribe y reconoce con `orb_store` |
| `build_orb_dataset.py`, `demo_orb_dataset.py` | `scripts/` | Construye y prueba un CSV ORB a partir de una carpeta de fotos |
| `eval_orb.py` | `scripts/` | Evaluación y calibración de T2 solo con ORB |
| `tests/` | `tests/test_orb_dataset.py`, `tests/test_orb_pipeline.py` | Tests de lo anterior |

`recognition/orb.py` y `recognition/matching.py`, también del integrante 3, **siguen en el sistema**: se usan en la cascada.

### `evaluacion_previa/`: estudio con particiones por identidad

| Archivo | Antes en | Qué hace |
|---|---|---|
| `run.py`, `data.py` | `evaluation/` | Estudio con identidades desconocidas disjuntas, variabilidad entre particiones y escenario de una foto por persona |
| `results/` | `evaluation/results/` | Resultados de ese estudio (con 50 personas, antes de añadir a `henry_cavill`) |
| `tests/test_run.py` | `tests/test_evaluation.py` (parte) | Test del pipeline del estudio |

La metodología y las cifras se resumen en `docs/evaluacion.md` ("Estudio previo").

### `docs/`

| Archivo | Por qué está aquí |
|---|---|
| `informe_tecnico.md` | Describe la versión anterior (SQLite + AES-GCM, sin verificación ni los 9 métodos). La documentación vigente está en `docs/README.md` |

## Cómo ejecutar lo que hay aquí

Desde la raíz del repositorio:

```bash
python -m pytest legacy                                                  # sus 16 tests
python -m legacy.integrante2.demo_svm data/fotos/adele/<foto>.jpg
python -m legacy.integrante2.train
python -m legacy.evaluacion_previa.run --images data/fotos --repeat 20
python -m legacy.integrante3.eval_orb data/fotos
```

## Por qué no forma parte del sistema

- **Base de datos duplicada:** la práctica exige que la base de datos sea un CSV. Tener otro CSV con caras confundiría al grupo rival.
- **`svm.joblib` es un pickle sin firmar:** cargarlo si alguien lo ha sustituido permite ejecutar código. El SVM del sistema se regenera desde la base de datos y su meta va firmado con HMAC.
- **`orb_store` guarda templates sin integridad:** el sistema usa `database/csv_store.py`, con HMAC por fila.
- **Es reproducible:** todo se puede regenerar a partir de las fotos (`data/fotos/`).

## Opciones pendientes de decidir con el equipo

1. **Eliminar la carpeta** (`git rm -r legacy`). Seguiría disponible en el historial de git.
2. **Mantenerla** como material de los estudios previos.
