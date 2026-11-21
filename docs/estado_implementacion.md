# Estado de la implementación

**Última actualización:** 21-11-2026

Documento de seguimiento del equipo: actualízalo al cerrar cada tarea. Las vulnerabilidades no van aquí, sino en [vulnerabilidades.md](vulnerabilidades.md).

## Requisitos de la práctica

| Requisito | Estado | Dónde |
|---|---|---|
| Python | ✅ | Python 3.12 |
| Repositorio fácil de ejecutar, con Docker | ✅ Probado por el equipo | `docker compose up -d --build` |
| Base de datos en `.csv` | ✅ | `data/famosos.csv` (semilla) → `db.csv` en uso, con HMAC |
| 51 personas inscritas (50 del docente + 1 del equipo) | ✅ | La persona 51 es `henry_cavill` |
| Evaluación propia, incluidos ataques (saltarse partes, leer la BD, escribir la BD) | ✅ | [evaluacion.md](evaluacion.md) y `evaluation/` |
| Evaluación del sistema rival | ⬜ | Pendiente: aún no lo tenemos |
| Exposición de resultados y conclusiones | ⬜ | Pendiente |

### Los 9 métodos

Todos están en `sistema.py` como funciones de módulo y como métodos de `SistemaBiometrico`, y también como órdenes de `python -m sistema`, incluido dentro de Docker. Las firmas las fija `tests/test_sistema.py`, y el funcionamiento con fotos reales lo comprueba `tests/test_real_photos.py::test_nine_methods_with_real_photos`.

| Método (especificación) | Implementación | Comprobado |
|---|---|---|
| `inscribir(id, foto)` | Detector + extractor. Solo ids nuevos | ✅ Sintético y real |
| `verificar(id, foto)` | Cascada 1:1 contra `id` (T1v/T2v) | ✅ Sintético y real |
| `identificar(foto)` | Cascada 1:N (T1/T2), con rechazo y candidato | ✅ Sintético y real |
| `inscribir_sd(id, foto)` | Sin detector: la foto es la cara recortada | ✅ Sintético y real |
| `verificar_sd(id, foto)` | Ídem | ✅ Sintético y real |
| `identificar_sd(foto)` | Ídem | ✅ Sintético y real |
| `inscribir_t(id, template)` | Sin detector ni extractor; template en base64 (`extraer_template`) | ✅ Sintético y real |
| `verificar_t(id, template)` | Ídem. La especificación dice "foto", pero la firma es `template` | ✅ Sintético y real |
| `identificar_t(template)` | Ídem | ✅ Sintético y real |

La práctica no pide eliminar, y no se ofrece ni en la app ni en la API.

## Estado técnico

- **Tests:** 151 en `tests/` (se ejecutan en la CI) más 16 en `legacy/` (`python -m pytest legacy`). `ruff` y `mypy` sin errores.
- **Umbrales calibrados y adoptados** (`data/thresholds.json`): T1 = −0,271, T2 = 0,272, T1v = −0,608 y T2v = 0,029.
- **Resultados en test:**
  - Verificación: FAR del 1,0 % y FRR del 45 %.
  - Identificación: IR del 23,5 % y FAR de 1/51.
- **Docker:** funciona. El build comprueba que el detector arranca. La CI (`.github/workflows/ci.yml`) aún no se ha ejecutado en GitHub.

## Pendiente

| # | Tarea | Responsable |
|---|---|---|
| 1 | Decidir qué hacer con `legacy/`: eliminarlo o conservarlo | Equipo |
| 2 | Pruebas sobre el sistema desplegado; las novedades van a [vulnerabilidades.md](vulnerabilidades.md) | Equipo |
| 3 | Commit, PR y comprobar que la CI pasa en GitHub | — |
| 4 | Evaluar el sistema rival | Equipo |
| 5 | Preparar la exposición | Equipo |

## Cómo retomar el trabajo

- **Entorno:** `.venv` con Python 3.12 (`uv venv --python 3.12 .venv` y `uv pip install -p .venv/bin/python -r requirements-dev.txt`).
- **Fotos:** `data/fotos/<id>/`. No se versionan y hacen falta para regenerar la semilla y para evaluar.
- **Regenerar todo:**
  ```bash
  python -m scripts.seed_db
  python -m evaluation.deployed --write-thresholds
  python -m evaluation.attacks
  rm -rf var/
  ```
- **Estado local de ejecución:** `var/` (BD en uso, clave, manifiesto y SVM). Borrarlo equivale a un primer arranque.

## Historial

| Fecha | Hito |
|---|---|
| 21-11-2026 | Integración inicial: tipos, motor de la cascada, BD SQLite cifrada, app y Docker |
| 21-11-2026 | Adaptación a la práctica: BD CSV con HMAC, los 9 métodos, verificación, 51 personas, umbrales calibrados y evaluación de ataques |
| 21-11-2026 | Filtro de evidencia ORB contra entradas que no son caras. Código antiguo movido a `legacy/`. Documentación unificada y [vulnerabilidades.md](vulnerabilidades.md) creado |
| 21-11-2026 | V-01 corregida: la inscripción solo admite identidades nuevas. Eliminada la carpeta `secrets/`, ya sin uso |
| 21-11-2026 | Retirada la eliminación de la app. Revisión de los 9 métodos: corregido un fallo con ids numéricos, añadidos la línea de comandos (`python -m sistema`), la carpeta `entrada/` en Docker y un bloqueo entre procesos |
