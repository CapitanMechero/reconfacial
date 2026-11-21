# Seguridad y protección de datos

Este documento describe **qué protege el sistema y cómo**. Lo que no protege está en [vulnerabilidades.md](vulnerabilidades.md).

## Qué se guarda y dónde

| Dato | Dónde | Protección |
|---|---|---|
| Fotografías | **En ningún sitio.** Solo existen en memoria durante la petición. `data/fotos/` (fotos del docente) está fuera de git y de la imagen | — |
| Templates (HOG + ORB) y nombre de cada muestra | BD CSV: `var/db.csv` en local, volumen `facerec-data` en Docker | **En claro** (decisión del equipo) con HMAC por fila |
| BD semilla | `data/famosos.csv` (versionado y dentro de la imagen) | Sin firmar; se firma en el primer arranque |
| Clave HMAC y manifiesto | `var/state/` en local, volumen `facerec-state` en Docker | Permisos 0600, **separados** de la BD |
| Modelo SVM | Junto a la BD (`models/svm_runtime.joblib`) | SHA-256 en un meta firmado con HMAC |

## Integridad de la base de datos

Implementada en `database/csv_store.py` y `database/integrity.py`.

- **HMAC-SHA256 por fila** sobre `sample_id`, `identity`, `template` y `created_at`, con prefijos de longitud. Como el nombre está firmado, renombrar o sustituir un template invalida la firma.
- **Manifiesto firmado** (`manifest.json`): guarda la `generation` y un resumen HMAC del conjunto de filas. Detecta filas borradas o añadidas y la restauración de una copia anterior del CSV.
- **Escritura atómica y recuperable:**
  1. Se escribe el manifiesto con el resumen nuevo marcado como `pending`.
  2. Se escribe el CSV con `os.replace`.
  3. Se escribe el manifiesto final.

  Si el proceso se corta entre medias, al reabrir se completa la operación. Las escrituras de varios procesos (la app y `python -m sistema`) se serializan con `fcntl.flock`.
- **Primer arranque (TOFU):**
  - Se genera la clave (32 bytes aleatorios, archivo 0600 creado en exclusiva).
  - Se copia la semilla.
  - Si no hay manifiesto y ninguna fila está firmada, se firman todas.
- **Cualquier fallo de integridad** produce un `ERROR` con el código `template_corrupt`. Nunca se convierte en "desconocido".

Resultado de los ataques de escritura ([evaluacion.md](evaluacion.md#5-ataques)): se detectan todos los hechos sin la clave (D1–D6), y también el borrado del manifiesto con filas firmadas por quien tiene la clave (D7). Los casos que no se detectan son V-05, V-06 y V-07.

## Integridad del modelo SVM

El SVM se guarda con `joblib`, es decir, como *pickle*, y cargar un pickle manipulado permite ejecutar código. Por eso el meta (`svm_runtime.meta.json`) contiene el SHA-256 del modelo y está firmado con la clave HMAC. Si el modelo o el meta se alteran sin la clave, el modelo no se carga, T1 se desactiva y el modelo se reentrena desde la BD (ataque D9).

Además, un modelo cuya `generation` no coincide con la de la BD **nunca decide**. Así se evita que un SVM antiguo confunda a personas recién inscritas.

## Validación de entradas

- **Imágenes:** formato, tamaño mínimo y un máximo de 50 Mpx (`preprocessing.load_image`). Con detector se exige exactamente una cara válida.
- **Templates (`_t`):** se comprueban la cabecera, la versión, la longitud, que el HOG sea finito y esté en [0, 1], y que los puntos ORB caigan dentro de la cara de 128×128. Los templates malformados se rechazan (ataque B5: 5/5).
- **Nombres:** normalización NFKC, de 1 a 100 caracteres y sin caracteres de control.
- **Filtro de evidencia ORB:** antes de que decida el SVM, la entrada debe tener puntos ORB y alguna coincidencia geométrica con una cara inscrita. Frena las entradas que no son caras por `_sd` y los templates fabricados por `_t` (ver V-09 y V-10).

## Despliegue

- Docker publica el puerto solo en `127.0.0.1`.
- El contenedor se ejecuta sin privilegios (UID 10001), con el sistema de archivos de solo lectura salvo los volúmenes y `/tmp`, sin capacidades (`cap_drop: ALL`) y con `no-new-privileges`.
- El modelo de MediaPipe se verifica por SHA-256 al construir la imagen.
- Los logs no contienen imágenes, templates ni la clave.

## Protección de datos (contexto RGPD)

Las plantillas faciales usadas para identificar a una persona son **datos biométricos de categoría especial** (art. 9 RGPD).

En esta práctica las identidades son personajes públicos y las fotos las proporciona el docente. Aun así, una BD en claro y versionada **no sería aceptable en un despliegue real**: haría falta cifrado, una base jurídica y consentimiento explícito, una EIPD (art. 35), control de acceso y un plazo de conservación.

Se aplica minimización de datos: no se guardan fotos, solo las plantillas necesarias. El borrado de una identidad elimina sus filas y la saca del SVM, pero no alcanza a copias anteriores de la BD ni al historial de git de la semilla.
