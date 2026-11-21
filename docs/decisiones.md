# Decisiones técnicas

Cada entrada indica el problema, la elección y su motivo. Las decisiones marcadas como **(equipo)** las tomó el equipo; el resto son técnicas. El diseño inicial, que planteaba muchas de estas preguntas, está en [historico/diseno_inicial.md](historico/diseno_inicial.md).

## Datos y base de datos

### 1. Base de datos: CSV en claro (equipo)

- **Problema:** la práctica exige que la BD sea un `.csv` precargado con las 51 personas, y el grupo rival tiene que poder ejecutar el sistema.
- **Elección:** CSV en claro y versionado (`data/famosos.csv`), con una fila por foto: `sample_id,identity,template,created_at,hmac`. Sustituye a una versión anterior con SQLite y cifrado AES-256-GCM.
- **Consecuencia:** quien lea la BD obtiene los templates (V-03, V-04).

### 2. Integridad sin confidencialidad: HMAC con clave generada en el primer arranque (equipo)

- **Problema:** con la BD en claro, quien pueda escribir en ella puede renombrar identidades o inscribir su cara como otra persona.
- **Alternativas consideradas:**
  - No proteger la BD.
  - Firmarla con nuestra clave y entregársela al rival.
  - Generar la clave en el primer arranque (TOFU).
- **Elección:** TOFU. La clave se genera en un volumen **separado** de la BD, y en ese momento se firman todas las filas y un manifiesto con la generación y el resumen del conjunto.
- **Límites:** V-05, V-06 y V-07.

### 3. Formato de template público

- **Elección:** formato binario versionado (`recognition/template.py`): `FRT`, versión, HOG float32[8100] y `OrbTemplate.to_bytes()`, en base64. Es el mismo formato del CSV y el que reciben los métodos `_t`.
- **Motivo:** sin `pickle` y con validación de rangos al leerlo.

### 4. Fotos inscritas: todas menos 2 por persona (equipo)

- **Elección:** `scripts/seed_db.py` inscribe todas las fotos válidas menos 2, elegidas al azar con semilla fija y listadas en `data/holdout.csv`. Una se usa para calibrar y la otra para el test.
- **Motivo:** con una sola foto por persona, el rank-1 del SVM cae a la mitad. El SVM no puede aprender la variación intra-clase, y los aumentos sintéticos no la sustituyen. Las fotos reservadas permiten una evaluación sin fuga de información.

### 5. No se conservan las fotografías

- **Elección:** las fotos solo existen en memoria durante la petición.
- **Coste:** si cambia el preprocesamiento, hay que volver a inscribir (`scripts/seed_db.py`).
- **Motivo:** minimización de datos.

## Reconocimiento

### 6. Cascada HOG+SVM → ORB y verificación 1:1

- **Identificación:**
  1. Filtro de evidencia ORB.
  2. Si el margen del SVM es ≥ T1, se identifica.
  3. Si no, se identifica cuando ORB da best ≥ T2 y un margen suficiente sobre el 2.º candidato.
  4. Si no, desconocido.
- **Verificación:** la misma idea, pero comparando solo con la identidad reclamada (T1v y T2v).
- Las puntuaciones **no** se presentan como probabilidades.

### 7. Cuatro desenlaces distintos

- **Elección:** `IDENTIFIED`, `UNCERTAIN`, `UNKNOWN` y `ERROR`.
  - Un error técnico (imagen inválida, BD manipulada) **nunca** se convierte en "desconocido".
  - "Incierto" se usa cuando el sistema no puede afirmar que la persona no está inscrita: coincidencia ORB ambigua, o entrada sin evidencia ORB.

### 8. Identificación con rechazo, mostrando siempre el candidato (equipo)

- **Elección:** `identificar` puede responder "desconocido", pero el resultado siempre incluye `candidate`, la identidad más parecida aunque no se acepte.

### 9. Exactamente una cara por foto

- **Elección:**
  - 0 caras → error "no se detectó ninguna cara".
  - 2 o más caras válidas → error "sube una foto con una sola persona".
  - Las caras de menos de 48 px se ignoran.
- **Motivo:** evita inscribir a una persona con la cara de otra y simplifica el modelo de resultados.

### 10. Filtro de evidencia ORB

- **Problema:** el SVM puntúa cualquier entrada, y las no-caras (`_sd`) o los templates fabricados (`_t`) superaban T1 (94–100 %).
- **Alternativas medidas:**
  - Un filtro HOG de "parece una cara": no separa y se descartó.
  - Exigir alguna coincidencia geométrica ORB: sí separa.
- **Elección:** el filtro ORB, aplicado antes de T1 en todos los niveles. No tiene coste medible en genuinos. Ver [evaluacion.md](evaluacion.md#6-mitigaciones-evaluadas).

### 11. Umbrales calibrados con fotos reservadas

- **Elección:** `evaluation.deployed --write-thresholds` elige T1, T2, T1v y T2v **solo con calibración** (FAR ≤ 1 %), evalúa test una vez y escribe `data/thresholds.json`.
- Sin ese archivo se usan valores provisionales (T = 0,0 y 0,05), marcados como "no calibrados" en la interfaz. Las variables `FACEREC_T1` y `FACEREC_T2` también marcan los umbrales como no calibrados.

## Ciclo de vida del SVM

### 12. Reentrenamiento completo y publicación atómica

- **Elección:** tras cada alta o baja, el SVM se reentrena desde la BD, que es la fuente de verdad.
- **Validaciones:** el modelo debe tener las clases exactas de la BD, al menos un 90 % de exactitud sobre su propio entrenamiento y superar un *round-trip* de guardado y carga.
- **Publicación:** con `os.replace`, junto a un meta firmado que guarda el hash y la generación de la BD.
- **Coherencia:**
  - Un modelo cuyo hash o generación no coincide no decide: se omite T1 y queda una nota.
  - Si falla un reentrenamiento tras un alta, se conserva el último modelo válido.
  - Tras una baja, el modelo anterior se **descarta**, porque contiene a la persona eliminada.
  - Con menos de 2 identidades no hay SVM.
- **Coste:** unos 3 s con unas 50 identidades, aceptable en altas y bajas.

### 13. Solo se inscriben identidades nuevas (equipo)

- **Elección:** `inscribir*` y la pestaña Inscribir solo crean identidades **nuevas**. Un id existente se rechaza (`DuplicateIdentityError`). Corrige V-01, que permitía añadir la cara de otra persona a una identidad ajena.
- **Alternativa descartada:** añadir muestras solo si se verifican como esa persona. Rechazaba el 45 % de las fotos legítimas y dejaba pasar alrededor del 1 % de las ajenas.
- **Consecuencia:** todas las fotos de una persona se aportan al inscribirla (varias a la vez en la app). No se pueden cambiar después, porque tampoco se puede eliminar (decisión 14).
- Antes de guardar también se rechaza una cara que ORB ya acepta como **otra** identidad (`DuplicateFaceError`), salvo que se fuerce.

### 14. Sin eliminación de identidades (equipo)

- **Elección:** la app y `sistema.py` ofrecen exactamente lo que pide la práctica (inscribir, verificar e identificar en sus tres variantes), sin eliminar. La pestaña Eliminar, que venía del diseño inicial, se retiró.
- **Motivo:** la práctica no lo pide, y exponerlo permitía borrar a una persona e inscribirse con su nombre (riesgo residual de V-01).
- **Para volver al estado inicial:** `docker compose down -v` restaura la BD de famosos. El borrado sigue existiendo como operación interna (`service.delete`) para los tests.

## Infraestructura

### 15. Versiones y Docker

- **Python 3.12:** MediaPipe no tiene ruedas para 3.14. Las versiones están fijadas en `requirements.txt` y se verificaron juntas.
- **OpenCV:** solo `opencv-contrib-python`, que es el que arrastra MediaPipe.
- **Imagen:** `python:3.12-slim-bookworm`, más las bibliotecas nativas que necesita MediaPipe (EGL y GLES) y OpenCV. Se obtuvieron de las dependencias de sus `.so`, y el build comprueba que el detector arranca.
- **Volúmenes:** `facerec-data` (BD y SVM) y `facerec-state` (clave y manifiesto).

### 16. Línea de comandos y carpeta `entrada/`

- **Elección:** `python -m sistema <método> …` expone los 9 métodos con salida JSON. Docker monta `./entrada` en `/entrada` en solo lectura, para que quien evalúe pueda usar sus fotos dentro del contenedor con `docker compose exec`.
- Como la app y la línea de comandos son procesos distintos, las escrituras de la BD y el reentrenamiento del SVM usan un bloqueo entre procesos (`fcntl.flock`).

### 17. Código anterior en `legacy/` (equipo)

- **Elección:** el código que el sistema ya no usa (flujo CSV del integrante 2, almacén ORB provisional, estudio previo y docs superados) se movió a `legacy/` con un README, a la espera de que el equipo decida si se elimina o se conserva.
