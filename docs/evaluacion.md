# Evaluación

Este documento cubre la evaluación del sistema **tal como se entrega**: 51 personas inscritas y los umbrales de `data/thresholds.json`. Todas las cifras se pueden regenerar (requiere `data/fotos/`):

```bash
python -m evaluation.deployed --write-thresholds   # calibra, evalúa en test y adopta los umbrales
python -m evaluation.attacks                       # escenarios de ataque
python -m evaluation.ood_study                     # estudio del filtro HOG descartado
```

Los resultados se guardan en `evaluation/results/`: `deployed.md`/`.json` y `ataques.md`. Las vulnerabilidades derivadas están en [vulnerabilidades.md](vulnerabilidades.md).

## 1. Datos

| Recurso | Contenido |
|---|---|
| `data/fotos/<id>/` (no versionado) | 288 fotos de 51 personas: las 50 del docente y `henry_cavill`, elegido por el equipo. Son válidas 276; se descartan 11 sin cara detectable y 1 con dos caras |
| `data/famosos.csv` | BD semilla: **174 fotos inscritas**, que son todas las válidas de cada persona menos 2 |
| `data/holdout.csv` | Las **2 fotos reservadas** por persona (102), que nunca se inscriben. La 1.ª se usa para **calibrar** y la 2.ª para el **test** final |

Las fotos reservadas se eligieron al azar con semilla fija (`scripts/seed_db.py`). Ninguna foto de calibración o de test está inscrita, así que no hay fuga de información.

## 2. Métricas

Cada consulta termina en "aceptada como X" o en "no aceptada". *Desconocido*, *incierto* y *error* cuentan como no aceptada.

| Métrica | Definición | Denominador |
|---|---|---|
| **FAR** | Impostores aceptados | Consultas impostoras |
| **FRR** | Genuinos no aceptados | Consultas genuinas |
| **IR** (identification rate) | Genuinos identificados como **su** identidad | Consultas genuinas |
| **FIR** (falsa identificación) | Genuinos identificados como **otra** identidad | Consultas genuinas (IR + FRR + FIR = 1) |

Todas las tasas llevan un intervalo de confianza de Wilson al 95 % (`evaluation/metrics.py`). Las puntuaciones del SVM y de ORB **no son probabilidades**.

## 3. Protocolo

### Verificación (1:1)

- **Genuina:** cada foto reservada reclama su propia identidad (51 por split).
- **Impostora:** la misma foto reclama cada una de las **otras** 50 identidades (2550 por split; impostor de esfuerzo cero).
- **Regla:** pasa el filtro de evidencia ORB **y** el margen del SVM para la clase reclamada es ≥ T1v, o, si no, la puntuación ORB contra sus templates es ≥ T2v.

### Identificación (1:N, conjunto abierto)

Para medir el FAR hacen falta personas no inscritas. Se simulan con 5 particiones de unas 10 identidades: en cada una se reentrena el SVM y se retiran las referencias ORB de esas personas, y sus fotos reservadas pasan a ser impostoras. Las genuinas son las fotos reservadas del resto.

**Limitaciones:**
- Los impostores de calibración y de test son las mismas personas, aunque con fotos distintas.
- Cada SVM tiene unas 41 clases en lugar de 51.

### Elección de umbrales

Se recorre una rejilla de candidatos (puntos medios entre puntuaciones observadas, más `inf` para desactivar una etapa) y se elige la combinación que **maximiza los genuinos aceptados con FAR ≤ 1 % en calibración**. Los empates se resuelven tomando la combinación central, para no colocar el umbral pegado a la puntuación genuina más baja (eso sobreajustaba). Después, **test se evalúa una sola vez** con los umbrales ya fijados.

La regla de decisión de la evaluación (`evaluation/cascade.py`) es la misma que la del motor. Un test lo comprueba sobre 300 casos aleatorios, incluido el filtro de evidencia ORB.

## 4. Resultados del sistema entregado

### Verificación

| | T1v | T2v | FAR test | FRR test |
|---|---|---|---|---|
| **Adoptado** (SVM o ORB) | −0,608 | 0,029 | **1,0 %** (26/2550; IC95 0,7–1,5) | **45,1 %** (23/51; IC95 32,3–58,6) |
| Solo SVM (mismo T1v) | −0,608 | — | 0,6 % (16/2550) | 49,0 % (25/51) |

En calibración el FAR fue del 0,9 % (24/2550). **El objetivo de FAR se mantiene en test.** ORB rescata a 2 genuinos a cambio de 10 falsas aceptaciones más.

### Identificación

| Modo | T1 | T2 | IR test | FIR test | FAR test |
|---|---|---|---|---|---|
| HOG + SVM | −0,271 | — | 21,6 % (44/204) | 1,5 % | 1/51 |
| ORB | — | 0,070 | 9,8 % (20/204) | 0,0 % | 0/51 |
| **Cascada (adoptada)** | −0,271 | 0,272 | **23,5 %** (48/204; IC95 18,2–29,8) | 1,5 % | **1/51 = 2,0 %** (IC95 0,3–10,3) |

Con el sistema completo (51 clases), sobre las 51 fotos de test se identifican bien 11 (21,6 %) y se verifican bien 28 (54,9 %).

**Inestabilidad.** En una ejecución anterior con 50 personas y otra selección de fotos reservadas, la cascada obtuvo una IR del 36,5 % con T2 = 0,055. Al añadir a la persona 51 cambió esa selección: la IR bajó al 23,5 % y T2 subió a 0,27, tan alto que ORB casi nunca decide. Con unas 50 consultas por split, la IR varía más de 10 puntos según qué fotos se reserven (V-14).

## 5. Ataques

Los ataques se ejecutan con `python -m evaluation.attacks` sobre una copia nueva del sistema desplegado, con los umbrales adoptados. Las fotos genuinas son las 51 reservadas de test.

| ID | Ataque | Resultado | Vulnerabilidad |
|---|---|---|---|
| **A. Saltarse el detector (`_sd`)** | | | |
| A1 | Foto genuina completa (sin recortar) por `verificar_sd` | aceptada el 9,8 % (con detector: 54,9 %) | Esperado: `_sd` supone una cara ya recortada |
| A2 | Imágenes que no son caras por `identificar_sd` | identificadas el **10 %** (antes del filtro ORB: 94 %) | V-09 🟡 |
| A3 | Imágenes que no son caras por `verificar_sd` (identidad al azar) | aceptadas el **0 %** (antes: 50 %) | V-09 🟡 |
| A4 | Impostor con cara sin alinear por `verificar_sd` | aceptado el **11,8 %** (con detector: 0 %) | V-08 🔴 |
| **B. Inyectar templates (`_t`)** | | | |
| B1/B2 | Replay de un template leído de la BD | aceptado e identificado el **100 %** | V-04 🔴 |
| B3/B4 | Templates aleatorios con formato válido | aceptados e identificados el **0 %** (antes: 4 % y 100 %) | — |
| B5 | Templates malformados | rechazados 5/5 | — |
| **C. Leer la BD** | Templates y nombres en claro | permite B1/B2 | V-03 ⚪ |
| **D. Escribir la BD o el volumen** | | | |
| D1–D4 | Renombrar, sustituir, falsificar o borrar filas sin la clave | **detectado** | — |
| D5 | Restaurar una copia anterior del CSV | **detectado** | — |
| D6 | Sustituir la BD por un CSV sin firmar | **detectado** | — |
| D7 | Con la clave: añadir una fila firmada y borrar el manifiesto | **detectado** | — |
| D8 | Con la clave: inscribir la cara del atacante como la víctima | **no detectado**; el atacante se verifica como la víctima | V-05 ⚪ |
| D9 | Sustituir el modelo SVM y recalcular su hash sin la clave | **detectado**; el modelo no se carga | — |
| **E. Abuso de la API** | | | |
| E1 | `inscribir(id_víctima, foto_atacante)` | antes de corregir: el atacante **se verificaba como la víctima**. Ahora: **bloqueado**, porque solo se inscriben identidades nuevas (`DuplicateIdentityError`; prueba con fotos reales) | V-01 🟢 |

## 6. Mitigaciones evaluadas

### Filtro HOG de "parece una cara": descartado

Se probó con `evaluation/ood_study.py`, fijando el umbral para que pase el 99 % de las caras de calibración:

| Filtro | Caras desconocidas que pasan | No-caras (`_sd`) que pasan | Templates aleatorios que pasan |
|---|---|---|---|
| Coseno con la cara inscrita más cercana | 100 % | 32 % | **100 %** |
| Coseno centrado | 98 % | 99,5 % | 100 % |
| Error de reconstrucción PCA (k = 10 y 40) | 98–100 % | 97,5–99,5 % | 100 % |

HOG no separa caras de no-caras: los templates aleatorios se parecen *más* a las caras inscritas que las caras reales de desconocidos.

### Filtro de evidencia ORB: adoptado

Antes de que decida el SVM, la entrada debe tener puntos ORB y una puntuación ORB estrictamente positiva contra alguna identidad (`orb_min_evidence = 0.0`). Se aplica a identificación y verificación, en los tres niveles de entrada.

| | Sin filtro | Con filtro |
|---|---|---|
| A2: no-caras identificadas | 94 % | 10 % |
| A3: no-caras aceptadas | 50 % | 0 % |
| B4: templates aleatorios identificados | 100 % | 0 % |
| Métricas de genuinos (FAR, FRR, IR) | — | **idénticas**: todas las caras reservadas tienen evidencia ORB |

Un atacante que conozca el filtro puede superarlo (V-10).

## 7. Estudio previo (archivado en `legacy/evaluacion_previa/`)

Antes de adaptar el sistema a la práctica se hizo un estudio con particiones de identidades disjuntas: 50 personas, unas 2,6 fotos de entrenamiento por persona y 20 repeticiones. Sus conclusiones motivaron el diseño actual:

- **Rank-1 sin umbral:** HOG 46 % y ORB 31 %. Con una sola foto por persona baja al 23 % y al 10 %, de ahí la decisión de inscribir todas las fotos menos 2.
- **Umbrales calibrados con unos 54 impostores:** en una partición alcanzaron un FAR del 11,3 % en test, y en 20 repeticiones el FAR medio estuvo entre el 1,4 % y el 2,6 %. De ahí la cautela con el FAR de identificación (V-14).
- **IR media en 20 repeticiones:** 17 % con HOG, 13 % con ORB y 22 % con la cascada.

Para reproducirlo: `python -m legacy.evaluacion_previa.run --images data/fotos --repeat 20`. Con la persona 51, las cifras cambiarán.
