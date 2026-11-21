# Resultados de evaluación (HOG CSV)

Generado por `python -m evaluation.run`; no editar a mano.

# Escenario: todas las fotos de entrenamiento

Semilla 42 · FAR objetivo en calibración ≤ 1% · rank-1 del SVM en conjunto cerrado (test, sin umbral): 52.6 %

| Conjunto | Consultas | Identidades |
|---|---|---|
| Entrenamiento / referencias | 80 | 29 |
| Calibración genuinas | 38 | 29 |
| Calibración impostoras | 56 | 10 |
| Test genuinas | 38 | 29 |
| Test impostoras | 50 | 10 |

## todas las fotos de entrenamiento — hog

Umbrales elegidos en calibración: T1 = -0.1612, T2 = inf (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 0.0 % (0/38; IC95 0.0–9.2) |
| FRR | 100.0 % (38/38; IC95 90.8–100.0) |
| Falsa identificación (FIR) | 0.0 % (0/38; IC95 0.0–9.2) |
| FAR | 0.0 % (0/50; IC95 0.0–7.1) |

Calibración: IR 18.4 % (7/38; IC95 9.2–33.4), FAR 0.0 % (0/56; IC95 0.0–6.4).

# Escenario: una foto por identidad

Semilla 42 · FAR objetivo en calibración ≤ 1% · rank-1 del SVM en conjunto cerrado (test, sin umbral): 28.9 %

| Conjunto | Consultas | Identidades |
|---|---|---|
| Entrenamiento / referencias | 29 | 29 |
| Calibración genuinas | 38 | 29 |
| Calibración impostoras | 56 | 10 |
| Test genuinas | 38 | 29 |
| Test impostoras | 50 | 10 |

## una foto por identidad — hog

Umbrales elegidos en calibración: T1 = -0.4105, T2 = inf (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 5.3 % (2/38; IC95 1.5–17.3) |
| FRR | 94.7 % (36/38; IC95 82.7–98.5) |
| Falsa identificación (FIR) | 0.0 % (0/38; IC95 0.0–9.2) |
| FAR | 4.0 % (2/50; IC95 1.1–13.5) |

Calibración: IR 15.8 % (6/38; IC95 7.4–30.4), FAR 0.0 % (0/56; IC95 0.0–6.4).

## Variabilidad entre 20 particiones adicionales (test)

| Modo | IR media (mín–máx) | FAR media (mín–máx) |
|---|---|---|
| hog | 20.2 % (2.5–37.8) | 1.4 % (0.0–7.5) |
