# Resultados de evaluación (fotografías)

Generado por `python -m evaluation.run`; no editar a mano.

# Escenario: todas las fotos de entrenamiento

Semilla 42 · FAR objetivo en calibración ≤ 1% · rank-1 del SVM en conjunto cerrado (test, sin umbral): 46.2 %

| Conjunto | Consultas | Identidades |
|---|---|---|
| Entrenamiento / referencias | 78 | 30 |
| Calibración genuinas | 39 | 30 |
| Calibración impostoras | 54 | 10 |
| Test genuinas | 39 | 30 |
| Test impostoras | 53 | 10 |

## todas las fotos de entrenamiento — hog

Umbrales elegidos en calibración: T1 = -0.2445, T2 = inf (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 7.7 % (3/39; IC95 2.7–20.3) |
| FRR | 92.3 % (36/39; IC95 79.7–97.3) |
| Falsa identificación (FIR) | 0.0 % (0/39; IC95 0.0–9.0) |
| FAR | 11.3 % (6/53; IC95 5.3–22.6) |

Calibración: IR 17.9 % (7/39; IC95 9.0–32.7), FAR 0.0 % (0/54; IC95 0.0–6.6).

## todas las fotos de entrenamiento — orb

Umbrales elegidos en calibración: T1 = inf, T2 = 0.07493 (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 2.6 % (1/39; IC95 0.5–13.2) |
| FRR | 97.4 % (38/39; IC95 86.8–99.5) |
| Falsa identificación (FIR) | 0.0 % (0/39; IC95 0.0–9.0) |
| FAR | 0.0 % (0/53; IC95 0.0–6.8) |

Calibración: IR 10.3 % (4/39; IC95 4.1–23.6), FAR 0.0 % (0/54; IC95 0.0–6.6).

## todas las fotos de entrenamiento — cascade

Umbrales elegidos en calibración: T1 = -0.2445, T2 = 0.1373 (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 7.7 % (3/39; IC95 2.7–20.3) |
| FRR | 92.3 % (36/39; IC95 79.7–97.3) |
| Falsa identificación (FIR) | 0.0 % (0/39; IC95 0.0–9.0) |
| FAR | 11.3 % (6/53; IC95 5.3–22.6) |

Calibración: IR 17.9 % (7/39; IC95 9.0–32.7), FAR 0.0 % (0/54; IC95 0.0–6.6).

# Escenario: una foto por identidad

Semilla 42 · FAR objetivo en calibración ≤ 1% · rank-1 del SVM en conjunto cerrado (test, sin umbral): 23.1 %

| Conjunto | Consultas | Identidades |
|---|---|---|
| Entrenamiento / referencias | 30 | 30 |
| Calibración genuinas | 39 | 30 |
| Calibración impostoras | 54 | 10 |
| Test genuinas | 39 | 30 |
| Test impostoras | 53 | 10 |

## una foto por identidad — hog

Umbrales elegidos en calibración: T1 = -0.3367, T2 = inf (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 0.0 % (0/39; IC95 0.0–9.0) |
| FRR | 100.0 % (39/39; IC95 91.0–100.0) |
| Falsa identificación (FIR) | 0.0 % (0/39; IC95 0.0–9.0) |
| FAR | 1.9 % (1/53; IC95 0.3–9.9) |

Calibración: IR 10.3 % (4/39; IC95 4.1–23.6), FAR 0.0 % (0/54; IC95 0.0–6.6).

## una foto por identidad — orb

Umbrales elegidos en calibración: T1 = inf, T2 = 0.03992 (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 0.0 % (0/39; IC95 0.0–9.0) |
| FRR | 100.0 % (39/39; IC95 91.0–100.0) |
| Falsa identificación (FIR) | 0.0 % (0/39; IC95 0.0–9.0) |
| FAR | 1.9 % (1/53; IC95 0.3–9.9) |

Calibración: IR 10.3 % (4/39; IC95 4.1–23.6), FAR 0.0 % (0/54; IC95 0.0–6.6).

## una foto por identidad — cascade

Umbrales elegidos en calibración: T1 = -0.3367, T2 = 0.03992 (`inf` = etapa desactivada).

| Métrica (test) | Valor |
|---|---|
| Identification rate | 0.0 % (0/39; IC95 0.0–9.0) |
| FRR | 100.0 % (39/39; IC95 91.0–100.0) |
| Falsa identificación (FIR) | 0.0 % (0/39; IC95 0.0–9.0) |
| FAR | 3.8 % (2/53; IC95 1.0–12.8) |

Calibración: IR 12.8 % (5/39; IC95 5.6–26.7), FAR 0.0 % (0/54; IC95 0.0–6.6).

## Variabilidad entre 20 particiones adicionales (test)

| Modo | IR media (mín–máx) | FAR media (mín–máx) |
|---|---|---|
| hog | 17.3 % (5.1–30.8) | 1.4 % (0.0–7.0) |
| orb | 13.2 % (0.0–25.0) | 1.6 % (0.0–9.3) |
| cascade | 21.8 % (10.8–30.8) | 2.6 % (0.0–9.3) |
