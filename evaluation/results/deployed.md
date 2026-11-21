# Evaluación del sistema desplegado

Generado por `python -m evaluation.deployed`; no editar a mano.

51 identidades, 174 fotos inscritas; fotos reservadas por split: {'calibracion': 51, 'test': 51}. FAR objetivo en calibración ≤ 1%.

## Verificación (1:1)

Umbrales elegidos en calibración: T1v = -0.6078, T2v = 0.02938.

| Split | FAR | FRR | FAR solo SVM | FRR solo SVM |
|---|---|---|---|---|
| calibracion | 0.9 % (24/2550; IC95 0.6–1.4) | 41.2 % (21/51; IC95 28.8–54.8) | 0.6 % (16/2550; IC95 0.4–1.0) | 47.1 % (24/51; IC95 34.1–60.5) |
| test | 1.0 % (26/2550; IC95 0.7–1.5) | 45.1 % (23/51; IC95 32.3–58.6) | 0.6 % (16/2550; IC95 0.4–1.0) | 49.0 % (25/51; IC95 35.9–62.3) |

## Identificación (1:N, conjunto abierto)

| Modo | T1 | T2 | IR test | FRR test | FIR test | FAR test | IR calib. | FAR calib. |
|---|---|---|---|---|---|---|---|---|
| hog | -0.2706 | inf | 21.6 % (44/204; IC95 16.5–27.7) | 77.0 % (157/204; IC95 70.7–82.2) | 1.5 % (3/204; IC95 0.5–4.2) | 2.0 % (1/51; IC95 0.3–10.3) | 18.1 % (37/204; IC95 13.5–24.0) | 0.0 % (0/51; IC95 0.0–7.0) |
| orb | inf | 0.07001 | 9.8 % (20/204; IC95 6.4–14.7) | 90.2 % (184/204; IC95 85.3–93.6) | 0.0 % (0/204; IC95 0.0–1.8) | 0.0 % (0/51; IC95 0.0–7.0) | 3.9 % (8/204; IC95 2.0–7.5) | 0.0 % (0/51; IC95 0.0–7.0) |
| cascade | -0.2706 | 0.2716 | 23.5 % (48/204; IC95 18.2–29.8) | 75.0 % (153/204; IC95 68.6–80.4) | 1.5 % (3/204; IC95 0.5–4.2) | 2.0 % (1/51; IC95 0.3–10.3) | 18.1 % (37/204; IC95 13.5–24.0) | 0.0 % (0/51; IC95 0.0–7.0) |
