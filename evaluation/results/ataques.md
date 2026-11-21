# Evaluación de ataques

Generado por `python -m evaluation.attacks`; no editar a mano.

Umbrales: calibrados (evaluation/results/deployed.json). Fotos genuinas: las reservadas de test (51).

| Ataque | Resultado | Observación |
|---|---|---|
| A1 Fotos genuinas completas por `verificar_sd` (sin alinear) | aceptadas 9.8 % (5/51; IC95 4.3–21.0) | referencia con detector: aceptadas 54.9 % (28/51; IC95 41.4–67.7) |
| A2 Imágenes que no son caras por `identificar_sd` | identificadas como alguien 10.0 % (5/50; IC95 4.3–21.4) | con detector: rechazadas por «sin cara» 100.0 % (50/50; IC95 92.9–100.0) |
| A3 Imágenes que no son caras por `verificar_sd` (identidad al azar) | aceptadas 0.0 % (0/50; IC95 0.0–7.1) |  |
| A4 Impostor reclamando otra identidad por `verificar_sd` | aceptados 11.8 % (6/51; IC95 5.5–23.4) | con detector: 0.0 % (0/51; IC95 0.0–7.0) |
| B1 Replay: template leído de la BD por `verificar_t` (víctima) | aceptados 100.0 % (51/51; IC95 93.0–100.0) | el CSV está en claro: cualquier lector lo obtiene |
| B2 Replay: template leído de la BD por `identificar_t` | identificados como la víctima 100.0 % (51/51; IC95 93.0–100.0) |  |
| B3 Templates aleatorios por `verificar_t` (identidad al azar) | aceptados 0.0 % (0/50; IC95 0.0–7.1) |  |
| B4 Templates aleatorios por `identificar_t` | identificados 0.0 % (0/50; IC95 0.0–7.1) |  |
| B5 Templates malformados (vacío, truncado, cabecera falsa, valores extremos) | rechazados con error de formato 5/5 |  |
| D1 Renombrar las filas de una identidad a otra (sin la clave) | detectado | HMAC por fila / manifiesto |
| D2 Sustituir el template de la víctima por el del atacante (sin la clave) | detectado | HMAC por fila / manifiesto |
| D3 Añadir una fila falsificada (firma inventada) (sin la clave) | detectado | HMAC por fila / manifiesto |
| D4 Borrar todas las filas de una identidad (sin la clave) | detectado | HMAC por fila / manifiesto |
| D5 Restaurar una copia anterior del CSV (resucitar a un eliminado) | detectado | generación en el manifiesto |
| D6 Sustituir la BD por un CSV sin firmar (p. ej. la semilla manipulada) | detectado | el manifiesto ya existe |
| D7 Con la clave: añadir fila firmada y borrar el manifiesto | detectado | filas firmadas sin manifiesto -> se rechaza |
| D8 Con la clave: inscribir la cara del atacante como la víctima | no detectable; verificación del atacante como víctima: ACEPTADA | riesgo residual: quien controla la clave controla el sistema |
| D9 Sustituir el modelo SVM y recalcular su hash (sin la clave) | detectado (no se carga) | meta firmado con HMAC; el pickle no se deserializa |
