# Vulnerabilidades y limitaciones conocidas

Documento vivo. Recoge todo lo que sabemos que el sistema **no** resiste o hace mal, para que el grupo rival no encuentre nada que nosotros no conozcamos.

- Añade las novedades al final de la sección correspondiente, con el siguiente ID libre y la [plantilla](#plantilla-para-nuevas-entradas).
- No borres entradas: si algo se corrige, cambia su **Estado** y anota la fecha en el [registro de cambios](#registro-de-cambios).
- Las cifras salen de `evaluation/results/` (sistema desplegado: 51 personas y umbrales de `data/thresholds.json`), salvo que la entrada diga otra cosa.

**Estados:** 🔴 Abierta · 🟡 Mitigada parcialmente · 🟢 Corregida · ⚪ Aceptada (por diseño o por decisión del equipo)

**Severidad** (estimación del equipo): Crítica · Alta · Media · Baja

## Resumen

| ID | Título | Tipo | Severidad | Estado |
|---|---|---|---|---|
| [V-01](#v-01) | Cualquiera puede añadir su cara a otra identidad con `inscribir` | Control de acceso | Crítica | 🟢 Corregida |
| [V-02](#v-02) | La interfaz y la API no tienen autenticación | Control de acceso | Alta | ⚪ Aceptada |
| [V-03](#v-03) | Base de datos en claro: cualquiera que la lea obtiene los templates | Lectura de la BD | Alta | ⚪ Aceptada |
| [V-04](#v-04) | Replay de templates leídos de la BD por los métodos `_t` | Inyección de templates | Alta | 🔴 Abierta |
| [V-05](#v-05) | Quien tenga la clave HMAC controla la BD y el modelo | Escritura de la BD | Alta | ⚪ Aceptada |
| [V-06](#v-06) | Manipular la semilla antes del primer arranque no se detecta | Escritura de la BD | Media | ⚪ Aceptada |
| [V-07](#v-07) | Restaurar a la vez BD y estado, o borrar el estado, no se detecta | Escritura de la BD | Media | 🔴 Abierta |
| [V-08](#v-08) | Impostores con caras sin alinear por `verificar_sd` | Saltarse el detector | Media | 🔴 Abierta |
| [V-09](#v-09) | Algunas imágenes que no son caras se identifican por `identificar_sd` | Saltarse el detector | Media | 🟡 Mitigada |
| [V-10](#v-10) | El filtro de evidencia ORB se puede superar | Inyección de templates | Media | 🔴 Abierta |
| [V-11](#v-11) | Las puntuaciones devueltas permiten ataques de escalada (*hill-climbing*) | Inyección de templates | Media | 🔴 No evaluada |
| [V-12](#v-12) | Sin detección de vida: fotos de fotos o pantallas | Presentación | Media | ⚪ Fuera de alcance |
| [V-13](#v-13) | Precisión baja: casi la mitad de los genuinos se rechazan | Rendimiento | Alta | ⚪ Limitación del método |
| [V-14](#v-14) | FAR de identificación con mucha incertidumbre y resultados inestables | Evaluación | Media | ⚪ Limitación de datos |
| [V-15](#v-15) | Los umbrales no se recalibran al inscribir personas nuevas | Rendimiento | Media | 🔴 Abierta |
| [V-16](#v-16) | Algunas fotos no se pueden procesar (sin cara detectable) | Rendimiento | Baja | ⚪ Limitación |
| [V-17](#v-17) | Nombres visualmente idénticos se tratan como personas distintas | Identidades | Baja | 🔴 No evaluada |
| [V-18](#v-18) | Sin límites de uso: denegación de servicio y fuerza bruta | Disponibilidad | Baja | 🔴 Abierta |

## Control de acceso

### V-01

**Cualquiera puede añadir su cara a otra identidad con `inscribir`.** Crítica · 🟢 Corregida el 21-11-2026

- **Descripción original:** `inscribir(id, foto)` con un id existente **añadía** la muestra a esa identidad, sin comprobar que la foto fuera de esa persona.
- **Reproducción antes de corregir** (fotos reales): una foto reservada de `henry_cavill` que reclamaba ser `adele` se rechazaba. Tras `inscribir("adele", <otra foto de henry_cavill>)`, la misma foto **se aceptaba como Adele**.
- **Corrección (decisión del equipo):** la inscripción **solo crea identidades nuevas**. Un id existente se rechaza con `DuplicateIdentityError` en la app y en los tres `inscribir*`, sin opción para saltárselo. Lo cubren `test_cannot_add_any_face_to_existing_identity` y `tests/test_sistema.py`.
  - Un primer intento fue exigir que cada foto nueva se verificara contra la identidad, pero se descartó: rechazaba el 45 % de las fotos legítimas y dejaba pasar alrededor del 1 % de las ajenas.
- **Comprobación tras corregir:** el ataque se bloquea, y la otra foto de Henry Cavill sigue rechazándose como Adele.
- **Consecuencia:** las fotos de una persona solo se pueden aportar al inscribirla: varias a la vez en la app, una con la API.
- **Riesgo residual eliminado:** la vía "eliminar a la víctima e inscribirse con su nombre" se cerró al retirar la eliminación de la app (la práctica no la pide; `sistema.py` nunca la tuvo).

### V-02

**La interfaz y la API no tienen autenticación.** Alta · ⚪ Aceptada

- **Descripción:** cualquiera que llegue a la app (puerto 8501) o pueda ejecutar `sistema.py` puede inscribir personas nuevas, verificar e identificar sin identificarse. Desde el 21-11-2026 no se puede eliminar ni modificar identidades existentes.
- **Mitigación actual:** Docker publica el puerto solo en `127.0.0.1` y no hay TLS.
- **Impacto restante:** inscribir identidades nuevas sin control (por ejemplo, llenar la BD) y probar sin límite (V-11, V-18).
- **Motivo de aceptación:** la práctica exige que el rival pueda invocar los 9 métodos. Un despliegue real necesitaría autenticación y roles.

## Lectura de la base de datos

### V-03

**Base de datos en claro: cualquiera que la lea obtiene los templates.** Alta · ⚪ Aceptada (decisión del equipo)

- **Descripción:** `data/famosos.csv` (versionado y dentro de la imagen) y la BD en uso (volumen `facerec-data`) guardan nombres y templates HOG y ORB en claro, en base64. No hay confidencialidad.
- **Impacto:** permite V-04. Además, los vectores HOG permiten aproximar la apariencia de la cara (la visualización de HOG es reconocible).
- **Motivo de aceptación:** la práctica exige una BD en `.csv` que el rival pueda usar. El cifrado AES-256-GCM de una versión anterior se retiró. Ver [decisiones.md](decisiones.md#1-base-de-datos-csv-en-claro-equipo).

## Inyección de templates (`_t`)

### V-04

**Replay de templates leídos de la BD por los métodos `_t`.** Alta · 🔴 Abierta

- **Reproducción:** ataques B1 y B2 de `evaluation.attacks`. Se toma de la BD un template de cada víctima y se envía tal cual.
- **Resultado:** `verificar_t` lo acepta en el **100 %** de los casos (51/51) e `identificar_t` identifica a la víctima en el **100 %** (51/51).
- **Mitigación posible:** cifrar la BD (descartado, V-03) o rechazar templates idénticos o casi idénticos a uno inscrito. Lo segundo solo obliga a perturbarlos un poco.

### V-10

**El filtro de evidencia ORB se puede superar.** Media · 🔴 Abierta

- **Descripción:** el filtro que frena las entradas que no son caras exige alguna coincidencia geométrica ORB con una cara inscrita. Un atacante que conozca el filtro puede fabricar un template con HOG arbitrario y descriptores ORB copiados de una cara real (por ejemplo, de la BD, V-03) para superarlo.
- **Estado:** razonado, no evaluado experimentalmente.

### V-11

**Las puntuaciones devueltas permiten ataques de escalada (*hill-climbing*).** Media · 🔴 No evaluada

- **Descripción:** `verificar*` e `identificar*` devuelven el margen del SVM y la puntuación ORB. Un atacante puede modificar un template poco a poco y quedarse con los cambios que suben la puntuación, hasta superar el umbral sin tener ninguna foto de la víctima.
- **Mitigación posible:** no exponer las puntuaciones en la API, o redondearlas, y limitar el número de intentos (V-18).

## Escritura de la base de datos

### V-05

**Quien tenga la clave HMAC controla la BD y el modelo.** Alta · ⚪ Aceptada

- **Reproducción:** ataque D8. Con la clave del volumen `facerec-state`, el atacante firma una fila con su cara bajo el nombre de la víctima: **no se detecta**, y su verificación como víctima **se acepta**. Con la clave también puede firmar un modelo SVM malicioso; como el modelo es un *pickle*, eso permite **ejecutar código**.
- **Motivo de aceptación:** la integridad depende del secreto de la clave. Está en un volumen separado de la BD y con permisos 0600.

### V-06

**Manipular la semilla antes del primer arranque no se detecta.** Media · ⚪ Aceptada (TOFU)

- **Descripción:** `data/famosos.csv` se versiona sin firmar y se firma en el primer arranque. Lo que se modifique antes (en el repositorio o en la imagen) se adopta como legítimo.
- **Mitigación posible:** publicar un hash de la semilla, por ejemplo en el README, y comprobarlo al arrancar.

### V-07

**Restaurar a la vez BD y estado, o borrar el estado, no se detecta.** Media · 🔴 Abierta

- **Descripción:**
  - Restaurar a la vez una copia antigua de la BD **y** del manifiesto no se detecta, porque no hay un contador monotónico externo.
  - Borrar la clave y el manifiesto **y** sustituir la BD por un CSV sin firmar provoca una nueva adopción, como en un primer arranque.

  Ambos casos requieren escribir en los dos volúmenes.
- **Sí se detectan** (ataques D1–D7): renombrados, sustituciones, filas falsificadas o borradas, restauración solo del CSV, sustitución por un CSV sin firmar con el estado presente, y borrado del manifiesto con filas firmadas.

## Saltarse el detector (`_sd`)

### V-08

**Impostores con caras sin alinear por `verificar_sd`.** Media · 🔴 Abierta

- **Reproducción:** ataque A4. Fotos reales de una persona, completas y sin recortar, reclamando otra identidad.
- **Resultado:** `verificar_sd` las **acepta el 11,8 %** (6/51; IC95 5,5–23,4), frente al 0 % (0/51) con detector.
- **Causa:** los umbrales se calibraron con caras alineadas por MediaPipe. Una imagen sin alinear produce puntuaciones fuera de esa distribución.

### V-09

**Algunas imágenes que no son caras se identifican por `identificar_sd`.** Media · 🟡 Mitigada parcialmente

- **Reproducción:** ataques A2 y A3, con ruido, superficies lisas, degradados y recortes de fondo.
- **Antes del filtro de evidencia ORB:** se identificaban como alguien el **94 %** (`identificar_sd`) y se aceptaban el **50 %** (`verificar_sd`).
- **Ahora:** se identifican el **10 %** (5/50) y se aceptan el **0 %** (0/50).
- **Mitigaciones evaluadas:** un filtro HOG de "parece una cara" **no funciona** (`evaluation/ood_study.py`). El filtro ORB está en [evaluacion.md](evaluacion.md#6-mitigaciones-evaluadas).

## Presentación

### V-12

**Sin detección de vida: fotos de fotos o pantallas.** Media · ⚪ Fuera de alcance

- **Descripción:** el sistema trabaja con fotografías estáticas y no distingue una cara real de una foto impresa o de una pantalla. Cualquier foto de una persona inscrita (por ejemplo, de internet) sirve para intentar suplantarla.

## Rendimiento y evaluación

### V-13

**Precisión baja: casi la mitad de los genuinos se rechazan.** Alta · ⚪ Limitación del método

- **Resultados en test** (FAR objetivo ≤ 1 %):
  - Verificación: se rechaza al **45 %** de los genuinos (23/51), con un FAR del 1,0 %.
  - Identificación: solo se reconoce al **23,5 %** de las personas inscritas.
  - Sin umbral, el SVM acierta la identidad (rank-1) el 46 % de las veces y ORB el 31 %.
- **Causa:** HOG sobre caras de 128 px y ORB son técnicas clásicas poco robustas a cambios de pose, iluminación y expresión, y hay pocas fotos por persona (unas 3,4).

### V-14

**FAR de identificación con mucha incertidumbre y resultados inestables.** Media · ⚪ Limitación de datos

- **Descripción:**
  - El FAR de identificación se estima con solo 51 consultas impostoras: 1/51, con un IC95 del 0,3 al 10,3 %.
  - Cambiar qué fotos se reservan mueve la tasa de identificación más de 10 puntos (del 36,5 % al 23,5 % al añadir a la persona 51).
  - En el estudio previo, umbrales con un FAR de 0/54 en calibración llegaron a un 11,3 % en test.
- **Implicación:** las cifras de identificación hay que leerlas con sus intervalos de confianza. En verificación, con 2550 pares impostores, la estimación es mucho más fiable.

### V-15

**Los umbrales no se recalibran al inscribir personas nuevas.** Media · 🔴 Abierta

- **Descripción:** la escala del margen del SVM depende del número de clases y de muestras. Si el rival inscribe muchas personas nuevas, T1 y T1v dejan de corresponder a lo calibrado y el FAR real puede cambiar. La interfaz sigue mostrando los umbrales como "calibrados".
- **Mitigación posible:** marcar los umbrales como no calibrados cuando la población cambie significativamente, o recalibrar con `evaluation.deployed`.

### V-16

**Algunas fotos no se pueden procesar (sin cara detectable).** Baja · ⚪ Limitación

- **Descripción:** MediaPipe no encuentra cara en 11 de las 288 fotos de `data/fotos`, y en otra encuentra dos. Esas fotos no se pueden inscribir ni identificar con detector. La política es exigir exactamente una cara válida por foto.

## Identidades

### V-17

**Nombres visualmente idénticos se tratan como personas distintas.** Baja · 🔴 No evaluada

- **Descripción:** los nombres se normalizan (NFKC y sin distinguir mayúsculas), pero las letras de otros alfabetos que se ven igual no. Por ejemplo, `adele` con una "е" cirílica es otra identidad que en la interfaz se ve igual, y podría confundir a quien verifica, o permitir inscribir un «doble» con un nombre engañoso.

## Disponibilidad

### V-18

**Sin límites de uso: denegación de servicio y fuerza bruta.** Baja · 🔴 Abierta

- **Descripción:**
  - No hay límite de peticiones ni de intentos de verificación.
  - Cada consulta compara ORB contra todas las fotos inscritas (coste lineal), y cada inscripción reentrena el SVM (unos 3 s).
  - Muchas inscripciones o consultas seguidas degradan el servicio.
  - El tamaño de subida está limitado a 10 MB y a 50 Mpx por imagen.

## Plantilla para nuevas entradas

```markdown
### V-XX

**Título corto.** Severidad · Estado

- **Encontrada por / fecha:** nombre, DD-MM-AAAA
- **Componente:** (detector, `_sd`, `_t`, BD, SVM, ORB, interfaz, Docker…)
- **Descripción:** qué ocurre y por qué.
- **Reproducción:** pasos o comando exacto, y con qué datos.
- **Resultado:** cifras (con n y, si se puede, IC95) o comportamiento observado.
- **Impacto:** qué puede conseguir un atacante o qué falla para un usuario.
- **Mitigación posible / aplicada:**
```

Recuerda añadir la fila correspondiente en la tabla de resumen.

## Registro de cambios

| Fecha | Cambio |
|---|---|
| 21-11-2026 | Documento creado con lo conocido hasta la fecha (V-01 a V-18). V-09 mitigada parcialmente con el filtro de evidencia ORB. V-01 verificada con fotos reales |
| 21-11-2026 | V-01 corregida: la inscripción solo admite identidades nuevas. Se descartó la variante de verificar las fotos añadidas por su coste (45 % de fotos legítimas rechazadas) |
| 21-11-2026 | Retirada la eliminación de la app (la práctica no la pide): cierra el riesgo residual de V-01 y reduce V-02. Corregido un fallo con ids numéricos en `verificar*`. Las escrituras concurrentes de la app y la línea de comandos se serializan con un bloqueo entre procesos |
