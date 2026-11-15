# Sistema de reconocimiento facial (HOG + SVM, ORB y MediaPipe)

Aplicación local que identifica a personas registradas a partir de fotografías estáticas y rechaza a las no registradas («persona desconocida»). Utiliza una cascada de dos técnicas clásicas de visión por computador, con MediaPipe para detectar y alinear las caras, una interfaz web en Streamlit y todo el entorno empaquetado en Docker.

> **Estado:** propuesta de diseño (v1.0). Ver [`docs/diseno.md`](docs/diseno.md) para el documento completo.

## Cómo funciona

```text
Fotografía -> Streamlit -> validación -> MediaPipe (detección + landmarks)
          -> alineamiento y recorte -> normalización (gris, 128x128)
          -> HOG -> SVM -> ¿confianza >= T1? -- sí --> IDENTIDAD
                                   |
                                   no
                                   v
                          ORB + matching con la BD
                                   |
                          ¿confianza >= T2? -- sí --> IDENTIDAD
                                   |
                                   no
                                   v
                              DESCONOCIDO
```

1. **Detección y preprocesamiento:** MediaPipe localiza la cara y sus landmarks; se alinea a partir de los ojos, se recorta y se normaliza.
2. **HOG + SVM (primera etapa):** HOG extrae un vector global de la cara y el SVM propone una identidad. Solo se acepta si la puntuación supera el umbral conservador **T1**.
3. **ORB (segunda comprobación):** si HOG + SVM no es suficientemente fiable, ORB compara puntos clave locales con las referencias (`BFMatcher`, distancia Hamming). Solo se acepta si supera **T2**.
4. **Rechazo:** si ninguna etapa alcanza su umbral, el resultado es «persona desconocida».

T1 y T2 se determinan experimentalmente (ver `evaluation/`), no se fijan a priori.

## Arquitectura del repositorio

```text
reconocimiento-facial/
├── app.py                      # Interfaz Streamlit (solo llama a módulos)
├── recognition/                # Núcleo de reconocimiento
│   ├── detector.py             # MediaPipe: detección y landmarks
│   ├── preprocessing.py        # Alineamiento, recorte y normalización
│   ├── hog.py                  # Extracción de características HOG
│   ├── svm.py                  # Clasificador SVM y puntuación de decisión
│   ├── orb.py                  # Puntos clave y descriptores ORB
│   ├── matching.py             # Matching, filtros y puntuación ORB
│   ├── recognition.py          # Cascada completa de reconocimiento
│   ├── config.py               # Umbrales T1/T2 y parámetros
│   └── types.py                # Estructuras de datos compartidas
├── database/                   # Persistencia y seguridad
│   ├── manager.py              # Registro, eliminación y carga de identidades
│   ├── encryption.py           # Cifrado AES-256-GCM de templates
│   └── storage/                # Datos persistentes (volumen Docker)
├── training/
│   └── train.py                # Entrenamiento del SVM
├── evaluation/                 # Métricas (FAR, FRR, identificación) y ajuste de T1/T2
├── data/                       # Fotos de referencia y consulta (fuera de git)
├── models/                     # Modelos de MediaPipe y SVM entrenado (fuera de git)
├── tests/                      # Pruebas por módulo + fixtures
├── docs/                       # Diseño, interfaces entre módulos y memoria
├── .github/                    # CI (workflows) y plantilla de pull request
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Qué hace cada bloque

| Carpeta | Responsabilidad |
|---|---|
| `recognition/` | Toda la lógica de reconocimiento. Cada archivo es un módulo independiente que se puede probar por separado. `types.py` y `config.py` fijan los contratos y parámetros comunes para poder trabajar en paralelo. |
| `database/` | Almacenamiento de identidades y templates cifrados (SQLite, pendiente de confirmar). La clave de cifrado nunca vive en la imagen Docker ni junto a los datos. |
| `training/` | Entrena el SVM con las identidades registradas y lo reentrena al registrar o eliminar personas. |
| `evaluation/` | Calcula FAR, FRR y tasa de identificación correcta, y sirve para calibrar T1 y T2 con consultas conocidas y desconocidas. |
| `app.py` | Tres apartados: Reconocimiento, Registro y Eliminación. No contiene lógica de reconocimiento. |
| `data/`, `models/` | Fotografías, modelos y claves: **nunca se suben a git** (están en `.gitignore`). |
| `tests/` | Pruebas unitarias por módulo y pruebas funcionales de la cascada. |
| `.github/` | CI que ejecuta tests y lint en cada pull request. |

## Puesta en marcha

Requisitos: Docker Desktop (Windows) o Docker Engine (Linux).

```bash
cp .env.example .env     # configura aquí el origen de la clave de cifrado
docker compose up --build
```

Después abre <http://localhost:8501>.

El procesamiento se realiza en CPU y los datos persisten en volúmenes de Docker.

## Alcance y limitaciones

- Solo fotografías estáticas: sin cámara en tiempo real, vídeo, emociones ni detección de vida.
- No es un mecanismo de autenticación de alta seguridad.
- El dataset disponible tiene **una foto por persona** (50 identidades), lo que limita la capacidad de generalización del SVM y la calibración de umbrales. Son necesarias fotografías de consulta adicionales, de personas registradas y no registradas; los aumentos artificiales deben presentarse como pruebas de robustez sintética.
- El cifrado de templates no protege por sí solo la clave, la interfaz ni las fotografías originales.

## Equipo y reparto de trabajo

| Integrante | Bloque | Archivos principales |
|---|---|---|
| 1 | Detección y preprocesamiento facial | `detector.py`, `preprocessing.py` |
| 2 | HOG y SVM | `hog.py`, `svm.py`, `training/train.py` |
| 3 | ORB y matching | `orb.py`, `matching.py` |
| 4 | Base de datos, cifrado, interfaz y Docker | `manager.py`, `encryption.py`, `app.py`, `Dockerfile`, `docker-compose.yml` |

Integración, pruebas del sistema completo, evaluación y documentación son responsabilidad compartida.

## Flujo de trabajo

- `main` protegida y `dev` como rama de integración.
- Una rama por módulo (`feat/detector`, `feat/hog-svm`, `feat/orb`, `feat/db-ui`) con pull request a `dev` y al menos una revisión.
- Cada módulo debe poder probarse de forma independiente antes de integrarse.
- Los contratos entre módulos están en [`docs/interfaces.md`](docs/interfaces.md); cualquier cambio se acuerda con el equipo.

## Decisiones pendientes

Entrenamiento del SVM con una muestra por identidad, calibración de T1/T2, confirmación de SQLite, gestión de la clave de cifrado, conservación de fotos originales, política de varias caras, identidades duplicadas, actualización del SVM y compatibilidad de versiones en Docker.
