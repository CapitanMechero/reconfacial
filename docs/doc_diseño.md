# Documento de diseño
## Sistema de reconocimiento facial mediante HOG + ORB y MediaPipe

**Versión:** 1.0  
**Estado:** Propuesta de diseño  
**Equipo:** 4 integrantes

---

# 1. Descripción general

El proyecto consiste en desarrollar un sistema de reconocimiento facial capaz de identificar a personas registradas en una base de datos y rechazar fotografías de personas cuya identidad no esté registrada.

El sistema utilizará una arquitectura de reconocimiento en cascada:

1. MediaPipe detectará la cara y permitirá obtener puntos de referencia faciales para su alineamiento.
2. HOG extraerá características globales de la cara.
3. Un clasificador SVM intentará identificar a la persona.
4. Si la confianza de HOG + SVM no supera un umbral elevado, se utilizará ORB como segunda comprobación.
5. Si ORB tampoco encuentra coincidencias suficientemente fiables, el sistema devolverá el resultado «persona desconocida».

La aplicación se desarrollará en Python, utilizando Streamlit como interfaz de usuario y Docker para distribuir y ejecutar todo el sistema.

## 1.1. Objetivos

- Identificar personas a partir de fotografías.
- Reconocer las identidades registradas en la base de datos.
- Rechazar personas desconocidas, evitando identificaciones incorrectas.
- Combinar dos técnicas clásicas de reconocimiento facial: HOG y ORB.
- Utilizar MediaPipe para detectar y alinear las caras.
- Permitir registrar y eliminar personas.
- Almacenar los templates biométricos cifrados.
- Facilitar la ejecución del sistema mediante Docker, sin requerir instalaciones manuales de las dependencias de Python.
- Proporcionar una interfaz sencilla y multiplataforma.

## 1.2. Alcance

El sistema trabajará exclusivamente con fotografías estáticas.

No se contempla:
- Reconocimiento mediante cámara en tiempo real.
- Seguimiento de personas en vídeo.
- Reconocimiento de emociones.
- Detección de vida (*liveness detection*).
- Protección frente a ataques de presentación mediante fotografías impresas o pantallas.

El sistema no debe interpretarse como un mecanismo de autenticación de alta seguridad sin incorporar posteriormente controles adicionales.

---

# 2. Tecnologías

| Tecnología | Responsabilidad |
|---|---|
| Python | Lenguaje principal |
| MediaPipe | Detección facial y landmarks |
| OpenCV | Procesamiento de imágenes y ORB |
| NumPy | Operaciones numéricas |
| scikit-image | Extracción de características HOG |
| scikit-learn | Clasificador SVM |
| Streamlit | Interfaz de usuario |
| SQLite | Posible almacenamiento estructurado de identidades y templates cifrados |
| AES-256-GCM | Cifrado e integridad de los templates |
| Docker | Entorno de ejecución |
| Docker Compose | Configuración y lanzamiento de la aplicación |

La selección de SQLite es una propuesta para simplificar el almacenamiento local. Deberá confirmarse durante la implementación.

---

# 3. Arquitectura general

El sistema estará dividido en cinco bloques funcionales:

1. Interfaz de usuario.
2. Detección y preprocesamiento facial.
3. Reconocimiento mediante HOG + SVM.
4. Reconocimiento mediante ORB.
5. Gestión y almacenamiento de identidades.

## 3.1. Flujo general de reconocimiento

```text
                 FOTOGRAFÍA
                      |
                      v
               INTERFAZ STREAMLIT
                      |
                      v
              VALIDACIÓN DE IMAGEN
                      |
                      v
              MEDIAPIPE FACE DETECTOR
                      |
                      v
             LANDMARKS Y BOUNDING BOX
                      |
                      v
             ALINEAMIENTO Y RECORTE
                      |
                      v
             NORMALIZACIÓN FACIAL
                      |
                      v
                 EXTRACCIÓN HOG
                      |
                      v
                  SVM
                      |
               ¿Confianza >= T1?
                  /       \
                SÍ         NO
                |           |
                v           v
           IDENTIDAD      EXTRACCIÓN ORB
                            |
                            v
                    MATCHING CON BD
                            |
                     ¿Confianza >= T2?
                        /       \
                      SÍ         NO
                      |           |
                      v           v
                  IDENTIDAD   DESCONOCIDO
```

Donde:

- T1 es el umbral de aceptación de HOG + SVM.
- T2 es el umbral de aceptación de ORB.

Los dos umbrales deberán determinarse experimentalmente. No se fijarán arbitrariamente durante el diseño.

**Importante:** la primera etapa debe ser conservadora. Si HOG + SVM no tiene suficiente confianza, no se aceptará automáticamente su predicción, sino que se solicitará la segunda comprobación mediante ORB.

---

# 4. Detección y preprocesamiento facial

## 4.1. MediaPipe

Se utilizará MediaPipe para localizar las caras presentes en las fotografías.

El detector deberá proporcionar:

- Bounding box de cada cara.
- Puntos de referencia faciales (*landmarks*).
- Coordenadas de los ojos y otros puntos necesarios para el alineamiento.

Se propone utilizar MediaPipe Face Landmarker o una solución equivalente que proporcione suficientes landmarks para realizar el alineamiento.

La elección definitiva del modelo de MediaPipe y su API se realizará al comenzar la implementación.

## 4.2. Alineamiento

El objetivo es conseguir que las caras de entrada presenten una geometría similar.

Se utilizarán los landmarks de los ojos para estimar la inclinación de la cabeza y corregirla mediante una transformación geométrica.

Como primera aproximación, se utilizará una rotación que sitúe ambos ojos aproximadamente a la misma altura.

Si las pruebas lo justifican, podrá utilizarse una transformación afín basada en más puntos faciales.

## 4.3. Normalización

Después del alineamiento:

1. Recortar la región facial.
2. Redimensionar todas las caras a una resolución común.
3. Convertirlas a escala de grises.
4. Aplicar, si resulta necesario, técnicas moderadas de normalización de iluminación.

Se propone inicialmente una resolución de 128 × 128 píxeles. Este valor deberá validarse mediante pruebas, ya que puede afectar de forma diferente a HOG y ORB.

La normalización deberá conservar suficiente información de textura para no perjudicar los descriptores ORB.

## 4.4. Casos especiales

El detector deberá contemplar:

- Fotografías sin caras detectables.
- Fotografías con varias caras.
- Caras demasiado pequeñas.
- Imágenes dañadas o con formatos no compatibles.

La política de procesamiento de fotografías con varias caras deberá definirse en la implementación. Una posibilidad es procesar cada cara de forma independiente.

---

# 5. Reconocimiento mediante HOG + SVM

## 5.1. HOG

HOG (*Histogram of Oriented Gradients*) extraerá características globales de la cara normalizada.

El procedimiento será:

1. Recibir la cara normalizada.
2. Calcular los gradientes de intensidad.
3. Dividir la imagen en celdas.
4. Calcular histogramas de orientaciones.
5. Normalizar los bloques de histogramas.
6. Obtener un vector de características de longitud fija.

Este vector será la entrada del clasificador SVM.

## 5.2. SVM

El clasificador SVM se encargará de determinar qué identidad registrada resulta más compatible con el vector HOG recibido.

Se deberá estudiar una configuración multiclase adecuada para las identidades registradas.

El sistema no aceptará directamente la clase predicha por el SVM. Será necesario aplicar un criterio de rechazo.

Se utilizará una puntuación de decisión o un mecanismo de confianza definido durante las pruebas.

La decisión será:

- Puntuación superior o igual a T1: aceptar la identidad propuesta.
- Puntuación inferior a T1: solicitar una segunda comprobación mediante ORB.

No debe confundirse la puntuación de decisión del SVM con una probabilidad real de que la identidad sea correcta. Si se necesitan probabilidades calibradas, habrá que implementar y validar esa calibración.

## 5.3. Limitación del entrenamiento

El dataset disponible contiene:

- 50 personas diferentes.
- 50 fotografías.
- Una fotografía por persona.
- Ninguna identidad repetida.

Esto implica que cada identidad dispone de un único ejemplo de referencia.

Un SVM multiclase puede entrenarse con esos ejemplos, pero no dispondrá de suficientes muestras reales para aprender la variabilidad de cada persona.

Por tanto:

- No podrá aprender adecuadamente los cambios de iluminación, expresión o posición de cada identidad.
- La capacidad de generalización será limitada.
- Los umbrales no podrán calibrarse correctamente utilizando únicamente esas 50 fotografías.

Se podrán estudiar aumentos artificiales de datos, pero deberán identificarse como datos sintéticos y no considerarse equivalentes a nuevas fotografías reales.

La limitación deberá reflejarse en la memoria del proyecto.

---

# 6. Reconocimiento mediante ORB

## 6.1. Funcionamiento

ORB (*Oriented FAST and Rotated BRIEF*) extraerá características locales de la cara normalizada.

A diferencia de HOG, ORB no generará un único vector global de tamaño fijo, sino un conjunto de puntos clave y descriptores binarios.

El funcionamiento será:

1. Recibir la cara normalizada.
2. Detectar puntos clave.
3. Calcular sus descriptores.
4. Comparar los descriptores con los almacenados para cada identidad.
5. Evaluar la calidad y cantidad de coincidencias.
6. Determinar si existe una identidad suficientemente compatible.

## 6.2. Matching

Se propone utilizar `cv2.BFMatcher` con distancia Hamming, adecuada para los descriptores binarios de ORB.

Se estudiarán mecanismos de filtrado de coincidencias, como el ratio test o el matching con comprobación cruzada.

La puntuación de coincidencia podrá considerar:

- Número de coincidencias válidas.
- Proporción de coincidencias respecto a los descriptores disponibles.
- Distancia entre descriptores.
- Consistencia geométrica de las coincidencias, si resulta viable.

## 6.3. Umbral de aceptación

ORB actuará como segunda opinión y no aceptará una identidad simplemente porque tenga más coincidencias que las demás.

Será necesario establecer un umbral T2 suficientemente exigente.

Si ninguna identidad supera T2, se devolverá «persona desconocida».

El umbral deberá determinarse mediante pruebas con personas conocidas y desconocidas.

---

# 7. Base de datos y gestión de identidades

## 7.1. Organización

La base de datos contendrá la información necesaria para identificar a las personas registradas.

Cada identidad tendrá:

- Identificador único.
- Nombre o etiqueta identificativa.
- Template HOG.
- Descriptores ORB.
- Información auxiliar necesaria para reconstruir y utilizar los templates.

Se propone almacenar los templates cifrados en SQLite.

El formato definitivo de almacenamiento se establecerá durante la implementación.

## 7.2. Cifrado

Se utilizará AES-256-GCM para proteger los templates biométricos.

Este algoritmo proporciona:

- Confidencialidad.
- Autenticación del contenido cifrado.
- Detección de modificaciones no autorizadas en los datos cifrados.

Cada operación de cifrado deberá utilizar un nonce único.

La clave de cifrado no se almacenará dentro de la imagen Docker ni junto a los templates. Se proporcionará mediante un mecanismo externo, como un secreto o un archivo protegido montado en el contenedor.

La gestión y copia de seguridad de esa clave deberá documentarse, ya que perderla implicaría no poder recuperar los templates cifrados.

**Alcance del cifrado:** el requisito confirmado es cifrar los templates biométricos. No se ha establecido que las fotografías originales deban almacenarse ni que deban cifrarse.

## 7.3. Registro de personas

El procedimiento será:

1. Introducir una identidad.
2. Seleccionar una fotografía.
3. Detectar y procesar la cara mediante MediaPipe.
4. Extraer el vector HOG.
5. Extraer los descriptores ORB.
6. Cifrar ambos templates.
7. Almacenar los datos asociados a la identidad.
8. Actualizar el modelo o estructura de reconocimiento que utilice el sistema.

Se deberá impedir registrar una fotografía si no contiene una cara válida o si se produce un error durante el procesamiento.

La política para evitar identidades duplicadas deberá definirse durante la implementación.

## 7.4. Eliminación de personas

La aplicación permitirá seleccionar una identidad y eliminar su registro.

La eliminación deberá afectar a:

- Los templates HOG.
- Los descriptores ORB.
- Los metadatos asociados.
- Cualquier estructura de reconocimiento que dependa de esa identidad.

El sistema deberá mantener la coherencia entre la base de datos y el modelo SVM.

---

# 8. Interfaz de usuario

## 8.1. Tecnología

Se utilizará Streamlit.

La interfaz se ejecutará localmente y el usuario accederá a ella mediante su navegador web.

No será necesario desarrollar una aplicación de escritorio independiente.

## 8.2. Funcionalidades

Se proponen tres apartados principales:

**Reconocimiento**
- Subir una fotografía.
- Mostrar la imagen recibida.
- Mostrar la cara detectada.
- Ejecutar el reconocimiento.
- Mostrar la identidad obtenida o el resultado «desconocido».

**Registro**
- Introducir el nombre o identificador.
- Seleccionar la fotografía.
- Procesar la cara.
- Registrar los templates.

**Eliminación**
- Mostrar las identidades registradas.
- Seleccionar una identidad.
- Solicitar confirmación.
- Eliminar sus datos.

La interfaz no contendrá la lógica de reconocimiento. Se limitará a llamar a los módulos correspondientes.

---

# 9. Despliegue mediante Docker

## 9.1. Objetivo

Todo el sistema se ejecutará dentro de Docker.

El usuario no tendrá que instalar Python, MediaPipe, OpenCV ni las demás dependencias del proyecto.

El procesamiento se realizará en CPU.

## 9.2. Arquitectura de ejecución

```text
             ORDENADOR DEL USUARIO
                      |
                      v
                 DOCKER
                      |
             CONTENEDOR PRINCIPAL
                      |
              +-------+-------+
              |               |
              v               v
          STREAMLIT       MOTOR ML
              |               |
              +-------+-------+
                      |
                      v
             BASE DE DATOS LOCAL
                CIFRADA
```

La interfaz Streamlit estará disponible mediante un puerto local, por ejemplo, `8501`.

El contenedor tendrá acceso persistente a los datos mediante volúmenes.

## 9.3. Distribución

Se proporcionarán:

- `Dockerfile`.
- `docker-compose.yml`.
- `requirements.txt`.
- Scripts de inicio para Windows y Linux, si resultan necesarios.
- Instrucciones de instalación y ejecución.

El usuario deberá instalar Docker Desktop en Windows o Docker Engine en Linux.

El procedimiento de ejecución propuesto será:

```bash
docker compose up --build
```

Después podrá acceder a:

```text
http://localhost:8501
```

La configuración definitiva deberá comprobarse en ambos sistemas operativos.

---

# 10. Estructura propuesta del proyecto

```text
reconocimiento-facial/
│
├── app.py
│
├── recognition/
│   ├── detector.py
│   ├── preprocessing.py
│   ├── hog.py
│   ├── svm.py
│   ├── orb.py
│   └── recognition.py
│
├── database/
│   ├── manager.py
│   ├── encryption.py
│   └── storage/
│
├── training/
│   └── train.py
│
├── tests/
│   ├── test_detector.py
│   ├── test_hog.py
│   ├── test_orb.py
│   ├── test_database.py
│   └── test_recognition.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

La estructura podrá adaptarse durante el desarrollo, pero se intentará mantener la separación de responsabilidades.

---

# 11. Reparto de trabajo entre los cuatro integrantes

Se propone dividir el proyecto en cuatro bloques funcionales. Cada integrante tendrá un módulo principal y será responsable de sus pruebas, documentación e integración.

## Integrante 1: Detección y preprocesamiento facial

**Responsabilidades:**

- Integrar MediaPipe.
- Detectar las caras.
- Obtener los landmarks.
- Implementar el alineamiento facial.
- Recortar y normalizar las imágenes.
- Gestionar errores de detección.
- Definir la interfaz de entrada y salida del módulo.

**Entregables:**

- `detector.py`.
- `preprocessing.py`.
- Pruebas con distintas fotografías.
- Documentación de parámetros y funcionamiento.

**Criterio de finalización:** dado un archivo de imagen válido, el módulo debe devolver las caras detectadas y normalizadas, o informar de forma controlada de que no ha podido procesarlas.

## Integrante 2: HOG y SVM

**Responsabilidades:**

- Implementar la extracción de características HOG.
- Preparar el vector de entrada del SVM.
- Implementar el entrenamiento del clasificador.
- Gestionar las predicciones multiclase.
- Definir el mecanismo de puntuación y rechazo.
- Estudiar la limitación de una única fotografía por identidad.

**Entregables:**

- `hog.py`.
- `svm.py`.
- `training/train.py`.
- Pruebas del clasificador.
- Documentación del entrenamiento y los umbrales.

**Criterio de finalización:** dado un vector HOG, el módulo debe proporcionar la identidad candidata y la información necesaria para decidir si supera el umbral de aceptación.

## Integrante 3: ORB y matching

**Responsabilidades:**

- Implementar la extracción de puntos clave y descriptores ORB.
- Diseñar el almacenamiento lógico de los descriptores.
- Implementar el matching contra las identidades registradas.
- Estudiar los filtros de coincidencias.
- Definir la puntuación de matching.
- Determinar experimentalmente el umbral de aceptación.
- Implementar el rechazo cuando no existan coincidencias fiables.

**Entregables:**

- `orb.py`.
- Pruebas de matching.
- Documentación del cálculo de puntuaciones.
- Propuesta de umbral de aceptación.

**Criterio de finalización:** dado un conjunto de descriptores ORB y una base de referencias, el módulo debe devolver una identidad candidata o indicar que no encuentra ninguna coincidencia suficientemente fiable.

## Integrante 4: Base de datos, cifrado e interfaz

**Responsabilidades:**

- Diseñar la estructura de almacenamiento.
- Implementar SQLite, si se confirma esta opción.
- Implementar AES-256-GCM.
- Gestionar la clave de cifrado.
- Implementar el registro y eliminación de identidades.
- Desarrollar la interfaz Streamlit.
- Preparar Docker y Docker Compose.
- Elaborar las instrucciones de instalación y ejecución.

**Entregables:**

- `database/manager.py`.
- `database/encryption.py`.
- `app.py`.
- `Dockerfile`.
- `docker-compose.yml`.
- `README.md`.

**Criterio de finalización:** el usuario debe poder registrar, consultar y eliminar identidades desde la interfaz, y ejecutar el sistema mediante Docker.

## 11.1. Equilibrio del reparto

La división anterior asigna a cada integrante un bloque de responsabilidad independiente.

Para evitar que el cuarto integrante acumule demasiada carga, se establecen estas condiciones:

- El integrante 1 entregará el preprocesamiento completamente independiente de la interfaz.
- Los integrantes 2 y 3 implementarán sus algoritmos sin depender de la base de datos definitiva.
- El integrante 4 desarrollará inicialmente la interfaz utilizando datos de prueba y contratos de módulos provisionales.
- La integración final y las pruebas generales serán responsabilidad compartida.

El objetivo es que cada integrante tenga aproximadamente una cuarta parte del trabajo total. Las tareas de integración, documentación y evaluación se repartirán entre todos, en lugar de asignarlas exclusivamente a una persona.

---

# 12. Interfaces entre módulos

Para facilitar el trabajo en paralelo, se establecerán contratos comunes.

Se propone utilizar estas interfaces iniciales:

```python
# detector.py

def detect_and_align(image):
    """
    Entrada:
        Imagen original.

    Salida:
        Lista de caras normalizadas.
    """
```

```python
# hog.py

def extract_hog(face):
    """
    Entrada:
        Cara normalizada.

    Salida:
        Vector de características HOG.
    """
```

```python
# svm.py

def predict_identity(hog_features):
    """
    Entrada:
        Vector HOG.

    Salida:
        Identidad candidata y puntuación de decisión.
    """
```

```python
# orb.py

def extract_orb(face):
    """
    Entrada:
        Cara normalizada.

    Salida:
        Puntos clave y descriptores ORB.
    """

def match_identity(descriptors, database):
    """
    Entrada:
        Descriptores de consulta y referencias.

    Salida:
        Identidad candidata y puntuación de matching.
    """
```

```python
# recognition.py

def recognize(image):
    """
    Ejecuta la cadena completa:
        detección
        -> HOG + SVM
        -> ORB si es necesario
        -> resultado final
    """
```

```python
# manager.py

def register_person(person_id, name, image):
    pass

def delete_person(person_id):
    pass

def get_registered_people():
    pass

def load_templates():
    pass
```

Estas firmas son orientativas. El equipo deberá acordar los tipos de datos y estructuras de retorno antes de implementar los módulos.

---

# 13. Plan de pruebas y evaluación

La evaluación deberá contemplar tanto el reconocimiento de personas conocidas como el rechazo de personas desconocidas.

## 13.1. Pruebas funcionales

- Detección de una cara.
- Detección de varias caras.
- Imagen sin caras.
- Fotografía corrupta.
- Registro de una identidad.
- Eliminación de una identidad.
- Reconocimiento de una identidad registrada.
- Rechazo de una identidad desconocida.
- Fallo al descifrar un template.
- Persistencia de datos al reiniciar Docker.

## 13.2. Pruebas de reconocimiento

Se deberán utilizar fotografías de consulta independientes de las imágenes de referencia.

Los escenarios a estudiar serán:

- Cambios de iluminación.
- Variaciones de posición.
- Inclinación de la cabeza.
- Cambios moderados de expresión.
- Diferencias de resolución.
- Personas desconocidas.

## 13.3. Métricas

Se propone calcular:

**False Acceptance Rate (FAR):** proporción de intentos de personas desconocidas que son aceptados incorrectamente.

**False Rejection Rate (FRR):** proporción de intentos de personas registradas que son rechazados.

**Tasa de identificación correcta:** proporción de consultas de personas conocidas identificadas correctamente.

**Accuracy:** podrá utilizarse como métrica complementaria, pero no será suficiente por sí sola para evaluar un sistema abierto.

También se registrarán los resultados de HOG y ORB por separado y los del sistema combinado.

## 13.4. Limitaciones experimentales

Con una sola fotografía por persona no se puede construir una división convencional de entrenamiento y prueba por identidad que conserve ejemplos independientes de cada persona.

Por ello, será necesario conseguir fotografías de consulta adicionales:

- De personas registradas, para evaluar la identificación correcta.
- De personas no registradas, para evaluar la resistencia a falsas aceptaciones.

Si no se consiguen fotografías adicionales, las pruebas con transformaciones artificiales deberán presentarse como pruebas de robustez sintética, no como una evaluación equivalente a fotografías reales.

---

# 14. Planificación del desarrollo

Se propone organizar el trabajo en seis fases.

| Fase | Trabajo | Resultado |
|---|---|---|
| 1. Diseño | Definir contratos, estructura y dependencias | Arquitectura acordada |
| 2. Desarrollo paralelo | Implementar MediaPipe, HOG, ORB y almacenamiento/interfaz | Módulos independientes |
| 3. Integración | Conectar los módulos y construir la cascada | Sistema funcional |
| 4. Evaluación | Probar identidades conocidas y desconocidas | Métricas y umbrales |
| 5. Despliegue | Construir y probar Docker | Aplicación ejecutable |
| 6. Documentación | Memoria, instrucciones y resultados | Entrega final |

Las fases podrán solaparse. Por ejemplo, la interfaz y Docker pueden desarrollarse con datos simulados mientras se implementan los algoritmos.

---

# 15. Decisiones pendientes y riesgos

Los siguientes puntos no deben considerarse cerrados hasta que el equipo los valide:

1. **Entrenamiento SVM con una muestra por identidad:** hay que comprobar si el planteamiento ofrece resultados útiles y estudiar alternativas compatibles con el requisito HOG + SVM.
2. **Calibración de umbrales:** hacen falta fotografías de consulta independientes para determinar T1 y T2 de forma experimental.
3. **Almacenamiento:** confirmar SQLite y el formato exacto de los templates.
4. **Clave de cifrado:** definir cómo se genera, proporciona, protege y respalda.
5. **Fotografías originales:** decidir si se conservarán después de extraer los templates.
6. **Múltiples caras:** decidir si se procesarán todas o si se exigirá una sola cara por fotografía.
7. **Identidades duplicadas:** definir cómo se evitarán registros repetidos.
8. **Actualización del SVM:** establecer cómo se reconstruye o actualiza el clasificador cuando se registra o elimina una persona.
9. **Compatibilidad:** comprobar versiones compatibles de MediaPipe, OpenCV, scikit-learn y Python dentro de la imagen Docker.
10. **Seguridad:** el cifrado de templates no protege por sí solo la clave, la interfaz, las fotografías originales ni el acceso al sistema.

---

# 16. Resultado esperado

Al finalizar el proyecto se espera disponer de una aplicación local que permita:

- Introducir una fotografía desde una interfaz web.
- Detectar y normalizar las caras mediante MediaPipe.
- Intentar identificar a cada persona mediante HOG + SVM.
- Utilizar ORB como segunda comprobación cuando HOG no alcance su umbral.
- Rechazar personas desconocidas.
- Registrar nuevas identidades.
- Eliminar identidades existentes.
- Almacenar los templates cifrados.
- Ejecutar el conjunto de la aplicación mediante Docker en Windows y Linux.

La arquitectura deberá mantener separados el preprocesamiento, los algoritmos de reconocimiento, la persistencia de datos y la interfaz, facilitando el desarrollo paralelo y las futuras modificaciones.

**Principio de diseño:** cada módulo debe poder probarse de forma independiente antes de integrarse en la aplicación completa.

## 17. Reparto de tareas entre los integrantes del equipo

El desarrollo se dividirá en cuatro bloques de trabajo, asignando a cada integrante la responsabilidad principal de un módulo. Aunque cada persona tendrá unas tareas específicas, la integración, las pruebas finales y la documentación se realizarán de forma conjunta para equilibrar la carga de trabajo.

### Integrante 1: Detección y preprocesamiento facial

**Objetivo:** desarrollar el módulo encargado de localizar los rostros en las imágenes y prepararlos para su posterior reconocimiento.

Tareas:
- Integrar MediaPipe para la detección facial.
- Obtener los puntos de referencia faciales necesarios para la alineación.
- Implementar la alineación del rostro utilizando los puntos de referencia.
- Recortar y normalizar las imágenes faciales.
- Gestionar los errores de detección, como imágenes sin rostros o con varios rostros.
- Realizar pruebas con diferentes condiciones de iluminación, posición y orientación.

**Entregables:**
- `detector.py`: detección facial y obtención de puntos de referencia.
- `preprocessing.py`: alineación, recorte y normalización.
- Pruebas del módulo.

### Integrante 2: Reconocimiento mediante HOG y SVM

**Objetivo:** implementar el primer sistema de reconocimiento facial, basado en características HOG y un clasificador SVM.

Tareas:
- Implementar la extracción de características mediante HOG.
- Desarrollar el clasificador SVM para identificar candidatos.
- Implementar el cálculo y la interpretación de las puntuaciones de decisión.
- Establecer un umbral conservador de aceptación (T1) para evitar identificaciones incorrectas.
- Investigar la viabilidad del entrenamiento con una única imagen de referencia por persona.
- Evaluar el rendimiento del sistema con personas registradas y desconocidas.

**Entregables:**
- `hog.py`: extracción de características HOG.
- `svm.py`: implementación del clasificador.
- `train.py`: entrenamiento y preparación del modelo.
- Pruebas y resultados de evaluación.

### Integrante 3: Reconocimiento mediante ORB

**Objetivo:** implementar el segundo sistema de reconocimiento, basado en características locales ORB, que actuará como segunda opinión cuando el primer método no alcance el umbral establecido.

Tareas:
- Implementar la extracción de puntos clave y descriptores mediante ORB.
- Desarrollar la comparación de descriptores entre la imagen consultada y las referencias almacenadas.
- Implementar el emparejamiento de características mediante BFMatcher y distancia de Hamming.
- Diseñar una puntuación que permita valorar la fiabilidad de las coincidencias.
- Establecer un umbral de aceptación (T2).
- Evaluar el comportamiento del método ante variaciones de iluminación, orientación y calidad de imagen.

**Entregables:**
- `orb.py`: extracción de características ORB.
- `matching.py`: comparación y emparejamiento de descriptores.
- Pruebas y resultados de evaluación.

### Integrante 4: Aplicación, base de datos y seguridad

**Objetivo:** desarrollar la parte de gestión de usuarios y la interfaz que permitirá utilizar el sistema.

Tareas:
- Diseñar e implementar la estructura de almacenamiento de la base de datos.
- Implementar el cifrado de las plantillas biométricas mediante AES-256-GCM.
- Gestionar de forma segura la clave de cifrado, manteniéndola fuera de la imagen Docker y separada de los datos cifrados.
- Desarrollar las funciones para registrar y eliminar personas.
- Implementar la interfaz gráfica mediante Streamlit.
- Permitir la carga de imágenes para realizar reconocimientos.
- Preparar Docker y Docker Compose para facilitar la ejecución del proyecto en Windows y Linux.
- Elaborar las instrucciones de instalación y uso.

**Entregables:**
- `manager.py`: gestión de la base de datos y de los registros.
- `encryption.py`: cifrado y descifrado de las plantillas.
- `app.py`: interfaz de usuario.
- `Dockerfile` y `docker-compose.yml`.
- `README.md` con las instrucciones de ejecución.

---

### Tareas compartidas por todo el equipo

Además de las responsabilidades individuales, todos los integrantes participarán en las siguientes actividades:

1. **Definición de interfaces:** acordar desde el principio los formatos de entrada y salida de cada módulo para facilitar su integración.
2. **Integración:** conectar la detección y el preprocesamiento facial con los dos métodos de reconocimiento y la aplicación.
3. **Pruebas del sistema completo:** comprobar el funcionamiento de extremo a extremo, incluyendo registro, reconocimiento, rechazo de personas desconocidas y eliminación de registros.
4. **Evaluación:** analizar conjuntamente las métricas FAR, FRR y tasa de identificación correcta, así como ajustar los umbrales de aceptación.
5. **Documentación y presentación:** elaborar la memoria del proyecto y preparar la demostración final.

### Orden de desarrollo recomendado

| Fase | Actividad | Responsables |
|---|---|---|
| 1 | Definición de interfaces y formatos de datos | Todos |
| 2 | Desarrollo independiente de los módulos | Cada integrante |
| 3 | Integración de los módulos | Todos |
| 4 | Pruebas y evaluación individual de los métodos | Integrantes 1, 2 y 3, con apoyo del 4 |
| 5