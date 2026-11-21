"""Excepciones del módulo de detección y preprocesamiento facial."""


class FaceProcessingError(Exception):
    """Error controlado al procesar una imagen. La interfaz puede mostrar str(e)."""


class InvalidImageError(FaceProcessingError):
    """Imagen inexistente, corrupta, vacía, de formato no soportado o demasiado pequeña."""


class NoFaceDetectedError(FaceProcessingError):
    """No se ha detectado ninguna cara en la imagen."""


class FaceTooSmallError(FaceProcessingError):
    """Se detectaron caras, pero todas por debajo del tamaño mínimo."""


class ModelNotFoundError(RuntimeError):
    """Falta el modelo de MediaPipe (problema de configuración, no de la imagen)."""

class MultipleFacesError(FaceProcessingError):
    """La imagen contiene más de una cara; se exige exactamente una."""

    def __init__(self, count: int):
        self.count = count
        super().__init__(
            f"Se han detectado {count} caras en la imagen. "
            "Sube una fotografía en la que aparezca una sola persona."
        )

class SvmUnavailableError(RuntimeError):
    """No hay un modelo SVM utilizable (no entrenado, ausente o ilegible)."""


class StorageError(RuntimeError):
    """Fallo de persistencia (base de datos o modelo en disco)."""


class TemplateCorruptError(StorageError):
    """Un template almacenado no se puede descifrar o interpretar."""


class KeyConfigurationError(RuntimeError):
    """Clave de cifrado ausente, mal formada o incorrecta para los datos."""


class DuplicateIdentityError(ValueError):
    """Ya existe una identidad con ese nombre."""


class IdentityNotFoundError(LookupError):
    """La identidad solicitada no está registrada."""
