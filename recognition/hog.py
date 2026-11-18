"""Extracción HOG de caras normalizadas (paso anterior)"""
import numpy as np
from skimage.feature import hog

FACE_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
N_FEATURES = 8100


def extract_hog(face, visualize=False):
    """Devuelve un vector HOG; con visualize=True, también su representación."""
    if not isinstance(face, np.ndarray) or face.shape != (128, 128) or face.dtype != np.uint8:
        raise ValueError('Se requiere una cara gris uint8 de 128x128.')
    result = hog(face, visualize=visualize, **HOG_PARAMS)
    if visualize:
        features, image = result
        return features.astype(np.float32), image
    return result.astype(np.float32)