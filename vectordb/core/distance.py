import numpy as np


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l2_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix - query, axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
    return float(1.0 - np.dot(a, b) / denom)


def cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query) or 1e-10
    m_norm = np.linalg.norm(matrix, axis=1)
    m_norm[m_norm == 0] = 1e-10
    return 1.0 - (matrix @ query) / (m_norm * q_norm)


DISTANCE_FNS = {"l2": l2, "cosine": cosine}
DISTANCE_BATCH_FNS = {"l2": l2_batch, "cosine": cosine_batch}
