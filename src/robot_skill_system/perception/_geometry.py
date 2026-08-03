"""Small fixed geometry helpers for local perception adapters."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def rotation_matrix_to_quaternion_xyzw(
    matrix: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""

    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3)")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = float(matrix[2, 1] - matrix[1, 2]) / scale
        y = float(matrix[0, 2] - matrix[2, 0]) / scale
        z = float(matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + float(matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            w = float(matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = float(matrix[0, 1] + matrix[1, 0]) / scale
            z = float(matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + float(matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            w = float(matrix[0, 2] - matrix[2, 0]) / scale
            x = float(matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = float(matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + float(matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            w = float(matrix[1, 0] - matrix[0, 1]) / scale
            x = float(matrix[0, 2] + matrix[2, 0]) / scale
            y = float(matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]
