import numpy as np
from scipy.spatial.transform import Rotation


def rotation_difference(rotation1, rotation2):
    # Cortana: 3x3 Rotationsmatrix von Cortana
    # otherRotation: 3x3 andere Rotationsmatrix
    # angle: berechneter Winkel in Grad

    # Rotationen in Quaternionen umwandeln
    q1 = Rotation.from_dcm(rotation1).as_quat()
    q2 = Rotation.from_dcm(rotation2).as_quat()

    # Quaternionen normalisieren (sollte bereits der Fall sein, aber sicherheitshalber)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Winkel zwischen den Quaternionen berechnen
    dot_product = np.dot(q1, q2)
    angle_cos = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Winkel in Grad umwandeln
    angle = np.degrees(np.abs(angle_cos))
    return angle
