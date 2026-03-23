import cv2
import mediapipe as mp
import numpy as np

def calc_angle_2d(a,b,c):
    angle_ba = np.array([a.x - b.x, a.y - b.y])
    angle_bc = np.array([c.x - b.x, c.y - b.y])
    magnitude_ba = np.linalg.norm(angle_ba)
    magnitude_bc = np.linalg.norm(angle_bc)
    cosine_angle = np.dot(angle_ba, angle_bc) / (magnitude_ba * magnitude_bc)

    # Constraint the angle to
    bounded_angle = np.clip(cosine_angle, 0.0, 1.0)

    # Convert decimal to radians
    rads = np.arccos(bounded_angle)

    # Convert radians to degrees
    degrees = np.degrees(rads)

    return degrees