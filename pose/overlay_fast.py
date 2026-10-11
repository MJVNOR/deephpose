"""Opaque XYZ axes without full-image masks or RGB/BGR round trips."""
import cv2
import numpy as np


def draw_opaque_axes(rgb, pose, K, scale=0.1, thickness=3):
    def project(point):
        point = np.asarray(point)
        point = (pose @ point.reshape(4, 1)).reshape(-1)[:3]
        point = K @ point.reshape(3, 1)
        return tuple(np.round(point[:2] / point[2]).astype(int).reshape(-1))
    origin = project([0, 0, 0, 1])
    # Same order, integer endpoints, antialiasing and zero-length arrow tips as Utils.
    for point, color in [([scale, 0, 0, 1], (255, 0, 0)),
                         ([0, scale, 0, 1], (0, 255, 0)),
                         ([0, 0, scale, 1], (0, 0, 255))]:
        cv2.arrowedLine(rgb, origin, project(point), color, thickness,
                        line_type=cv2.LINE_AA, tipLength=0)
    return rgb
