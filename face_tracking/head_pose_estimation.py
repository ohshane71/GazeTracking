from __future__ import division
import os
import cv2
import dlib
import numpy as np

from .face_tracking import FaceTracking

class HeadPoseEstimation(FaceTracking):
    POINTS = [
        30,     # Nose tip
        8,      # Chin
        36,     # Left eye left corner
        45,     # Right eye right corner
        48,     # Left Mouth corner
        54      # Right mouth corner
    ]

    def __init__(self):
        super().__init__()
        self.selected_landmarks = None
        self.p1 = (0, 0)
        self.p2 = (0, 0)

    def _analyze(self):
        super()._analyze()
        if self.landmarks is not None and self.landmarks_points is not None:
            self.selected_landmarks = [(self.landmarks_points[i].x, self.landmarks_points[i].y) for i in HeadPoseEstimation.POINTS]

            image_points = np.array(self.selected_landmarks, dtype="double")
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corne
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            focal_length = self.width
            center = (self.width/2, self.height/2)

            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], 
                dtype = "double")

            dist_coeffs = np.zeros((4,1))

            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.cv2.SOLVEPNP_ITERATIVE)

            (nose_end_point2D, jacobian) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs)

            self.p1 = (int(image_points[0][0]), int(image_points[0][1]))
            self.p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
        else:
            pass

    def annotated_frame(self):
        super().annotated_frame()

        if self.selected_landmarks is not None:
            for landmark in self.selected_landmarks:
                cv2.circle(self.frame, (int(landmark[0]), int(landmark[1])), 3, (0,0,255), -1)
        else:
            pass

        cv2.line(self.frame, self.p1, self.p2, (255,0,0), 2)
        return self.frame