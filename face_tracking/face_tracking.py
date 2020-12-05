from __future__ import division
import os
import cv2
import dlib

class FaceTracking(object):
    def __init__(self):
        self.frame = None
        self.width = 0
        self.height = 0
        self.faces = []
        self.landmarks = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def face_exists(self):
        return len(self.faces) > 0

    @property
    def number_of_faces(self):
        return len(self.faces)

    @property
    def faces_points(self):
        faces_points = []
        if self.face_exists:
            for face in self.faces:
                face_points = []
                face_points.append( (face.left(), face.top()) )
                face_points.append( (face.right(), face.top()) )
                face_points.append( (face.right(), face.bottom()) )
                face_points.append( (face.left(), face.bottom()) )
                faces_points.append(face_points)
        return faces_points

    @property
    def landmarks_points(self):
        """
        Array of 68 length
        throws exception
        """
        if self.face_exists:
            return self.landmarks.parts()
        else:
            return None

    def _analyze(self):
        """Detects the face and initialize face objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self._face_detector(frame)
        
        if self.face_exists:
            self.landmarks = self._predictor(frame, self.faces[0])

    def refresh(self, frame, width, height):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self.width = width
        self.height = height
        self._analyze()

    def annotated_frame(self):
        """Returns the main frame with face highlighted"""
        color = (0, 255, 0)
        if self.face_exists:
            for face in self.faces:
                cv2.line(self.frame, (face.left(), face.top()), (face.right(), face.top()), color)
                cv2.line(self.frame, (face.left(), face.bottom()), (face.right(), face.bottom()), color)
                cv2.line(self.frame, (face.left(), face.top()), (face.left(), face.bottom()), color)
                cv2.line(self.frame, (face.right(), face.top()), (face.right(), face.bottom()), color)

        return self.frame
