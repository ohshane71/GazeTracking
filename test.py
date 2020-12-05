import cv2
from face_tracking.head_pose_estimation import FaceTracking
from face_tracking.head_pose_estimation import HeadPoseEstimation

# tracker = FaceTracking()
tracker = HeadPoseEstimation()
webcam = cv2.VideoCapture(0)

width  = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    tracker.refresh(frame, width, height)
    frame = tracker.annotated_frame()

    cv2.imshow("Demo", frame)

    print("------")
    print(tracker.number_of_faces)
    print(tracker.faces_points)
    print(tracker.landmarks_points)
    print(tracker.p1, tracker.p2)

    print(cv2.CAP_PROP_FPS)

    if cv2.waitKey(1) == 27:
        break
