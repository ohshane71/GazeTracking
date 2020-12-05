import cv2
import cheatah

ft = cheatah.FaceTracking()
gt = cheatah.GazeTracking()
hpe = cheatah.HeadPoseEstimation()

webcam = cv2.VideoCapture(0)

width  = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    ft.refresh(frame, width, height)
    gt.refresh(frame, width, height)
    hpe.refresh(frame, width, height)

    frame = ft.annotated_frame()
    frame = gt.annotated_frame()
    frame = hpe.annotated_frame()

    cv2.imshow("Demo", frame)

    print("------")
    print(ft.number_of_faces)
    print(gt.faces_points)
    print(gt.faces)
    print(gt.landmarks_points)
    print(hpe.p1, hpe.p2)
    print(gt.horizontal_ratio(), gt.vertical_ratio())

    if cv2.waitKey(1) == 27:
        break
