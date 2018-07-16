import cv2
from Detectors.Gestures import GestureRec

from Web_camera.WebCamera import   VideoStream

stream = cv2.VideoCapture(0)
op = GestureRec()
classifier = op.load_classifier()

RUN = True

bb = [0, 100, 450, 350]

while RUN:
    ret, img = stream.read()
    if not ret:
        RUN = not RUN
    else:
        hand, k_points, img = op.get_hand_skeleton(img, bb)
        cv2.rectangle(img, (bb[0], bb[1]), ((bb[0] + bb[2], bb[1]+bb[3])), (0, 0, 255),6)
        cv2.imshow('video', img)
        if cv2.waitKey(10) & 0xFF == 27:
            RUN = not RUN
