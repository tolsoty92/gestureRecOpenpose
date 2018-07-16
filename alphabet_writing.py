import cv2
from Detectors.Gestures import GestureRec
import numpy as np

stream = cv2.VideoCapture(0)
op = GestureRec()
classifier = op.load_classifier()

RUN = True
counter = 1
bb = [0, 100, 450, 350]

save = False

while RUN and counter <= 500:
    print(save)
    ret, img = stream.read()
    if not ret:
        RUN = not RUN
    else:
        hand, k_points, img = op.get_hand_skeleton(img, bb)
        if len(k_points) > 0:
            if save:
                f_name = '/home/user/Documents/Gesture_recognition/signs/y/R'+str(counter)+'.npy'


                print(np.array(k_points))
                np.save(f_name, np.array(k_points))
                counter += 1
        cv2.rectangle(img, (bb[0], bb[1]), ((bb[0] + bb[2], bb[1]+bb[3])), (0, 0, 255),6)
        cv2.putText(img, 'Counter = %i' % counter, (20, 30), 0, 0.8, (0, 0, 255), 4)
        cv2.imshow('video', img)
        if cv2.waitKey(10) & 0xFF == 27:
            RUN = not RUN
        elif cv2.waitKey(10) & 0xFF == ord('s'):
            save = not save