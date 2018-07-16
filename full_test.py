import cv2
import joblib
import string
from sklearn.neighbors import KNeighborsClassifier
from time import time
from Detectors.Gestures import GestureRec
from Detectors.Hands import HandDetector
from Web_camera.WebCamera import VideoStream


if __name__ == '__main__':

    knn = joblib.load('alph_knn20.sav')

    width, height = 640, 480
    stream = VideoStream(width=width, height=height, camera_num=0)
    RUN = True
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    detector = HandDetector()
    op = GestureRec()

    classifier = op.load_classifier()

    TF = True
    OP = False
    k_points = []
    op_box = None

    alphabet = {}
    for i in range(26):
        alphabet[i] = string.ascii_lowercase[i]

    while RUN:
        gesture = ''
        t = time()
        ret, img = stream.get_img()
        if ret:
            if TF:
                actual_boxes = detector.detect_hands(img)
                if len(actual_boxes) > 0:
                    box = actual_boxes[0]
                   # cv2.rectangle(img, (box[0], box[1]), (box[i], box[3]), (255, 0, 0), 6)
                    x, y, d1, d2 = [box[0]-int(abs(box[2] - box[0])*0.2),
                              box[1]-int(abs(box[3] - box[1])*0.2),
                              int(abs(box[2] - box[0])*1.4),
                              int(abs(box[3]-box[1])*1.4)]
                    #print(d1, d2)
                    x, y, d1, d2 = x-10, y-10, d1+20, d2+20
                    op_box = [x, y, d1, d2]
                    cv2.rectangle(img, (x, y),(x+d1, y+d2), (0, 0, 255), 5)
                    k_points, img = op.get_hand_skeleton(img, op_box)
                    if len(k_points) > 0:
                        op_box = op.compute_BB(k_points)[1]
                        TF = not TF
            else:
                k_points, img = op.get_hand_skeleton(img, op_box)
                if len(k_points) > 0:
                    op_box = op.compute_BB(k_points)[1]
                    distance = op.compute_distanse20(k_points)
                    gesture = knn.predict(distance)[0]
                else:
                    TF = not TF

        t = time() - t
        fps = 1.0 / t

        cv2.putText(img, 'FPS = %f' % fps, (20, 20), 0, 0.5, (0, 0, 255))
        cv2.putText(img, 'Gesture = %s' % gesture, (20, 55), 0, 1, (255, 0, 0), thickness=6)
        cv2.imshow('Video', img)

        if cv2.waitKey(10) & 0xFF == 27:
            RUN = not RUN

    cv2.destroyAllWindows()
    stream.stop()