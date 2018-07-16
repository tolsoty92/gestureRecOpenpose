import cv2
import time
from Detectors.Hands import HandDetector
from Web_camera.WebCamera import   VideoStream

if __name__ == '__main__':

    print('\tStart!')

    width, height = 640, 480
    detector = HandDetector()
    stream = VideoStream(width=width, height=height, camera_num=0)
    RUN = True

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while RUN:
        t = time.time()
        ret, img = stream.get_img()
        if ret:
            actual_boxes = detector.detect_hands(img)

            for box in actual_boxes:
                detector.draw_box(img, box)

            t = time.time() - t
            fps = 1.0 / t

            cv2.putText(img, 'FPS = %f' % fps, (20, 20), 0, 0.5, (0, 0, 255))
            cv2.imshow('Video', img)

            if cv2.waitKey(1) & 0xFF == 27:
                RUN = not RUN
        else:
            RUN = not RUN

    stream.stop()
    detector.stop()
    cv2.destroyAllWindows()