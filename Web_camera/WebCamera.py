import cv2
from threading import Thread


class VideoStream:
    def __init__(self, width=180, height=135, camera_num=0):
        # Initialize the video camera stream
        self.stream = cv2.VideoCapture(camera_num)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_img(self):
        ret, img = self.stream.read()
        return ret, img

    def stop(self):
        self.stream.release()

