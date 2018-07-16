import PyOpenPose as OP
import os
import joblib
from math import sqrt
import numpy as np
import cv2


class GestureRec():
    """Class to getting hand scelet and gesture rcognition."""
    # Init OenPose neural network
    def __init__(self, input_im_size=(640, 480), net_res=(240, 240),
                                    output_im_size=(640, 480), heatmaps=False,
                                                                    face=False, hands=True):
        OPENPOSE_ROOT= "/home/user/openpose"
        self.op = OP.OpenPose((640, 480), net_res, output_im_size,
                              "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                              heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, face, hands)

    def load_classifier(self, path_to_classifier='/home/user/Documents/Gesture_recognition/data/g_classifier/g_c2'):
        classifier = joblib.load(path_to_classifier)
        return classifier

    def compute_BB(self, hand, padding=1.5):
        """Compute new boundong box to founded hand
            It returns score and box"""
        minX = np.min(hand[:, 0])
        minY = np.min(hand[:, 1])

        maxX = np.max(hand[:, 0])
        maxY = np.max(hand[:, 1])

        width = maxX - minX
        height = maxY - minY

        cx = minX + width / 2
        cy = minY + height / 2

        width = height = max(width, height) * padding

        minX = cx - width / 2
        minY = cy - height / 2

        score = np.mean(hand[:, 2])

        if minX > 15:
            minX -= 15
        else:
            minX = 0

        if minY > 15:
            minY -= 15
        else:
            minY = 0
        return score, [int(minX), int(minY), int(width) + 30, int(height) + 30]

    def what_hand(self, img, box):
        self.op.detectHands(img, np.array(box + box, dtype=np.int32).reshape((1, 8)))
        leftHand = self.op.getKeypoints(self.op.KeypointType.HAND)[0].reshape(-1, 3)
        rightHand = self.op.getKeypoints(self.op.KeypointType.HAND)[1].reshape(-1, 3)
        scoreL, newHandBBL = self.compute_BB(leftHand)
        scoreR, newHandBBR = self.compute_BB(rightHand)
        if scoreL >= scoreR:
            hand = 'Left'
            return scoreL, hand, leftHand
        else:
            hand = 'Right'
            return scoreR, hand, rightHand

    def left_hand_skeleton(self, img, box):
        self.op.detectHands(img, np.array(box + [0, 0, 0, 0] , dtype=np.int32).reshape((1, 8)))
        hand = self.op.getKeypoints(self.op.KeypointType.HAND)[0].reshape(-1, 3)
        score, newHandBB = self.compute_BB(hand)
        k_points = []
        rendered_img = img
        if score > 0.5:
            k_points = hand
            rendered_img = self.op.render(img)
            return  k_points, rendered_img
        else:
            return [], img

    def right_hand_skeleton(self, img, box):
        self.op.detectHands(img, np.array([0, 0, 0, 0] + box, dtype=np.int32).reshape((1, 8)))
        hand = self.op.getKeypoints(self.op.KeypointType.HAND)[1].reshape(-1, 3)
        score, newHandBB = self.compute_BB(hand)
        k_points = []
        rendered_img = img
        if score > 0.5:
            k_points = hand
            rendered_img = self.op.render(img)
            return k_points, rendered_img
        else:
            return [], img


    def get_hand_skeleton(self, img, box):
        """ Double detection """
        self.op.detectHands(img, np.array(box + box, dtype=np.int32).reshape((1, 8)))
        leftHand = self.op.getKeypoints(self.op.KeypointType.HAND)[0].reshape(-1, 3)
        rightHand = self.op.getKeypoints(self.op.KeypointType.HAND)[1].reshape(-1, 3)
        scoreL, newHandBBL = self.compute_BB(leftHand)
        scoreR, newHandBBR =self.compute_BB(rightHand)
        k_points = []
        rendered_img = img
        if scoreL > scoreR:
            if scoreL > 0.5:
                k_points = leftHand
                #self.op.detectHands(img, np.array(box + [0, 0, 0, 0], dtype=np.int32).reshape((1, 8)))
                rendered_img = self.op.render(img)
        else:
            if scoreR > 0.5:
                k_points = rightHand
                #self.op.detectHands(img, np.array([0, 0, 0, 0] + box, dtype=np.int32).reshape((1, 8)))
                rendered_img = self.op.render(img)
        return    k_points, rendered_img

    def gesture_classification(self, k_points, classifier):
        if len(k_points):
            distace = self.compute_distanse(k_points)
            gesture = classifier.predict(distace)[0]
            labels = {0: 'rock', 1: 'palm', 2: 'fist', 3: '1 finger', 4: 'i fingers'}
            return labels[gesture]
        else:
            return  None

    @staticmethod
    def compute_distanse(hand):
        x = []
        y = []
        width = []
        for k in hand:
            x.append(k[0])
            y.append(k[1])
        for i in range(1, 21):
            width.append(sqrt((x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2))
        width = np.array(width)
        return width.reshape(1, 20)

    @staticmethod
    def compute_distanse20(hand):
        x = []
        y = []
        width = []
        for k in hand:
            x.append(k[0])
            y.append(k[1])
        for i in range(1, 21):
            width.append(sqrt((x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2))
        width.append(sqrt((x[8] - x[4]) ** 2 + (y[8] - y[4]) ** 2))
        width.append(sqrt((x[12] - x[8]) ** 2 + (y[12] - y[8]) ** 2))
        width.append(sqrt((x[16] - x[12]) ** 2 + (y[16] - y[12]) ** 2))
        width.append(sqrt((x[20] - x[16]) ** 2 + (y[20] - y[16]) ** 2))
        width = np.array(width)
        return width.reshape(1, 24)


