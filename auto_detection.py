import PyOpenPose as OP
import cv2
import tensorflow as tf
import numpy as np
import os
import time
from utils.web_camera import VideoStream
from utils.hands_detect import HandsDetector
from object_detection.utils import label_map_util
import joblib
from math import sqrt


RUN = True

PATH_TO_CKPT = os.path.join('./data', 'faster_rcnn.pb')
PATH_TO_LABELS = os.path.join('./data', 'label_map.pbtxt')
PATH_TO_CLASSIFIER = os.path.join('./data', 'g_classifier/g_c2')

NUM_CLASSES = 5
OPENPOSE_ROOT = "/home/user/openpose"

classifier = joblib.load(PATH_TO_CLASSIFIER)


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detector = HandsDetector()

# Init graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Init camera
width, height = 640, 480
stream = cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Init PyOpenPose
download_heatmaps = False
with_hands = True
with_face = False
op = OP.OpenPose((640, 480), (368, 368), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                 download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)

# Init params for main loop
tensor = True
compBB = False
left_hand_BB = [0, 0, 0, 0]
right_hand_BB = [0, 0, 0, 0]

#arr_count = 0   # Counter to create dataset

labels = {0: 'rock', 1: 'palm', 2: 'fist', 3: '1 finger', 4: 'i fingers'}

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

# Main loop
while RUN:

    #print(tensor, compBB)

    t = time.time()     # Init time for fps

    # Get image fro camera
    ret, img = stream.read()
    img = cv2.resize(img, (width, height))
    imgSize = img.shape
    outSize = imgSize[1::-1]

    gesture_class = None

    # Looking for hands
    if tensor:
        # get data from hand detection network
        boxed_img, class_name, left_box, boxes_lst = detector.detect_objects(img.copy(),
                                                            sess, detection_graph, category_index)

        actual_fps = 0
        paused = False
        delay = {True: 0, False: 1}

        # If hand was found
        if len(boxes_lst) > 0:
            print(len(boxes_lst))
            boxes = detector.searching_area_by_cnts(boxes_lst, width, height)
            #print(boxes)
            left_box = boxes[0]     # Get the first box for hand
            left_hand_BB = [left_box[0], left_box[1], left_box[2] - left_box[0], left_box[3] - left_box[1]]
            cv2.circle(img, (left_box[0], left_box[1]), 15, (0, 255, 0))

            boxes = detector.searching_area_by_cnts(boxes_lst, width, height)

            op.detectHands(img, np.array(left_hand_BB + [0, 0, 0, 0], dtype=np.int32).reshape((1, 8)))
            leftHand = op.getKeypoints(op.KeypointType.HAND)[0].reshape(-1, 3)
            score, newHandBB = detector.compute_BB(leftHand)
            if score > 0.3:
                left_hand_BB = newHandBB
                tensor = not tensor
                compBB = not compBB
            #img = op.render(img)

    if compBB:
        #rightHand = [50, 180, 250, 380]
        #rightHand = [0, 0, 0, 0]
        #cv2.rectangle(img, (rightHand[0], rightHand[1]), (rightHand[i], rightHand[3]), (0, 255, 255))
        op.detectHands(img, np.array(left_hand_BB + [0, 0, 0, 0], dtype=np.int32).reshape((1, 8)))
        leftHand = op.getKeypoints(op.KeypointType.HAND)[0].reshape(-1, 3)
        #rightHand = op.getKeypoints(op.KeypointType.HAND)[1].reshape(-1, 3)
        #print(rightHand)
        score, newHandBB = detector.compute_BB(leftHand)
        cv2.rectangle(img, (left_hand_BB[0], left_hand_BB[1]), (left_hand_BB[2] + left_hand_BB[0], left_hand_BB[3] + left_hand_BB[1]), (0, 255, 255))
        if score > 0.5:
            img = op.render(img)
            left_hand_BB = newHandBB
            l_distance = compute_distanse(leftHand)
            gesture_class = classifier.predict(l_distance)[0]
        else:
            tensor = not tensor
            compBB = not compBB

    t = time.time() - t
    op_fps = 1.0 / t

    cv2.putText(img, str(op_fps), (20, 20), 0, 0.5, (0, 0, 255))
    if gesture_class != None:
         cv2.putText(img, labels[gesture_class], (520, 20), 0, 0.5, (0, 0, 255))

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == 27:
        RUN = not RUN

sess.close()
cv2.destroyAllWindows()
stream.release()
