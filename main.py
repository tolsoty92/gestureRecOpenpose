import os
import cv2
import multiprocessing
#import numpy as np
import tensorflow as tf
import time
from Detectors import Hands

from utils.web_camera import VideoStream
#from utils.mouse_control import Mouse
from utils.hands_detect import HandsDetector
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('./','faster_rcnn.pb')
PATH_TO_LABELS = os.path.join('./','label_map.pbtxt')

NUM_CLASSES = 5

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print('\tData downloaded!')

detector = HandsDetector()

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    while True:
        img = input_q.get()
        output_q.put(detector.detect_hands(img, sess, detection_graph, category_index))

    sess.close()

if __name__ == '__main__':

    print('\tStart!')

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=5)
    output_q = Queue(maxsize=5)
    pool = Pool(2, worker, (input_q, output_q))

    width, height = 640, 480

    stream = VideoStream(width=width, height=height, camera_num=0).start()
    RUN = True

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while RUN:
        t = time.time()
        img = stream.get_img()

        input_q.put(img)
        boxed_img, class_name, actual_boxes, boxes, actual_boxes2 = output_q.get()

        # print(actual_boxes)
        # print(boxes)

        # for cnt in detector.searching_area_by_cnts(boxes_lst, width, height):
        #     cv2.circle(boxed_img, cnt, 20, (0, 255, 255))



        t = time.time() - t
        fps = 1.0 / t
        cv2.putText(boxed_img, 'FPS = %f' % fps, (20, 20), 0, 0.5, (0, 0, 255))

        cv2.imshow('Video', boxed_img)

        #RUN = not RUN

        if cv2.waitKey(1) & 0xFF == 27:
            RUN = not RUN

    stream.stop()
    pool.terminate()
    cv2.destroyAllWindows()
