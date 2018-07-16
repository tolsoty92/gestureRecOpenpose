import numpy as np
import tensorflow as tf
import cv2


class HandDetector:
    """ Class contains functions for detecting a hand on an image.
        Args:   PATH_TO_GRAPH - path to neural network graph
                PATH_TO_LABELS - path to label map
                NUM_CLASSES - num classes in label map """

    def __init__(self, path_to_graph='./data/faster_rcnn.pb'):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

    def detect_hands(self, img_np, score_thresh=0.5):
        """ Function return a list with all founded hands.
            Args:
                image_np - image as numpy array. In openCV image is already numpy array
                score_thresh - min score of network's confidence
                             that founded object is hand (0: 1)"""

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(img_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        actual_boxes = []
        for i in range(0, len(boxes[0])):
            if scores[0][i] > score_thresh:
                box = boxes[0][i]
                cv_box = []
                cv_box.append(int(box[1] * 640))
                cv_box.append(int(box[0] * 480))
                cv_box.append(int(box[3] * 640))
                cv_box.append(int(box[2] * 480))
                actual_boxes.append(cv_box)

        return actual_boxes

    def draw_box(self, img, box):
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=6)

    def stop(self):
        self.sess.close()