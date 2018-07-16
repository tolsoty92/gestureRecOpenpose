import numpy as np
from object_detection.utils import visualization_utils as vis_util


class HandsDetector:

    @staticmethod
    def detect_objects(image_np, sess, detection_graph, category_index):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.

        (class_name, box, boxes_lst) = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores), category_index,
            use_normalized_coordinates=True,
            line_thickness=4, max_boxes_to_draw=2)

        return image_np, class_name, box, boxes_lst

    @staticmethod
    def searching_area_by_cnts(boxes_lst, width, height):
        boxes = []
        for box in boxes_lst:
            y_min, x_min, y_max, x_max = box[0], box[1], box[2], box[3]
            #cnts.append((int(x_max * width), int(y_max * height)))
            #cnts.append((int(x_min * width), int(y_min * height)))

            maxX = int(x_max * width)
            maxY = int(y_max * height)
            minX = int(x_min * width)
            minY = int(y_min * height)

            area_size = max(maxY-minY, maxX-minX)
            area = (minX-10, minY-10, minX + area_size+10, minY + area_size+10)

            boxes.append(area)

        return boxes

    @staticmethod
    def searching_area_by_box(boxes_lst, width, height):
        boxes = []
        for box in boxes_lst:
            y_min, x_min, y_max, x_max = \
                int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)
            boxes.append(((x_min, y_min), (x_max, y_max)))
        return boxes

    def compute_BB(self,hand, padding=1.5):
        minX = np.min(hand[:, 0])
        minY = np.min(hand[:, 1])

        maxX = np.max(hand[:, 0])
        maxY = np.max(hand[:, 1])

        width = maxX - minX
        height = maxY - minY

        cx = minX + width / 2
        cy = minY + height / 2

        width = height = max(width, height) * padding
        # width = height = width * padding

        minX = cx - width / 2
        minY = cy - height / 2

        score = np.mean(hand[:, 2])

        if minX > 20:
            minX -= 20
        else:
            minX = 0

        if minY > 20:
            minY -= 20
        else:
            minY = 0
        return score, [int(minX), int(minY), int(width) + 40, int(height) + 40]