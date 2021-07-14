import cv2
import numpy as np 
import os


class LidarCameraFusion:
    def __init__(self):
        pass

    def load_yolo(self):
        net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
        classes = []

        with open("yolo/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

        return net, classes, output_layers


    def detect_objects(self, img, net, outputLayers):			
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)
        return blob, outputs


    def get_box_dimensions(self, outputs, height, width, conf_score=0.65):
        boxes = []

        for output in outputs:
            for detect in output:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    conf = scores[class_id]
                    if conf > conf_score:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        x = int(center_x - w/2)
                        y = int(center_y - h / 2)
                        boxes.append(Box2D(x,y,w,h,float(conf),class_id))
        return boxes


    def enclose_pcs(self, img, bboxes, pointcloud, calib, indexes, shrink_factor=0.1):
        for point in pointcloud:
            x_lid, y_lid, z_lid = point[0], point[1], point[2]
            in_ = np.array([x_lid, y_lid, z_lid, 1])
            out_homog = np.dot(calib['T_velo_to_cam'], in_)
            out_homog = np.dot(calib['P'], out_homog)
            x_cam, y_cam = int(out_homog[0] / out_homog[2]), int(out_homog[1] / out_homog[2])

            for it in range(len(bboxes)):
                if it in indexes:
                    # 2D Rect (BBOX) Shrinkage
                    x1= bboxes[it].x + shrink_factor * (bboxes[it].w / 2.0)
                    y1 = bboxes[it].y + shrink_factor * (bboxes[it].h / 2.0)
                    small_w, small_h = (bboxes[it].w * (1-shrink_factor)), (bboxes[it].h * (1-shrink_factor))
                    x2, y2 = (x1 + small_w), (y1 + small_h)

                    if x1 < x_cam and x2 > x_cam and y1 < y_cam and y2 > y_cam:
                        bboxes[it].enclosing_pcs.append((x_cam, y_cam))
                        img = cv2.circle(img, (x_cam, y_cam), 1, color=(162,162,162))           
        return img
    

    def perform_nms(self, bboxes, score_T=0.5, nms_T=0.3):
        boxes = []
        scores = []
        for box in bboxes:
            boxes.append([box.x, box.y, box.w, box.h])
            scores.append(box.score)
        indexes = cv2.dnn.NMSBoxes(boxes, scores, score_T, nms_T)
        return indexes


    def draw_labels(self, img, bboxes, nms_box_indexes, colors, classes): 

        for i in range(len(bboxes)):
            if i in nms_box_indexes:
                x, y, w, h = bboxes[i].x, bboxes[i].y, bboxes[i].w, bboxes[i].h
                label = (bboxes[i].classID)

                if classes[label] not in colors.keys():
                    continue
                img = cv2.rectangle(img, (x,y), (x+w, y+h), colors[classes[label]], 2)
                img = cv2.putText(img, classes[label], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

        return img
