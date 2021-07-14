import cv2
import numpy as np 
import os

from data_utils import VideoData
from lidar_camera_det import LidarCameraFusion


if __name__ == '__main__':
    import os
    base_dir = '../../'
    imgs_dir = os.path.join(base_dir, 'images')
    lidar_dir = os.path.join(base_dir, 'point_cloud_npy')
    video_obj = VideoData(imgs_dir, lidar_dir)
    calib = video_obj.get_calibration()

    colors = {
        "car": (6,190,240),
        "person": (240,240,6),
        "bicycle": (183,6,240),
        "bus": (6,240,20),
        "truck": (6,240,20)

    }

    lid_cam_fusion = LidarCameraFusion()
    
    model, classes, output_layers = lid_cam_fusion.load_yolo()
    out_imgs = []

    for idx in range(len(video_obj)):
        img, pointcloud = video_obj[idx]

        height, width, channels = img.shape
        blob, outputs = lid_cam_fusion.detect_objects(img, model, output_layers)
        
        bboxes = lid_cam_fusion.get_box_dimensions(outputs, height, width)
        indexes = lid_cam_fusion.perform_nms(bboxes)
        img = lid_cam_fusion.enclose_pcs(img, bboxes, pointcloud, calib, indexes)
        
        img = lid_cam_fusion.draw_labels(img, bboxes, indexes, colors, classes)
        out_imgs.append(img)
        print(f"1-Image: {idx} Finished")
        # cv2.imshow('WIND',img)
        # if cv2.waitKey():
        #     cv2.destroyAllWindows

    # Video Generation
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # outVideo = cv2.VideoWriter('demo_video.mp4', fourcc, 15, (width, height))
    # for idx, img in enumerate(out_imgs):
    #     print(f"2-Image: {idx} Finished")
    #     outVideo.write(img)


