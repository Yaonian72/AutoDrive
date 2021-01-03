#!/usr/bin/env python3
import os
from collections import defaultdict

from kitti_util import *
from parseTrackletXML import *
from pub_utils import *
from inference import *

import cv2
import numpy as np
import pandas as pd
import rospy
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray


DATA_PATH = '/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync/'
CKPT_PATH = '/home/yzy/PycharmProjects/AutoDrive/mmdetection3d/checkpoints/second/epoch_40.pth'
CONF_PATH = '/home/yzy/PycharmProjects/AutoDrive/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'


def read_object(tracklet_file):
    calib = Calibration("/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync", from_video=True)
    tracklets = parseXML(trackletFile=tracklet_file)
    frame_dict_2d = defaultdict(list)
    frame_dict_3d = defaultdict(list)
    frame_dict_obj= defaultdict(list)
    frame_dict_id = defaultdict(list)
    for iTracklet, tracklet in enumerate(tracklets):
        # print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        h,w,l = tracklet.size
        trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
                in tracklet:
            #if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
            #    continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]   # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([\
              [np.cos(yaw), -np.sin(yaw), 0.0], \
              [np.sin(yaw),  np.cos(yaw), 0.0], \
              [        0.0,          0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

            # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
            #   car-centered yaw (i.e. 0 degree = same orientation as car).
            #   makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yawVisual = ( yaw - np.arctan2(y, x) ) % (2.*np.pi)
            frame_dict_3d[absoluteFrameNumber].append(cornerPosInVelo.T)
            cornerPosInImg = calib.project_velo_to_image(cornerPosInVelo.T)
            frame_dict_2d[absoluteFrameNumber].append([np.min(cornerPosInImg, axis=0)[0], np.min(cornerPosInImg, axis=0)[1],
                                                        np.max(cornerPosInImg, axis=0)[0], np.max(cornerPosInImg, axis=0)[1]])
            frame_dict_obj[absoluteFrameNumber].append(tracklet.objectType)
            frame_dict_id[absoluteFrameNumber].append(iTracklet)
    return frame_dict_2d, frame_dict_3d, frame_dict_obj, frame_dict_id


def read_pred(pred_file):
    calib = Calibration("/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync", from_video=True)
    frame_dict_2d = defaultdict(list)
    frame_dict_3d = defaultdict(list)
    frame_dict_obj= defaultdict(list)
    frame_dict_id = defaultdict(list)
    
    df = pd.read_csv(pred_file, header=0, index_col=None)
    for index, row in df.iterrows():
        absoluteFrameNumber = row['id']
        h,w,l = row['h'], row['w'], row['l']
        trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])
        yaw = row['yaw']
        translation = np.array([row['x'], row['y'], row['z']])
        rotMat = np.array([\
              [np.cos(yaw), -np.sin(yaw), 0.0], \
              [np.sin(yaw),  np.cos(yaw), 0.0], \
              [        0.0,          0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T
        cornerPosInImg = calib.project_velo_to_image(cornerPosInVelo.T)
        x, y, z = row['x'], row['y'], row['z']
        frame_dict_3d[absoluteFrameNumber].append(cornerPosInVelo.T)
        frame_dict_2d[absoluteFrameNumber].append([np.min(cornerPosInImg, axis=0)[0], np.min(cornerPosInImg, axis=0)[1],
                                                        np.max(cornerPosInImg, axis=0)[0], np.max(cornerPosInImg, axis=0)[1]])
        frame_dict_id[absoluteFrameNumber].append(index)
        frame_dict_obj[absoluteFrameNumber].append("Prediction")
    return frame_dict_2d, frame_dict_3d, frame_dict_obj, frame_dict_id
        

if __name__ == "__main__":

    rospy.init_node('kitti_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_pcl', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego', MarkerArray, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3dboxes', MarkerArray, queue_size=100)
    box3d_pred_pub = rospy.Publisher('kitti_pred_3dboxes', MarkerArray, queue_size=100)
    bridge = CvBridge()

    detector = Detector(checkpoint=CKPT_PATH,
                        config=CONF_PATH,
                        calib_file=DATA_PATH)

    frame_dict_2d, frame_dict_3d, frame_dict_obj, frame_dict_id = read_object(os.path.join(DATA_PATH, "tracklet_labels.xml"))
    # frame_dict_2d_pred, frame_dict_3d_pred, frame_dict_obj_pred, frame_dict_id_pred = read_pred("/home/yzy/PycharmProjects/AutoDrive/pred.csv")
    rate = rospy.Rate(10)
    frame = 0

    while not rospy.is_shutdown():

        pc_path = os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin' % frame)
        img_path = os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame)

        # load data
        img = cv2.imread(img_path)
        point_cloud = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

        # run inference
        frame_dict_2d_pred, frame_dict_3d_pred = detector.run(pc_path)

        # publish data
        publish_point_cloud(pcl_pub, point_cloud)
        publish_camera(cam_pub, bridge, img, frame_dict_2d[frame] + frame_dict_2d_pred,
                       frame_dict_obj[frame]+['Prediction' for i in range(len(frame_dict_3d_pred))])
        publish_3dbox(box3d_pub, frame_dict_3d[frame], frame_dict_id[frame], frame_dict_obj[frame], publish_id=False, publish_distance=False)
        # publish_3dbox(box3d_pred_pub, frame_dict_3d_pred[frame], frame_dict_id_pred[frame], frame_dict_obj_pred[frame], publish_id=False, publish_distance=False)
        publish_3dbox(box3d_pred_pub, frame_dict_3d_pred,
                      [-i for i in range(1, len(frame_dict_3d_pred)+1)],
                      ['Prediction' for i in range(len(frame_dict_3d_pred))],
                      publish_id=False, publish_distance=False)
        publish_ego_car(ego_pub)
        rospy.loginfo('camera image published')
        rate.sleep()
        frame += 1
        frame %= 473
