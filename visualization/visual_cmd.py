import os

import numpy as np
import time
from mayavi import mlab
from collections import defaultdict

from visualization.kitti_object import kitti_object, show_lidar_with_depth, show_lidar_on_image, \
                         show_image_with_boxes, show_lidar_topview_with_boxes, draw_gt_boxes3d, draw_lidar
from visualization.kitti_object import kitti_object_video
from visualization.kitti_util import *
from parseTrackletXML import *
import logging

logging.getLogger(mlab.__name__).setLevel(logging.WARNING)


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

    return frame_dict_2d, frame_dict_3d, frame_dict_obj, frame_dict_id


video_dir = "/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync"
frame_dict_2d, frame_dict_3d, frame_dict_obj, _ = read_object(os.path.join(video_dir, "tracklet_labels.xml"))
dataset = kitti_object_video(os.path.join(video_dir, "image_02/data"),
                             os.path.join(video_dir, "velodyne_points/data"),
                             video_dir,
                             os.path.join(video_dir, "tracklet_labels.xml"))

if __name__ == '__main__':


    mmdet3d_dir = '../mmdetection3d/'
    config_file = os.path.join(mmdet3d_dir, "configs/3dssd/3dssd_kitti-3d-car.py")
    ckpt_file = os.path.join(mmdet3d_dir, "checkpoints/3dssd_kitti-3d-car/latest.pth")
    # build the model from a config file and a checkpoint file
    from mmdet3d.apis import init_detector, inference_detector, show_result_meshlab

    from tqdm import tqdm
    model = init_detector(config_file, ckpt_file, device='cuda:0')
    preds = []
    for i in tqdm(range(len(dataset))):
        result, data = inference_detector(model, dataset.get_lidar_filename(i))
        threshold = 0.5
        bboxes, scores = result[0]['boxes_3d'].corners, result[0]['scores_3d']
        pred = bboxes[scores > threshold]
        preds.append(pred)

    def anim(t):
        global fig_3d
        mlab.clf()
        t = int(t)
        img = dataset.get_image(t)
        pcd = dataset.get_lidar(t)
        calib = dataset.get_calibration(t)
        _, gt, _, _ = dataset.get_object(t)
        pred = preds[t]
        img_height, img_width, _ = img.shape
        draw_gt_boxes3d(gt, fig=fig_3d)
        draw_gt_boxes3d(pred, fig=fig_3d, color=(0, 1, 0))
        draw_lidar(pcd, color_by_intensity=True, fig=fig_3d)
        # show_lidar_with_depth(pcd, objects, calib, fig_3d, False, img_width, img_height)
        fig_3d = mlab.gcf()
        fig_3d.scene._lift()
        return mlab.screenshot()

    import moviepy.editor as mpy

    fig_3d = mlab.figure(bgcolor=(0, 0, 0), size=(800, 450))
    animation = mpy.VideoClip(anim, duration=20)
    animation.write_videofile("lidar_with_depth.mp4", fps=10)

    @mlab.show
    @mlab.animate(delay=100)
    def anim_video():
        for l in range(len(dataset)):
            img = dataset.get_image(l)
            pcd = dataset.get_lidar(l)
            calib = dataset.get_calibration(l)
            vis.mlab_source.reset(x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], scalars=pcd[:, 3])
            num = len(frame_dict_3d[l])
            for n in range(num):
                if n >= 1:
                    continue

                b = frame_dict_3d[l][n]

                for k in range(0, 4):
                    # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                    if l == 0:
                        i, j = k, (k + 1) % 4
                        boxes_list[n][i].mlab_source.reset(x=np.array([b[i, 0], b[j, 0]]), y=np.array([b[i, 1], b[j, 1]]),
                                                         z=np.array([b[i, 2], b[j, 2]]))

                        i, j = k + 4, (k + 1) % 4 + 4
                        boxes_list[n][i].mlab_source.reset(x=np.array([b[i, 0], b[j, 0]]), y=np.array([b[i, 1], b[j, 1]]),
                                                         z=np.array([b[i, 2], b[j, 2]]))

                        i, j = k, k + 4
                        boxes_list[n][i].mlab_source.reset(x=np.array([b[i, 0], b[j, 0]]), y=np.array([b[i, 1], b[j, 1]]),
                                                         z=np.array([b[i, 2], b[j, 2]]))
                    else:
                        i, j = k, (k + 1) % 4
                        boxes_list[n][i].mlab_source.trait_set(x=np.array([b[i, 0], b[j, 0]]), y=np.array([b[i, 1], b[j, 1]]),
                                                         z=np.array([b[i, 2], b[j, 2]]))

                        i, j = k + 4, (k + 1) % 4 + 4
                        boxes_list[n][i].mlab_source.trait_set(x=np.array([b[i, 0], b[j, 0]]), y=np.array([b[i, 1], b[j, 1]]),
                                                         z=np.array([b[i, 2], b[j, 2]]))

                        i, j = k, k + 4
                        boxes_list[n][i].mlab_source.trait_set(x=np.array([b[i, 0], b[j, 0]]), y=np.array([b[i, 1], b[j, 1]]),
                                                         z=np.array([b[i, 2], b[j, 2]]))
                    '''
                    boxes_list.append(mlab.plot3d(
                        [b[i, 0], b[j, 0]],
                        [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=(1, 1, 1),
                        tube_radius=None,
                        line_width=1,
                        figure=fig,
                    ))'''
            yield

