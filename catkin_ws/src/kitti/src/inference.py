import numpy as np
from mmdet3d.apis import inference_detector, init_detector

from kitti_util import *


class Detector:

    def __init__(self, checkpoint, config, calib_file, from_video=True):

        self.model = init_detector(config, checkpoint)
        self.calib = Calibration(calib_file, from_video=from_video)

    def run(self, data_bin, threshold=0.3):
        result, data = inference_detector(self.model, data_bin)
        obj_ind = result[0]['scores_3d'] >= threshold
        pred_3d = result[0]['boxes_3d'].corners[obj_ind, ...]
        pred_2d = []
        for obj in pred_3d:
            obj_2d = self.calib.project_velo_to_image(obj)
            pred_2d.append([np.min(obj_2d, axis=0)[0], np.min(obj_2d, axis=0)[1],
                            np.max(obj_2d, axis=0)[0], np.max(obj_2d, axis=0)[1]])
        return pred_2d, [xx.squeeze() for xx in np.split(pred_3d, pred_3d.shape[0], axis=0)]


if __name__ == "__main__":
    detector = Detector(checkpoint="/home/yzy/PycharmProjects/AutoDrive/mmdetection3d/checkpoints/second/epoch_40.pth",
                        config="/home/yzy/PycharmProjects/AutoDrive/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py",
                        calib_file="/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync")
    pred_2d, pred_3d = detector.run("/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync/velodyne_points/data/0000000000.bin", 0.5)

    print(len(pred_2d), pred_3d)