import mmengine

from mmdet3d.registry import DATASETS
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes

import numpy as np

@DATASETS.register_module()
class DonaSet(Det3DDataset):

    # replace with all the classes in customized pkl info file
    METAINFO = {
        'classes':('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc'),
    }

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        # gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        # ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        gt_bboxes_3d = CameraInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                 np.linalg.inv(lidar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info