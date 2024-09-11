_base_ = [
    '../../_base_/schedules/cyclic-40e.py', '../../_base_/default_runtime.py'
]

################## MODEL ##################

voxel_size = [0.16, 0.16, 9]
point_cloud_range = [-92, -19.84, -4, 100, 19.84, 5]  
anchor_range = [-92, -19.84, -1.5, 100, 19.84, -1.5]
anchor_size  = [5, 2, 1.5]
output_shape = [
    int((point_cloud_range[4] - point_cloud_range[1])/voxel_size[1]),
    int((point_cloud_range[3] - point_cloud_range[0])/voxel_size[0]),
    ]


model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range= point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(20000, 40000),
            )),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=output_shape),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                anchor_range,
            ],
            sizes=[anchor_size],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.4,
        min_bbox_size=0,
        nms_pre=10,
        max_num=10))

################## DATASET ##################

# dataset settings
dataset_type = 'KittiLIDAR'
data_root = '/mnt/ssd/Datasets/DatasetForTraining/HYDRIS_v4/'
batch_size = 2
times = 1
class_names = ['Car']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kittilidar_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points']) # Need gt
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True), # Need Annotation
    dict(type='Pack3DDetInputs', keys=['points','gt_bboxes_3d', 'gt_labels_3d']) # Need gt
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=times,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kittilidar_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in donaset and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=3,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kittilidar_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kittilidar_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='LossesMetric',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

################## SCHEDULER ##################

lr = 0.001
epoch_num = 80

optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1000,by_epoch = False),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='Det3DVisualizationHook',show=True,vis_task='lidar_det',draw=True,interval=50)
    )
