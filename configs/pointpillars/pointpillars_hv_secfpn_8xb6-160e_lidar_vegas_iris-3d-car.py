# model settings
_base_ = './pointpillars_hv_secfpn_8xb6-160e_lidar_vegas_iris-3d-3class.py'

# dataset settings
dataset_type = 'KittiLIDAR'
data_root = 'data/Dataset_Lidar_20240111_vegas_ontrack_01/'
class_names = ['Car']
metainfo = dict(classes=class_names)
backend_args = None

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

model = dict(
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False))

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
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
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
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
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
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=3,
    dataset=dict(
        times = 3,
        dataset=dict(
            pipeline=train_pipeline, 
            metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))


# HANDCHANGED
default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=250,by_epoch = False),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='Det3DVisualizationHook',show=True,vis_task='lidar_det',draw=True,interval=50)
    )
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

# Disable validation 
# val_dataloader = None
# val_cfg= None
# val_evaluator = None
# # test_dataloader = None
# # test_cfg = None
#test_evaluator = {}

################################ Changes for Losses metric ################################
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
val_dataloader = dict(batch_size=3,dataset=dict(pipeline=eval_pipeline,test_mode = False, metainfo=metainfo)) # test_mode = False
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True), # Need Annotation
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
test_dataloader = dict(batch_size=1,dataset=dict(pipeline=test_pipeline,test_mode = False, metainfo=metainfo)) # test_mode = False

val_evaluator = dict(
    _delete_=True,
    type='LossesMetric',
    backend_args=backend_args)
test_evaluator = val_evaluator
############################################################################################
