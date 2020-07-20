fp16 = dict(loss_scale=512.)
# model settings
# conv_cfg = dict(type='ConvWS')
# norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='FasterRCNN',
    pretrained=  # NOQA
    'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',  # NOQA
    backbone=dict(
        type='Res2Net',
        depth=50,
        scale=4,
        baseWidth=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        # zero_init_residual=False,
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        gcb=dict(ratio=1. / 4., ),
        stage_with_gcb=(False, True, True, True),
        # conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        ),
    rpn_head=dict(
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        target_means=(.0, .0, .0, .0),
        target_stds=[0.07, 0.07, 0.11, 0.11],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=
        dict(
            type='TSDSharedFCBBoxHead',
            featmap_strides=[4, 8, 16, 32],
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            cls_pc_margin=0.3,
            loc_pc_margin=0.3,
            norm_cfg=norm_cfg,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    )

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        ga_assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        ga_sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        center_ratio=0.2,
        ignore_ratio=0.5,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=350,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.58,
                neg_iou_thr=0.58,
                min_pos_iou=0.58,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=300,
                pos_fraction=0.3,
                neg_pos_ub=-1,
                add_gt_as_proposals=True), #ToDO False
            pos_weight=-1,
            debug=False),
        
    stage_loss_weights=[1, 0.75, 0.5])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=350,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=1e-4, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/media/dereyly/data_ssd/ImageDB/COCO/'
img_norm_cfg = dict(
    # The mean and std is used in PyCls when training RegNets
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1600, 450), (1600, 1100)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=(800, 800)), #crop_size=(1280, 1280)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=6,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=50, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer = dict(
#     type='Adam',
#     lr=.0001
# )
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1.0 / 32,
    step=[18, 23])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir  = '/media/dereyly/data_hdd/models/COCO/tsd_r2_50_ga_dcn_gc'
load_from = None
resume_from = '/media/dereyly/data_hdd/models/COCO/tsd_r2_50_ga_dcn_gc/epoch_21.pth'
workflow = [('train', 1)]
