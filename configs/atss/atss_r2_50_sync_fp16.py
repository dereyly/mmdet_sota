fp16 = dict(loss_scale=512.)
# model settings
# conv_cfg = dict(type='ConvWS')
# norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='ATSS',
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
        norm_cfg=norm_cfg,
        # dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True),
        # gcb=dict(ratio=1. / 4., ),
        # stage_with_gcb=(False, True, True, True),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        # conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=1,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/media/dereyly/data_ssd/ImageDB/COCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1600, 500), (1600, 1100)],
        multiscale_mode='range',
        keep_ratio=True),
    # dict( type='bbaug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=(1024, 1024)), #crop_size=(1280, 1280)
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
evaluation = dict(interval=1, metric='bbox')
# # optimizer
# optimizer = dict(
#     type='Adam',
#     lr=.0001
# )
# optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1.0 / 10,
    step=[16, 22])
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
work_dir = '/media/dereyly/data_hdd/models/COCO/atss_r2_50_sync_fp16'
load_from = None
resume_from = None
workflow = [('train', 1)]
