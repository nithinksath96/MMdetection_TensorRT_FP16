# model settings
model = dict(
    type='YoloV3UA',
)
# training and testing settings
train_cfg = dict()
test_cfg = dict()
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    # mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_train2017.json',
    #     img_prefix=data_root + 'train2017/',
    #     img_scale=(1333, 800),
    #     img_norm_cfg=img_norm_cfg,
    #     size_divisor=32,
    #     flip_ratio=0.5,
    #     with_mask=False,
    #     with_crowd=False,
    #     with_label=True),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'val2017/',
    #     img_scale=(1333, 800),
    #     img_norm_cfg=img_norm_cfg,
    #     size_divisor=32,
    #     flip_ratio=0,
    #     with_mask=False,
    #     with_crowd=False,
    #     with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        # img_scale=(416, 416),
        img_scale = (1920, 1920),
        resize_keep_ratio=True,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11])
# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# # yapf:enable
# # runtime settings
# total_epochs = 12
# device_ids = range(8)
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = './work_dirs/retinanet_r50_fpn_1x'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
