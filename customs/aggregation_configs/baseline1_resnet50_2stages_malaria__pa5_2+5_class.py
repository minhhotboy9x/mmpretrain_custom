from mmengine.config import read_base


default_scope = 'mmpretrain' # need to set

with read_base():
    # from mmpretrain.configs._base_.datasets.imagenet_bs32 import *
    from mmpretrain.configs._base_.default_runtime import *
    from ...configs._base_.models.resnet50 import *
    from mmpretrain.configs._base_.schedules.imagenet_bs256 import *
from customs.customed_modules.linear_head import LinearClsHead2
from customs.customed_modules.two_stages import TwoStagesClassifier




work_dir = '/kaggle/working/experiment_result'
data_root = 'dataset/final_malaria_pa5_2+5_class_classification'
batch_size = 32
training_epochs = 100
lr0 = 0.01 # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf = 0.01 # (float) final learning rate (lr0 * lrf)
momentum = 0.937 # (float) SGD momentum/Adam beta1
weight_decay = 0.0005 # (float) optimizer weight decay 5e-4
num_workers = 4

#----------------------------------------------data settings----------------------------------

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale = 224, crop_ratio_range=(0.8, 1.0)),
    dict(type='RandomFlip', prob= [0.25, 0.25], direction=['horizontal', 'vertical']), 
    #dict(type='RandAugment', policies=[dict(type='Rotate', magnitude_range=(0, 360))]),
    #dict(type='RandomRotate', max_angle = 180),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = 224),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = 224),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size = batch_size,
    num_workers = num_workers,
    dataset=dict(
        type='CustomDataset',
        data_root = data_root, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "train_annotation.txt",
        with_label=True,
        pipeline=train_pipeline,
       ),
    sampler=dict(type='DefaultSampler',
                 shuffle=True,),

    collate_fn=dict(type='default_collate')
)
train_cfg = dict(
  type= "EpochBasedTrainLoop",
  #type = "CustomTrainLoop", 
  max_epochs = training_epochs,
  val_interval = 1,
)

val_dataloader = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "val_annotation.txt",
        with_label=True, 
        pipeline=val_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate')
)
val_cfg = dict(type = "ValLoop")
val_evaluator = [dict(type='Accuracy'), dict(type='ConfusionMatrix')]

test_dataloader = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "test_annotation.txt",
        with_label=True, 
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler'  , shuffle=False),
    collate_fn=dict(type='default_collate')
)
test_cfg = dict(type = "TestLoop")
test_evaluator = [dict(type='Accuracy'), dict(type='ConfusionMatrix')]

#-------------------------------------------- model settings --------------------------------------------


model = dict(
    type= TwoStagesClassifier,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        #Config to load pretrained model
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    backbone2=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        #Config to load pretrained model
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    neck=dict(type='GlobalAveragePooling'),
 
    head=dict(
        type=LinearClsHead2,
        num_classes=2,
        in_channels=2048),

    head2=dict(
        type=LinearClsHead2,
        num_classes=5,
        in_channels=2048),

    num_all_classes=6,
    first_stage_map_gt=[4],
    first2second_stage_class=1,
    second_stage_map_gt=[0, 1, 2, 3, 5],
    loss = dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    data_preprocessor = dict(
        type='ClsDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        num_classes=6,
        to_onehot=True,
        ),
    )

# optimizer
optim_wrapper = dict(
    type = 'AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type=SGD, lr=lr0, momentum=momentum, weight_decay=weight_decay),
    clip_grad=dict(max_norm=10, norm_type=2)
)


param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=training_epochs, 
    eta_min=lrf * lr0,
)


#-------------------------------------------- runtime settings --------------------------------------------
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),  # Log every 10 iterations
    checkpoint=dict(type='CheckpointHook', 
                    interval=1, 
                    max_keep_ckpts=2,
                    save_best='accuracy/top1'),  # Save checkpoints every epoch
    visualization=dict(type='VisualizationHook', enable=True)
)

randomness = dict(seed=0, deterministic=False)