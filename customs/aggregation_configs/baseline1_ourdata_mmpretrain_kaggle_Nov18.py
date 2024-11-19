_base_ = [
    #'mmpretrain/configs/resnet/resnet50_8xb32_in1k.py',
    '/kaggle/working/Multi_Domain_Learning_Malaria_Parasite/configs/_base_/default_runtime.py',
    '/kaggle/working/Multi_Domain_Learning_Malaria_Parasite/configs/_base_/models/resnet50.py',
    '/kaggle/working/Multi_Domain_Learning_Malaria_Parasite/configs/_base_/schedules/imagenet_bs256.py'
    #'mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
]

work_dir='/kaggle/working/experiment_result'
data_root = '/kaggle/input/malaria-parasite/final_malaria_full_class_classification_cropped'
batch_size = 128 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale = 224, crop_ratio_range=(0.08, 1.0)),
    dict(type='RandomFlip', prob= [0.25, 0.25], direction=['horizontal', 'vertical']), 
    #dict(type='RandAugment', policies=[dict(type='Rotate', magnitude_range=(0, 360))]),
    #dict(type='RandomRotate', max_angle = 180),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale = 224),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size = batch_size,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root= data_root, # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "train_annotation.txt",
        with_label=True,
        pipeline=train_pipeline,
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
train_cfg = dict(
  type= "EpochBasedTrainLoop",
  #type = "CustomTrainLoop", 
  max_epochs = 50,
  val_interval = 5,
  _delete_ = True
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
        pipeline=test_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
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
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)
test_cfg = dict(type = "TestLoop")
test_evaluator = [dict(type='Accuracy'), dict(type='ConfusionMatrix')]


model = dict(
    #type='CustomClassifier',
    type= 'ImageClassifier',
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
    neck=dict(type='GlobalAveragePooling'),
    #neck=dict(type='Custom_Pooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    #Normalization 
    data_preprocessor = dict(
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    )

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001, _delete_ = True)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer= optimizer,
    clip_grad=None)

param_scheduler = dict(by_epoch=True, gamma=0.1, milestones=[25], type='MultiStepLR')

visualizer = dict(
    type='Visualizer', 
    vis_backends=[dict(type='TensorboardVisBackend')])

# the default value of by_epoch is True
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10, by_epoch=True))