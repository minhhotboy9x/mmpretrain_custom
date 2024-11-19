# This is a BETA new format config file, and the usage may change recently.
from mmengine.dataset import DefaultSampler

from mmpretrain.datasets import (CenterCrop, LoadImageFromFile, CustomDataset,
                                 PackInputs, RandomFlip, RandomResizedCrop,
                                 ResizeEdge)
from mmpretrain.evaluation import Accuracy


dataset_type = CustomDataset
data_preprocessor = dict(
    num_classes=3,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=224),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='./dataset/data_draft',
        with_label=True,   # or False for unsupervised tasks
        pipeline=train_pipeline
    ),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_cfg = None
val_evaluator = None

test_dataloader = None
test_cfg = None
test_evaluator = None