# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmengine.model import BaseModel
from mmpretrain.models.losses import CrossEntropyLoss


@MODELS.register_module()
class TwoCascadeStagesClassifier(BaseModel):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.

        num_classes (int): The number of classes in the dataset.
        first_stage_map_gt (list): The mapping from the grountruth to 
            first stage class. Defaults to None.
        first2second_stage_class (int): The number of classes to switch to 
            second stage.
        second_stage_map_gt (list): The mapping from the grountruth to
            second stage class. Defaults to None.

        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:
            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 backbone2: dict,
                 neck: dict = None,
                 head: dict = None,
                 head2: dict = None,
                 num_all_classes: int = None,
                 first_stage_map_gt: list = None,
                 first2second_stage_class: int = None,
                 second_stage_map_gt: list = None,
                 loss: dict = dict(type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0),
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(TwoCascadeStagesClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if not isinstance(backbone2, nn.Module):
            backbone2 = MODELS.build(backbone2)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)
        if head2 is not None and not isinstance(head2, nn.Module):
            head2 = MODELS.build(head2)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)

        self.backbone = backbone
        self.backbone2 = backbone2
        self.neck = neck
        self.head = head
        self.head2 = head2

        self.num_all_classes = num_all_classes
        self.num_first_classes = self.head.num_classes
        self.num_second_classes = self.head2.num_classes

        self.first_stage_map_gt = torch.tensor(first_stage_map_gt)
        self.first2second_stage_class = first2second_stage_class
        self.second_stage_map_gt = torch.tensor(second_stage_map_gt)
        self.loss_module = loss

        # If the model needs to load pretrain weights from a third party,
        # the key can be modified with this hook
        if hasattr(self.backbone, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone._checkpoint_filter)
        if hasattr(self.backbone2, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone2._checkpoint_filter)
    
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        """Extract features from the input tensor."""
        device = next(self.parameters()).device
        preds = torch.zeros(inputs.size(0), self.num_all_classes).float().to(device)
        out1 = self.head(self.neck(self.backbone(inputs)))  # (N, num_first_classes)

        first_stage_map_gt = self.first_stage_map_gt.to(device)
        stage1_class = torch.argmax(out1, dim=1) # (N, 1)
        mask = stage1_class != self.first2second_stage_class  # (N, )

        preds[mask] = preds[mask].scatter(1, first_stage_map_gt.repeat(mask.sum(), 1), 
                                          out1[mask, stage1_class[mask]].resize(mask.sum(), 1).float())

        mask2 = ~mask

        if mask2.sum() == 0:
            return preds
        
        second_stage_map_gt = self.second_stage_map_gt.to(device)
        out2 = self.head2(self.neck(self.backbone2(inputs[mask2])))
        
        preds[mask2] = preds[mask2].scatter(1, second_stage_map_gt.repeat(mask2.sum(), 1), 
                                            out2.float())

        return preds
    
    def loss(self, input: torch.Tensor, data_samples: List[DataSample]) -> dict:
        """Calculate the loss."""
        cls_score = self.extract_feat(input)

        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0))
        losses['loss'] = loss

        return losses
    
    def predict(
        self,
        input: torch.Tensor,
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        cls_score = self.extract_feat(input)

        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
        
        

    def get_layer_depth(self, param_name: str, stage: int) -> int:
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            stage (int): The stage index.
        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if stage == 0:
            if hasattr(self.backbone, 'get_layer_depth'):
                return self.backbone.get_layer_depth(param_name, 'backbone.')
            else:
                raise NotImplementedError(
                    f"The backbone {type(self.backbone)} doesn't "
                    'support `get_layer_depth` by now.')
        elif stage == 1:
            if hasattr(self.backbone2, 'get_layer_depth'):
                return self.backbone2.get_layer_depth(param_name, 'backbone2.')
            else:
                raise NotImplementedError(
                    f"The backbone2 {type(self.backbone2)} doesn't "
                    'support `get_layer_depth` by now.')
        else:
            raise ValueError(f"Invalid stage index {stage}.")