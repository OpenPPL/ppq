# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry
from torch.utils.data import DataLoader
from .coco import CocoDataset
from mmcv.parallel import collate


# if platform.system() != 'Windows':
#     # https://github.com/pytorch/pytorch/issues/973
#     import resource
#     rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#     base_soft_limit = rlimit[0]
#     hard_limit = rlimit[1]
#     soft_limit = min(max(4096, base_soft_limit), hard_limit)
#     resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(ann_file,data_root,input_size,batch_size):
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=input_size,
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    dataset = CocoDataset(ann_file=ann_file, pipeline=test_pipeline,data_root=data_root)
    data_loader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate)
    return dataset,data_loader

# def _concat_dataset(cfg, default_args=None):
#     from .dataset_wrappers import ConcatDataset
#     ann_files = cfg['ann_file']
#     img_prefixes = cfg.get('img_prefix', None)
#     seg_prefixes = cfg.get('seg_prefix', None)
#     proposal_files = cfg.get('proposal_file', None)
#     separate_eval = cfg.get('separate_eval', True)

#     datasets = []
#     num_dset = len(ann_files)
#     for i in range(num_dset):
#         data_cfg = copy.deepcopy(cfg)
#         # pop 'separate_eval' since it is not a valid key for common datasets.
#         if 'separate_eval' in data_cfg:
#             data_cfg.pop('separate_eval')
#         data_cfg['ann_file'] = ann_files[i]
#         if isinstance(img_prefixes, (list, tuple)):
#             data_cfg['img_prefix'] = img_prefixes[i]
#         if isinstance(seg_prefixes, (list, tuple)):
#             data_cfg['seg_prefix'] = seg_prefixes[i]
#         if isinstance(proposal_files, (list, tuple)):
#             data_cfg['proposal_file'] = proposal_files[i]
#         datasets.append(build_dataset(data_cfg, default_args))

#     return ConcatDataset(datasets, separate_eval)


# def build_dataset(cfg, default_args=None):
#     from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
#                                    MultiImageMixDataset, RepeatDataset)
#     if isinstance(cfg, (list, tuple)):
#         dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
#     elif cfg['type'] == 'ConcatDataset':
#         dataset = ConcatDataset(
#             [build_dataset(c, default_args) for c in cfg['datasets']],
#             cfg.get('separate_eval', True))
#     elif cfg['type'] == 'RepeatDataset':
#         dataset = RepeatDataset(
#             build_dataset(cfg['dataset'], default_args), cfg['times'])
#     elif cfg['type'] == 'ClassBalancedDataset':
#         dataset = ClassBalancedDataset(
#             build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
#     elif cfg['type'] == 'MultiImageMixDataset':
#         cp_cfg = copy.deepcopy(cfg)
#         cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
#         cp_cfg.pop('type')
#         dataset = MultiImageMixDataset(**cp_cfg)
#     elif isinstance(cfg.get('ann_file'), (list, tuple)):
#         dataset = _concat_dataset(cfg, default_args)
#     else:
#         dataset = build_from_cfg(cfg, DATASETS, default_args)

#     return dataset


