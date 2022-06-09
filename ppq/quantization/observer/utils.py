# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# https://github.com/megvii-research/FQ-ViT
def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()
