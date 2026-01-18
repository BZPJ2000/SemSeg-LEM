"""
Loss functions module for SemSeg-LEM project.
Provides various loss functions for semantic segmentation.
"""

# Dice-based losses
from .dice_loss import (
    GDiceLoss,
    GDiceLossV2,
    SSLoss,
    SoftDiceLoss,
    IoULoss,
    TverskyLoss,
    FocalTversky_loss,
    AsymLoss,
    DC_and_CE_loss,
    PenaltyGDiceLoss,
    DC_and_topk_loss,
    ExpLog_loss,
)

# Boundary losses
from .boundary_loss import (
    BDLoss,
    DC_and_BD_loss,
    DistBinaryDiceLoss,
)

# Focal loss
from .focal_loss import FocalLoss

# Lovasz loss
from .lovasz_loss import LovaszSoftmax

# Cross-entropy losses
from .ND_Crossentropy import (
    CrossentropyND,
    TopKLoss,
    WeightedCrossEntropyLoss,
    WeightedCrossEntropyLossV2,
    DisPenalizedCE,
)

# Hausdorff losses
from .hausdorff import (
    HausdorffDTLoss,
    HausdorffERLoss,
)

__all__ = [
    # Dice losses
    'GDiceLoss',
    'GDiceLossV2',
    'SSLoss',
    'SoftDiceLoss',
    'IoULoss',
    'TverskyLoss',
    'FocalTversky_loss',
    'AsymLoss',
    'DC_and_CE_loss',
    'PenaltyGDiceLoss',
    'DC_and_topk_loss',
    'ExpLog_loss',
    # Boundary losses
    'BDLoss',
    'DC_and_BD_loss',
    'DistBinaryDiceLoss',
    # Focal loss
    'FocalLoss',
    # Lovasz loss
    'LovaszSoftmax',
    # Cross-entropy losses
    'CrossentropyND',
    'TopKLoss',
    'WeightedCrossEntropyLoss',
    'WeightedCrossEntropyLossV2',
    'DisPenalizedCE',
    # Hausdorff losses
    'HausdorffDTLoss',
    'HausdorffERLoss',
]
