import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, warnings

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    # https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    true = true.long()
    num_classes = logits.shape[1]
    if num_classes == 1:
        t_device = true.device
        true_1_hot = torch.eye(num_classes + 1).to(t_device)[true.squeeze(1)].to(t_device)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

class Dice_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        return dice_loss(target, preds)

'''
# Dice généralisée (https://arxiv.org/pdf/1707.03237.pdf par exemple)
def dice_loss(input, target, weights, eps=1e-4):
    wb, wf = weights
    iflat = input.view(-1)
    tflat = target.view(-1)
    inter_fore = (iflat * tflat).sum()
    inter_back = ((1.-iflat)*(1.-tflat)).sum()

    return 1. - 2.*((wf*inter_fore+wb*inter_back + eps) / (wf*(iflat.sum() + tflat.sum())+wb*((1.-iflat).sum()+(1-tflat).sum()) + eps))

class DiceLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(DiceLoss,self).__init__()
        if weights == None:
            self.weights=[1.0, 1.0]
        else:
            self.weights=weights

    def forward(self,input,target):
        return dice_loss(torch.sigmoid(input), target, weights=self.weights)
'''
