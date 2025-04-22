import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()

class AskOFA(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(AskOFA, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):