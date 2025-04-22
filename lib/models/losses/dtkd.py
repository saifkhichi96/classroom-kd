import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceDTKD(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',adaptive=False
    ):
        super(KLDivergenceDTKD, self).__init__()
        self.tau = tau
        self.adaptive = adaptive
        self.alpha = 0.5

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T,tau_scale = [1.0,1.0]):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        

        T_stu = tau_scale[1]*self.tau
        T_tea = tau_scale[0]*self.tau
        p_stu = F.log_softmax(preds_S / T_stu,dim=1)
        p_tea = F.softmax(preds_T / T_tea,dim=1)
        # DTKD
        loss = F.kl_div(p_stu, p_tea,reduction=self.reduction)* T_tea * T_stu
        loss = loss.mean() 
        # dtkd_loss = alpha * loss * T_tea * T_stu
        
        # softmax_pred_T = F.softmax(preds_T / (T_tea), dim=1)
        # softmax_preds_S = F.softmax(preds_S / (T_stu), dim=1)
        # loss = ((1.0 * tau_scale*self.tau)**2) * F.kl_div(
        #     logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return loss
