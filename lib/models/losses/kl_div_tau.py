import torch.nn as nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
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

    def __init__(self, tau=1.0, reduction="batchmean", adaptive=False):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.adaptive = adaptive

        accept_reduction = {"none", "batchmean", "sum", "mean"}
        assert reduction in accept_reduction, (
            f"KLDivergence supports reduction {accept_reduction}, but gets {reduction}."
        )
        self.reduction = reduction

    def forward(self, preds_S, preds_T, tau_scale=1.0):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        # l_stu_mx, _ = l_stu.max(dim=1, keepdim=True)
        # l_tea_mx, _ = l_tea.max(dim=1, keepdim=True)

        # T_stu = 2 * l_stu_mx / (l_tea_mx+l_stu_mx)*T
        # T_tea = 2 * l_tea_mx / (l_tea_mx+l_stu_mx)*T
        # p_stu = F.softmax(l_student / T_stu)
        # p_tea = F.softmax(l_teacher / T_tea)
        # # DTKD
        # loss = kl_div(log(p_stu), p_tea)
        # dtkd_loss = alpha * loss * T_tea * T_stu
        preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / (1.0 + tau_scale * self.tau), dim=1)
        logsoftmax_preds_S = F.log_softmax(
            preds_S / (1.0 + tau_scale * self.tau), dim=1
        )
        loss = ((1.0 * tau_scale * self.tau) ** 2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction
        )
        return loss
