import torch.nn as nn
import torch.nn.functional as F


class KLDiv(nn.Module):
    def __init__(self, temperature=1.0):
        super(KLDiv, self).__init__()
        self.temperature = temperature

    def forward(self, z_s, z_t, **kwargs):
        log_pred_student = F.log_softmax(z_s / self.temperature, dim=1)
        pred_teacher = F.softmax(z_t / self.temperature, dim=1)
        kd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        kd_loss *= self.temperature ** 2
        return kd_loss
