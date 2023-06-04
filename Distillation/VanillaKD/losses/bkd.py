import torch.nn as nn
import torch.nn.functional as F


class BinaryKLDiv(nn.Module):
    def __init__(self):
        super(BinaryKLDiv, self).__init__()

    def forward(self, z_s, z_t, **kwargs):
        kd_loss = F.binary_cross_entropy_with_logits(z_s, z_t.softmax(1))
        return kd_loss
