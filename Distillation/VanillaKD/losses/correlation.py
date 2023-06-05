import torch
import torch.nn as nn


class Correlation(nn.Module):
    scale = 0.02
    def __init__(self, feat_s_channel, feat_t_channel, feat_dim=128):
        super(Correlation, self).__init__()
        self.embed_s = LinearEmbed(feat_s_channel, feat_dim)
        self.embed_t = LinearEmbed(feat_t_channel, feat_dim)

    def forward(self, z_s, z_t, **kwargs):
        f_s = self.embed_s(kwargs['feature_student'][-1])
        f_t = self.embed_t(kwargs['feature_teacher'][-1])

        delta = torch.abs(f_s - f_t)
        kd_loss = self.scale * torch.mean((delta[:-1] * delta[1:]).sum(1))
        return kd_loss


class LinearEmbed(nn.Module):
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
