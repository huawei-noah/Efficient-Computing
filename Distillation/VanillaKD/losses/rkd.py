import torch
import torch.nn as nn
import torch.nn.functional as F


def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKD(nn.Module):
    def __init__(self, distance_weight=25, angle_weight=50, eps=1e-12, squared=False):
        super(RKD, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.eps = eps
        self.squared = squared

    def forward(self, z_s, z_t, **kwargs):
        f_s = kwargs['feature_student'][-1]
        f_t = kwargs['feature_teacher'][-1]

        stu = f_s.view(f_s.shape[0], -1)
        tea = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = _pdist(tea, self.squared, self.eps)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = _pdist(stu, self.squared, self.eps)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = tea.unsqueeze(0) - tea.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = stu.unsqueeze(0) - stu.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        kd_loss = self.distance_weight * loss_d + self.angle_weight * loss_a
        return kd_loss
