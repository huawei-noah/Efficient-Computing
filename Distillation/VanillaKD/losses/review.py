import torch
import torch.nn as nn
import torch.nn.functional as F


class ReviewKD(nn.Module):
    pre_act_feat = True
    def __init__(self, feat_index_s, feat_index_t, in_channels, out_channels,
                 shapes=(1, 7, 14, 28, 56), out_shapes=(1, 7, 14, 28, 56),
                 warmup_epochs=1, max_mid_channel=512):
        super(ReviewKD, self).__init__()
        self.feat_index_s = feat_index_s
        self.feat_index_t = feat_index_t
        self.shapes = shapes
        self.out_shapes = out_shapes
        self.warmup_epochs = warmup_epochs

        abfs = nn.ModuleList()
        mid_channel = min(max_mid_channel, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1))

        self.abfs = abfs[::-1]

    def forward(self, z_s, z_t, **kwargs):
        f_s = [kwargs['feature_student'][i] for i in self.feat_index_s]
        pre_logit_feat_s = kwargs['feature_student'][-1]
        if len(pre_logit_feat_s.shape) == 2:
            pre_logit_feat_s = pre_logit_feat_s.unsqueeze(-1).unsqueeze(-1)
        f_s.append(pre_logit_feat_s)

        f_s = f_s[::-1]
        results = []
        out_features, res_features = self.abfs[0](f_s[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(f_s[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)

        f_t = [kwargs['feature_teacher'][i] for i in self.feat_index_t]
        pre_logit_feat_t = kwargs['feature_teacher'][-1]
        if len(pre_logit_feat_t.shape) == 2:
            pre_logit_feat_t = pre_logit_feat_t.unsqueeze(-1).unsqueeze(-1)
        f_t.append(pre_logit_feat_t)

        kd_loss = min(kwargs["epoch"] / self.warmup_epochs, 1.0) * hcl_loss(results, f_t)

        return kd_loss


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all
