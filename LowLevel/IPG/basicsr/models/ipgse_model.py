import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class IPGSEModel(SRModel):

    def test(self):
        # pad to multiplication of window_size
        if self.opt['val']['selfensemble_testing']:
            with torch.no_grad():
                # from https://github.com/thstkdgus35/EDSR-Pytorch
                lr_list = [self.lq]
                for tf in 'v', 'h', 't':
                    lr_list.extend([self._test_transform(t, tf) for t in lr_list])

                sr_list = [self._test_pad(aug) for aug in lr_list]
                for i in range(len(sr_list)):
                    if i > 3:
                        sr_list[i] = self._test_transform(sr_list[i], 't')
                    if i % 4 > 1:
                        sr_list[i] = self._test_transform(sr_list[i], 'h')
                    if (i % 4) % 2 == 1:
                        sr_list[i] = self._test_transform(sr_list[i], 'v')

                output_cat = torch.cat(sr_list, dim=0)
                self.output = output_cat.mean(dim=0, keepdim=True)

        else:
            # pad to multiplication of patch_size (window)
            # patch_size1 = max(self.opt['network_g']['split_size_0'])
            # patch_size2 = max(self.opt['network_g']['split_size_1'])
            # patch_size = max(patch_size1, patch_size2)
            patch_size = self.opt['network_g']['sample_size']
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % patch_size != 0:
                mod_pad_h = patch_size - h % patch_size
            if w % patch_size != 0:
                mod_pad_w = patch_size - w % patch_size
            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(img)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(img)
                self.net_g.train()

            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def _test_transform(self, v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(v.device)

        return ret

    def _test_pad(self, lq):
        # pad to multiplication of patch_size (window)
        # patch_size1 = max(self.opt['network_g']['split_size_0'])
        # patch_size2 = max(self.opt['network_g']['split_size_1'])
        # patch_size = max(patch_size1, patch_size2)
        patch_size = self.opt['network_g']['sample_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = lq.size()
        if h % patch_size != 0:
            mod_pad_h = patch_size - h % patch_size
        if w % patch_size != 0:
            mod_pad_w = patch_size - w % patch_size
        img = lq
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        return output