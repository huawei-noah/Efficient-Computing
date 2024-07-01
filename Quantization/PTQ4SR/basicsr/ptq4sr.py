# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
import torch
import torch.nn as nn
from os import path as osp
import math
from basicsr.data.data_sampler import EnlargedSampler

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

from functools import partial
import numpy as np

import torch.nn.functional as F

def create_quant_dataloader(opt):
    # create train and val dataloaders
    calib_train_loader, paq_train_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'stage1_dbdc':
            dataset_opt['phase'] = 'train'
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            # train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            calib_train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                # sampler=train_sampler,
                sampler=None,
                seed=opt['manual_seed'])
        elif phase == 'stage2_paq':
            dataset_opt['phase'] = 'train'
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            # train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            paq_train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                # sampler=train_sampler,
                sampler=None,
                seed=opt['manual_seed'])

    return calib_train_loader, paq_train_loader

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if "test" in phase:
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
            test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # create stage1_dbdc and stage2_paq dataloaders
    logger.info("calibration and finetune dataset prepareing......")
    calib_train_loader, paq_train_loader = create_quant_dataloader(opt)
    from basicsr.quant.convert2quant import disable_calibration, print_range

    logger.info("start calibration")
    for iter, train_data in enumerate(calib_train_loader):
        logger.info(f"calibrate ({iter+1}/{model.q_config['calib_iter']})")
        model.calibration(train_data)
        if iter+1==model.q_config['calib_iter']:
            break

    model.calibrate = False
    model.model_quant = True

    if model.quant:
        disable_calibration(model.net_g)
        print_range(model.net_g, logger)

    # validation
    for test_loader in test_loaders[:2]:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    # paq
    qparams_w, qparams_x = [], []
    for name, param in model.net_g.named_parameters():
        if "wgt_quantizer" in name:
            qparams_w.append(param)
        elif "act_quantizer" in name:
            qparams_x.append(param)

    model.net_g.train()

    print(f"train : qparams_w : {len(qparams_w)}, qparams_x : {len(qparams_x)}")

    paq_w_opt_iter = model.q_config['paq_w_opt_iter']
    paq_a_opt_iter = model.q_config['paq_a_opt_iter']

    opt_qparams_w = torch.optim.Adam(qparams_w, lr=model.q_config['opt_w_lr'][0])
    opt_qparams_x = torch.optim.Adam(qparams_x, lr=model.q_config['opt_a_lr'][0])
    scheduler_qparams_w = torch.optim.lr_scheduler.CosineAnnealingLR(opt_qparams_w, T_max=paq_w_opt_iter, eta_min = model.q_config['opt_w_lr'][1])
    scheduler_qparams_x = torch.optim.lr_scheduler.CosineAnnealingLR(opt_qparams_x, T_max=paq_a_opt_iter, eta_min = model.q_config['opt_a_lr'][1])

    from basicsr.archs.arch_util import ResidualBlockNoBN
    from basicsr.quant.layers import feature_loss
    module_names, fp_modules, quant_modules = [], [], []
    for name, module in model.net_g.named_modules():
        if isinstance(module,ResidualBlockNoBN):
            module_names.append(name)
            quant_modules.append(module)
    for name, module in model.net_fp.named_modules():
        if isinstance(module,ResidualBlockNoBN):
            fp_modules.append(module)
    print(f"names : {len(module_names)}, fp_modules: {len(fp_modules)}, quant_modules : {len(quant_modules)}")

    losses_w, losses_w_out, losses_w_fea = [], [], []
    losses_x, losses_x_out, losses_x_fea = [], [], []
    factor = 5

    # insert hook
    fp32_output = []
    def hook(name, module, input, output):
        fp32_output.append(output.detach().cpu())

    quant_output = []
    def Qhook(name, module, input, output):
        quant_output.append(output.detach().cpu())

    fp_handle_list, quant_handle_list = [], []
    for idx, module_name in enumerate(module_names):
        fp_handle_list.append(fp_modules[idx].register_forward_hook(partial(hook, module_name)))
        quant_handle_list.append(quant_modules[idx].register_forward_hook(partial(Qhook, module_name)))

    # optimize activation
    for iter, train_data in enumerate(paq_train_loader):
        out_fp = model.fp_inference(train_data)
        out_quant = model.quant_inference(train_data)

        loss_out = F.l1_loss(out_fp, out_quant)

        fea_loss = 0
        for layer_idx in range(len(fp32_output)):
            fea_loss += feature_loss(fp32_output[layer_idx], quant_output[layer_idx])
        fea_loss = fea_loss/len(fp32_output)

        loss = loss_out + factor*fea_loss
        fp32_output = []
        quant_output = []

        losses_x.append(loss.item())
        losses_x_out.append(loss_out.item())
        losses_x_fea.append(factor*fea_loss.item())
        opt_qparams_x.zero_grad()

        loss.backward()

        opt_qparams_x.step()

        print(f"optimize x, iter : {iter+1:2d}/{paq_w_opt_iter}, lr_x : {opt_qparams_x.param_groups[0]['lr']:.7f}, cur train loss : {loss:.4f}, avg train loss : {np.mean(losses_x):.4f}, avg out loss : {np.mean(losses_x_out):.4f}, avg fea loss : {np.mean(losses_x_fea):.4f}")
        if iter==paq_w_opt_iter-1:
            break
        
        scheduler_qparams_x.step()
        

    # optimize weight
    for iter, train_data in enumerate(paq_train_loader):
        out_fp = model.fp_inference(train_data)
        out_quant = model.quant_inference(train_data)

        loss_out = F.l1_loss(out_fp, out_quant)

        fea_loss = 0
        for layer_idx in range(len(fp32_output)):
            fea_loss += feature_loss(fp32_output[layer_idx], quant_output[layer_idx])
        fea_loss = fea_loss/len(fp32_output)

        loss = loss_out + factor*fea_loss
        fp32_output = []
        quant_output = []

        losses_w.append(loss.item())
        losses_w_out.append(loss_out.item())
        losses_w_fea.append(factor*fea_loss.item())
        opt_qparams_w.zero_grad()

        loss.backward()

        opt_qparams_w.step()

        print(f"optimize w, iter : {iter+1:2d}/{paq_a_opt_iter}, lr_w : {opt_qparams_w.param_groups[0]['lr']:.7f}, cur train loss : {loss:.4f}, avg train loss : {np.mean(losses_w):.4f}, avg out loss : {np.mean(losses_w_out):.4f}, avg fea loss : {np.mean(losses_w_fea):.4f}")
        if iter==paq_a_opt_iter-1:
            break

        scheduler_qparams_w.step()

    for handle in fp_handle_list:
        handle.remove()
    for handle in quant_handle_list:
        handle.remove()

    # final validation
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    torch.save(model.net_g.state_dict(), f"./experiments/quant/{opt['name']}.pth")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
