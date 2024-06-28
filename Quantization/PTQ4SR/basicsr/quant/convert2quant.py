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

import torch
import torch.nn as nn
from .layers import Quant_Conv2d, Quant_Linear

def convert_quantization(module, q_config, layer_name=""):
    if isinstance(module, nn.Conv2d):
        q_config["layer_name"] = layer_name
        return Quant_Conv2d(in_channels  = module.in_channels,
                            out_channels = module.out_channels,
                            kernel_size  = module.kernel_size,
                            stride       = module.stride,
                            padding      = module.padding,
                            dilation     = module.dilation,
                            groups       = module.groups,
                            bias         = False if module.bias is None else True,
                            q_config     = q_config,
                            module       = module)
    elif isinstance(module, nn.Linear):
        q_config["layer_name"] = layer_name
        return Quant_Linear(in_features  = module.in_features,
                            out_features = module.out_features,
                            bias         = False if module.bias is None else True,
                            q_config     = q_config,
                            module       = module)
    else:
        for name, submodule in module.named_children():
            if layer_name!="":
                sub_layer_name = layer_name + "." + name
            else:
                sub_layer_name = name
            setattr(module, name, convert_quantization(submodule, q_config, sub_layer_name))
        return module

def enable_calibration(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if hasattr(module, "a_bit") and module.a_bit!=32:
            module.act_quantizer.calibration = True
        if hasattr(module, "w_bit") and module.w_bit!=32:
            module.wgt_quantizer.calibration = True
    else:
        for submodule in module.children():
            enable_calibration(submodule)

def disable_calibration(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if hasattr(module, "a_bit") and module.a_bit!=32:
            module.act_quantizer.calibration = False
        if hasattr(module, "w_bit") and module.w_bit!=32:
            module.wgt_quantizer.calibration = False
    else:
        for submodule in module.children():
            disable_calibration(submodule)

def print_range(module, logger=None):
    if logger is None:
        print_info = print
    else:
        print_info = logger.info
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        print()
        if module.w_bit!=32 or module.a_bit!=32:
            print_info(f"{module.layer_name}")
        else:
            print_info(f"{module.layer_name} : keep FP32")
        if hasattr(module, "w_bit") and module.w_bit!=32:
            if module.wgt_quantizer.per_channel:
                print_info(f"==> wgt calibration : {module.wgt_quantizer.calibration}, bit : {module.w_bit}, init : {module.w_init:10s}, clip : per_channel (not show)")
            elif hasattr(module.wgt_quantizer, "clip_val") :
                print_info(f"==> wgt calibration : {module.wgt_quantizer.calibration}, bit : {module.w_bit}, init : {module.w_init:10s}, clip : {module.wgt_quantizer.clip_val.cpu().detach().numpy():.4f}")
            elif hasattr(module.wgt_quantizer, "upper_clip_val") and hasattr(module.wgt_quantizer, "lower_clip_val"):
                print_info(f"==> wgt calibration : {module.wgt_quantizer.calibration}, bit : {module.w_bit}, init : {module.w_init:10s}, lower clip : {module.wgt_quantizer.lower_clip_val.cpu().detach().numpy():.4f}, upper clip : {module.wgt_quantizer.upper_clip_val.cpu().detach().numpy():.4f}")
        if hasattr(module, "a_bit") and module.a_bit!=32:
            if hasattr(module.act_quantizer, "clip_val") :
                print_info(f"==> act calibration : {module.act_quantizer.calibration}, bit : {module.a_bit}, init : {module.a_init:10s}, clip : {module.act_quantizer.clip_val.cpu().detach().numpy():.4f}")
            elif hasattr(module.act_quantizer, "upper_clip_val") and hasattr(module.act_quantizer, "lower_clip_val"):
                print_info(f"==> act calibration : {module.act_quantizer.calibration}, bit : {module.w_bit}, init : {module.w_init:10s}, lower clip : {module.act_quantizer.lower_clip_val.cpu().detach().numpy():.4f}, upper clip : {module.act_quantizer.upper_clip_val.cpu().detach().numpy():.4f}")
    else:
        for submodule in module.children():
            print_range(submodule, logger)
