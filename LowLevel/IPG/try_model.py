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

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from basicsr.models import build_model

from basicsr.utils.options import parse_options


if __name__=='__main__':
    opt, args = parse_options('.', is_train=False)
    device = 'cuda:0'
    opt['network_g']['img_size'] = 512 // 4
    model = build_model(opt)


    print(model.net_g)
    model.net_g = model.net_g.to(device)
    model.net_g.eval()
    
    try_input = torch.randn((1,3,32,32)).float().to(device)

    A = model.net_g(try_input)
    print(A.shape)
    print(model.net_g.flops()/1e9)
