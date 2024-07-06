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
import argparse
from copy import deepcopy
# import moxing as mox
import numpy as np


def get_no():
    import torch
    return torch.cuda.device_count()

def run_cmd(args, exclude, unparsed):
    parallel_str = ''
    if args.init_method is not None:
        print(f'ROMA PARALLEL init method: {args.init_method}')
        init_method = args.init_method
        master_addr, master_port = init_method.replace('tcp://', '').split(':')
        if not (master_addr == '' or master_addr is None):
            # master_addr = '192.168.0.8'
            parallel_str = f' --master_addr={master_addr} --master_port={master_port}'
            print(f'PROCESSED parallel INFO: {parallel_str}')

    cmd = f'NCCL_IB_TIMEOUT=22 python -m torch.distributed.launch --nproc_per_node={get_no()}{parallel_str} basicsr/train.py '

    for k, v in args.__dict__.items():
        if k in exclude:
            continue
        if v == '':
            continue
        if isinstance(v, bool):
            if v==True:
                cmd += f'--{k} '
            else:
                continue
        elif v is not None:
            cmd += f'--{k} '
            cmd += f'{v} '
    # deal with unparsed
    cmd += ' '.join(unparsed)
    # run
    print('='*30)
    print(cmd)
    print('-'*30)
    os.system(cmd)
    print('-'*30)

def swinir_tester(task, scale, training_patch_size, model_path, l_folder, h_folder, eval_save_dir):

    cmd = f'python main_test_swinir.py --task {task} --scale {scale} --training_patch_size {training_patch_size} --model_path {model_path} --folder_lq {l_folder} --folder_gt {h_folder} --save_dir {eval_save_dir}'
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    json_path = './options/train_mod/train_ART_SR_x2.yml' # default
    eval_json_path = 'options/test_mod/test_{}_SR_x{}.yml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--install', action='store_true', default=False)
    parser.add_argument('--no_sync', action='store_true', default=False)
    parser.add_argument('--init_method', type=str, default=None) # receive init_method


    parser.add_argument('--train_data', type=str, default='DF2K')
    parser.add_argument('--task', type=str, default='SR')
    parser.add_argument('--model_type', type=str, default='IPG')
    parser.add_argument('--data_dir', type=str, default='../SRdata') # relative to codebase, not container
    # crucial parsers for running
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--train__total_iter', type=int, default=500000)

    # autoset if None
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--datasets__train__dataroot_gt', type=str, default=None)
    parser.add_argument('--datasets__train__dataroot_lq', type=str, default=None)
    # parser.add_argument('--datasets__val__dataroot_gt', type=str, default=None)
    # parser.add_argument('--datasets__val__dataroot_lq', type=str, default=None)

    # eval from folder
    parser.add_argument('--eval_folder', type=str, default=None)

    # eval opt
    parser.add_argument('--eval_opt', type=str, default=None)
    parser.add_argument('--eval_opt2', type=str, default=None)
    # install pytorch for A
    parser.add_argument('--force_install', type=int, default=0)


    args, unparsed = parser.parse_known_args()
    # some processing
    if args.eval_opt is None:
        args.eval_opt = eval_json_path.format(args.model_type, args.scale)


    exclude = ['install', 'no_sync', 'init_method', 'model_type', 'task', 'train_data', 'data_dir', 'eval_mod', 'local_rank', 'eval_folder', 'eval_opt', 'eval_opt2']

    if args.name is None:
        args.name = f'train_{args.model_type}_{args.task}_{args.train_data}_x{args.scale}_{args.train__total_iter}'

    dataset2folder = {
        'DIV2K': {'name':'DIV2K_800', 'appendix':'.tar', 'HR':'DIV2K_train_HR', 'LR':'DIV2K_train_LR_bicubic'}, # DIV2K
        'DF2K': {'name':'DF2K', 'appendix':'.tar', 'HR':'DF2K_train_HR', 'LR':'DF2K_train_LR_bicubic'}, # DF2K
        'DF2K_lmdb': {'name':'DF2K_lmdb', 'appendix':'.tar', 'HR':'DF2K_train_HR.lmdb', 'LR':'DF2K_train_LR_bicubic'}, # DF2K lmdb
        'DIV2K_lmdb': {'name':'DIV2K_lmdb', 'appendix':'.tar', 'HR':'DIV2K_train_HR.lmdb', 'LR':'DIV2K_train_LR_bicubic'}, # DIV2K lmdb
    }
    if False:
        # set training set lmdb
        if 'lmdb' not in args.train_data: # normal dataset: use taylored template
            args.datasets__train__filename_tmpl = '{}x'+f'{args.scale}'
        else: # autoset lmdb
            print('** lmdb data loading activated!')
            args.datasets__train__io_backend__type = 'lmdb'


    bucket_no = None
    folder_name = None
    if args.install: # install packages automatically
        os.system('pip install wheel==0.26')
        os.system('pip install -r requirements.txt')
        os.system('python setup.py develop') # install hat


    if False:
        # autoset
        if args.datasets__train__dataroot_gt is None:
            args.datasets__train__dataroot_gt = os.path.join(args.data_dir, dataset2folder[args.train_data]['name'], dataset2folder[args.train_data]['HR'])
            if 'lmdb' in args.train_data: # lmdb cat
                datamdb_dir = os.path.join(args.datasets__train__dataroot_gt, 'data.mdb')
                
        if args.datasets__train__dataroot_lq is None:
            if 'lmdb' not in args.train_data:
                args.datasets__train__dataroot_lq = os.path.join(args.data_dir, dataset2folder[args.train_data]['name'], dataset2folder[args.train_data]['LR'], f'X{args.scale}')
            else: # lmdb format
                args.datasets__train__dataroot_lq = os.path.join(args.data_dir, dataset2folder[args.train_data]['name'], dataset2folder[args.train_data]['LR']+f'_X{args.scale}.lmdb')
                datamdb_dir = os.path.join(args.datasets__train__dataroot_lq, 'data.mdb')


    if args.eval_folder is None: # normal training, run training script
        run_cmd(args, exclude, unparsed)

    if (args.eval_folder is not None) and args.eval_folder.endswith('.pth'):
        final_model_path = args.eval_folder
    else:
        final_model_path = os.path.join('experiments', args.name, 'models', 'net_g_latest.pth' if args.eval_folder is None else f'net_g_{args.train__total_iter}.pth')


    print('-'*50)
    eval_cmd = f'python basicsr/test.py --opt {args.eval_opt} --name {args.name} ' + ' '.join(unparsed) + f' --path__pretrain_network_g {final_model_path}'
    print(eval_cmd)
    print('================>')
    os.system(eval_cmd)

    if args.eval_opt2 is not None:
        print('-'*50)
        eval_cmd = f'python basicsr/test.py --opt {args.eval_opt2} --name {args.name} ' + ' '.join(unparsed) + f' --path__pretrain_network_g {final_model_path}'
        print(eval_cmd)
        print('================>')
        os.system(eval_cmd)


    saver_folder = os.path.join('experiments', args.name)
    print(f'Saved at: {saver_folder}')




