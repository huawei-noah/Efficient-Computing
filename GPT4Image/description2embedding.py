#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import clip
import time
import argparse
from tqdm import tqdm
from torch.nn import functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


@torch.no_grad()
def main(args):
    # load the list of descriptions
    caption_list = torch.load(args.caption)
    assert isinstance(caption_list, list)
    num_captions = len(caption_list)
    print(f'load caption file {args.caption} ...\n containing {num_captions} items.')
    
    model, preprocess = clip.load(args.model, device=args.device)

    all_result = []
    for i in tqdm(range(0, num_captions, args.bs)):
        cur_end_idx = min(i + args.bs, num_captions)

        cur_captions = caption_list[i:cur_end_idx]
        assert len(cur_captions) <= args.bs
        tokens = clip.tokenize(cur_captions, truncate=True).to(args.device)
        text_emb = model.encode_text(tokens)
        text_emb = F.normalize(text_emb, dim=-1, p=2)
        text_emb = text_emb.cpu()
        
        all_result += [text_emb]

    all_result = torch.cat(all_result, dim=0)
    print(all_result.shape)
    torch.save(all_result, './imagenet_clip_text_emb_0_1281166.pth')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--model', default='RN50', type=str, choices=['ViT-L/14', 'RN50'])
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--caption', default='minigpt4_caption_imagenet_train_0_1281166.pth', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    start_time = time.time()
    print("{}".format(args).replace(', ', ',\n'))

    main(args)
    
    end_time = time.time()
    time_used = end_time-start_time
    print(f'Time used: {time_used:.1f}s')
