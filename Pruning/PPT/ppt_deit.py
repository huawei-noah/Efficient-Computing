# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome import bipartite_soft_matching, merge_source, merge_wavg
from tome import parse_r

# Copy and refine from DynamicViT
def batch_index_select(x, idx):
    if x is not None:
        if len(x.size()) == 3:
            B, N, C = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
            return out
        elif len(x.size()) == 2:
            B, N = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
            return out
        else:
            raise NotImplementedError
    else:
        return x


class PPTAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
     - Return the tokens scores
    """
    # copy and refine from the ATS
    @staticmethod
    def score_assignment_step(attn, v):
        """
        Token Score Assignment Step.
        :param attn: attention matrix
        :param v: values
        :return: sorted significance scores and their corresponding indices
        """

        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
        )  # value norm of size [B x T]
        significance_score = attn[:, :, 0].sum(
            dim=1
        )  # attention weights of CLS token of size [B x T]
        significance_score = significance_score * v_norm  # [B x T]
        significance_score = significance_score[:, 1:]  # [B x T-1]

        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x T-1]
        
        return significance_score
    
    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        scores = self.score_assignment_step(attn, v) # [B, N-1]
        cls_score = 2*torch.ones(scores.shape[0], 1).to(scores.device) # confirm cls tokens are reserved
        scores = torch.cat([cls_score, scores], dim=-1) # [B, N]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k, scores as well here
        return x, k.mean(1), scores

class block(Block):
    """
    Modifications:
     - Consider the token size.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._pp_info["size"] if self._pp_info["prop_attn"] else None
        x_attn, _, _ = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        self._pp_info["block_index"] = self._pp_info["block_index"]+1
        return x

class pp_block_adaptive(Block):
    """
    Modifications:
     - Consider the token size.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._pp_info["size"] if self._pp_info["prop_attn"] else None
        if attn_size is None:
            attn_size = torch.ones(x.shape[0], x.shape[1], 1).to(x.device)
        x_attn, metric, scores = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        
        x_pooling = None
        x_pruning = None
        attn_size_pooling = None
        attn_size_pruning = None

        r = self._pp_info["r"].pop(0)
        r = min(r, (x.shape[1] - 1) // 2)

        # calculate the var of feature scores
        S_op = torch.var(scores[:, 1:], dim=1)      # [B] 

        if r > 0:    
            # Apply pooling here
            pooling_imgs_indices = torch.nonzero(S_op < self._pp_info["OP_threshold"]).squeeze(-1)
            if pooling_imgs_indices.shape[0] > 0:
                x_pooling = x[pooling_imgs_indices]
                metric = metric[pooling_imgs_indices]
                attn_size_pooling = attn_size[pooling_imgs_indices]
                merge, _ = bipartite_soft_matching(metric, r, self._pp_info["class_token"], self._pp_info["distill_token"])
                if self._pp_info["trace_source"]:
                    self._pp_info["source"] = merge_source(merge, x_pooling, self._pp_info["source"])
                x_pooling, attn_size_pooling = merge_wavg(merge, x_pooling, attn_size_pooling)

            # Apply pruning here
            pruning_imgs_indices = torch.nonzero(S_op >= self._pp_info["OP_threshold"]).squeeze(-1)
            if pruning_imgs_indices.shape[0] > 0:
                x_pruning = x[pruning_imgs_indices]
                scores = scores[pruning_imgs_indices]
                attn_size_pruning = attn_size[pruning_imgs_indices]
                _, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                pruning_indices = sorted_indices[:, :-r] # reserved tokens
                if self._pp_info["trace_source"]:
                    if self._pp_info["source"] is None:
                        n, t, _ = x_pruning.shape
                        self._pp_info["source"] = torch.eye(t, device=x_pruning.device)[None, ...].expand(n, t, t)
                    self._pp_info["source"] = batch_index_select(self._pp_info["source"], pruning_indices)
                x_pruning = batch_index_select(x_pruning, pruning_indices)
                attn_size_pruning = batch_index_select(attn_size_pruning, pruning_indices)

            # Merge two subbatch
            if x_pooling is not None and x_pruning is not None:
                x = torch.zeros(x_pooling.shape[0]+x_pruning.shape[0], x_pooling.shape[1], x_pooling.shape[2]).to(x.device)
                x[pooling_imgs_indices], x[pruning_imgs_indices] = x_pooling, x_pruning
                self._pp_info["size"] = torch.ones(x.shape[0], x.shape[1], 1).to(x.device)
                self._pp_info["size"][pooling_imgs_indices], self._pp_info["size"][pruning_imgs_indices] = attn_size_pooling, attn_size_pruning
                
            elif x_pooling is not None:
                x = x_pooling
                self._pp_info["size"] = attn_size_pooling
            elif x_pruning is not None:
                x = x_pruning
                self._pp_info["size"] = attn_size_pruning
            else:
                pass

        x = x + self._drop_path2(self.mlp(self.norm2(x)))

        return x

class PPTVisionTransformer(VisionTransformer):
    """
    Modifications:
    - Initialize r, token size, and token sources.
    """
    def forward(self, x, *args, **kwdargs) -> torch.Tensor:
        self._pp_info["r"] = parse_r(len(self.blocks), self.r)
        self._pp_info["size"] = None
        self._pp_info["source"] = None
        self._pp_info["block_index"] = 0
        self._pp_info["epoch"] = self.epoch

        return super().forward(x, *args, **kwdargs)


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, pp_loc_list: list = [], threshold: float = 0
):
    """
    Applies PPT to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._pp_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    model.__class__ = PPTVisionTransformer
    model.r = 0
    model.epoch = 0
    model._pp_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
        "OP_threshold": threshold,
        "block_index": 0,
        "epoch": 0,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._pp_info["distill_token"] = True

    blk_idx = 0
    for module in model.modules():
        if isinstance(module, Block):
            if blk_idx in pp_loc_list:
                module.__class__ = pp_block_adaptive
                module._pp_info = model._pp_info
            else:
                module.__class__ = block
                module._pp_info = model._pp_info
            blk_idx = blk_idx + 1

        elif isinstance(module, Attention):
            module.__class__ = PPTAttention
