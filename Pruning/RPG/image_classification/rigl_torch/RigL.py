# 2023.11-Modified some parts in the code
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from .util import get_W
from .math_policies import *
from functools import partial




def magnitude_scorer(w):
    return w.data.abs()


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step



class RigLScheduler:

    def __init__(self, model, optimizer, dense_allocation=1, T_end=None, sparsity_distribution='uniform', ignore_linear_layers=True, delta=100, alpha=0.3, static_topo=False, grad_accumulation_n=1, state_dict=None, args=None):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception('Dense allocation must be on the interval (0, 1]. Got: %f' % dense_allocation)
        self.args=args
        self.model = model
        self.optimizer = optimizer
        self.stationary = True
        self.W, self._linear_layers_mask = get_W(model, return_linear_layers_mask=True)

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)
            
        self.dense_allocation = dense_allocation
        self.N = [torch.numel(w) for w in self.W]
        self.scorer = magnitude_scorer # default: magnitude prune
        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None

            # define sparsity allocation

            if sparsity_distribution == "uniform":
                self.S = []
                for i, (W, is_linear) in enumerate(zip(self.W, self._linear_layers_mask)):
                    # when using uniform sparsity, the first layer is always 100% dense
                    # UNLESS there is only 1 layer
                    is_first_layer = i == 0
                    if is_first_layer and self.sparsity_distribution == 'uniform' and len(self.W) > 1:
                        self.S.append(0)

                    elif is_linear and self.ignore_linear_layers:
                        # if choosing to ignore linear layers, keep them 100% dense
                        self.S.append(0)

                    else:
                        self.S.append(1-dense_allocation)
            elif sparsity_distribution == 'alluniform':
                self.S = [1-dense_allocation] * len(self.W)
            elif sparsity_distribution in ["erk"]:
                self.S = self.get_erdos_renyi_dist(is_kernel=True)
            # dynamic methods
            elif '-' in sparsity_distribution:
                self.stationary = False
                sparsity_hyper = sparsity_distribution.split('-')
                score_type = sparsity_hyper[0]
                policy_type = sparsity_hyper[1]
                self.scorer = {'magnitude':magnitude_scorer}[score_type]
                self.math_policy = {'exponential':exponential}[policy_type]
                if policy_type == 'acdc': # acdc: delta stands for a round of ac/dc, 5 epoch
                    assert hasattr(args, 'train_loader_len') # for safety
                    args.delta = delta = delta*args.train_loader_len # delta is epoch for acdc; change to iter

                if len(sparsity_hyper) > 2 and sparsity_hyper[2] == 'warm':
                    # e.g. magnitude-exponential-warm-0.2
                    self.math_policy = partial(warming, policy=self.math_policy, warmup_proportion=float(sparsity_hyper[3]))
                self.S = [0] * len(self.W)
            else:
                raise NotImplementedError(f'{sparsity_distribution} not implemented!')

            # randomly sparsify model according to S
            self.random_sparsify(stationary=self.stationary)
            
            # scheduler keeps a log of how many times it's called. this is how it does scheduling
            self.step = 0
            self.rigl_steps = 0

            # define the actual schedule
            if args.delta == 0:
                args.delta = delta = 10 * args.train_loader_len # never update if delta=0
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end


        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] < 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_rigl_backward_hook', False):
                raise Exception("This model already has been registered to a RigLScheduler.")

            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_rigl_backward_hook',True)



    def get_erdos_renyi_dist(
        self, is_kernel: bool = True, is_reshape: bool = False, dense_first=False, dense_fc=False,
    ):
        _erk_power_scale =1.0
        epsilon = 1.0
        is_epsilon_valid=False
        _dense_layers=set()
        if dense_fc:
            _dense_layers.add(len(self.W) - 1)
        if dense_first:
            _dense_layers.add(0)
        while not is_epsilon_valid:
            divisor =0
            rhs = 0
            raw_probabilities = {}
            for name, mask in enumerate(self.W):
                n_param=np.prod(mask.shape)
                n_zeros=int(n_param*(1-self.dense_allocation))
                n_ones=int(n_param*self.dense_allocation)

                if name in _dense_layers:

                    rhs -= n_zeros

                else:
                    rhs+=n_ones

                    if is_reshape and is_kernel: # erv2
                        raw_probabilities[name] = (
                            (mask.shape[0] + np.prod(mask.shape[1:])) / np.prod(mask.shape)
                        ) ** _erk_power_scale

                    elif is_kernel:

                        raw_probabilities[name]=(
                            np.sum(mask.shape) / np.prod(mask.shape)
                        ) ** _erk_power_scale
                    else:

                        n_in, n_out = mask.shape[:2]
                        raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs/divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid =False
                for mask_name,mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:

                        _dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        prob_dict = [None] * len(self.W)

        for name,weight in enumerate(self.W):
            if name in _dense_layers:
                prob = 1.0
            else:
                prob = epsilon * raw_probabilities[name]

            prob_dict[name] = 1 - prob # return proportion to be pruned!
        return prob_dict

    def state_dict(self):
        obj = {
            'dense_allocation': self.dense_allocation,
            'S': self.S,
            'N': self.N,
            'hyperparams': {
                'delta_T': self.delta_T,
                'alpha': self.alpha,
                'T_end': self.T_end,
                'ignore_linear_layers': self.ignore_linear_layers,
                'static_topo': self.static_topo,
                'sparsity_distribution': self.sparsity_distribution,
                'grad_accumulation_n': self.grad_accumulation_n,
            },
            'step': self.step,
            'rigl_steps': self.rigl_steps,
            'backward_masks': self.backward_masks,
            '_linear_layers_mask': self._linear_layers_mask,
        }

        return obj

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)


    @torch.no_grad()
    def random_sparsify(self,stationary=True):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] < 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n)
            perm = perm[:s]
            flat_mask = torch.ones(n, device=w.device)
            if self.args.iterative_T_end_percent == 0. and stationary: # added, support both RigL and iterative
                flat_mask[perm] = 0 # disabled 
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)
    @torch.no_grad()
    def layerwise_sorted_init_sparsify(self, scorer=magnitude_scorer):
        ''' Use sorted score and layerwise sparsity self.S to init sparsify. '''
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] < 0:
                self.backward_masks.append(None)
                continue
            sparsity = self.S[l]
            top_k = round(w.numel() * (1.-sparsity)) # params to save
            if sparsity != 0:
                threshold, _ = scorer(w).view(-1).kthvalue(top_k)
            else:
                threshold = 0.
            mask = (scorer(w) >= threshold).bool()

            if is_dist:
                dist.broadcast(mask, 0)
            w *= mask
            self.backward_masks.append(mask)
            
    @torch.no_grad()
    def sorted_init_sparsify(self, sparsity, scorer=magnitude_scorer):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        scores = []

        # accumulate score for threshold
        for l,w in enumerate(self.W):
            scores.append(scorer(w).view(-1))




        scores = torch.cat(scores, dim=0)
        top_k = round(sparsity*scores.size(0))
        if sparsity != 0:
            threshold, _ = scores.kthvalue(top_k)
        else:
            threshold = 0.

        for l, w in enumerate(self.W):
            mask = (scorer(w) >= threshold)
            
            if is_dist:
                dist.broadcast(mask,0)
            mask=mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    def __str__(self):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return '' # return nothing for other ranks in distribution mode
        s = 'RigLScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_conv_params = 0
        total_nonzero = 0
        total_conv_nonzero = 0

        for N, S, mask, W, is_linear in zip(self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S
            if not is_linear:
                total_conv_nonzero += N-actual_S
                total_conv_params += N

        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        
        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        s += 'total_CONV_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_conv_nonzero, total_conv_params, float(total_conv_nonzero)/float(total_conv_params)*100)) + ',\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_rigl_steps=' + str(self.rigl_steps) + ',\n'
        s += 'ignoring_linear_layers=' + str(self.ignore_linear_layers) + ',\n'
        s += 'sparsity_distribution=' + str(self.sparsity_distribution) + ',\n'

        return s + ')'


    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s < 0:
                continue

            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                # mask the momentum matrix
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s < 0:
                continue
                
            w *= mask


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s < 0:
                continue

            w.grad *= mask

    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next rigl step is, 
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_rigl_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def iterative_cosine_annealing(self,D):
        # for gradual decay with fixed layerwise distribution

        S = 1. - D

        if self.step < self.args.iterative_T_end:
            rate = S / 2 * (1+np.cos((self.step*np.pi)/self.args.iterative_T_end))
        else:
            rate = 0

        return D + rate


    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step < self.T_end: # check schedule
            self._rigl_step()
            self.rigl_steps += 1
            if self.alpha == 0.: # still train for the case of acdc and gmp
                return True
            return False
        return True


    @torch.no_grad()
    def _rigl_step(self):
        if not self.stationary:
            remaining = self.math_policy(self.rigl_steps, self.dense_allocation, self.args.iterative_T_end//self.delta_T, self.args)
            self.sorted_init_sparsify(1. - remaining, self.scorer)
            # generate self.S
            self.S = []
            for mask in self.backward_masks:
                self.S.append((mask==0).sum().item() / mask.numel())
            print("LAYERWISE SPARSITY: ", self.S)
        drop_fraction = self.cosine_annealing()

        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:

                continue

            current_mask = self.backward_masks[l]

            # calculate raw scores
            score_drop = self.scorer(w)
            score_grow = self.backward_hook_objects[l].dense_grad


            score_grow = torch.abs(score_grow)


            # if is distributed, synchronize scores
            if is_dist:
                dist.all_reduce(score_drop)  # get the sum of all drop scores
                score_drop /= world_size     # divide by world size (average the drop scores)

                dist.all_reduce(score_grow)  # get the sum of all grow scores
                score_grow /= world_size     # divide by world size (average the grow scores)

            # calculate drop/grow quantities
            n_total = self.N[l]

            if self.stationary:
                n_iterative_remain = int(self.iterative_cosine_annealing((1-self.S[l])) * n_total)
            else: # dynamic
                n_iterative_remain = int((1-self.S[l])*n_total)
                if remaining == self.dense_allocation:
                    self.stationary = True
            n_prune = int((1-self.S[l]) * n_total * drop_fraction)
            n_keep = n_iterative_remain - n_prune

            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values=torch.where(
                torch.arange(n_total,device=w.device) < n_keep,
                torch.ones_like(sorted_indices),
                torch.zeros_like(sorted_indices))
            mask1 = new_values.scatter(0, sorted_indices, new_values)
            if self.alpha != 0.:
                # flatten grow scores
                score_grow = score_grow.view(-1)

                # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
                score_grow_lifted = torch.where(
                                    mask1 == 1, 
                                    torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                    score_grow)

                # create grow mask
                _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
                new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_prune,
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
                mask2 = new_values.scatter(0, sorted_indices, new_values)

                mask2_reshaped = torch.reshape(mask2, current_mask.shape)
                grow_tensor = torch.zeros_like(w)
                
                REINIT_WHEN_SAME = False
                if REINIT_WHEN_SAME:
                    raise NotImplementedError()
                else:
                    new_connections = ((mask2_reshaped == 1) & (current_mask == 0))
                # update new weights to be initialized as zeros and update the weight tensors
                new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
                w.data = new_weights

                mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()
            else:  # no grow parameters
                mask_combined = torch.reshape(mask1, current_mask.shape).bool()
            # update the mask
            current_mask.data = mask_combined

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients() 