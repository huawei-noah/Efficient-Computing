#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.
import math as m


def exponential(current_step, dense_allocation, update_times, args):
    factor = m.pow(dense_allocation, 1./update_times)
    remaining = m.pow(factor, current_step) if current_step < update_times else dense_allocation
    return remaining


def warming(current_step, dense_allocation, update_times, args, policy, warmup_proportion):
    warmup_times = round(update_times * warmup_proportion)
    if current_step < warmup_times:
        return 1.
    else:
        return policy(current_step-warmup_times, dense_allocation, update_times-warmup_times, args)
