# 2023.3.20-Written for building NetworkExpansion
#            Huawei Technologies Co., Ltd. <foss@huawei.com>


deit_base_reproduce = {
    'target_model': 'deit_base_patch16_224',
    'restart_lr': False,
    'width_fn': 'interp',
    'depth_fn': 'insert',
    'schedule': [
        {'num_heads': 12, 'depth': 12}
    ]
}

deit_base_depth_6_9_12 = {
    'target_model': 'deit_base_patch16_224',
    'restart_lr': False,
    'width_fn': 'fpi',
    'depth_fn': 'insert',
    'schedule': [
        {'num_heads': 12, 'depth': 6},
        {'num_heads': 12, 'depth': 9},
        {'num_heads': 12, 'depth': 12}
    ]
}

deit_base_depth_6_12 = {
    'target_model': 'deit_base_patch16_224',
    'restart_lr': False,
    'width_fn': 'fpi',
    'depth_fn': 'insert',
    'schedule': [
        {'num_heads': 12, 'depth': 6},
        {'num_heads': 12, 'depth': 12}
    ]
}

deit_base_depth_8_12 = {
    'target_model': 'deit_base_patch16_224',
    'restart_lr': False,
    'width_fn': 'fpi',
    'depth_fn': 'insert',
    'schedule': [
        {'num_heads': 12, 'depth': 8},
        {'num_heads': 12, 'depth': 12}
    ]
}

deit_base_depth_4_8_12 = {
    'target_model': 'deit_base_patch16_224',
    'restart_lr': False,
    'width_fn': 'fpi',
    'depth_fn': 'insert',
    'schedule': [
        {'num_heads': 12, 'depth': 4},
        {'num_heads': 12, 'depth': 8},
        {'num_heads': 12, 'depth': 12}
    ]
}
