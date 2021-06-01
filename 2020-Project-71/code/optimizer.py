import torch


OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def get_optimizer(opt_params, model_params):
    opt_type = opt_params['type']
    assert opt_type in OPTIMIZERS, "Optimizer type is unknown: '{}'".format(opt_type)
    kwargs = opt_params['kwargs']
    return OPTIMIZERS[opt_type](model_params, **kwargs)