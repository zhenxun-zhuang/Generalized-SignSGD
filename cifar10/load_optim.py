"""
Load the desired optimizer.
"""

import torch.optim as optim
from sgd_clip import SGDClipGrad
from generalized_signsgd import GeneralizedSignSGD
from sgd_normalized import SGDNormalized


def load_optim(params, optim_method, eta0, eps, nesterov, momentum, beta2,
               weight_decay, clipping_param):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use.
        eta0: starting step size.
        eps: epsilon used in Adam/AdamW.
        nesterov: whether to use nesterov momentum (True) or not (False).
        momentum: momentum factor used in variants of SGD.
        beta2: used in Adam and GeneralizedSignSGD.
        weight_decay: weight decay factor.
        clipping_param: used in SGDClipGrad to control gradient clipping.

    Outputs:
        an optimizer
    """

    if optim_method == 'SGD':
        optimizer = optim.SGD(params=params, lr=eta0, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    if optim_method == 'SGDNormalized':
        optimizer = SGDNormalized(params=params, lr=eta0, momentum=momentum,
                                  weight_decay=weight_decay, nesterov=nesterov)
    elif optim_method == 'SGDClipGrad':
        optimizer = SGDClipGrad(params, lr=eta0, momentum=0,
                                weight_decay=weight_decay, nesterov=nesterov,
                                clipping_param=clipping_param)
    elif optim_method == 'SGDClipMomentum':
        optimizer = SGDClipGrad(params, lr=eta0, momentum=momentum,
                                weight_decay=weight_decay, nesterov=nesterov,
                                clipping_param=clipping_param)
    elif optim_method == 'Adam':
        optimizer = optim.Adam(params=params, lr=eta0, eps=eps, betas=(momentum, beta2),
                               weight_decay=weight_decay)
    elif optim_method == 'GeneralizedSignSGD':
        optimizer = GeneralizedSignSGD(params=params, lr=eta0, eps=eps,
                                       betas=(momentum, beta2), weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_method))

    return optimizer
