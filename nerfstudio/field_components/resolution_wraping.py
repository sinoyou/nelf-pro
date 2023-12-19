import numpy as np 
import torch
import torch.nn.functional as F

def apply_theta_wraping(theta: torch.Tensor, factor: int, period = (0, np.pi)):
    """
    theta: [..., 1] in [0, pi]
    factor: int
    """
    if factor <= 1:
        return theta
    assert factor > 1
    mapping_factor_one = lambda x: x % (period[1] - period[0])
    return mapping_factor_one((theta - period[0]) * factor) + period[0]

def apply_phi_wraping(phi: torch.Tensor, factor: int, period = (-np.pi, np.pi)):
    """
    phi: [..., 1] in [-pi, pi]
    factor: int
    """
    if factor <= 1:
        return phi
    assert factor > 1
    mapping_factor_one = lambda x: x % (period[1] - period[0])
    return mapping_factor_one((phi - period[0]) * factor) + period[0]