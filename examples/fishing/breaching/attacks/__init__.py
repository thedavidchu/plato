"""Load attacker code and instantiate appropriate objects."""
import torch

from .optimization_based_attack import OptimizationBasedAttacker



def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    # NOTE: cfg_attack.attack_type == "optimization"
    attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)

    return attacker


__all__ = ["prepare_attack"]
