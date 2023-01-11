"""Load attacker code and instantiate appropriate objects."""
import torch

from .optimization_based_attack import OptimizationBasedAttacker
from .multiscale_optimization_attack import MultiScaleOptimizationAttacker
from .optimization_with_label_attack import OptimizationJointAttacker
from .optimization_permutation_attack import OptimizationPermutationAttacker
from .analytic_attack import AnalyticAttacker, ImprintAttacker, DecepticonAttacker, AprilAttacker
from .recursive_attack import RecursiveAttacker


def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    # cfg_attack.attack_type == "optimization":   # NOTE(dchu): FISHING
    attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)

    return attacker


__all__ = ["prepare_attack"]
