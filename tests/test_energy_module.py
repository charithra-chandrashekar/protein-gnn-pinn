"""Standalone decoy-ranking test for EnergyModule."""

import pytest

from src.physics.energy_module import EnergyModule
from src.utils.metrics import decoy_ranking_score


def test_decoy_ranking_perfect():
    native = 1.0
    decoys = [2.0, 3.0, 4.0]
    assert decoy_ranking_score(native, decoys) == 1.0


def test_energy_module_requires_terms():
    module = EnergyModule({"energy_scales": {"bonded": 1.0, "nonbonded": 1.0, "solvation": 1.0}})
    with pytest.raises(NotImplementedError):
        module.forward(pos=None, edge_index=None)
