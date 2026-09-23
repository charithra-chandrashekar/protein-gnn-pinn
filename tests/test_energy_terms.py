"""Tests for individual energy terms."""

import pytest

from src.physics.energy_terms import bonded_energy, nonbonded_energy, solvation_energy


@pytest.mark.parametrize(
    "fn",
    [bonded_energy, nonbonded_energy, solvation_energy],
)
def test_energy_terms_not_implemented(fn):
    with pytest.raises(NotImplementedError):
        fn(None, None, {})
