"""Tests for graph construction."""

import pytest

from src.data.graph_builder import build_graph


def test_build_graph_not_implemented():
    with pytest.raises(NotImplementedError):
        build_graph([])
