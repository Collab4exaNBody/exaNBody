"""pyexanbody: run the microCosmos solar-system sample and read results back."""

import numpy as np
import pytest

import pyexanbody as xnb


@pytest.fixture
def solar_system(tmp_path, monkeypatch):
    """Run a few iterations of solar_system.msp and yield the live context."""
    monkeypatch.chdir(tmp_path)  # plugins.db and dumps are written in the working directory
    msp = xnb.find_example("solar_system.msp")
    argv = xnb.pyonika.command_line(msp, program="test", overrides={
        "global": {"simulation_end_iteration": 3, "simulation_dump_frequency": -1},
        "simulation_epilog": [{"extract_particle_fields": {"fields": ["rx", "ry", "rz", "vx", "type"]}}],
    })
    ctx = xnb.init(argv)
    assert ctx.error_code < 0
    xnb.run(ctx)
    yield ctx
    xnb.end(ctx)


def test_particles(solar_system):
    p = xnb.particles(solar_system)
    assert p is not None
    assert set(p) == {"rx", "ry", "rz", "vx", "type", "id"}
    n = len(p["id"])
    assert n > 0
    for name in ("rx", "ry", "rz", "vx"):
        assert p[name].shape == (n,) and p[name].dtype == np.float64
    assert p["id"].dtype == np.uint64
    assert len(np.unique(p["id"])) == n


def test_sim_stats(solar_system):
    stats = xnb.read_sim_stats(solar_system.graph)
    assert stats["particle_count"] == len(xnb.particles(solar_system)["id"])
    assert np.isfinite(stats["kinetic_energy"])


def test_unknown_field_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    msp = xnb.find_example("solar_system.msp")
    argv = xnb.pyonika.command_line(msp, program="test", overrides={
        "global": {"simulation_end_iteration": 1, "simulation_dump_frequency": -1},
        "simulation_epilog": [{"extract_particle_fields": {"fields": ["no_such_field"]}}],
    })
    ctx = xnb.init(argv)
    with pytest.raises(RuntimeError, match="no_such_field"):
        xnb.run(ctx)
    xnb.end(ctx)


def test_examples_listed():
    assert any(e.endswith("solar_system.msp") for e in xnb.list_examples())
