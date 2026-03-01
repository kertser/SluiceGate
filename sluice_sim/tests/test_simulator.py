"""Tests for the simulator orchestrator."""

import json
import tempfile
from pathlib import Path

import pytest

from sluice_sim.sim.simulator import SimConfig, Simulator


@pytest.fixture
def default_sim() -> Simulator:
    return Simulator(SimConfig())


class TestSimulatorRun:
    def test_returns_dataframe_with_correct_columns(self, default_sim: Simulator):
        df = default_sim.run()
        expected_cols = {"t", "H", "a", "a_cm", "Qin", "Qout", "e", "u", "duty", "cmd"}
        assert expected_cols.issubset(set(df.columns))

    def test_dataframe_not_empty(self, default_sim: Simulator):
        df = default_sim.run()
        assert len(df) > 0

    def test_time_column_increases(self, default_sim: Simulator):
        df = default_sim.run()
        assert (df["t"].diff().dropna() > 0).all()

    def test_H_stays_positive(self, default_sim: Simulator):
        df = default_sim.run()
        assert (df["H"] >= 0).all()

    def test_a_within_bounds(self, default_sim: Simulator):
        df = default_sim.run()
        assert (df["a"] >= 0).all()
        assert (df["a"] <= default_sim.config.plant.a_max + 1e-9).all()


class TestEquilibriumReached:
    def test_default_scenario_converges(self):
        """With the default ramp scenario, H should approach H_set."""
        cfg = SimConfig(t_end=300.0)  # longer run
        sim = Simulator(cfg)
        df = sim.run()
        # Last 10% of data should be close to setpoint
        tail = df.tail(int(len(df) * 0.1))
        mean_H = tail["H"].mean()
        assert abs(mean_H - cfg.H_set) < 0.1, f"Mean H={mean_H} not close to H_set={cfg.H_set}"


class TestStepMode:
    def test_step_once(self, default_sim: Simulator):
        row = default_sim.step_once()
        assert "t" in row
        assert "H" in row

    def test_step_accumulates(self, default_sim: Simulator):
        for _ in range(10):
            default_sim.step_once()
        df = default_sim.get_dataframe()
        assert len(df) == 10


class TestExportImport:
    def test_csv_export(self, default_sim: Simulator):
        default_sim.run()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        default_sim.export_csv(path)
        content = Path(path).read_text()
        assert "t,H,a,a_cm,Qin,Qout,e,u,duty,cmd" in content

    def test_scenario_roundtrip(self, default_sim: Simulator):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        default_sim.save_scenario_json(path)
        loaded = Simulator.load_scenario_json(path)
        assert loaded.H_set == default_sim.config.H_set
        assert loaded.plant.b == default_sim.config.plant.b
        assert loaded.inflow_type == default_sim.config.inflow_type


class TestNoise:
    def test_noisy_run_completes(self):
        cfg = SimConfig(noise_std=0.01, noise_seed=42, t_end=10.0)
        sim = Simulator(cfg)
        df = sim.run()
        assert len(df) > 0

    def test_deterministic_with_seed(self):
        cfg1 = SimConfig(noise_std=0.01, noise_seed=42, t_end=5.0)
        cfg2 = SimConfig(noise_std=0.01, noise_seed=42, t_end=5.0)
        df1 = Simulator(cfg1).run()
        df2 = Simulator(cfg2).run()
        assert (df1["H"] == df2["H"]).all()

