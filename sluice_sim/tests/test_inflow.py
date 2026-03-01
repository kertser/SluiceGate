"""Tests for inflow profiles."""

import math
import pytest

from sluice_sim.profiles.inflow import (
    ConstantInflow,
    StepInflow,
    RampInflow,
    SineInflow,
    M3H_TO_M3S,
)


class TestConstantInflow:
    def test_returns_constant(self):
        profile = ConstantInflow(q_m3h=500.0)
        assert profile(0.0) == pytest.approx(500.0 * M3H_TO_M3S)
        assert profile(100.0) == pytest.approx(500.0 * M3H_TO_M3S)

    def test_default(self):
        profile = ConstantInflow()
        assert profile(0.0) == pytest.approx(1000.0 * M3H_TO_M3S)


class TestStepInflow:
    def test_before_step(self):
        p = StepInflow(q_base_m3h=1000.0, q_step_m3h=200.0, t_step=5.0)
        assert p(4.99) == pytest.approx(1000.0 * M3H_TO_M3S)

    def test_at_step(self):
        p = StepInflow(q_base_m3h=1000.0, q_step_m3h=200.0, t_step=5.0)
        assert p(5.0) == pytest.approx(1200.0 * M3H_TO_M3S)

    def test_after_step(self):
        p = StepInflow(q_base_m3h=1000.0, q_step_m3h=200.0, t_step=5.0)
        assert p(100.0) == pytest.approx(1200.0 * M3H_TO_M3S)


class TestRampInflow:
    def test_before_ramp(self):
        p = RampInflow(q_base_m3h=1000.0, q_delta_m3h=100.0, t_start=10.0, duration=10.0)
        assert p(5.0) == pytest.approx(1000.0 * M3H_TO_M3S)

    def test_mid_ramp(self):
        p = RampInflow(q_base_m3h=1000.0, q_delta_m3h=100.0, t_start=10.0, duration=10.0)
        # At t=15 → halfway → base + 50
        assert p(15.0) == pytest.approx(1050.0 * M3H_TO_M3S)

    def test_after_ramp(self):
        p = RampInflow(q_base_m3h=1000.0, q_delta_m3h=100.0, t_start=10.0, duration=10.0)
        assert p(25.0) == pytest.approx(1100.0 * M3H_TO_M3S)

    def test_at_ramp_start(self):
        p = RampInflow(q_base_m3h=1000.0, q_delta_m3h=100.0, t_start=10.0, duration=10.0)
        assert p(10.0) == pytest.approx(1000.0 * M3H_TO_M3S)

    def test_at_ramp_end(self):
        p = RampInflow(q_base_m3h=1000.0, q_delta_m3h=100.0, t_start=10.0, duration=10.0)
        assert p(20.0) == pytest.approx(1100.0 * M3H_TO_M3S)


class TestSineInflow:
    def test_at_zero(self):
        p = SineInflow(q_base_m3h=1000.0, amplitude_m3h=100.0, period=60.0)
        assert p(0.0) == pytest.approx(1000.0 * M3H_TO_M3S)

    def test_quarter_period(self):
        p = SineInflow(q_base_m3h=1000.0, amplitude_m3h=100.0, period=60.0)
        assert p(15.0) == pytest.approx(1100.0 * M3H_TO_M3S, rel=1e-6)

    def test_no_negative_flow(self):
        p = SineInflow(q_base_m3h=50.0, amplitude_m3h=200.0, period=10.0)
        # At 3/4 period the sine goes to -200 → base - 200 = -150 → clamped to 0
        assert p(7.5) == pytest.approx(0.0, abs=1e-10)

    def test_full_period(self):
        p = SineInflow(q_base_m3h=1000.0, amplitude_m3h=100.0, period=60.0)
        assert p(60.0) == pytest.approx(1000.0 * M3H_TO_M3S, abs=1e-10)

