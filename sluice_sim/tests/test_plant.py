"""Tests for the plant model."""

import math
import pytest

from sluice_sim.models.plant import Plant, PlantParams, PlantState, MotorCommand


@pytest.fixture
def plant() -> Plant:
    return Plant(PlantParams())


@pytest.fixture
def default_state() -> PlantState:
    return PlantState(H=1.5, a=0.1)


class TestDischarge:
    def test_increases_with_H(self, plant: Plant):
        q1 = plant.compute_discharge(H=1.0, a=0.1)
        q2 = plant.compute_discharge(H=2.0, a=0.1)
        assert q2 > q1

    def test_increases_with_a(self, plant: Plant):
        q1 = plant.compute_discharge(H=1.5, a=0.1)
        q2 = plant.compute_discharge(H=1.5, a=0.2)
        assert q2 > q1

    def test_zero_when_gate_closed(self, plant: Plant):
        assert plant.compute_discharge(H=2.0, a=0.0) == 0.0

    def test_zero_when_no_water(self, plant: Plant):
        assert plant.compute_discharge(H=0.0, a=0.2) == 0.0

    def test_known_value(self):
        """Check against hand-calculated value."""
        p = PlantParams(Cd=0.65, b=0.75, g=9.81)
        plant = Plant(p)
        H, a = 2.0, 0.1
        expected = 0.65 * 0.75 * 0.1 * math.sqrt(2 * 9.81 * 2.0)
        assert plant.compute_discharge(H, a) == pytest.approx(expected)


class TestActuator:
    def test_open_increases_a(self, plant: Plant, default_state: PlantState):
        dt = 0.1
        Qin = 0.2
        new_state, _ = plant.step(default_state, MotorCommand.OPEN, Qin, dt)
        assert new_state.a > default_state.a

    def test_close_decreases_a(self, plant: Plant, default_state: PlantState):
        dt = 0.1
        Qin = 0.2
        new_state, _ = plant.step(default_state, MotorCommand.CLOSE, Qin, dt)
        assert new_state.a < default_state.a

    def test_stop_keeps_a(self, plant: Plant, default_state: PlantState):
        dt = 0.1
        Qin = 0.2
        new_state, _ = plant.step(default_state, MotorCommand.STOP, Qin, dt)
        assert new_state.a == default_state.a

    def test_clamp_upper(self):
        p = PlantParams(a_max=0.2)
        plant = Plant(p)
        state = PlantState(H=1.0, a=0.19)
        # Big dt so opening overshoots
        new_state, _ = plant.step(state, MotorCommand.OPEN, 0.0, 10.0)
        assert new_state.a == pytest.approx(0.2)

    def test_clamp_lower(self, plant: Plant):
        state = PlantState(H=1.0, a=0.005)
        new_state, _ = plant.step(state, MotorCommand.CLOSE, 0.0, 10.0)
        assert new_state.a == 0.0


class TestMassBalance:
    def test_H_never_negative(self, plant: Plant):
        """Even with zero inflow and large outflow, H must not go negative."""
        state = PlantState(H=0.001, a=0.45)
        for _ in range(1000):
            state, _ = plant.step(state, MotorCommand.STOP, 0.0, 0.02)
        assert state.H >= 0.0

    def test_level_rises_with_excess_inflow(self, plant: Plant):
        state = PlantState(H=1.0, a=0.0)  # gate closed
        new_state, _ = plant.step(state, MotorCommand.STOP, 1.0, 0.1)
        assert new_state.H > state.H

    def test_H_clamped_at_H_max(self):
        """Water level must not exceed H_max (overflow protection)."""
        params = PlantParams(H_max=2.5)
        plant = Plant(params)
        state = PlantState(H=2.49, a=0.0)  # gate closed, near max
        # Large inflow should push level above H_max → clamped
        new_state, _ = plant.step(state, MotorCommand.STOP, 10.0, 1.0)
        assert new_state.H <= params.H_max


class TestEquilibrium:
    def test_known_equilibrium(self):
        Qin = 1000.0 / 3600.0  # m³/s
        Cd, b, g, H_set = 0.65, 0.75, 9.81, 2.0
        a_eq = Plant.compute_equilibrium_opening(Qin, Cd, b, g, H_set)
        # Verify round-trip
        Qout = Cd * b * a_eq * math.sqrt(2 * g * H_set)
        assert Qout == pytest.approx(Qin, rel=1e-6)

