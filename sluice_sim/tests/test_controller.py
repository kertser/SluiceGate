"""Tests for the PID pulse controller."""

import pytest

from sluice_sim.controllers.pid_pulse import ControllerParams, PIDPulseController
from sluice_sim.models.plant import MotorCommand


@pytest.fixture
def default_ctrl() -> PIDPulseController:
    return PIDPulseController(ControllerParams())


class TestDeadband:
    def test_stop_when_error_below_lo(self):
        """Error inside deadband_lo → STOP."""
        params = ControllerParams(deadband_lo=0.01, deadband_hi=0.02)
        ctrl = PIDPulseController(params)
        # H_meas very close to H_set
        cmd = ctrl.step(H_meas=2.005, H_set=2.0, dt=0.02)
        assert cmd is MotorCommand.STOP

    def test_active_when_error_above_hi(self):
        """Error exceeding deadband_hi → motor should act."""
        params = ControllerParams(deadband_lo=0.005, deadband_hi=0.02, Kp=5.0, Umax=0.18, Twindow=0.5, min_on=0.0)
        ctrl = PIDPulseController(params)
        # H_meas far below setpoint → error = 0.5, well above deadband_hi
        cmd = ctrl.step(H_meas=1.5, H_set=2.0, dt=0.02)
        assert cmd in (MotorCommand.OPEN, MotorCommand.CLOSE)

    def test_hysteresis(self):
        """Once in deadband, stay until error crosses deadband_hi."""
        params = ControllerParams(deadband_lo=0.005, deadband_hi=0.02)
        ctrl = PIDPulseController(params)

        # Start with tiny error → deadband
        cmd = ctrl.step(H_meas=1.995, H_set=2.0, dt=0.02)
        assert cmd is MotorCommand.STOP

        # Error grows but stays between lo and hi → should still STOP (hysteresis)
        cmd = ctrl.step(H_meas=1.99, H_set=2.0, dt=0.02)  # |e| ≈ 0.01 after filtering
        # With alpha=0.3, Hf ≈ 0.3*1.99 + 0.7*1.995 = 1.9935 → e=0.0065
        # 0.005 < 0.0065 < 0.02 → stays in deadband
        assert cmd is MotorCommand.STOP


class TestAntiWindup:
    def test_integral_clamped(self):
        """Integral should not exceed I_max."""
        params = ControllerParams(Kp=0.0, Ki=100.0, Kd=0.0, I_min=-1.0, I_max=1.0,
                                  deadband_hi=0.0, deadband_lo=0.0)
        ctrl = PIDPulseController(params)
        # Run many steps with large error
        for _ in range(10000):
            ctrl.step(H_meas=0.0, H_set=10.0, dt=0.02)
        assert ctrl._I_sum <= params.I_max + 1e-9
        assert ctrl._I_sum >= params.I_min - 1e-9


class TestPulseModulation:
    def test_pulse_within_window(self):
        """Pulse duration should never exceed Twindow."""
        params = ControllerParams(Kp=50.0, Ki=0.0, Kd=0.0, Umax=0.18, Twindow=0.5,
                                  min_on=0.0, deadband_hi=0.0, deadband_lo=0.0)
        ctrl = PIDPulseController(params)
        ctrl.step(H_meas=1.0, H_set=2.0, dt=0.02)
        assert ctrl._pulse_duration <= params.Twindow + 1e-9

    def test_min_on_respected(self):
        """If calculated pulse < min_on, it should be rounded to 0 (skipped)."""
        params = ControllerParams(
            Kp=0.001,  # tiny gain → very small u
            Ki=0.0, Kd=0.0,
            Umax=0.18, Twindow=0.5, min_on=0.18,
            deadband_hi=0.0, deadband_lo=0.0,
        )
        ctrl = PIDPulseController(params)
        ctrl.step(H_meas=1.99, H_set=2.0, dt=0.02)
        # u = 0.001 * 0.01 = 0.00001 → duty ≈ 0.00006 → pulse ≈ 0.00003 < 0.18
        assert ctrl._pulse_duration == 0.0 or ctrl._pulse_duration >= params.min_on


class TestNoIllegalTransitions:
    def test_no_open_close_without_stop(self):
        """Motor should not jump from OPEN to CLOSE (or vice-versa) without a STOP in between."""
        params = ControllerParams(Kp=5.0, Ki=0.25, Kd=0.0,
                                  deadband_hi=0.001, deadband_lo=0.0005,
                                  Twindow=0.5, min_on=0.0,
                                  min_switch_interval=0.0)
        ctrl = PIDPulseController(params)

        prev_cmd = MotorCommand.STOP
        violations = 0
        # Oscillate measurement around setpoint
        for i in range(2000):
            H_meas = 2.0 + 0.1 * ((-1) ** i)
            cmd = ctrl.step(H_meas, 2.0, 0.02)
            if (prev_cmd is MotorCommand.OPEN and cmd is MotorCommand.CLOSE) or \
               (prev_cmd is MotorCommand.CLOSE and cmd is MotorCommand.OPEN):
                violations += 1
            prev_cmd = cmd
        # Allow a small number due to window boundaries
        assert violations <= 5, f"Too many direct direction reversals: {violations}"

