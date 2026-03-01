"""PI/PID controller with pulse modulation for a discrete motor.

Implements:
- Low-pass measurement filter
- PID computation with anti-windup
- Pulse-width modulation over fixed time windows
- Deadband / hysteresis logic
- Minimum on-time and minimum switch-interval constraints
"""

from __future__ import annotations

from dataclasses import dataclass

from sluice_sim.models.plant import MotorCommand


@dataclass
class ControllerParams:
    """Tunable controller parameters.

    Attributes
    ----------
    Kp, Ki, Kd : float
        PID gains.
    alpha : float
        Low-pass filter coefficient (0..1). Higher → more responsive.
    Umax : float
        Controller output corresponding to 100 % duty.
    Twindow : float
        Pulse-modulation window length [s].
    min_on : float
        Minimum motor on-time per window [s].
    min_switch_interval : float
        Minimum time between direction reversals [s].
    deadband_hi : float
        Outer (activation) deadband threshold [m].
    deadband_lo : float
        Inner (de-activation) deadband threshold [m].
    I_min, I_max : float
        Integral-term clamp bounds.
    """

    Kp: float = 5.0
    Ki: float = 0.25
    Kd: float = 0.0
    alpha: float = 0.3
    Umax: float = 0.18
    Twindow: float = 0.5
    min_on: float = 0.18
    min_switch_interval: float = 0.5
    deadband_hi: float = 0.02
    deadband_lo: float = 0.005
    I_min: float = -10.0
    I_max: float = 10.0


class PIDPulseController:
    """PID controller that outputs discrete motor commands via pulse modulation."""

    def __init__(self, params: ControllerParams | None = None) -> None:
        self.params = params or ControllerParams()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal controller state."""
        self._Hf: float | None = None  # filtered measurement
        self._e_prev: float = 0.0
        self._I_sum: float = 0.0

        # Pulse-modulation state
        self._window_elapsed: float = 0.0
        self._pulse_duration: float = 0.0  # how long to keep motor on this window
        self._pulse_elapsed: float = 0.0
        self._current_cmd: MotorCommand = MotorCommand.STOP
        self._scheduled_cmd: MotorCommand = MotorCommand.STOP

        # Direction-change tracking
        self._last_direction: MotorCommand = MotorCommand.STOP
        self._time_since_last_switch: float = float("inf")

        # Deadband hysteresis memory
        self._in_deadband: bool = True

        # Diagnostics (latest computed values)
        self.last_u: float = 0.0
        self.last_duty: float = 0.0
        self.last_error: float = 0.0

    def step(self, H_meas: float, H_set: float, dt: float) -> MotorCommand:
        """Compute the motor command for the current time step.

        Parameters
        ----------
        H_meas : float
            Measured water level [m].
        H_set : float
            Setpoint water level [m].
        dt : float
            Time-step size [s].

        Returns
        -------
        MotorCommand
            OPEN, CLOSE, or STOP.
        """
        p = self.params

        # --- Low-pass filter ---
        if self._Hf is None:
            self._Hf = H_meas
        else:
            self._Hf = p.alpha * H_meas + (1.0 - p.alpha) * self._Hf

        # --- Error (positive → water is low → need to CLOSE gate) ---
        e = H_set - self._Hf
        self.last_error = e

        # --- Deadband / hysteresis ---
        abs_e = abs(e)
        if abs_e < p.deadband_lo:
            self._in_deadband = True
        elif abs_e >= p.deadband_hi:
            self._in_deadband = False
        # else: keep previous state (hysteresis)

        if self._in_deadband:
            self.last_u = 0.0
            self.last_duty = 0.0
            self._current_cmd = MotorCommand.STOP
            self._window_elapsed = 0.0
            self._pulse_elapsed = 0.0
            self._pulse_duration = 0.0
            self._time_since_last_switch += dt
            return MotorCommand.STOP

        # --- Track window timing ---
        self._window_elapsed += dt
        self._time_since_last_switch += dt

        new_window = self._window_elapsed >= p.Twindow or self._pulse_duration == 0.0

        if new_window:
            self._window_elapsed = 0.0 if self._window_elapsed >= p.Twindow else self._window_elapsed
            self._pulse_elapsed = 0.0

            # --- PID calculation ---
            P = p.Kp * e
            self._I_sum += p.Ki * e * dt
            # Anti-windup: clamp integral
            self._I_sum = max(p.I_min, min(self._I_sum, p.I_max))

            D = 0.0
            if dt > 0:
                D = p.Kd * (e - self._e_prev) / dt
            self._e_prev = e

            u = P + self._I_sum + D

            # Anti-windup: conditional integration (stop integrating when saturated)
            if abs(u) >= p.Umax and (u * e > 0):
                # Undo the integration step that pushed us further into saturation
                self._I_sum -= p.Ki * e * dt
                self._I_sum = max(p.I_min, min(self._I_sum, p.I_max))
                u = P + self._I_sum + D

            self.last_u = u

            # --- Direction ---
            if u > 0:
                # e > 0 → water low → close gate to raise level
                desired_dir = MotorCommand.CLOSE
            else:
                desired_dir = MotorCommand.OPEN

            # --- Duty cycle ---
            duty = min(abs(u) / p.Umax, 1.0) if p.Umax > 0 else 0.0
            pulse = duty * p.Twindow

            # Enforce minimum on-time
            if 0 < pulse < p.min_on:
                pulse = 0.0  # too short → skip
                duty = 0.0

            # Enforce min_switch_interval
            if (
                desired_dir != self._last_direction
                and self._last_direction != MotorCommand.STOP
                and self._time_since_last_switch < p.min_switch_interval
            ):
                pulse = 0.0
                duty = 0.0

            self.last_duty = duty
            self._pulse_duration = pulse
            self._scheduled_cmd = desired_dir

            if pulse > 0:
                if desired_dir != self._last_direction:
                    self._time_since_last_switch = 0.0
                self._last_direction = desired_dir

        # --- Determine command for this step ---
        if self._pulse_elapsed < self._pulse_duration:
            self._current_cmd = self._scheduled_cmd
            self._pulse_elapsed += dt
        else:
            self._current_cmd = MotorCommand.STOP

        return self._current_cmd


