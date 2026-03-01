"""Inflow disturbance profiles for the sluice gate simulator.

All profiles accept parameters in m³/h (user-facing) and convert
internally to m³/s for the simulation engine.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Conversion factor
M3H_TO_M3S = 1.0 / 3600.0


class InflowProfile(ABC):
    """Base class for inflow profiles."""

    @abstractmethod
    def __call__(self, t: float) -> float:
        """Return inflow rate in m³/s at time *t* [s]."""

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the profile."""


@dataclass
class ConstantInflow(InflowProfile):
    """Constant inflow rate.

    Parameters
    ----------
    q_m3h : float
        Constant inflow in m³/h.
    """

    q_m3h: float = 1000.0

    def __call__(self, t: float) -> float:  # noqa: ARG002
        return self.q_m3h * M3H_TO_M3S

    def describe(self) -> str:
        return f"Constant {self.q_m3h:.1f} m³/h"


@dataclass
class StepInflow(InflowProfile):
    """Step change in inflow.

    Parameters
    ----------
    q_base_m3h : float
        Base inflow in m³/h.
    q_step_m3h : float
        Step magnitude in m³/h (added to base).
    t_step : float
        Time [s] at which the step occurs.
    """

    q_base_m3h: float = 1000.0
    q_step_m3h: float = 100.0
    t_step: float = 10.0

    def __call__(self, t: float) -> float:
        q = self.q_base_m3h
        if t >= self.t_step:
            q += self.q_step_m3h
        return q * M3H_TO_M3S

    def describe(self) -> str:
        return (
            f"Step: {self.q_base_m3h:.1f} → "
            f"{self.q_base_m3h + self.q_step_m3h:.1f} m³/h at t={self.t_step:.1f}s"
        )


@dataclass
class RampInflow(InflowProfile):
    """Ramp change in inflow over a specified duration.

    Parameters
    ----------
    q_base_m3h : float
        Base inflow in m³/h.
    q_delta_m3h : float
        Total ramp magnitude in m³/h.
    t_start : float
        Ramp start time [s].
    duration : float
        Ramp duration [s].
    """

    q_base_m3h: float = 1000.0
    q_delta_m3h: float = 100.0
    t_start: float = 10.0
    duration: float = 10.0

    def __call__(self, t: float) -> float:
        q = self.q_base_m3h
        if t >= self.t_start:
            elapsed = t - self.t_start
            if elapsed < self.duration:
                q += self.q_delta_m3h * (elapsed / self.duration)
            else:
                q += self.q_delta_m3h
        return q * M3H_TO_M3S

    def describe(self) -> str:
        return (
            f"Ramp: +{self.q_delta_m3h:.1f} m³/h over {self.duration:.1f}s "
            f"starting at t={self.t_start:.1f}s"
        )


@dataclass
class SineInflow(InflowProfile):
    """Sinusoidal disturbance on top of a base inflow.

    Parameters
    ----------
    q_base_m3h : float
        Base (mean) inflow in m³/h.
    amplitude_m3h : float
        Sine amplitude in m³/h.
    period : float
        Oscillation period [s].
    """

    q_base_m3h: float = 1000.0
    amplitude_m3h: float = 100.0
    period: float = 60.0

    def __call__(self, t: float) -> float:
        q = self.q_base_m3h + self.amplitude_m3h * math.sin(2.0 * math.pi * t / self.period)
        return max(q, 0.0) * M3H_TO_M3S

    def describe(self) -> str:
        return (
            f"Sine: {self.q_base_m3h:.1f} ± {self.amplitude_m3h:.1f} m³/h, "
            f"T={self.period:.1f}s"
        )


# Registry for UI convenience
PROFILE_TYPES: dict[str, type[InflowProfile]] = {
    "Constant": ConstantInflow,
    "Step": StepInflow,
    "Ramp": RampInflow,
    "Sine": SineInflow,
}

