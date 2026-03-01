"""PID auto-tuning via relay feedback (Åström–Hägglund method).

Runs a relay-feedback experiment on the plant to identify the ultimate
gain (Ku) and ultimate period (Tu), then applies Ziegler–Nichols or
other tuning rules to compute Kp, Ki, Kd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sluice_sim.models.plant import MotorCommand, Plant, PlantParams, PlantState
from sluice_sim.profiles.inflow import InflowProfile


@dataclass
class AutotuneResult:
    """Outcome of a relay-feedback auto-tune experiment."""

    Ku: float  # ultimate gain
    Tu: float  # ultimate period [s]
    Kp: float
    Ki: float
    Kd: float
    rule: str  # tuning rule name
    oscillation_amplitude: float  # measured amplitude [m]
    success: bool
    message: str


# ──────────────────────────────────────────────────────────────────────
# Tuning rules
# ──────────────────────────────────────────────────────────────────────

def _ziegler_nichols_pi(Ku: float, Tu: float) -> tuple[float, float, float]:
    Kp = 0.45 * Ku
    Ki = Kp / (0.83 * Tu)
    return Kp, Ki, 0.0


def _ziegler_nichols_pid(Ku: float, Tu: float) -> tuple[float, float, float]:
    Kp = 0.6 * Ku
    Ki = Kp / (0.5 * Tu)
    Kd = Kp * 0.125 * Tu
    return Kp, Ki, Kd


def _tyreus_luyben_pi(Ku: float, Tu: float) -> tuple[float, float, float]:
    Kp = Ku / 3.2
    Ki = Kp / (2.2 * Tu)
    return Kp, Ki, 0.0


TUNING_RULES: dict[str, Any] = {
    "Ziegler-Nichols PI": _ziegler_nichols_pi,
    "Ziegler-Nichols PID": _ziegler_nichols_pid,
    "Tyreus-Luyben PI": _tyreus_luyben_pi,
}


# ──────────────────────────────────────────────────────────────────────
# Relay-feedback experiment
# ──────────────────────────────────────────────────────────────────────

def run_autotune(
    plant_params: PlantParams,
    inflow: InflowProfile,
    H_set: float,
    H0: float = 1.5,
    a0: float = 0.0,
    dt: float = 0.02,
    relay_amplitude: float = 0.005,
    max_time: float = 600.0,
    min_cycles: int = 4,
    rule: str = "Ziegler-Nichols PI",
) -> AutotuneResult:
    """Execute relay-feedback auto-tuning.

    The relay toggles between OPEN and CLOSE based on the sign of the
    error (H_set − H).  After enough oscillation cycles are recorded
    the ultimate gain Ku and period Tu are estimated.

    Parameters
    ----------
    plant_params : PlantParams
        Physical plant configuration.
    inflow : InflowProfile
        The inflow profile to use during the experiment.
    H_set : float
        Desired water-level setpoint [m].
    H0, a0 : float
        Initial conditions.
    dt : float
        Simulation time-step.
    relay_amplitude : float
        Relay hysteresis band [m].  The relay switches when |e| crosses
        this threshold.
    max_time : float
        Maximum experiment duration [s].
    min_cycles : int
        Minimum number of full oscillation cycles before declaring success.
    rule : str
        Name of the tuning rule (key in ``TUNING_RULES``).
    """
    plant = Plant(plant_params)
    state = PlantState(H=H0, a=a0)

    # First, bring the gate to approximate equilibrium
    Qin_init = inflow(0.0)
    a_eq = Plant.compute_equilibrium_opening(
        Qin_init, plant_params.Cd, plant_params.b, plant_params.g, H_set,
    )
    # Pre-position the gate near equilibrium
    state = PlantState(H=H_set, a=min(a_eq, plant_params.a_max))

    # Relay state
    relay_high = True  # True → commanding CLOSE (raise level)
    zero_crossings: list[float] = []
    peaks: list[float] = []
    valleys: list[float] = []
    prev_error_sign: int = 0

    t = 0.0
    n_max = int(max_time / dt)

    for _ in range(n_max):
        Qin = inflow(t)
        e = H_set - state.H

        # Relay switching with hysteresis
        if relay_high and e < -relay_amplitude:
            relay_high = False  # switch to OPEN (lower level)
            zero_crossings.append(t)
            peaks.append(state.H)
        elif not relay_high and e > relay_amplitude:
            relay_high = True  # switch to CLOSE (raise level)
            zero_crossings.append(t)
            valleys.append(state.H)

        cmd = MotorCommand.CLOSE if relay_high else MotorCommand.OPEN
        state, _ = plant.step(state, cmd, Qin, dt)
        t += dt

        # Check if we have enough cycles
        n_crossings = len(zero_crossings)
        if n_crossings >= 2 * min_cycles + 1:
            break

    # ── Analyse results ──
    n_crossings = len(zero_crossings)
    if n_crossings < 4:
        return AutotuneResult(
            Ku=0, Tu=0, Kp=0, Ki=0, Kd=0, rule=rule,
            oscillation_amplitude=0, success=False,
            message=f"Not enough oscillation cycles detected ({n_crossings} crossings). "
                    "Try increasing max_time or adjusting relay_amplitude.",
        )

    # Ultimate period: average half-period × 2
    half_periods = [
        zero_crossings[i + 1] - zero_crossings[i]
        for i in range(n_crossings - 1)
    ]
    Tu = float(2.0 * np.mean(half_periods))

    # Oscillation amplitude
    if peaks and valleys:
        osc_amplitude = (np.mean(peaks) - np.mean(valleys)) / 2.0
    else:
        osc_amplitude = relay_amplitude  # fallback

    osc_amplitude = max(osc_amplitude, 1e-9)

    # Ultimate gain:  Ku = 4d / (π·a)  where d = relay output, a = oscillation amplitude
    # Here relay output is effectively the gate movement rate × Twindow ≈ open_rate
    # For a position-based relay: d corresponds to the control effort.
    # We use the classic formula where d = the relay "amplitude" in process-output
    # units. Since our relay toggles between OPEN/CLOSE at rate open_rate,
    # the effective relay output in m/s is the rate itself.
    d = plant_params.open_rate  # effective relay amplitude [m/s of gate]
    Ku = (4.0 * d) / (math.pi * osc_amplitude)

    # Apply tuning rule
    rule_fn = TUNING_RULES.get(rule)
    if rule_fn is None:
        return AutotuneResult(
            Ku=Ku, Tu=Tu, Kp=0, Ki=0, Kd=0, rule=rule,
            oscillation_amplitude=osc_amplitude, success=False,
            message=f"Unknown tuning rule: {rule}",
        )

    Kp, Ki, Kd = rule_fn(Ku, Tu)

    return AutotuneResult(
        Ku=round(Ku, 6),
        Tu=round(Tu, 4),
        Kp=round(Kp, 4),
        Ki=round(Ki, 4),
        Kd=round(Kd, 4),
        rule=rule,
        oscillation_amplitude=round(osc_amplitude, 6),
        success=True,
        message=f"Auto-tune OK — {len(zero_crossings)} crossings, "
                f"Ku={Ku:.4f}, Tu={Tu:.2f}s",
    )


