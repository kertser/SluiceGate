"""Physical plant model for the sluice gate simulator.

Contains the channel geometry, gate discharge, motor actuator,
and mass-balance integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MotorCommand(Enum):
    """Discrete motor commands."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"
    STOP = "STOP"


@dataclass
class PlantParams:
    """Immutable physical parameters for the plant.

    Attributes
    ----------
    b : float
        Channel width [m].
    L : float
        Effective upstream storage length [m].
    Cd : float
        Discharge coefficient (dimensionless).
    g : float
        Gravitational acceleration [m/s²].
    a_max : float
        Maximum gate opening [m].
    open_rate : float
        Gate opening speed [m/s].
    close_rate : float
        Gate closing speed [m/s].
    """

    b: float = 0.75
    L: float = 50.0
    Cd: float = 0.65
    g: float = 9.81
    a_max: float = 0.45
    open_rate: float = 0.01  # 1 cm/s
    close_rate: float = 0.01  # 1 cm/s
    H_max: float = 2.5  # maximum allowed water level [m] (overflow limit)


@dataclass
class PlantState:
    """Mutable state of the plant at a given instant.

    Attributes
    ----------
    H : float
        Water level [m].
    a : float
        Current gate opening [m].
    """

    H: float = 1.5
    a: float = 0.0


class Plant:
    """Sluice-gate plant model.

    Encapsulates the physics: gate actuation, discharge calculation,
    and water-level integration via forward Euler.
    """

    def __init__(self, params: PlantParams | None = None) -> None:
        self.params = params or PlantParams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        state: PlantState,
        cmd: MotorCommand,
        Qin: float,
        dt: float,
    ) -> tuple[PlantState, float]:
        """Advance the plant by one time step.

        Parameters
        ----------
        state : PlantState
            Current state (will **not** be mutated).
        cmd : MotorCommand
            Motor command for this step.
        Qin : float
            Inflow rate [m³/s].
        dt : float
            Time-step size [s].

        Returns
        -------
        new_state : PlantState
            Updated plant state.
        Qout : float
            Gate discharge during this step [m³/s].
        """
        p = self.params

        # --- Actuator update ---
        a_new = state.a
        if cmd is MotorCommand.OPEN:
            a_new += p.open_rate * dt
        elif cmd is MotorCommand.CLOSE:
            a_new -= p.close_rate * dt
        a_new = max(0.0, min(a_new, p.a_max))

        # --- Gate discharge ---
        Qout = self.compute_discharge(state.H, a_new)

        # --- Mass balance (forward Euler) ---
        A_storage = p.b * p.L
        dHdt = (Qin - Qout) / A_storage
        H_new = state.H + dHdt * dt
        H_new = max(H_new, 0.0)  # physical clamp – floor
        H_new = min(H_new, p.H_max)  # overflow clamp – ceiling

        return PlantState(H=H_new, a=a_new), Qout

    def compute_discharge(self, H: float, a: float) -> float:
        """Compute gate discharge Q_out [m³/s].

        Uses the orifice equation:
            Qout = Cd * b * a * sqrt(2 * g * H)
        """
        p = self.params
        H_eff = max(H, 0.0)
        a_eff = max(0.0, min(a, p.a_max))
        return p.Cd * p.b * a_eff * math.sqrt(2.0 * p.g * H_eff)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_equilibrium_opening(
        Qin: float,
        Cd: float,
        b: float,
        g: float,
        H_set: float,
    ) -> float:
        """Compute the gate opening that balances *Qin* at level *H_set*.

        a_eq = Qin / (Cd * b * sqrt(2 * g * H_set))
        """
        denom = Cd * b * math.sqrt(2.0 * g * H_set)
        if denom <= 0:
            return 0.0
        return Qin / denom


