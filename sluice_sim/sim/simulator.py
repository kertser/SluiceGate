"""Simulation orchestrator for the sluice gate system.

Wires together the plant, controller, and inflow profile,
runs the time loop, and produces a logged DataFrame.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sluice_sim.controllers.pid_pulse import ControllerParams, PIDPulseController
from sluice_sim.models.plant import Plant, PlantParams, PlantState
from sluice_sim.profiles.inflow import (
    ConstantInflow,
    InflowProfile,
    RampInflow,
    SineInflow,
    StepInflow,
)


@dataclass
class SimConfig:
    """Top-level simulation configuration.

    Collects all parameters needed to fully define a scenario.
    """

    # Plant
    plant: PlantParams = field(default_factory=PlantParams)

    # Controller
    controller: ControllerParams = field(default_factory=ControllerParams)

    # Initial / target conditions
    H0: float = 1.5
    a0: float = 0.0
    H_set: float = 2.0

    # Timing
    dt: float = 0.02
    t_end: float = 120.0

    # Noise
    noise_std: float = 0.0
    noise_seed: int | None = None

    # Inflow profile descriptor (serialisable)
    inflow_type: str = "Ramp"
    inflow_params: dict[str, float] = field(
        default_factory=lambda: {
            "q_base_m3h": 1000.0,
            "q_delta_m3h": 100.0,
            "t_start": 10.0,
            "duration": 10.0,
        }
    )

    def build_inflow(self) -> InflowProfile:
        """Instantiate the inflow profile described by *inflow_type*."""
        match self.inflow_type:
            case "Constant":
                return ConstantInflow(**{k: v for k, v in self.inflow_params.items() if k in ("q_m3h",)})
            case "Step":
                return StepInflow(**{k: v for k, v in self.inflow_params.items() if k in ("q_base_m3h", "q_step_m3h", "t_step")})
            case "Ramp":
                return RampInflow(**{k: v for k, v in self.inflow_params.items() if k in ("q_base_m3h", "q_delta_m3h", "t_start", "duration")})
            case "Sine":
                return SineInflow(**{k: v for k, v in self.inflow_params.items() if k in ("q_base_m3h", "amplitude_m3h", "period")})
            case _:
                raise ValueError(f"Unknown inflow type: {self.inflow_type}")


# ──────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────

class Simulator:
    """Runs the sluice-gate simulation loop and records results."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self._build()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg = self.config
        self.plant = Plant(cfg.plant)
        self.controller = PIDPulseController(cfg.controller)
        self.inflow = cfg.build_inflow()

        self.state = PlantState(H=cfg.H0, a=cfg.a0)
        self.t: float = 0.0
        self._rng = np.random.default_rng(cfg.noise_seed)

        self._log: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Re-initialise everything from the current config."""
        self._build()

    # ------------------------------------------------------------------
    # Step / Run
    # ------------------------------------------------------------------

    def step_once(self) -> dict[str, Any]:
        """Advance the simulation by one time step and return the log row."""
        cfg = self.config
        dt = cfg.dt

        # Inflow
        Qin = self.inflow(self.t)

        # Measurement (with optional noise)
        H_meas = self.state.H
        if cfg.noise_std > 0:
            H_meas += self._rng.normal(0.0, cfg.noise_std)

        # Controller
        cmd = self.controller.step(H_meas, cfg.H_set, dt)

        # Plant
        new_state, Qout = self.plant.step(self.state, cmd, Qin, dt)

        # Log
        row = {
            "t": round(self.t, 6),
            "H": new_state.H,
            "a": new_state.a,
            "a_cm": new_state.a * 100.0,
            "Qin": Qin,
            "Qout": Qout,
            "e": self.controller.last_error,
            "u": self.controller.last_u,
            "duty": self.controller.last_duty,
            "cmd": cmd.value,
        }
        self._log.append(row)

        # Advance
        self.state = new_state
        self.t += dt
        return row

    def run(self) -> pd.DataFrame:
        """Run the full simulation and return a DataFrame."""
        self.reset()
        n_steps = int(math.ceil(self.config.t_end / self.config.dt))
        for _ in range(n_steps):
            self.step_once()
        return self.get_dataframe()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """Return logged data as a DataFrame."""
        return pd.DataFrame(self._log)

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_csv(self, path: str | Path) -> None:
        """Save logged data to CSV."""
        self.get_dataframe().to_csv(path, index=False)

    def save_scenario_json(self, path: str | Path) -> None:
        """Persist the current *SimConfig* as JSON."""
        data = _config_to_dict(self.config)
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def load_scenario_json(path: str | Path) -> SimConfig:
        """Load a *SimConfig* from a JSON file."""
        data = json.loads(Path(path).read_text())
        return _config_from_dict(data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def equilibrium_opening(self) -> float:
        """Compute the equilibrium gate opening for current config."""
        cfg = self.config
        Qin_m3s = self.inflow(0.0)  # use initial inflow
        return Plant.compute_equilibrium_opening(
            Qin_m3s,
            cfg.plant.Cd,
            cfg.plant.b,
            cfg.plant.g,
            cfg.H_set,
        )


# ──────────────────────────────────────────────────────────────────────
# JSON serialisation helpers
# ──────────────────────────────────────────────────────────────────────

def _config_to_dict(cfg: SimConfig) -> dict[str, Any]:
    return {
        "plant": asdict(cfg.plant),
        "controller": asdict(cfg.controller),
        "H0": cfg.H0,
        "a0": cfg.a0,
        "H_set": cfg.H_set,
        "dt": cfg.dt,
        "t_end": cfg.t_end,
        "noise_std": cfg.noise_std,
        "noise_seed": cfg.noise_seed,
        "inflow_type": cfg.inflow_type,
        "inflow_params": cfg.inflow_params,
    }


def _config_from_dict(d: dict[str, Any]) -> SimConfig:
    return SimConfig(
        plant=PlantParams(**d.get("plant", {})),
        controller=ControllerParams(**d.get("controller", {})),
        H0=d.get("H0", 1.5),
        a0=d.get("a0", 0.0),
        H_set=d.get("H_set", 2.0),
        dt=d.get("dt", 0.02),
        t_end=d.get("t_end", 120.0),
        noise_std=d.get("noise_std", 0.0),
        noise_seed=d.get("noise_seed"),
        inflow_type=d.get("inflow_type", "Ramp"),
        inflow_params=d.get("inflow_params", {}),
    )


