"""Streamlit UI for the Sluice Gate Level Control Simulator.

Run with:
    streamlit run sluice_sim/ui/app.py
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import streamlit as st

from sluice_sim.controllers.pid_pulse import ControllerParams
from sluice_sim.controllers.autotune import run_autotune, TUNING_RULES, AutotuneResult
from sluice_sim.models.plant import Plant, PlantParams
from sluice_sim.profiles.inflow import M3H_TO_M3S
from sluice_sim.sim.simulator import SimConfig, Simulator, _config_to_dict, _config_from_dict

# ──────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sluice Gate Simulator", layout="wide")

_LOGO_PATH = Path(__file__).parent / "AtlantiumLogo.png"
if _LOGO_PATH.exists():
    st.image(str(_LOGO_PATH), width="stretch")

st.title("🌊 Sluice Gate Level Control Simulator")

# ──────────────────────────────────────────────────────────────────────
# Session-state defaults for PID gains (so auto-tune can update them)
# ──────────────────────────────────────────────────────────────────────

if "pid_Kp" not in st.session_state:
    st.session_state.pid_Kp = 5.0
if "pid_Ki" not in st.session_state:
    st.session_state.pid_Ki = 0.25
if "pid_Kd" not in st.session_state:
    st.session_state.pid_Kd = 0.0

# Apply pending auto-tune values BEFORE widgets are created
if st.session_state.get("_apply_autotune"):
    pending = st.session_state._apply_autotune
    st.session_state.pid_Kp = pending["Kp"]
    st.session_state.pid_Ki = pending["Ki"]
    st.session_state.pid_Kd = pending["Kd"]
    del st.session_state._apply_autotune

# ──────────────────────────────────────────────────────────────────────
# Sidebar – parameter controls with tooltips
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parameters")

    # --- Geometry ---
    with st.expander("📐 Geometry", expanded=False):
        b = st.number_input(
            "Channel width b [m]", 0.1, 10.0, 0.75, 0.05,
            help="Width of the rectangular open channel cross-section.",
        )
        L = st.number_input(
            "Storage length L [m]", 1.0, 500.0, 50.0, 1.0,
            help="Effective upstream storage length. Together with b defines the storage area A = b × L.",
        )
        a_max_cm = st.number_input(
            "Max gate opening a_max [cm]", 1.0, 500.0, 45.0, 1.0,
            help="Maximum vertical distance the sluice gate can open.",
        )
        a_max = a_max_cm / 100.0
        Cd = st.number_input(
            "Discharge coefficient Cd", 0.1, 1.0, 0.65, 0.01,
            help="Dimensionless orifice discharge coefficient (typically 0.6-0.65 for a sharp-edged sluice gate).",
        )
        H_max = st.number_input(
            "Max water level H_max [m]", 0.5, 20.0, 2.5, 0.1,
            help="Maximum allowed water level before overflow. The simulation clamps H to H_max.",
        )

    # --- Motor ---
    with st.expander("⚡ Motor", expanded=False):
        open_rate = st.number_input(
            "Open rate [cm/s]", 0.1, 10.0, 1.0, 0.1,
            help="Speed at which the gate motor opens the gate.",
        ) / 100.0
        close_rate = st.number_input(
            "Close rate [cm/s]", 0.1, 10.0, 1.0, 0.1,
            help="Speed at which the gate motor closes the gate.",
        ) / 100.0

    # --- Controller ---
    with st.expander("🎛 Controller", expanded=True):
        Kp = st.number_input(
            "Kp", 0.0, 100.0, step=0.1, key="pid_Kp",
            help="Proportional gain. Higher = faster response but more overshoot.",
        )
        Ki = st.number_input(
            "Ki", 0.0, 50.0, step=0.01, key="pid_Ki",
            help="Integral gain. Eliminates steady-state error. Too high = oscillation.",
        )
        Kd = st.number_input(
            "Kd", 0.0, 50.0, step=0.01, key="pid_Kd",
            help="Derivative gain. Dampens oscillation. Sensitive to noise.",
        )
        alpha = st.slider(
            "Filter alpha", 0.01, 1.0, 0.3, 0.01,
            help="Low-pass filter coefficient on the measurement. 1 = no filtering; lower = smoother but slower.",
        )
        Umax = st.number_input(
            "Umax", 0.01, 10.0, 0.18, 0.01,
            help="Controller output value that maps to 100% duty cycle. Scales the PID output to the [0,1] duty range.",
        )
        Twindow = st.number_input(
            "Twindow [s]", 0.1, 10.0, 0.5, 0.1,
            help="Pulse-modulation window length. At each window boundary the controller recalculates the duty cycle.",
        )
        min_on = st.number_input(
            "min_on [s]", 0.0, 5.0, 0.18, 0.01,
            help="Minimum motor on-time per window. Pulses shorter than this are skipped to protect the motor.",
        )

    with st.expander("🔇 Deadband", expanded=False):
        deadband_hi = st.number_input(
            "deadband_hi [m]", 0.0, 1.0, 0.02, 0.001, format="%.3f",
            help="Outer (activation) threshold. Controller activates when |error| exceeds this value.",
        )
        deadband_lo = st.number_input(
            "deadband_lo [m]", 0.0, 1.0, 0.005, 0.001, format="%.3f",
            help="Inner (de-activation) threshold. Controller stops when |error| drops below this. Creates hysteresis with deadband_hi.",
        )

    # --- Auto-tune ---
    with st.expander("🔧 Auto-Tune PID", expanded=False):
        st.caption("Run a relay-feedback experiment to estimate optimal PID gains.")
        autotune_rule = st.selectbox(
            "Tuning rule",
            list(TUNING_RULES.keys()),
            help="Algorithm used to convert the measured ultimate gain Ku and period Tu into PID gains.",
        )
        relay_amp = st.number_input(
            "Relay hysteresis band [m]", 0.001, 0.5, 0.005, 0.001, format="%.3f",
            help="The relay switches direction when |error| crosses this threshold. Smaller = tighter oscillations.",
        )
        autotune_time = st.number_input(
            "Max experiment time [s]", 60.0, 3600.0, 600.0, 60.0,
            help="Maximum duration of the relay-feedback experiment.",
        )
        autotune_btn = st.button(
            "Run Auto-Tune",
            width="stretch",
            help="Execute the relay-feedback experiment and display recommended PID gains.",
        )

    # --- Inflow ---
    with st.expander("💧 Inflow Profile", expanded=True):
        inflow_type = st.selectbox(
            "Profile", ["Constant", "Step", "Ramp", "Sine"],
            help="Shape of the upstream inflow disturbance applied during the simulation.",
        )
        q_base = st.number_input(
            "Base Qin [m3/h]", 0.0, 50000.0, 1000.0, 10.0,
            help="Baseline (constant) component of the upstream inflow.",
        )

        inflow_params: dict[str, float] = {}
        if inflow_type == "Constant":
            inflow_params = {"q_m3h": q_base}
        elif inflow_type == "Step":
            q_step = st.number_input(
                "Step dQ [m3/h]", -5000.0, 5000.0, 100.0, 10.0,
                help="Magnitude of the sudden inflow change (positive = increase).",
            )
            t_step = st.number_input(
                "Step time [s]", 0.0, 1000.0, 10.0, 1.0,
                help="Simulation time at which the step disturbance occurs.",
            )
            inflow_params = {"q_base_m3h": q_base, "q_step_m3h": q_step, "t_step": t_step}
        elif inflow_type == "Ramp":
            q_delta = st.number_input(
                "Ramp dQ [m3/h]", -5000.0, 5000.0, 100.0, 10.0,
                help="Total change in inflow over the ramp duration.",
            )
            t_start = st.number_input(
                "Ramp start [s]", 0.0, 1000.0, 10.0, 1.0,
                help="Time at which the ramp begins.",
            )
            duration = st.number_input(
                "Ramp duration [s]", 0.1, 1000.0, 10.0, 1.0,
                help="Duration over which the inflow linearly changes by dQ.",
            )
            inflow_params = {"q_base_m3h": q_base, "q_delta_m3h": q_delta, "t_start": t_start, "duration": duration}
        elif inflow_type == "Sine":
            amplitude = st.number_input(
                "Amplitude [m3/h]", 0.0, 5000.0, 100.0, 10.0,
                help="Peak-to-mean amplitude of the sinusoidal inflow oscillation.",
            )
            period = st.number_input(
                "Period [s]", 1.0, 1000.0, 60.0, 1.0,
                help="Oscillation period of the sinusoidal inflow.",
            )
            inflow_params = {"q_base_m3h": q_base, "amplitude_m3h": amplitude, "period": period}

    # --- Simulation ---
    with st.expander("🕹 Simulation", expanded=False):
        H0 = st.number_input(
            "Initial level H0 [m]", 0.0, 20.0, 1.5, 0.1,
            help="Water level at the start of the simulation.",
        )
        H_set = st.number_input(
            "Setpoint H_set [m]", 0.0, 20.0, 2.0, 0.1,
            help="Target water level the controller tries to maintain.",
        )
        a0_cm = st.number_input(
            "Initial opening a0 [cm]", 0.0, 500.0, 0.0, 0.5,
            help="Gate opening at the start of the simulation.",
        )
        a0 = a0_cm / 100.0
        dt = st.number_input(
            "Time step dt [s]", 0.001, 0.1, 0.02, 0.001, format="%.3f",
            help="Euler integration time step. Smaller = more accurate but slower.",
        )
        t_end = st.number_input(
            "Sim duration [s]", 1.0, 3600.0, 120.0, 10.0,
            help="Total simulation time.",
        )
        noise_std = st.number_input(
            "Noise std [m]", 0.0, 1.0, 0.0, 0.001, format="%.4f",
            help="Standard deviation of Gaussian measurement noise added to H.",
        )

# ──────────────────────────────────────────────────────────────────────
# Build config from sidebar
# ──────────────────────────────────────────────────────────────────────


def _build_config() -> SimConfig:
    return SimConfig(
        plant=PlantParams(
            b=b, L=L, Cd=Cd, a_max=a_max,
            open_rate=open_rate, close_rate=close_rate,
            H_max=H_max,
        ),
        controller=ControllerParams(
            Kp=Kp, Ki=Ki, Kd=Kd, alpha=alpha, Umax=Umax,
            Twindow=Twindow, min_on=min_on,
            deadband_hi=deadband_hi, deadband_lo=deadband_lo,
        ),
        H0=H0, a0=a0, H_set=H_set,
        dt=dt, t_end=t_end,
        noise_std=noise_std,
        inflow_type=inflow_type,
        inflow_params=inflow_params,
    )


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def _compute_settling_time(df_in: pd.DataFrame, h_set: float, tol: float = 0.02) -> str:
    """Compute the 2%-settling time as a display string."""
    band = h_set * tol
    within = (df_in["H"] - h_set).abs() <= band
    outside = df_in.loc[~within, "t"]
    if outside.empty:
        return "0.0 s"
    last_outside = outside.iloc[-1]
    if last_outside >= df_in["t"].iloc[-1]:
        return f"> {df_in['t'].iloc[-1]:.0f} s"
    return f"{last_outside:.1f} s"


# ──────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = None
if "sim" not in st.session_state:
    st.session_state.sim = None

# ──────────────────────────────────────────────────────────────────────
# Buttons
# ──────────────────────────────────────────────────────────────────────

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    run_btn = st.button(
        "▶ Run", width="stretch",
        help="Run the full simulation from t=0 to t_end.",
    )
with col_btn2:
    step_btn = st.button(
        "⏭ Step x100", width="stretch",
        help="Advance the simulation by 100 time-steps (useful for step-by-step inspection).",
    )
with col_btn3:
    reset_btn = st.button(
        "🔄 Reset", width="stretch",
        help="Clear all results and reset the simulation.",
    )

# ──────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────

if reset_btn:
    st.session_state.df = None
    st.session_state.sim = None


if run_btn:
    cfg = _build_config()
    sim = Simulator(cfg)
    with st.spinner("Simulating..."):
        df_result = sim.run()
    st.session_state.df = df_result
    st.session_state.sim = sim

if step_btn:
    cfg = _build_config()
    if st.session_state.sim is None:
        st.session_state.sim = Simulator(cfg)
    sim = st.session_state.sim
    for _ in range(100):
        sim.step_once()
    st.session_state.df = sim.get_dataframe()

# ──────────────────────────────────────────────────────────────────────
# Auto-tune action
# ──────────────────────────────────────────────────────────────────────

if autotune_btn:
    cfg = _build_config()
    inflow_profile = cfg.build_inflow()
    with st.spinner("Running relay-feedback experiment..."):
        at_result: AutotuneResult = run_autotune(
            plant_params=cfg.plant,
            inflow=inflow_profile,
            H_set=cfg.H_set,
            H0=cfg.H0,
            a0=cfg.a0,
            dt=cfg.dt,
            relay_amplitude=relay_amp,
            max_time=autotune_time,
            rule=autotune_rule,
        )
    st.session_state.autotune_result = at_result

# Show auto-tune results (persisted across reruns)
if "autotune_result" in st.session_state and st.session_state.autotune_result is not None:
    at_result = st.session_state.autotune_result
    if at_result.success:
        st.success(at_result.message)
        col_a, col_b, col_c, col_apply = st.columns([1, 1, 1, 1])
        col_a.metric("Recommended Kp", f"{at_result.Kp:.4f}",
                     help="Proportional gain from auto-tune.")
        col_b.metric("Recommended Ki", f"{at_result.Ki:.4f}",
                     help="Integral gain from auto-tune.")
        col_c.metric("Recommended Kd", f"{at_result.Kd:.4f}",
                     help="Derivative gain from auto-tune.")
        with col_apply:
            st.write("")  # spacer to align with metrics
            if st.button("✅ Apply to Controller", width="stretch",
                         help="Set the recommended Kp, Ki, Kd values in the Controller section."):
                st.session_state._apply_autotune = {
                    "Kp": round(at_result.Kp, 4),
                    "Ki": round(at_result.Ki, 4),
                    "Kd": round(at_result.Kd, 4),
                }
                st.session_state.autotune_result = None
                st.rerun()
        st.caption(
            f"Ultimate gain Ku = {at_result.Ku:.4f}  |  "
            f"Ultimate period Tu = {at_result.Tu:.2f} s  |  "
            f"Oscillation amplitude = {at_result.oscillation_amplitude:.6f} m  |  "
            f"Rule: {at_result.rule}"
        )
    else:
        st.error(at_result.message)

# ──────────────────────────────────────────────────────────────────────
# Info panel
# ──────────────────────────────────────────────────────────────────────

cfg = _build_config()
Qin_base = cfg.inflow_params.get("q_base_m3h", cfg.inflow_params.get("q_m3h", 1000.0))
a_eq = Plant.compute_equilibrium_opening(
    Qin_base * M3H_TO_M3S,
    cfg.plant.Cd, cfg.plant.b, cfg.plant.g, cfg.H_set,
)

info_cols = st.columns(4)
info_cols[0].metric(
    "Equilibrium opening", f"{a_eq * 100:.2f} cm",
    help="Gate opening that exactly balances Qin at the target level H_set.",
)
info_cols[1].metric(
    "Max gate opening", f"{cfg.plant.a_max * 100:.1f} cm",
    help="Physical upper limit of the gate opening (a_max).",
)
info_cols[2].metric(
    "Max water level", f"{cfg.plant.H_max:.2f} m",
    help="Maximum allowed water level. Water is clamped at this level to model overflow prevention.",
)
info_cols[3].metric(
    "Base inflow", f"{Qin_base:.0f} m3/h",
    help="Baseline inflow rate before any disturbance is applied.",
)

# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

df = st.session_state.df

if df is not None and len(df) > 0:

    # Convert flows to m3/h and gate opening to cm for display
    df_plot = df.copy()
    df_plot["Qin_m3h"] = df_plot["Qin"] / M3H_TO_M3S
    df_plot["Qout_m3h"] = df_plot["Qout"] / M3H_TO_M3S
    df_plot["a_cm"] = df_plot["a"] * 100.0

    tab_main, tab_extra, tab_schematic = st.tabs([
        "📈 Main Plots",
        "📊 Error & Duty",
        "🎬 Schematic Animation",
    ])

    # ── TAB 1: Main Plots ──
    with tab_main:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # H(t) – FULL SCALE 0 to H_max
        ax = axes[0]
        ax.plot(df_plot["t"], df_plot["H"], label="H(t) - water level", linewidth=1)
        ax.axhline(cfg.H_set, color="r", linestyle="--", linewidth=1,
                    label=f"H_set = {cfg.H_set} m")
        ax.axhline(cfg.plant.H_max, color="darkred", linestyle=":", linewidth=1,
                    label=f"H_max = {cfg.plant.H_max} m (overflow)")
        ax.set_ylabel("Water Level [m]")
        ax.set_ylim(0, cfg.plant.H_max * 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Water Level", fontsize=10)

        # a(t) in cm
        ax = axes[1]
        ax.plot(df_plot["t"], df_plot["a_cm"],
                label="a(t) - gate opening", color="tab:orange", linewidth=1)
        ax.axhline(a_eq * 100, color="g", linestyle="--", alpha=0.7,
                    label=f"a_eq = {a_eq * 100:.2f} cm")
        ax.axhline(cfg.plant.a_max * 100, color="gray", linestyle=":", alpha=0.5,
                    label=f"a_max = {cfg.plant.a_max * 100:.1f} cm")
        ax.set_ylabel("Gate Opening [cm]")
        ax.set_ylim(0, cfg.plant.a_max * 100 * 1.1)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Gate Opening", fontsize=10)

        # Qin / Qout
        ax = axes[2]
        ax.plot(df_plot["t"], df_plot["Qin_m3h"], label="Qin - inflow", linewidth=1)
        ax.plot(df_plot["t"], df_plot["Qout_m3h"],
                label="Qout - gate discharge", linewidth=1, alpha=0.8)
        ax.set_ylabel("Flow [m3/h]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Flow Rates", fontsize=10)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 2: Error & Duty ──
    with tab_extra:
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax = axes2[0]
        ax.plot(df_plot["t"], df_plot["e"],
                label="Error (H_set - Hf)", linewidth=1, color="tab:red")
        ax.axhline(0, color="gray", linestyle=":")
        ax.axhline(deadband_hi, color="orange", linestyle="--", alpha=0.5,
                    label=f"deadband_hi = {deadband_hi}")
        ax.axhline(-deadband_hi, color="orange", linestyle="--", alpha=0.5)
        ax.set_ylabel("Error [m]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Controller Error", fontsize=10)

        ax = axes2[1]
        ax.plot(df_plot["t"], df_plot["duty"],
                label="Duty cycle", linewidth=1, color="tab:purple")
        ax.set_ylabel("Duty [-]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Pulse Modulation Duty Cycle", fontsize=10)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── TAB 3: Schematic Animation (GIF) ──
    with tab_schematic:
        st.caption(
            "Animated GIF of the channel cross-section over time. "
            "Blue = water, brown = gate, red dashed = setpoint, "
            "dark red dotted = overflow limit."
        )

        # Down-sample for manageable GIF size
        max_frames = 120
        step_size = max(1, len(df_plot) // max_frames)
        frames_idx = list(range(0, len(df_plot), step_size))

        channel_w = 4.0
        channel_h = cfg.plant.H_max * 1.1

        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))

        def _draw_frame(idx: int) -> list:
            ax3.clear()
            row = df_plot.iloc[idx]
            h_val = row["H"]
            a_val = row["a"]
            t_val = row["t"]

            # Channel outline
            ax3.add_patch(mpatches.Rectangle(
                (0, 0), channel_w, channel_h,
                linewidth=2, edgecolor="black",
                facecolor="lightyellow", zorder=0,
            ))
            # Water fill
            water_h = min(h_val, channel_h)
            ax3.add_patch(mpatches.Rectangle(
                (0, 0), channel_w, water_h,
                facecolor="deepskyblue", alpha=0.6, zorder=1,
            ))
            # Gate (right side)
            gate_x = channel_w - 0.15
            ax3.add_patch(mpatches.Rectangle(
                (gate_x, a_val), 0.3, channel_h - a_val,
                facecolor="saddlebrown", edgecolor="black",
                linewidth=1.5, zorder=2,
            ))
            # Setpoint line
            ax3.axhline(cfg.H_set, color="red", linestyle="--",
                         linewidth=1, zorder=3)
            # Overflow line
            ax3.axhline(cfg.plant.H_max, color="darkred", linestyle=":",
                         linewidth=1, zorder=3)

            # Annotations
            ax3.text(
                channel_w / 2, water_h + 0.05,
                f"H = {h_val:.3f} m",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="navy", zorder=4,
            )
            ax3.text(
                gate_x + 0.5, a_val + 0.02,
                f"a = {a_val * 100:.1f} cm",
                ha="left", va="bottom",
                fontsize=8, color="saddlebrown", zorder=4,
            )

            ax3.set_xlim(-0.3, channel_w + 1.5)
            ax3.set_ylim(-0.1, channel_h + 0.3)
            ax3.set_aspect("equal")
            ax3.set_title(f"t = {t_val:.1f} s", fontsize=10)
            ax3.set_ylabel("Height [m]")
            return []

        with st.spinner("Generating animation..."):
            ani = animation.FuncAnimation(
                fig3, _draw_frame, frames=frames_idx,
                interval=80, blit=False,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".gif", delete=False
            ) as tmp:
                gif_path = tmp.name
            ani.save(gif_path, writer="pillow", fps=12)
            plt.close(fig3)

        st.image(gif_path, caption="Channel schematic animation",
                 width="stretch")

    # ──────────────────────────────────────────────────────────────────
    # Summary metrics
    # ──────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("📊 Summary")
    sum_cols = st.columns(5)
    sum_cols[0].metric(
        "Final H", f"{df_plot['H'].iloc[-1]:.3f} m",
        help="Water level at the end of the simulation.",
    )
    sum_cols[1].metric(
        "Final gate", f"{df_plot['a_cm'].iloc[-1]:.1f} cm",
        help="Gate opening at the end of the simulation.",
    )
    peak_H = df_plot["H"].max()
    overflow_flag = peak_H >= cfg.plant.H_max * 0.99
    sum_cols[2].metric(
        "Peak H", f"{peak_H:.3f} m",
        delta="OVERFLOW!" if overflow_flag else "OK",
        delta_color="inverse" if overflow_flag else "off",
        help="Maximum water level reached during the simulation.",
    )
    tail_10 = df_plot.tail(max(1, int(len(df_plot) * 0.1)))
    sse = abs(tail_10["H"].mean() - cfg.H_set)
    sum_cols[3].metric(
        "Steady-state error", f"{sse:.4f} m",
        help="Average |H - H_set| over the last 10% of the simulation. Closer to 0 = better.",
    )
    sum_cols[4].metric(
        "Settling time (2%)",
        _compute_settling_time(df_plot, cfg.H_set, tol=0.02),
        help="Time after which H stays within +/-2% of H_set.",
    )

    # ──────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────

    st.divider()
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="sluice_sim_log.csv",
            mime="text/csv",
            help="Download the full simulation log as a CSV file.",
        )

    with col_exp2:
        scenario_json = json.dumps(_config_to_dict(cfg), indent=2)
        st.download_button(
            "Download Scenario JSON",
            data=scenario_json,
            file_name="sluice_scenario.json",
            mime="application/json",
            help="Download the current parameter set as a JSON file. Can be loaded back later.",
        )

    # Load a saved scenario
    st.caption("Load a previously saved scenario:")
    load_file = st.file_uploader(
        "Load scenario JSON", type=["json"],
        help="Upload a scenario JSON file to restore all parameters. Press Run after loading.",
    )
    if load_file is not None:
        data = json.loads(load_file.read())
        loaded_cfg = _config_from_dict(data)
        sim = Simulator(loaded_cfg)
        st.session_state.sim = sim
        st.session_state.df = None
        st.info("Scenario loaded! Press **▶ Run** to simulate.")

else:
    st.info(
        "Configure parameters in the sidebar, then press "
        "**▶ Run** to start the simulation."
    )

# ──────────────────────────────────────────────────────────────────────
# Load scenario (also available when no simulation has run yet)
# ──────────────────────────────────────────────────────────────────────

if df is None or len(df) == 0:
    st.divider()
    st.caption("Or load a previously saved scenario:")
    load_file_empty = st.file_uploader(
        "Load scenario JSON", type=["json"], key="load_empty",
        help="Upload a scenario JSON file to restore all parameters. Press Run after loading.",
    )
    if load_file_empty is not None:
        data = json.loads(load_file_empty.read())
        loaded_cfg = _config_from_dict(data)
        sim = Simulator(loaded_cfg)
        st.session_state.sim = sim
        st.session_state.df = None
        st.info("Scenario loaded! Press **▶ Run** to simulate.")






