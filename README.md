# 🌊 Sluice Gate Level Control Simulator

Simulate and tune a sluice gate that controls water level in an open channel.

## Quick Start

```bash
uv sync --dev
uv run streamlit run sluice_sim/ui/app.py
```

A browser tab opens with the simulator dashboard.

## How to Use

### 1. Run a Simulation

1. Click **▶ Run** — the simulation runs with default settings
2. Three tabs appear:
   - **📈 Main Plots** — water level, gate opening (cm), flow rates
   - **📊 Error & Duty** — controller error and motor duty cycle
   - **🎬 Schematic Animation** — animated GIF of the channel

### 2. Adjust Parameters

Open the **sidebar** (left side) to change settings. Each parameter has a **ⓘ tooltip** — hover over it for an explanation.

| Section | What you can change |
|---------|-------------------|
| **📐 Geometry** | Channel width, length, max gate opening, discharge coefficient, max water level |
| **⚡ Motor** | Gate open/close speed (cm/s) |
| **🎛 Controller** | PID gains (Kp, Ki, Kd), filter, pulse timing, duty scaling |
| **🔇 Deadband** | Error thresholds to prevent motor chattering |
| **💧 Inflow Profile** | Choose Constant / Step / Ramp / Sine disturbance |
| **🕹 Simulation** | Initial level, setpoint, time step, duration, noise |

After changing parameters, click **▶ Run** again.

### 3. Auto-Tune the Controller

Not sure what PID gains to use? Let the simulator find them:

1. In the sidebar, expand **🔧 Auto-Tune PID**
2. Pick a tuning rule (e.g. "Ziegler-Nichols PI")
3. Click **Run Auto-Tune**
4. Copy the recommended Kp / Ki / Kd values into the **🎛 Controller** section
5. Click **▶ Run** to test

### 4. Export Results

After running a simulation, scroll down to find:

- **Download CSV** — full time-series data
- **Download Scenario JSON** — all parameter settings (can be loaded back later)

To reload a saved scenario, use the **Load scenario** file uploader next to the Run button.

## Summary Metrics

After each run, a **📊 Summary** section shows:

| Metric | Meaning |
|--------|---------|
| **Final H** | Water level at the end |
| **Final gate** | Gate opening at the end (cm) |
| **Peak H** | Highest water level reached — shows ⚠ OVERFLOW if near limit |
| **Steady-state error** | How close H stays to the setpoint at the end |
| **Settling time (2%)** | How quickly H reaches and stays within ±2% of setpoint |

## Key Concepts

- **H_set** — the target water level the controller tries to maintain
- **H_max** — maximum water level before overflow (water is clamped here)
- **a_eq** — the equilibrium gate opening that perfectly balances inflow at the setpoint
- **Gate opening** — displayed in **cm** everywhere
- The controller uses **pulse modulation**: it sends OPEN/CLOSE/STOP commands to the motor in timed pulses, not continuous movement

## Running Tests

```bash
uv run pytest sluice_sim/tests/ -v
```

## Project Files

```
sluice_sim/
├── models/plant.py          # Physics: channel, gate, motor
├── controllers/
│   ├── pid_pulse.py         # PID controller with pulse modulation
│   └── autotune.py          # Auto-tuning via relay feedback
├── profiles/inflow.py       # Inflow disturbance profiles
├── sim/simulator.py         # Runs the simulation loop
├── ui/app.py                # Streamlit dashboard
└── tests/                   # Unit tests
```
