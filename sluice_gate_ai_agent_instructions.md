# Sluice Gate Level Control Simulator

## General Instructions for AI Coding Agent

------------------------------------------------------------------------

# 1. Objective

Build a modular Python application that simulates:

-   A rectangular open channel with upstream water storage\
-   A sluice gate driven by a discrete motor (`OPEN / CLOSE / STOP`)\
-   A PI/PID controller that regulates water level to a target setpoint\
-   Adjustable inflow disturbances (step, ramp, etc.)\
-   Real-time visualization (plots + animated schematic)\
-   Interactive UI with configurable parameters\
-   Export capability (CSV + scenario JSON)

The application must be structured, testable, extensible, and physically
consistent.

------------------------------------------------------------------------

# 2. Core System Model

## 2.1 Physical Model (Plant)

### State Variables

-   `H(t)` --- water level \[m\]
-   `a(t)` --- gate opening \[m\]

### Geometry

-   Channel width: `b` \[m\]
-   Effective upstream storage length: `L` \[m\]
-   Storage cross-section:

```{=html}
<!-- -->
```
    A_storage = b * L

### Mass Balance Equation

    dH/dt = (Qin(t) - Qout(H, a)) / A_storage

Integrate numerically using fixed timestep Euler (RK4 optional later).

### Gate Discharge Model (Initial Version)

    Qout = Cd * b * a * sqrt(2 * g * H)

Where: - `Cd` default = 0.65 - `g = 9.81` - `a ∈ [0, a_max]` - `H >= 0`

Add clamping to ensure physical validity.

------------------------------------------------------------------------

# 3. Inflow Profiles

Support selectable inflow models:

-   Constant
-   Step
-   Ramp (e.g., +100 m³/h over 10 s)
-   Sine disturbance
-   Optional: piecewise custom segments

### Units

-   UI: m³/h
-   Internal calculations: m³/s

------------------------------------------------------------------------

# 4. Actuator (Motor) Model

The motor accepts only:

OPEN\
CLOSE\
STOP

### Actuator Parameters

-   `open_rate` \[m/s\]
-   `close_rate` \[m/s\]

Update each time step:

    if cmd == OPEN:   a += open_rate * dt
    if cmd == CLOSE:  a -= close_rate * dt

Clamp:

    a ∈ [0, a_max]

------------------------------------------------------------------------

# 5. Controller Design

## 5.1 Signal Processing

### Measurement Filtering

Low-pass filter:

    Hf = alpha * H_meas + (1 - alpha) * Hf

### Error Definition

    e = H_set - Hf

Convention: - e \> 0 → water low → CLOSE - e \< 0 → water high → OPEN

------------------------------------------------------------------------

## 5.2 PI/PID Core

    P = Kp * e
    I += Ki * e * dt
    D = Kd * (e - e_prev) / dt
    u = P + I + D

### Anti-Windup

-   Clamp I within bounds
-   Conditional integration when saturated

------------------------------------------------------------------------

## 5.3 Pulse Modulation (Discrete Motor)

Operate in windows of length `Twindow`.

At each window boundary:

1.  Compute `u`
2.  Direction = sign(u)
3.  Duty:

```{=html}
<!-- -->
```
    duty = clamp(|u| / Umax, 0..1)

4.  Pulse duration:

```{=html}
<!-- -->
```
    pulse = duty * Twindow

5.  Enforce:
    -   `min_on`
    -   `pulse <= Twindow`
    -   `min_switch_interval` between direction changes

------------------------------------------------------------------------

## 5.4 Deadband + Hysteresis

Use two thresholds:

-   `deadband_hi`
-   `deadband_lo`

Logic:

    if |e| < deadband_lo → STOP
    if |e| < deadband_hi and already STOP → STOP

------------------------------------------------------------------------

# 6. Default Scenario

Provide a preset matching the engineering example:

  Parameter     Value
  ------------- ---------------------
  b             0.75 m
  L             50 m
  H0            1.5 m
  H_set         2.0 m
  Qin           1000 m³/h
  Disturbance   +100 m³/h over 10 s
  Cd            0.65
  a_max         0.45 m
  Motor speed   1 cm/s
  Kp            5.0
  Ki            0.25
  Kd            0
  Twindow       0.5 s
  min_on        0.18 s
  Umax          0.18

Also compute equilibrium opening:

    a_eq = Qin / (Cd * b * sqrt(2 * g * H_set))

Display this in UI.

------------------------------------------------------------------------

# 7. Application Architecture

Use modular structure:

    sluice_sim/
    │
    ├── models/
    │   └── plant.py
    │
    ├── controllers/
    │   └── pid_pulse.py
    │
    ├── profiles/
    │   └── inflow.py
    │
    ├── sim/
    │   └── simulator.py
    │
    ├── ui/
    │   └── app.py
    │
    ├── tests/
    └── README.md

Responsibilities:

-   plant.py → physics
-   pid_pulse.py → control logic
-   inflow.py → disturbance generation
-   simulator.py → orchestration + logging
-   app.py → UI only

------------------------------------------------------------------------

# 8. UI Requirements

Use Streamlit (recommended) or PyQt/Dash.

## Controls

-   Geometry parameters
-   Motor speeds
-   Controller gains
-   Deadband settings
-   Inflow profile selection
-   Simulation time
-   Noise level
-   Start / Pause / Reset
-   Step simulation
-   Apply changes button

## Visualization

-   Plot: H(t) + setpoint
-   Plot: a(t)
-   Plot: Qin(t) + Qout(t)
-   Optional: error and duty

## Animated Schematic

-   Channel outline
-   Water fill height
-   Gate position indicator

## Export

-   CSV log export
-   Scenario JSON save/load
-   Optional GIF/MP4 export

------------------------------------------------------------------------

# 9. Logging Format

Simulation output DataFrame must include:

t\
H\
a\
Qin\
Qout\
e\
u\
duty\
cmd

------------------------------------------------------------------------

# 10. Testing Requirements

Write pytest tests for:

-   Inflow ramp correctness
-   Qout increases with H and a
-   Deadband behavior
-   Pulse scheduling limits
-   Anti-windup clamping
-   No illegal state transitions

------------------------------------------------------------------------

# 11. Engineering Constraints

-   Internal units in SI
-   Stable for dt ≤ 0.05 s
-   Deterministic runs with seed option
-   Clamp all physical variables
-   No blocking infinite loops in UI
-   Maintain separation of concerns

------------------------------------------------------------------------

# 12. Stretch Goals (Optional)

-   Feedforward from Qin
-   Submerged vs free-flow gate model
-   Nonlinear storage geometry
-   Sensor delay
-   Motor backlash
-   Auto-tuning controller

------------------------------------------------------------------------

# 13. Final Deliverables

-   Runnable application
-   Modular codebase
-   Preset scenario
-   Unit tests
-   README with usage instructions
-   Export capability

------------------------------------------------------------------------

End of specification.
