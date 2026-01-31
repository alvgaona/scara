# SCARA Manipulator Simulation

<img width="1324" height="414" alt="scara-sim" src="https://github.com/user-attachments/assets/76f9be53-5441-488d-9c4b-cd7c8a28a423" />

## Setup

```bash
uv sync
```

## Generate Plots

```bash
uv run python scripts/generate_plots.py
```

Outputs PDF plots to `plots/`.

## Data

Place MATLAB data file (`info.mat`) in `scripts/`.

Expected variables: `time`, `q1`, `q1ref`, `q2`, `q2ref`, `prismaArt`, `prismaRef`, `prismaOp`, `gripperL`, `gripperR`, `x`, `y`.
