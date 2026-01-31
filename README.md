# SCARA Robot Analysis

Visualization tools for SCARA robot PID control data.

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
