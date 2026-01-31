# SCARA Manipulator Simulation

https://github.com/user-attachments/assets/020b8004-feb7-4a93-965b-c4e99ffbe820

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

## License

[MIT](LICENSE)

## Authors

- Alvaro J. Gaona
- Peter Pasuy-Quevedo
