import os
import scipy.io
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(SCRIPT_DIR, "info.mat")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path)
    print(f"Exported to plots/{name}")


def plot_tracking_error(data):
    time = data["time"].flatten()
    e1 = data["q1"].flatten() - data["q1ref"].flatten()
    e2 = data["q2"].flatten() - data["q2ref"].flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True)

    ax1.plot(time, e1, color="#1f77b4", linewidth=0.8)
    ax1.set_ylabel(r"$e_1$ (rad)")
    ax1.axhline(0, color="k", linewidth=0.5, linestyle="--")

    ax2.plot(time, e2, color="#d62728", linewidth=0.8)
    ax2.set_ylabel(r"$e_2$ (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.axhline(0, color="k", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    save_plot(fig, "tracking_error.pdf")


def plot_xy_trajectory(data):
    x = data["x"].flatten()
    y = data["y"].flatten()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.plot(x, y, color="#1f77b4", linewidth=0.8)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_aspect("equal")

    fig.tight_layout()
    save_plot(fig, "xy_trajectory.pdf")


def plot_prismatic(data):
    time = data["time"].flatten()
    prisma = data["prismaArt"].flatten()
    prisma_ref = data["prismaRef"].flatten()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(time, prisma, color="#d62728", linestyle="-", label=r"$z$")
    ax.plot(time, prisma_ref, color="#1f77b4", linestyle="--", linewidth=1.2, label=r"$z^{\mathrm{ref}}$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_plot(fig, "prismatic.pdf")


def plot_error_histogram(data):
    import numpy as np

    e1 = data["q1"].flatten() - data["q1ref"].flatten()
    e2 = data["q2"].flatten() - data["q2ref"].flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5))

    ax1.hist(e1, bins=50, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel(r"$e_1$ (rad)")
    ax1.set_ylabel("Count")
    ax1.text(0.95, 0.95, f"$\\mu$={np.mean(e1):.2e}\n$\\sigma$={np.std(e1):.2e}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8)

    ax2.hist(e2, bins=50, color="#d62728", edgecolor="white", linewidth=0.5)
    ax2.set_xlabel(r"$e_2$ (rad)")
    ax2.text(0.95, 0.95, f"$\\mu$={np.mean(e2):.2e}\n$\\sigma$={np.std(e2):.2e}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8)

    fig.tight_layout()
    save_plot(fig, "error_histogram.pdf")
    print(f"  e1: mean={np.mean(e1):.6f}, std={np.std(e1):.6f}")
    print(f"  e2: mean={np.mean(e2):.6f}, std={np.std(e2):.6f}")


def plot_gripper_states(data):
    time = data["time"].flatten()
    gripper_l = data["gripperL"].flatten()
    gripper_r = data["gripperR"].flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True)

    ax1.plot(time, gripper_l, color="#1f77b4", linewidth=0.8)
    ax1.set_ylabel("Left gripper")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Open", "Closed"])

    ax2.plot(time, gripper_r, color="#d62728", linewidth=0.8)
    ax2.set_ylabel("Right gripper")
    ax2.set_xlabel("Time (s)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Open", "Closed"])

    fig.tight_layout()
    save_plot(fig, "gripper_states.pdf")


def plot_3d_trajectory(data):
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    x = data["x"].flatten()
    y = data["y"].flatten()
    z = data["prismaArt"].flatten()
    time = data["time"].flatten()

    # Downsample for performance
    step = 100
    x, y, z, time = x[::step], y[::step], z[::step], time[::step]

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection="3d")

    # Create line segments colored by time
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(time.min(), time.max())
    lc = Line3DCollection(segments, cmap="viridis", norm=norm, linewidth=0.8)
    lc.set_array(time[:-1])
    ax.add_collection3d(lc)

    # Set limits manually since add_collection3d doesn't auto-scale
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_zlabel(r"$z$ (m)")

    cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.15)
    cbar.set_label("Time (s)")

    fig.tight_layout()
    save_plot(fig, "trajectory_3d.pdf")


def plot_prismatic_error(data):
    import numpy as np

    time = data["time"].flatten()
    e_z = data["prismaArt"].flatten() - data["prismaRef"].flatten()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(time, e_z, color="#2ca02c", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$e_z$ (m)")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    save_plot(fig, "prismatic_error.pdf")
    print(f"  e_z: mean={np.mean(e_z):.6f}, std={np.std(e_z):.6f}")


def plot_joint_velocities(data):
    import numpy as np

    time = data["time"].flatten()
    q1 = data["q1"].flatten()
    q2 = data["q2"].flatten()

    dt = np.diff(time)
    dq1 = np.diff(q1) / dt
    dq2 = np.diff(q2) / dt
    t_vel = time[:-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True)

    ax1.plot(t_vel, dq1, color="#1f77b4", linewidth=0.5)
    ax1.set_ylabel(r"$\dot{q}_1$ (rad/s)")

    ax2.plot(t_vel, dq2, color="#d62728", linewidth=0.5)
    ax2.set_ylabel(r"$\dot{q}_2$ (rad/s)")
    ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    save_plot(fig, "joint_velocities.pdf")


def plot_phase_portraits(data):
    import numpy as np

    time = data["time"].flatten()
    q1 = data["q1"].flatten()
    q2 = data["q2"].flatten()

    dt = np.diff(time)
    dq1 = np.diff(q1) / dt
    dq2 = np.diff(q2) / dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5))

    ax1.plot(q1[:-1], dq1, color="#1f77b4", linewidth=0.3)
    ax1.set_xlabel(r"$q_1$ (rad)")
    ax1.set_ylabel(r"$\dot{q}_1$ (rad/s)")

    ax2.plot(q2[:-1], dq2, color="#d62728", linewidth=0.3)
    ax2.set_xlabel(r"$q_2$ (rad)")
    ax2.set_ylabel(r"$\dot{q}_2$ (rad/s)")

    fig.tight_layout()
    save_plot(fig, "phase_portraits.pdf")


def plot_error_psd(data):
    import numpy as np
    from scipy import signal

    time = data["time"].flatten()
    e1 = data["q1"].flatten() - data["q1ref"].flatten()
    e2 = data["q2"].flatten() - data["q2ref"].flatten()

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    f1, psd1 = signal.welch(e1, fs, nperseg=min(len(e1)//4, 4096))
    f2, psd2 = signal.welch(e2, fs, nperseg=min(len(e2)//4, 4096))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True)

    ax1.semilogy(f1, psd1, color="#1f77b4", linewidth=0.8)
    ax1.set_ylabel(r"PSD $e_1$ (rad$^2$/Hz)")

    ax2.semilogy(f2, psd2, color="#d62728", linewidth=0.8)
    ax2.set_ylabel(r"PSD $e_2$ (rad$^2$/Hz)")
    ax2.set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    save_plot(fig, "error_psd.pdf")


def plot_xy_trajectory_colored(data):
    import numpy as np
    from matplotlib.collections import LineCollection

    x = data["x"].flatten()
    y = data["y"].flatten()
    time = data["time"].flatten()

    step = 100
    x, y, time = x[::step], y[::step], time[::step]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm, linewidth=0.8)
    lc.set_array(time[:-1])
    ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_aspect("equal")

    cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", shrink=0.8, pad=0.15)
    cbar.set_label("Time (s)")

    fig.tight_layout()
    save_plot(fig, "xy_trajectory_colored.pdf")


def plot_combined_joints(data):
    time = data["time"].flatten()
    q1 = data["q1"].flatten()
    q1ref = data["q1ref"].flatten()
    q2 = data["q2"].flatten()
    q2ref = data["q2ref"].flatten()
    z = data["prismaArt"].flatten()
    zref = data["prismaRef"].flatten()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True)

    ax1.plot(time, q1, color="#d62728", linestyle="-", label=r"$q_1$")
    ax1.plot(time, q1ref, color="#1f77b4", linestyle="--", linewidth=1.2, label=r"$q_1^{\mathrm{ref}}$")
    ax1.set_ylabel(r"$q_1$ (rad)")
    ax1.legend(frameon=False, loc="upper right", fontsize=7)

    ax2.plot(time, q2, color="#d62728", linestyle="-", label=r"$q_2$")
    ax2.plot(time, q2ref, color="#1f77b4", linestyle="--", linewidth=1.2, label=r"$q_2^{\mathrm{ref}}$")
    ax2.set_ylabel(r"$q_2$ (rad)")
    ax2.legend(frameon=False, loc="upper right", fontsize=7)

    ax3.plot(time, z, color="#d62728", linestyle="-", label=r"$z$")
    ax3.plot(time, zref, color="#1f77b4", linestyle="--", linewidth=1.2, label=r"$z^{\mathrm{ref}}$")
    ax3.set_ylabel(r"$z$ (m)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(frameon=False, loc="upper right", fontsize=7)

    fig.tight_layout()
    save_plot(fig, "combined_joints.pdf")


def plot_prisma_op(data):
    time = data["time"].flatten()
    prisma_op = data["prismaOp"].flatten()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(time, prisma_op, color="#2ca02c", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("prismaOp")

    fig.tight_layout()
    save_plot(fig, "prisma_op.pdf")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = scipy.io.loadmat(DATA_FILE)

    plot_tracking_error(data)
    plot_xy_trajectory(data)
    plot_prismatic(data)
    plot_error_histogram(data)
    plot_gripper_states(data)
    plot_3d_trajectory(data)
    plot_prismatic_error(data)
    plot_joint_velocities(data)
    plot_phase_portraits(data)
    plot_error_psd(data)
    plot_xy_trajectory_colored(data)
    plot_combined_joints(data)
    plot_prisma_op(data)


if __name__ == "__main__":
    main()
