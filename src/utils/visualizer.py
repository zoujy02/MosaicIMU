import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def print_evaluation_metrics(
    gt_positions, ekf_ate_3d, ekf_rte_3d, net_ate_3d, net_rte_3d,
    ekf_ate_2d, ekf_rte_2d, net_ate_2d, net_rte_2d,
    dt, output_dir, base_name, actual_start, actual_end,
):
    print("\n--- Quantitative error analysis ---")
    print(f"Segment Length     : {len(gt_positions)} frames ({len(gt_positions) * dt:.1f} sec)")
    print(f"[3D] EKF MSCKF          -> ATE: {ekf_ate_3d:.4f} m, T-RTE (10s): {ekf_rte_3d:.4f} m")
    print(f"[3D] Pure Net Integrate -> ATE: {net_ate_3d:.4f} m, T-RTE (10s): {net_rte_3d:.4f} m")
    print(f"[2D] EKF MSCKF          -> ATE: {ekf_ate_2d:.4f} m, T-RTE (10s): {ekf_rte_2d:.4f} m")
    print(f"[2D] Pure Net Integrate -> ATE: {net_ate_2d:.4f} m, T-RTE (10s): {net_rte_2d:.4f} m")
    print("--------------------\n")

    metrics_file_txt = os.path.join(output_dir, f"{base_name}_seg{actual_start}-{actual_end}_metrics.txt")
    with open(metrics_file_txt, "w") as f:
        f.write(f"Trajectory Segment: {actual_start} ~ {actual_end}\n")
        f.write(f"Duration: {len(gt_positions) * dt:.1f} seconds\n\n")
        f.write(f"[3D] EKF MSCKF          -> ATE: {ekf_ate_3d:.4f} m, T-RTE (10s): {ekf_rte_3d:.4f} m\n")
        f.write(f"[3D] Pure Net Integrate -> ATE: {net_ate_3d:.4f} m, T-RTE (10s): {net_rte_3d:.4f} m\n\n")
        f.write(f"[2D] EKF MSCKF          -> ATE: {ekf_ate_2d:.4f} m, T-RTE (10s): {ekf_rte_2d:.4f} m\n")
        f.write(f"[2D] Pure Net Integrate -> ATE: {net_ate_2d:.4f} m, T-RTE (10s): {net_rte_2d:.4f} m\n")
    print(f"  [OK] Quantitative metrics saved to: {metrics_file_txt}")


def plot_trajectory_3d(
    gt_positions, est_positions, net_pos_integrated,
    output_dir, base_name, actual_start, actual_end,
):
    print("Generating and saving 3D trajectory plot...")

    fig1 = plt.figure(figsize=(10, 8))
    ax2 = fig1.add_subplot(111, projection='3d')
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', color='black', linewidth=2)
    ax2.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], label='EKF Posterior', color='red', linewidth=1.5)
    ax2.plot(net_pos_integrated[:, 0], net_pos_integrated[:, 1], net_pos_integrated[:, 2], label='Pure Net Integration', color='darkblue', linestyle='--', linewidth=1.2)

    ax2.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='green', s=100, marker='o', zorder=5, label=f'Start (Frame {actual_start})')
    ax2.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='black', s=150, marker='*', zorder=5, label='GT End')
    ax2.scatter(est_positions[-1, 0], est_positions[-1, 1], est_positions[-1, 2], c='red', s=150, marker='*', zorder=5, label='EKF End')
    ax2.scatter(net_pos_integrated[-1, 0], net_pos_integrated[-1, 1], net_pos_integrated[-1, 2], c='darkblue', s=150, marker='*', zorder=5, label='Net End')

    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Z [m]')
    ax2.set_title(f'3D Trajectory Analysis\nSegment: {actual_start} ~ {actual_end}')
    ax2.legend()

    x_limits = ax2.get_xlim3d()
    y_limits = ax2.get_ylim3d()
    z_limits = ax2.get_zlim3d()
    plot_radius = 0.5 * max([abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0])])

    ax2.set_xlim3d([np.mean(x_limits) - plot_radius, np.mean(x_limits) + plot_radius])
    ax2.set_ylim3d([np.mean(y_limits) - plot_radius, np.mean(y_limits) + plot_radius])
    ax2.set_zlim3d([np.mean(z_limits) - plot_radius, np.mean(z_limits) + plot_radius])

    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{base_name}_seg{actual_start}-{actual_end}_traj.png"), dpi=150)
    plt.close(fig1)
    print(f"  [OK] Saved 3D trajectory plot to: {output_dir}\n")
