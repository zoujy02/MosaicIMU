import math

import torch


def apply_ultimate_augmentation_torch(imu, vel, pos, quat):
    """Yaw-rotate IMU/vel and add white + brownian + bias noise (PyTorch).

    Matches the former Triton kernel semantics, without Triton compile issues.
    imu: [B, 6, T], vel: [B, T, 3], pos/quat unused except passthrough.
    """
    B, _, T = imu.shape
    device = imu.device
    dtype = imu.dtype

    yaw = (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * math.pi
    cos_y = torch.cos(yaw).view(B, 1)
    sin_y = torch.sin(yaw).view(B, 1)

    # Channel-wise noise scales: ax,ay,az,gx,gy,gz
    white_std = imu.new_tensor([0.015, 0.015, 0.015, 0.005, 0.005, 0.005]).view(1, 6, 1)
    brown_std = imu.new_tensor([0.001, 0.001, 0.001, 0.0002, 0.0002, 0.0002]).view(1, 6, 1)
    bias_std = imu.new_tensor([0.05, 0.05, 0.05, 0.01, 0.01, 0.01]).view(1, 6, 1)

    white = torch.randn(B, 6, T, device=device, dtype=dtype) * white_std
    brown = torch.cumsum(torch.randn(B, 6, T, device=device, dtype=dtype) * brown_std, dim=-1)
    bias = torch.randn(B, 6, 1, device=device, dtype=dtype) * bias_std

    noisy = imu + white + brown + bias

    ax, ay, az = noisy[:, 0], noisy[:, 1], noisy[:, 2]
    gx, gy, gz = noisy[:, 3], noisy[:, 4], noisy[:, 5]

    rot_ax = ax * cos_y - ay * sin_y
    rot_ay = ax * sin_y + ay * cos_y
    rot_gx = gx * cos_y - gy * sin_y
    rot_gy = gx * sin_y + gy * cos_y

    out_imu = torch.stack([rot_ax, rot_ay, az, rot_gx, rot_gy, gz], dim=1)

    cos_y_t = cos_y.unsqueeze(-1)
    sin_y_t = sin_y.unsqueeze(-1)
    vx, vy, vz = vel[:, :, 0], vel[:, :, 1], vel[:, :, 2]
    out_vel = torch.stack(
        [
            vx * cos_y_t.squeeze(-1) - vy * sin_y_t.squeeze(-1),
            vx * sin_y_t.squeeze(-1) + vy * cos_y_t.squeeze(-1),
            vz,
        ],
        dim=-1,
    )

    return out_imu, out_vel, pos, quat


def apply_ultimate_augmentation_triton(imu, vel, pos, quat):
    """Public entry used by training; stable PyTorch path (Triton kernel retired)."""
    return apply_ultimate_augmentation_torch(imu, vel, pos, quat)
