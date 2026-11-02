import os
import glob
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class IMUMoENpzDataset(Dataset):
    def __init__(self, data_root, window_size=500, step_size=250, mode='train',
                 repeats_config=None, categories=None, print_style='finetune',
                 flat_layout=None, val_metric_weights=None, verbose=True):
        self.window_size = window_size
        self.step_size = step_size
        self.seq_len = window_size
        self.mode = mode
        self.samples = []
        self.data_root = data_root
        self.val_metric_weights = val_metric_weights if val_metric_weights is not None else None
        self._verbose = bool(verbose)

        if categories is None:
            self.categories = {'car': 0, 'quadrupedal': 1, 'pedestrian': 2, 'drone': 3}
        else:
            self.categories = categories

        self.repeats = repeats_config if repeats_config else {v: 1 for v in self.categories.values()}

        use_flat = self._resolve_flat_layout(data_root, flat_layout, self.categories)

        missing_categories = []
        if self._verbose and print_style == 'finetune':
            print(f"Scanning [{mode}] dataset directory: {data_root}"
                  + (" (flat layout)" if use_flat else ""))
        for category, label in self.categories.items():
            folder_path = data_root if use_flat else os.path.join(data_root, category)
            if not os.path.exists(folder_path):
                missing_categories.append((category, folder_path))
                if self._verbose:
                    if print_style == 'finetune':
                        print(f"   [WARN] folder not found: '{folder_path}', class '{category}' will be missing.")
                    else:
                        print(f"[INFO] folder not found: {folder_path}, skipped.")
                continue
            npz_files = sorted(glob.glob(os.path.join(folder_path, '*.npz')))
            for file in npz_files:
                self._process_file(file, label, category)
        if self._verbose:
            if print_style == 'finetune':
                print(f"Dataset ({self.mode}) loaded, total {len(self.samples)} windows.")
            else:
                print(f"Dataset ({self.mode}) loaded, total {len(self.samples)} sliding-window samples.")

        if not self._verbose and missing_categories:
            mc = ", ".join(c for c, _ in missing_categories)
            print(f"   [WARN] [{self.mode}] missing categories (no data): {mc}")

        self._print_class_breakdown()

    @staticmethod
    def _sliding_window_count(num_frames, window_size, step_size):
        if num_frames < window_size:
            return 0
        return len(range(0, num_frames - window_size, step_size))

    @staticmethod
    def _npz_num_frames(data) -> int:
        """Sequence length from NPZ; prefer duration (frames), not file count."""
        for key in ('imu', 'retargetted_imu'):
            if key in data.files:
                return int(len(data[key]))
        raise KeyError(
            f"NPZ missing IMU array; keys={list(data.files)}"
        )

    @classmethod
    def count_raw_frames_per_class(
        cls, data_root, categories, flat_layout=None,
    ):
        """Total IMU frames (∝ duration at fixed Hz) per class label."""
        use_flat = cls._resolve_flat_layout(data_root, flat_layout, categories)
        counts = {label: 0 for label in categories.values()}
        for category, label in categories.items():
            folder_path = data_root if use_flat else os.path.join(data_root, category)
            if not os.path.exists(folder_path):
                continue
            for file_path in sorted(glob.glob(os.path.join(folder_path, '*.npz'))):
                with np.load(file_path) as data:
                    counts[label] += cls._npz_num_frames(data)
        return counts

    @classmethod
    def count_raw_windows_per_class(
        cls, data_root, window_size, step_size, categories, flat_layout=None,
    ):
        use_flat = cls._resolve_flat_layout(data_root, flat_layout, categories)
        counts = {label: 0 for label in categories.values()}
        for category, label in categories.items():
            folder_path = data_root if use_flat else os.path.join(data_root, category)
            if not os.path.exists(folder_path):
                continue
            for file_path in sorted(glob.glob(os.path.join(folder_path, '*.npz'))):
                with np.load(file_path) as data:
                    num_frames = cls._npz_num_frames(data)
                counts[label] += cls._sliding_window_count(num_frames, window_size, step_size)
        return counts

    @staticmethod
    def _resolve_flat_layout(data_root, flat_layout, categories):
        if flat_layout is True or flat_layout is False:
            return bool(flat_layout)
        if len(categories) != 1:
            return False
        if not os.path.isdir(data_root):
            return False
        only_category = next(iter(categories.keys()))
        per_class_dir = os.path.join(data_root, only_category)
        if os.path.isdir(per_class_dir):
            return False
        return True

    @staticmethod
    def _load_trajectory_arrays(file_path):
        """Load IMU/pos/quat; supports both raw and retargetted_* NPZ schemas."""
        with np.load(file_path) as data:
            if 'imu' in data.files:
                imu_data = np.asarray(data['imu'], dtype=np.float64)
                pos_data = np.asarray(data['pos'], dtype=np.float64)
                quat_data = np.asarray(data['quat'], dtype=np.float64)
            elif 'retargetted_imu' in data.files:
                imu_data = np.asarray(data['retargetted_imu'], dtype=np.float64)
                pos_data = np.asarray(data['retargetted_pos'], dtype=np.float64)
                quat_data = np.asarray(data['retargetted_quat'], dtype=np.float64)
            else:
                raise KeyError(
                    f"{file_path}: expected imu/pos/quat or retargetted_*; "
                    f"keys={list(data.files)}"
                )
        return imu_data, pos_data, quat_data

    def _process_file(self, file_path, label, category_name):
        imu_data, pos_data, quat_data = self._load_trajectory_arrays(file_path)

        dt = 1.0 / 200.0
        vel_world_data = np.zeros_like(pos_data)
        vel_world_data[1:] = (pos_data[1:] - pos_data[:-1]) / dt
        vel_world_data[0] = vel_world_data[1]

        num_frames = len(imu_data)
        file_samples = []

        for start_idx in range(0, num_frames - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size

            w_imu = imu_data[start_idx:end_idx].copy()
            w_quat = quat_data[start_idx:end_idx]

            # quat is SciPy [x,y,z,w]: Rotation.apply maps body vectors -> world
            r0_B_to_W = R.from_quat(w_quat[0])
            yaw0, _, _ = r0_B_to_W.as_euler('ZYX', degrees=False)
            r_W_to_L0 = R.from_euler('Z', -yaw0, degrees=False)

            target_vel_world_seq = vel_world_data[start_idx:end_idx]
            target_vel_L0_seq = r_W_to_L0.apply(target_vel_world_seq)

            r_B_to_W = R.from_quat(w_quat)
            r_B_to_L0 = r_W_to_L0 * r_B_to_W

            acc_L0 = r_B_to_L0.apply(w_imu[:, 0:3])
            gyro_L0 = r_B_to_L0.apply(w_imu[:, 3:6])
            w_imu_L0 = np.concatenate([acc_L0, gyro_L0], axis=-1)

            file_samples.append({
                'imu': torch.FloatTensor(w_imu_L0).transpose(0, 1),
                'target_vel_seq': torch.FloatTensor(target_vel_L0_seq),
                'target_pos': torch.FloatTensor([0.0, 0.0]),
                'label': label,
                'raw_imu': torch.FloatTensor(w_imu),
                'init_q': torch.FloatTensor(w_quat[0]),
                'yaw0': yaw0,
                'pos_start': torch.FloatTensor(pos_data[max(0, start_idx - 1)]),
                'quat_seq': torch.FloatTensor(w_quat),
            })

        if len(file_samples) == 0:
            return

        if self.mode == 'train':
            repeat_times = self.repeats.get(label, 1)
            for _ in range(repeat_times):
                self.samples.extend(file_samples)
        else:
            self.samples.extend(file_samples)

    def _print_class_breakdown(self):
        total = len(self.samples)
        cat_counts = Counter()
        for s in self.samples:
            cat_counts[s['label']] += 1

        if not self._verbose:
            if total == 0:
                print(f"   [{self.mode}] {self.data_root} | 0 windows (empty)")
                return
            show_val_weights = (
                self.mode == 'val'
                and self.val_metric_weights is not None
                and isinstance(self.val_metric_weights, dict)
                and len(self.val_metric_weights) > 0
            )
            if show_val_weights:
                total_w_mass = sum(
                    cat_counts.get(label, 0) * float(self.val_metric_weights.get(label, 1.0))
                    for _, label in self.categories.items()
                )
                parts = []
                for category, label in self.categories.items():
                    count = cat_counts.get(label, 0)
                    w_mass = count * float(self.val_metric_weights.get(label, 1.0))
                    pct = 100.0 * w_mass / total_w_mass if total_w_mass > 0 else 0.0
                    mass_disp = int(w_mass) if abs(w_mass - round(w_mass)) < 1e-6 else w_mass
                    parts.append(f"{category}:{mass_disp}({pct:.1f}%)")
                total_disp = int(total_w_mass) if abs(total_w_mass - round(total_w_mass)) < 1e-6 else total_w_mass
                print(
                    f"   [{self.mode}] {self.data_root} | {total_disp} equiv windows | "
                    + " ".join(parts)
                )
            else:
                parts = []
                for category, label in self.categories.items():
                    count = cat_counts.get(label, 0)
                    pct = 100.0 * count / total
                    parts.append(f"{category}:{count}({pct:.1f}%)")
                print(f"   [{self.mode}] {self.data_root} | {total} windows | " + " ".join(parts))
            return

        weighting_tag = "post-weighting" if self.mode == 'train' else "raw (no oversampling)"
        print(f"   [{self.mode}] data root         : {self.data_root}")
        print(f"   [{self.mode}] per-class summary : {weighting_tag}, total = {total} windows")

        if total == 0:
            print(f"   [{self.mode}]    (no samples loaded)")
            return

        show_val_weights = (
            self.mode == 'val'
            and self.val_metric_weights is not None
            and isinstance(self.val_metric_weights, dict)
            and len(self.val_metric_weights) > 0
        )
        if show_val_weights:
            total_w_mass = sum(
                cat_counts.get(label, 0) * float(self.val_metric_weights.get(label, 1.0))
                for _, label in self.categories.items()
            )
            print(
                f"   [{self.mode}] val_metric_weights -> equivalent mass Σ(count×w): "
                f"{total_w_mass:.4f}"
            )

        for category, label in self.categories.items():
            count = cat_counts.get(label, 0)
            repeat = self.repeats.get(label, 1) if self.mode == 'train' else 1
            pct = 100.0 * count / total
            tag = f"x{repeat}" if self.mode == 'train' else "raw"
            line = (
                f"   [{self.mode}]    - {category:<14} (label={label}, {tag:<3}): "
                f"{count:>6} windows  ({pct:5.2f}%)"
            )
            if show_val_weights:
                w = float(self.val_metric_weights.get(label, 1.0))
                w_mass = count * w
                denom = total_w_mass if total_w_mass > 0 else 1.0
                w_pct = 100.0 * w_mass / denom
                line += f"   | equiv. mass ×w={w_mass:>10.4f}  ({w_pct:5.2f}% of Σ count×w)"
            print(line)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['imu'], s['target_vel_seq'], s['target_pos'], s['label'],
            s['raw_imu'], s['init_q'], s['yaw0'], s['pos_start'], s['quat_seq'],
        )
