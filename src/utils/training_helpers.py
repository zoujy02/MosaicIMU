import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.imu_dataset import IMUMoENpzDataset
from src.utils.loss import build_carrier_soft_priors_tensor


def compute_balance_repeats(class_counts: dict) -> dict:
    """Oversample so each class approaches the longest class (by duration/frames)."""
    positive = [c for c in class_counts.values() if c > 0]
    if not positive:
        return {label: 1 for label in class_counts}
    target = max(positive)
    return {
        label: max(1, int(round(target / count))) if count > 0 else 1
        for label, count in class_counts.items()
    }


def auto_pretrain_balance_config(
    train_root, window_size, step_size, categories, *, flat_layout=None,
):
    # Balance by total duration (frame count @ fixed Hz), not file or window count.
    # window_size/step_size kept for call-site compatibility; unused here.
    _ = (window_size, step_size)
    raw_counts = IMUMoENpzDataset.count_raw_frames_per_class(
        train_root, categories, flat_layout=flat_layout,
    )
    repeats = compute_balance_repeats(raw_counts)
    val_weights = {label: float(repeat) for label, repeat in repeats.items()}

    idx_to_name = {idx: name for name, idx in categories.items()}
    parts = []
    for label in sorted(raw_counts):
        frames = raw_counts[label]
        dur_min = frames / 200.0 / 60.0
        name = idx_to_name.get(label, str(label))
        parts.append(f"{name}:{dur_min:.1f}min→x{repeats[label]}")
    print("[balance] by duration (not file count) | " + " ".join(parts))

    return repeats, val_weights, raw_counts


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def imu_moe_datasets(
    train_root,
    val_root,
    window_size,
    step_size,
    categories,
    repeats_config,
    val_metric_weights,
    *,
    print_style='finetune',
    verbose=True,
):
    train_dataset = IMUMoENpzDataset(
        data_root=train_root, window_size=window_size, step_size=step_size, mode='train',
        repeats_config=repeats_config, categories=categories, print_style=print_style,
        verbose=verbose,
    )
    val_dataset = IMUMoENpzDataset(
        data_root=val_root, window_size=window_size, step_size=step_size, mode='val',
        categories=categories, print_style=print_style, val_metric_weights=val_metric_weights,
        verbose=verbose,
    )
    return train_dataset, val_dataset


def imu_moe_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    *,
    num_workers=8,
    seed_loaders=False,
):
    train_kw = dict(
        batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_kw = dict(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
    )
    if seed_loaders:
        g_train = torch.Generator()
        g_train.manual_seed(42)
        g_val = torch.Generator()
        g_val.manual_seed(42)
        train_kw.update(worker_init_fn=_seed_worker, generator=g_train, prefetch_factor=2)
        val_kw.update(worker_init_fn=_seed_worker, generator=g_val, prefetch_factor=2)
    return (
        DataLoader(train_dataset, **train_kw),
        DataLoader(val_dataset, **val_kw),
    )


def resolve_loss_hyperparams(config: dict) -> dict:
    lc = config.get('loss') or {}
    hz = float(lc.get('imu_sample_rate_hz', 200))
    return {
        'imu_sample_rate_hz': hz,
        'dt': 1.0 / hz,
        'lambda_ce': float(lc.get('lambda_ce', 0.05)),
        'alpha_orth': float(lc.get('alpha_orth', 0.1)),
        'lambda_vel': float(lc.get('lambda_vel', 1.0)),
        'lambda_nll': float(lc.get('lambda_nll', 1.0)),
        'lambda_disp': float(lc.get('lambda_disp', 10.0)),
        'lambda_int': float(lc.get('lambda_int', 10.0)),
        'variance_detach_epochs': int(lc.get('variance_detach_epochs', 10)),
        'router_log_eps': float(lc.get('router_log_eps', 1e-8)),
        'carrier_soft_priors': build_carrier_soft_priors_tensor(lc),
        'grad_clip_norm': float(lc.get('grad_clip_norm', 5.0)),
    }


def ensure_checkpoint_dir(name: str = "checkpoint") -> str:
    os.makedirs(name, exist_ok=True)
    return name


def load_model_checkpoint(path, device):
    """Load weights from plain state_dict or {'model_state_dict', 'epoch', ...}.

    Returns (state_dict, meta) where meta['epoch'] is 1-based last-completed
    epoch when present, else None.
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        meta = dict(obj)
        return obj["model_state_dict"], meta
    if isinstance(obj, dict) and "state_dict" in obj:
        meta = dict(obj)
        meta.setdefault("model_state_dict", obj["state_dict"])
        return obj["state_dict"], meta
    return obj, {"model_state_dict": obj, "epoch": None}


def save_model_checkpoint(path, model, *, epoch, **extra):
    """Save checkpoint with 1-based last-completed epoch for resume."""
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
    }
    payload.update(extra)
    torch.save(payload, path)


def snapshot_checkpoint(src_path, epoch):
    """Copy `src_path` to sibling `<stem>_eXXX.pth` so periodic evals keep their weights."""
    import shutil

    stem, ext = os.path.splitext(src_path)
    dst = f"{stem}_e{int(epoch):03d}{ext}"
    shutil.copy2(src_path, dst)
    return dst
