import argparse
import os

import torch
import yaml
from tqdm import tqdm

from src.datasets.augmentation import apply_ultimate_augmentation_triton
from src.models.finetune_net import WeightTuneMoEFineTuner
from src.models.moe_net import CarrierConditionedMoE_Seq
from src.utils.loss import (
    build_router_prior_tensor,
    compute_finetune_total_loss,
    router_prior_warmup_weight,
)
from src.utils.metrics import ExpertUsageAnalyzer
from src.utils.training_helpers import (
    ensure_checkpoint_dir,
    imu_moe_dataloaders,
    imu_moe_datasets,
    load_model_checkpoint,
    resolve_loss_hyperparams,
    seed_all,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MoE weight_tune (5 carriers incl. weight_robot)")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    return parser.parse_args()


def run_weight_tune(config, device):
    seed_all(42)
    checkpoint_dir = ensure_checkpoint_dir()

    paths_cfg = config['weight_tune_paths']
    train_cfg = config['weight_tune_training']
    loss_hp = resolve_loss_hyperparams(config)
    TRAIN_DATA_ROOT = paths_cfg['data_dir']
    VAL_DATA_ROOT = paths_cfg['val_dir']
    PRETRAINED_MODEL = paths_cfg['pretrained_weights']

    BATCH_SIZE = train_cfg['batch_size']
    EPOCHS = train_cfg['epochs']
    LR = train_cfg['lr']
    ETA_MIN = train_cfg.get('eta_min', 1e-5)
    PRIOR_WS = train_cfg.get('prior_warmup_start', 15)
    PRIOR_WE = train_cfg.get('prior_warmup_end', 30)
    PRIOR_MAX = float(train_cfg.get('prior_max_weight', 1.0))
    ROUTER_TEMP = float(train_cfg.get('router_temperature', 4.0))
    SEQ_LEN = config['data']['window_size']
    BIN_DT = loss_hp['dt']
    GCLIP = loss_hp['grad_clip_norm']
    LAMBDA_INT = loss_hp['lambda_int']
    MONITOR_LABEL = 4

    q_car = build_router_prior_tensor(train_cfg)

    category_to_idx = {
        'car': 0, 'quadrupedal': 1, 'pedestrian': 2, 'drone': 3, 'weight_robot': 4,
    }
    idx_to_category = {v: k for k, v in category_to_idx.items()}

    train_dataset, val_dataset = imu_moe_datasets(
        TRAIN_DATA_ROOT, VAL_DATA_ROOT, SEQ_LEN, config['data']['step_size'],
        category_to_idx, None, None,
        verbose=False,
    )

    if len(train_dataset) == 0:
        print("[ERROR] No training windows loaded for weight_tune. Check:")
        print(f"  - {TRAIN_DATA_ROOT}/<carrier>/*.npz  OR  flat single-class *.npz")
        return
    if len(val_dataset) == 0:
        print("[ERROR] No validation windows loaded for weight_tune. Check:")
        print(f"  - {VAL_DATA_ROOT}/...")
        return

    train_loader, val_loader = imu_moe_dataloaders(
        train_dataset, val_dataset, BATCH_SIZE, num_workers=4, seed_loaders=True,
    )

    state_dict, _meta = load_model_checkpoint(PRETRAINED_MODEL, device)
    base_model = CarrierConditionedMoE_Seq(
        in_channels=6, context_dim=128, expert_dim=64, num_experts=4, seq_len=SEQ_LEN,
    ).to(device)
    base_model.load_state_dict(state_dict)
    finetune_model = WeightTuneMoEFineTuner(base_model, q_car=q_car).to(device)

    gate_p_w = [p for n, p in finetune_model.named_parameters() if 'gate' in n and 'weight' in n and p.requires_grad]
    gate_p_b = [p for n, p in finetune_model.named_parameters() if 'gate' in n and 'bias' in n and p.requires_grad]
    expert_p = [p for n, p in finetune_model.named_parameters() if 'gate' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': expert_p, 'lr': LR, 'weight_decay': 1e-4},
        {'params': gate_p_w, 'lr': LR * 5, 'weight_decay': 1e-4},
        {'params': gate_p_b, 'lr': LR * 5, 'weight_decay': 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

    train_analyzer = ExpertUsageAnalyzer()
    val_analyzer = ExpertUsageAnalyzer()

    best_val_drift = float('inf')
    best_ckpt_name = 'best_weight_tune.pth'
    mon_name = 'weight_robot'

    for epoch in range(EPOCHS):
        current_lambda_prior = PRIOR_MAX * router_prior_warmup_weight(epoch, PRIOR_WS, PRIOR_WE)

        finetune_model.train()
        epoch_task_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(pbar):
            imu, target_vel, _, label, _, _, _, _, quat_seq = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            imu_aug, target_vel_aug, _, _ = apply_ultimate_augmentation_triton(
                imu, target_vel, torch.zeros_like(target_vel[:, :2]), quat_seq,
            )

            optimizer.zero_grad()
            mu, var, w, alphas, l_prior_val, _, _ = finetune_model(
                imu_aug, label, temperature=ROUTER_TEMP,
            )

            loss, loss_task, l_vel, l_nll, _ = compute_finetune_total_loss(
                mu, var, target_vel_aug, l_prior_val, current_lambda_prior, dt=BIN_DT,
                lambda_int=LAMBDA_INT,
            )

            train_analyzer.update(label, mu, target_vel_aug, w, dt=BIN_DT)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), GCLIP)
            optimizer.step()

            epoch_task_loss += loss_task.item()

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Alpha': f"{alphas.mean().item():.4f}",
            })

        scheduler.step()

        num_batches = max(1, len(train_loader))
        train_stats = train_analyzer.compute_epoch_stats(epoch + 1)
        train_analyzer.print_pretrain_report(
            train_stats, epoch_task_loss / num_batches, compact=True,
            line_prefix=f"Train E{epoch + 1}/{EPOCHS}", carriers=idx_to_category,
        )

        finetune_model.eval()
        epoch_val_task_loss = 0.0
        pbar_val = tqdm(
            val_loader,
            desc=f"Epoch [{epoch + 1}/{EPOCHS}] Val",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in pbar_val:
                imu, target_vel, _, label, _, _, _, _, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                mu, var, w, alphas, l_prior_val, _, _ = finetune_model(
                    imu, label, temperature=ROUTER_TEMP,
                )

                _, loss_task, _, _, _ = compute_finetune_total_loss(
                    mu, var, target_vel, l_prior_val, 0.0, dt=BIN_DT,
                    lambda_int=LAMBDA_INT,
                )

                val_analyzer.update(label, mu, target_vel, w, dt=BIN_DT)
                epoch_val_task_loss += loss_task.item()

        num_val_batches = max(1, len(val_loader))
        has_monitor = len(val_analyzer.label_drift_errors.get(MONITOR_LABEL, [])) > 0
        val_stats = val_analyzer.compute_epoch_stats(epoch + 1)
        val_analyzer.print_pretrain_report(
            val_stats, epoch_val_task_loss / num_val_batches, compact=True,
            line_prefix=f"Val E{epoch + 1}/{EPOCHS}", carriers=idx_to_category,
        )

        cur_monitor_drift = val_stats['drift_err_4'] if has_monitor else float('inf')

        if cur_monitor_drift < best_val_drift:
            best_val_drift = cur_monitor_drift
            best_path = os.path.join(checkpoint_dir, best_ckpt_name)
            torch.save(finetune_model.state_dict(), best_path)
            print(f"[BEST] {mon_name} drift {best_val_drift:.4f} m → {best_path}")

        periodic_every = int(train_cfg.get('periodic_ckpt_every', 5))
        if periodic_every > 0 and (epoch + 1) % periodic_every == 0:
            periodic_path = os.path.join(checkpoint_dir, f"weight_tune_epoch_{epoch + 1:04d}.pth")
            torch.save(finetune_model.state_dict(), periodic_path)
            print(f"[CKPT] every {periodic_every} ep → {periodic_path}")


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Weight tune | device: {device}")
    run_weight_tune(config, device)


if __name__ == "__main__":
    main()
