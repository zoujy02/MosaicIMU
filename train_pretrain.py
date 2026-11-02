import argparse
import os

import torch
import yaml
from tqdm import tqdm

from src.datasets.augmentation import apply_ultimate_augmentation_triton
from src.models.moe_net import CarrierConditionedMoE_Seq
from src.utils.loss import compute_moe_loss_l0
from src.utils.metrics import ExpertUsageAnalyzer
from src.utils.training_helpers import (
    auto_pretrain_balance_config,
    ensure_checkpoint_dir,
    imu_moe_dataloaders,
    imu_moe_datasets,
    load_model_checkpoint,
    resolve_loss_hyperparams,
    save_model_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MoE pretrain (4 native carriers)")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Load checkpoint and continue training from its epoch (or --resume_epoch)',
    )
    parser.add_argument(
        '--resume_epoch', type=int, default=None,
        help='1-based last-completed epoch for the resume checkpoint '
             '(overrides epoch stored in checkpoint; required for old state_dict-only files)',
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override pretrain_training.epochs (number of epochs in this run)',
    )
    return parser.parse_args()


def _load_weights(model, path, device):
    state_dict, meta = load_model_checkpoint(path, device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return meta, incompatible


def _make_optimizer(model, train_cfg, lr):
    wd = float(train_cfg.get('weight_decay', 0.0))
    lr_head = float(train_cfg.get('lr_head', lr))
    lr_router = float(train_cfg.get('lr_router', lr))
    head_params, router_params, backbone = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('decoder.head_mu') or name.startswith('decoder.head_var'):
            head_params.append(param)
        elif name.startswith('router'):
            router_params.append(param)
        else:
            backbone.append(param)
    groups = [
        {'params': backbone, 'lr': lr, 'weight_decay': wd},
        {'params': router_params, 'lr': lr_router, 'weight_decay': wd * 0.1},
        {'params': head_params, 'lr': lr_head, 'weight_decay': 0.0},
    ]
    groups = [g for g in groups if g['params']]
    return torch.optim.AdamW(groups)


def run_pretrain(config, device, *, resume_path=None, resume_epoch=None, epochs_override=None):
    train_cfg = config['pretrain_training']
    loss_hp = resolve_loss_hyperparams(config)

    BATCH_SIZE = train_cfg['batch_size']
    RUN_EPOCHS = int(epochs_override if epochs_override is not None else train_cfg['epochs'])
    LR = train_cfg['lr']
    ETA_MIN = train_cfg.get('eta_min', 1e-5)
    PATIENCE = train_cfg.get('patience', 15)
    STAGE_SWITCH_EPOCH = int(train_cfg.get('stage_switch_epoch', 50))
    ROUTER_TEMP = float(train_cfg.get('router_temperature', 4.0))
    SEQ_LEN = config['data']['window_size']
    BIN_DT = loss_hp['dt']
    GCLIP = loss_hp['grad_clip_norm']
    VAR_DETACH = loss_hp['variance_detach_epochs']
    ROUTER_LOG_EPS = loss_hp['router_log_eps']
    SOFT_PRIORS = loss_hp['carrier_soft_priors'].to(device)

    LAMBDA_CE = loss_hp['lambda_ce']
    ALPHA_ORTH = loss_hp['alpha_orth']
    LAMBDA_VEL = loss_hp['lambda_vel']
    LAMBDA_NLL = loss_hp['lambda_nll']
    LAMBDA_DISP = loss_hp['lambda_disp']

    DROPOUT_RATE = config['model']['dropout_rate']

    category_to_idx = {'car': 0, 'quadrupedal': 1, 'pedestrian': 2, 'drone': 3}
    current_repeats, VAL_CLASS_WEIGHTS, _raw_counts = auto_pretrain_balance_config(
        config['data']['train_dir'], SEQ_LEN, config['data']['step_size'], category_to_idx,
    )

    train_dataset, val_dataset = imu_moe_datasets(
        config['data']['train_dir'], config['data']['val_dir'], SEQ_LEN, config['data']['step_size'],
        category_to_idx, current_repeats, VAL_CLASS_WEIGHTS, print_style='moe', verbose=False,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("[ERROR] No data was loaded; please check the data paths.")
        return

    train_loader, val_loader = imu_moe_dataloaders(
        train_dataset, val_dataset, BATCH_SIZE, num_workers=8, seed_loaders=False,
    )

    model = CarrierConditionedMoE_Seq(
        in_channels=config['model']['in_channels'],
        context_dim=config['model']['context_dim'],
        expert_dim=config['model']['expert_dim'],
        num_experts=4, seq_len=SEQ_LEN,
        dropout_p=DROPOUT_RATE,
    ).to(device)

    resume_path = resume_path or train_cfg.get('resume_weights')
    init_path = train_cfg.get('init_weights')
    start_epoch = 0  # 0-based index of first epoch to run
    best_val_drift_ever = float('inf')
    best_val_drift_err_stage1 = float('inf')
    best_val_drift_err_stage2 = float('inf')

    if resume_path:
        if not os.path.isfile(resume_path):
            print(f"[ERROR] Resume checkpoint not found: {resume_path}")
            return
        try:
            meta, _inc = _load_weights(model, resume_path, device)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return
        except Exception as exc:
            print(f"[ERROR] Failed to load resume checkpoint: {exc}")
            return

        ckpt_epoch = meta.get('epoch', None)
        if resume_epoch is not None:
            last_completed = int(resume_epoch)
            epoch_src = "CLI --resume_epoch"
        elif ckpt_epoch is not None:
            last_completed = int(ckpt_epoch)
            epoch_src = "checkpoint"
        else:
            print(
                "[ERROR] Checkpoint has no 'epoch' field. "
                "Pass --resume_epoch N (1-based last completed epoch), e.g. --resume_epoch 35"
            )
            return

        if last_completed < 0:
            print(f"[ERROR] Invalid resume epoch: {last_completed}")
            return

        # last_completed is 1-based; continue from the next epoch (0-based index == last_completed)
        start_epoch = last_completed
        if meta.get('best_val_drift_ever') is not None:
            best_val_drift_ever = float(meta['best_val_drift_ever'])
        if meta.get('best_val_drift_err_stage1') is not None:
            best_val_drift_err_stage1 = float(meta['best_val_drift_err_stage1'])
        if meta.get('best_val_drift_err_stage2') is not None:
            best_val_drift_err_stage2 = float(meta['best_val_drift_err_stage2'])

        print(
            f"[RESUME] Loaded {resume_path} | last completed E{last_completed} ({epoch_src}) "
            f"→ continue from E{start_epoch + 1} | "
            f"stage_switch@E{STAGE_SWITCH_EPOCH + 1} var_detach<{VAR_DETACH}"
        )
    elif init_path:
        if not os.path.isfile(init_path):
            print(f"[ERROR] Init checkpoint not found: {init_path}")
            return
        try:
            meta, inc = _load_weights(model, init_path, device)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return
        print(
            f"[INIT] Loaded {init_path} | src_epoch={meta.get('epoch')} "
            f"missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)} | "
            f"fresh run from E1 (all params trainable)"
        )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable {n_train}/{n_all}")

    end_epoch = start_epoch + RUN_EPOCHS
    if start_epoch >= end_epoch:
        print(f"[ERROR] Nothing to run: start_epoch={start_epoch}, end_epoch={end_epoch}")
        return

    optimizer = _make_optimizer(model, train_cfg, LR)
    print(
        "[optim] AdamW "
        + " | ".join(
            f"group{i} lr={g['lr']:g} n={len(g['params'])} wd={g.get('weight_decay', 0):g}"
            for i, g in enumerate(optimizer.param_groups)
        )
    )
    cosine_t_max = int(train_cfg.get('cosine_t_max') or end_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cosine_t_max, 1), eta_min=ETA_MIN,
    )
    print(f"[optim] cosine T_max={max(cosine_t_max, 1)} eta_min={ETA_MIN:g}")
    for _ in range(start_epoch):
        scheduler.step()
    print(
        "[optim] lr for first epoch this run "
        + " | ".join(f"group{i}={g['lr']:.6e}" for i, g in enumerate(optimizer.param_groups))
        + f" | start E{start_epoch + 1}"
    )

    train_analyzer = ExpertUsageAnalyzer()
    val_analyzer = ExpertUsageAnalyzer()

    checkpoint_dir = ensure_checkpoint_dir()
    best_name = train_cfg.get('ckpt_best', 'best_moe.pth')
    last_name = train_cfg.get('ckpt_last', 'last_moe.pth')
    best_model_path = best_name if os.path.isabs(best_name) else os.path.join(checkpoint_dir, os.path.basename(best_name))
    last_model_path = last_name if os.path.isabs(last_name) else os.path.join(checkpoint_dir, os.path.basename(last_name))

    patience_counter = 0

    for epoch in range(start_epoch, end_epoch):
        model.train()
        in_stage2 = epoch >= STAGE_SWITCH_EPOCH

        epoch_vel_loss, epoch_nll_loss, epoch_aux_loss, epoch_orth_loss, epoch_disp_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        if epoch == STAGE_SWITCH_EPOCH:
            print(f"[stage] epoch {STAGE_SWITCH_EPOCH + 1}: enable displacement loss | patience reset")
            patience_counter = 0

        pbar_train = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{end_epoch}] Train",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (imu, target_vel_seq, target_pos, label, raw_imu, init_q, yaw0, pos_start, quat_seq) in enumerate(pbar_train):
            imu = imu.to(device)
            target_vel_seq = target_vel_seq.to(device)
            target_pos = target_pos.to(device)
            label = label.to(device)
            quat_seq = quat_seq.to(device)

            imu_aug, target_vel_seq_aug, _, _ = apply_ultimate_augmentation_triton(
                imu, target_vel_seq, target_pos, quat_seq
            )

            optimizer.zero_grad()
            pred_mu_seq, pred_var_seq, weights, orth_loss = model(imu_aug, temperature=ROUTER_TEMP)

            loss, l_vel, l_nll, l_aux, l_disp, loss_type = compute_moe_loss_l0(
                pred_mu_seq, pred_var_seq, weights, orth_loss,
                target_vel_seq_aug, label, epoch=epoch,
                lambda_ce=LAMBDA_CE, alpha_orth=ALPHA_ORTH,
                lambda_vel=LAMBDA_VEL, lambda_nll=LAMBDA_NLL, lambda_disp=LAMBDA_DISP,
                dt=BIN_DT,
                stage_switch_epoch=STAGE_SWITCH_EPOCH,
                variance_detach_epochs=VAR_DETACH,
                router_log_eps=ROUTER_LOG_EPS,
                soft_priors=SOFT_PRIORS,
            )

            train_analyzer.update(label, pred_mu_seq, target_vel_seq_aug, weights, dt=BIN_DT)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GCLIP)
            optimizer.step()

            epoch_vel_loss += l_vel.item()
            epoch_nll_loss += l_nll.item()
            epoch_aux_loss += l_aux.item()
            epoch_orth_loss += orth_loss.item()
            epoch_disp_loss += l_disp.item()

            pbar_train.set_postfix({
                'L_Vel': f"{l_vel.item():.4f}",
                'L_NLL': f"{l_nll.item():.4f}",
                'L_Disp': f"{l_disp.item():.4f}" if in_stage2 else "N/A",
            })

        scheduler.step()

        num_batches = len(train_loader)
        avg_stats = {
            'task': (epoch_vel_loss + epoch_nll_loss + epoch_disp_loss) / num_batches,
            'aux': epoch_aux_loss / num_batches,
            'orth': epoch_orth_loss / num_batches,
        }
        train_stats = train_analyzer.compute_epoch_stats(epoch + 1, class_weights=VAL_CLASS_WEIGHTS)

        train_analyzer.print_pretrain_report(
            train_stats, avg_stats['task'], compact=True,
            line_prefix=f"Train E{epoch + 1}/{end_epoch}",
        )

        model.eval()
        val_vel_loss, val_nll_loss, val_aux_loss, val_orth_loss, val_disp_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        pbar_val = tqdm(
            val_loader,
            desc=f"Epoch [{epoch + 1}/{end_epoch}] Val",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch_idx, (imu, target_vel_seq, target_pos, label, raw_imu, init_q, yaw0, pos_start, quat_seq) in enumerate(pbar_val):
                imu = imu.to(device)
                target_vel_seq = target_vel_seq.to(device)
                label = label.to(device)

                pred_mu_seq, pred_var_seq, weights, orth_loss = model(imu, temperature=ROUTER_TEMP)

                _loss, l_vel, l_nll, l_aux, l_disp, _ = compute_moe_loss_l0(
                    pred_mu_seq, pred_var_seq, weights, orth_loss,
                    target_vel_seq, label, epoch=epoch,
                    lambda_ce=LAMBDA_CE, alpha_orth=ALPHA_ORTH,
                    lambda_vel=LAMBDA_VEL, lambda_nll=LAMBDA_NLL, lambda_disp=LAMBDA_DISP,
                    dt=BIN_DT,
                    stage_switch_epoch=STAGE_SWITCH_EPOCH,
                    variance_detach_epochs=VAR_DETACH,
                    router_log_eps=ROUTER_LOG_EPS,
                    soft_priors=SOFT_PRIORS,
                )

                val_analyzer.update(label, pred_mu_seq, target_vel_seq, weights, dt=BIN_DT)

                val_vel_loss += l_vel.item()
                val_nll_loss += l_nll.item()
                val_aux_loss += l_aux.item()
                val_orth_loss += orth_loss.item()
                val_disp_loss += l_disp.item()

        num_val_batches = len(val_loader)
        val_avg_stats = {
            'task': (val_vel_loss + val_nll_loss + val_disp_loss) / num_val_batches,
            'aux': val_aux_loss / num_val_batches,
            'orth': val_orth_loss / num_val_batches,
        }
        val_stats = val_analyzer.compute_epoch_stats(epoch + 1, class_weights=VAL_CLASS_WEIGHTS)

        val_analyzer.print_pretrain_report(
            val_stats, val_avg_stats['task'], compact=True,
            line_prefix=f"Val E{epoch + 1}/{end_epoch}",
            show_router_weights=True,
        )

        current_val_drift_err = val_stats['weighted_drift_err']

        if not in_stage2:
            if current_val_drift_err < best_val_drift_err_stage1:
                best_val_drift_err_stage1 = current_val_drift_err
                patience_counter = 0
                print(f"  [S1] best drift {best_val_drift_err_stage1:.4f} m")
            else:
                patience_counter += 1
        else:
            if current_val_drift_err < best_val_drift_err_stage2:
                best_val_drift_err_stage2 = current_val_drift_err
                patience_counter = 0
                print(f"  [S2] best drift {best_val_drift_err_stage2:.4f} m")
            else:
                patience_counter += 1

        completed_epoch = epoch + 1  # 1-based
        if current_val_drift_err < best_val_drift_ever:
            best_val_drift_ever = current_val_drift_err
            save_model_checkpoint(
                best_model_path, model, epoch=completed_epoch,
                best_val_drift_ever=best_val_drift_ever,
                best_val_drift_err_stage1=best_val_drift_err_stage1,
                best_val_drift_err_stage2=best_val_drift_err_stage2,
                val_drift=current_val_drift_err,
            )
            print(f"  [BEST] saved {best_model_path} | drift {best_val_drift_ever:.4f} m | E{completed_epoch}")

        save_model_checkpoint(
            last_model_path, model, epoch=completed_epoch,
            best_val_drift_ever=best_val_drift_ever,
            best_val_drift_err_stage1=best_val_drift_err_stage1,
            best_val_drift_err_stage2=best_val_drift_err_stage2,
            val_drift=current_val_drift_err,
        )

        if patience_counter >= PATIENCE:
            print(f"[EARLY STOP] No improvement for {PATIENCE} consecutive epochs.")
            break

    print(
        f"Done | best_ckpt {best_model_path} | best drift {best_val_drift_ever:.4f} m | "
        f"S1 best {best_val_drift_err_stage1:.4f} m | S2 best {best_val_drift_err_stage2:.4f} m",
    )


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pretrain | device: {device}")
    # loss weights are resolved inside run_pretrain; echo from config for visibility
    loss_cfg = (config.get('loss') or {})
    train_cfg = (config.get('pretrain_training') or {})
    print(
        "[loss] "
        f"λ_vel={loss_cfg.get('lambda_vel', 1.0)} "
        f"λ_nll={loss_cfg.get('lambda_nll', 1.0)} "
        f"λ_disp={loss_cfg.get('lambda_disp', 10.0)} "
        f"λ_ce={loss_cfg.get('lambda_ce', 0.05)} "
        f"α_orth={loss_cfg.get('alpha_orth', 0.1)}"
    )
    print(f"[train] router_temperature={train_cfg.get('router_temperature', 4.0)}")
    if train_cfg.get('init_weights'):
        print(f"[train] init_weights={train_cfg.get('init_weights')} lr={train_cfg.get('lr')} lr_head={train_cfg.get('lr_head')}")
    run_pretrain(
        config, device,
        resume_path=args.resume,
        resume_epoch=args.resume_epoch,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
