import os
import glob
import yaml
import numpy as np
import torch
import argparse
from scipy.spatial.transform import Rotation

from src.models.moe_net import CarrierConditionedMoE_Seq
from src.models.finetune_net import WeightTuneMoEFineTuner
from src.filter.msckf import ImuMSCKF, FilterConfig, CARRIER_FILTER_PRESETS
from src.utils.metrics import compute_metrics
from src.utils.eval_aggregate import (
    gt_path_length_3d_m,
    group_metrics_for_summary,
    format_path_weighted_summary_line,
    weighted_mean_by_gt_path,
)
from src.utils.visualizer import (
    plot_trajectory_3d,
    print_evaluation_metrics,
)
from src.utils.training_helpers import load_model_checkpoint

PRETRAIN_CARRIER_TO_ROUTER_EXPERT = {
    'car': 0,
    'quadrupedal': 1,
    'pedestrian': 2,
    'drone': 3,
}
ROUTER_EXPERT_TO_CARRIER = {v: k for k, v in PRETRAIN_CARRIER_TO_ROUTER_EXPERT.items()}
NATIVE_CARRIERS = ('car', 'quadrupedal', 'pedestrian', 'drone')

CARRIER_EVAL_DEFAULTS = {
    'weight_robot': {'burn_in_sec': 0.0},
}

EVAL_BASE_DEFAULTS = {
    'pretrain_eval_router_temperature': 4.0,
    'weight_tune_eval_router_temperature': 4.0,
    'msckf_update_period_sec': 0.5,
    'msckf_history_window_sec': 2.5,
    'burn_in_sec': 10.0,
    'carrier_source': 'dataset',
}


def _base_cfg_get(base_cfg: dict, key: str):
    return base_cfg.get(key, EVAL_BASE_DEFAULTS[key])

_FILTER_CONFIG_KEYS = (
    'g_norm', 'sigma_na', 'sigma_ng', 'ita_ba', 'ita_bg',
    'init_attitude_sigma', 'init_yaw_sigma', 'init_vel_sigma',
    'init_pos_sigma', 'init_bg_sigma', 'init_ba_sigma',
)


def _build_filter_kwargs(carrier_name: str, carrier_cfg: dict) -> dict:
    kwargs = dict(CARRIER_FILTER_PRESETS.get(carrier_name, {}))
    kwargs.update({k: carrier_cfg[k] for k in _FILTER_CONFIG_KEYS if k in carrier_cfg})
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Evaluator")
    parser.add_argument('--config', type=str, default='configs/test_config.yaml')
    parser.add_argument(
        '--carrier', nargs='+', default=['all'],
        help=(
            "Carrier name(s): 'all' (mode_carrier_groups[mode]) | one name | "
            "multiple names space- or comma-separated (e.g. 'car pedestrian' or 'car,drone'). "
            "In --carrier-source router mode this filters which detected carriers to keep."
        ),
    )
    parser.add_argument('--mode', type=str,
                        choices=['pretrain', 'weight_tune'],
                        default='pretrain')
    parser.add_argument(
        '--carrier-source',
        type=str,
        default='yaml',
        choices=['yaml', 'dataset', 'router'],
        help=(
            "How to obtain carrier identity. "
            "'dataset': only read folders named after the 4 native carriers; "
            "'router': ignore folder names, classify each trajectory with the MoE router "
            "then apply that carrier's YAML config; "
            "'yaml' reads base.carrier_source (default dataset)."
        ),
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help=(
            "Test data root. dataset: <root>/<carrier>/*.npz only; "
            "router: recursive **/*.npz under root (nested dirs included). "
            "Default: base.test_data_root, else parent of native carrier data_dir."
        ),
    )
    parser.add_argument(
        '--weights', type=str, default=None,
        help='Override checkpoint path (pretrain: base.weights; weight_tune: base.weight_tune_weights)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override base.output_dir for this run',
    )
    return parser.parse_args()


def _resolve_msckf_timing(carrier_cfg: dict, base_cfg: dict):
    dt = float(base_cfg['dt'])
    if 'clone_interval' in carrier_cfg and 'window_size' in carrier_cfg:
        return int(carrier_cfg['clone_interval']), int(carrier_cfg['window_size'])

    update_sec = float(carrier_cfg.get(
        'msckf_update_period_sec',
        _base_cfg_get(base_cfg, 'msckf_update_period_sec'),
    ))
    history_sec = float(carrier_cfg.get(
        'msckf_history_window_sec',
        _base_cfg_get(base_cfg, 'msckf_history_window_sec'),
    ))
    clone_interval = max(1, round(update_sec / dt))
    window_size = max(1, round(history_sec / update_sec))
    return clone_interval, window_size


def _resolve_carrier_source(cli_value: str, base_cfg: dict) -> str:
    if cli_value == 'yaml':
        src = str(_base_cfg_get(base_cfg, 'carrier_source')).lower()
        if src not in ('dataset', 'router'):
            print(f"[WARN] Invalid carrier_source '{src}', falling back to 'dataset'.")
            return 'dataset'
        return src
    return cli_value


def _infer_test_data_root(base_cfg: dict, carriers_cfg: dict, cli_data_root):
    if cli_data_root:
        return os.path.normpath(cli_data_root)
    if base_cfg.get('test_data_root'):
        return os.path.normpath(base_cfg['test_data_root'])
    for name in NATIVE_CARRIERS:
        cfg = carriers_cfg.get(name) or {}
        data_dir = cfg.get('data_dir')
        if data_dir:
            return os.path.dirname(os.path.normpath(data_dir))
    return None


def _carrier_cfg_without_data_dir(carrier_cfg: dict) -> dict:
    cfg = dict(carrier_cfg)
    cfg.pop('data_dir', None)
    return cfg


def _collect_dataset_npz(carrier_name: str, carrier_cfg: dict, test_data_root):
    """Dataset-given carrier: only <carrier_folder>/*.npz (non-recursive).

    Native carriers must live in a folder whose basename is exactly the carrier name.
    Loose npz at the test root and differently named sibling folders are ignored.
    """
    if carrier_name in NATIVE_CARRIERS:
        if test_data_root:
            folder = os.path.join(test_data_root, carrier_name)
        else:
            folder = carrier_cfg.get('data_dir')
            if not folder:
                print(f"  [WARN] [{carrier_name}] no data_dir / test_data_root; skip")
                return []
            folder = os.path.normpath(folder)
            if os.path.basename(folder) != carrier_name:
                print(
                    f"  [WARN] [{carrier_name}] data_dir basename is "
                    f"'{os.path.basename(folder)}' (expected '{carrier_name}'); skip",
                )
                return []
        if not os.path.isdir(folder):
            print(f"  [WARN] [{carrier_name}] folder missing, skip: {folder}")
            return []
        files = sorted(glob.glob(os.path.join(folder, '*.npz')))
        print(f"  [dataset] [{carrier_name}] {len(files)} npz in {folder}")
        return files

    folder = carrier_cfg.get('data_dir')
    if not folder or not os.path.isdir(folder):
        print(f"  [WARN] [{carrier_name}] data_dir missing, skip: {folder}")
        return []
    files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    print(f"  [dataset] [{carrier_name}] {len(files)} npz in {folder}")
    return files


def _collect_router_npz(test_data_root: str):
    """Router mode: recursive **/*.npz under root (nested sequences included)."""
    if not test_data_root or not os.path.isdir(test_data_root):
        print(f"  [WARN] test data root missing: {test_data_root}")
        return []
    files = sorted(glob.glob(os.path.join(test_data_root, '**', '*.npz'), recursive=True))
    print(f"  [router] scanned {len(files)} npz under {test_data_root} (recursive)")
    return files


def _npz_output_stem(npz_path: str, data_root) -> str:
    if data_root:
        try:
            rel = os.path.relpath(npz_path, data_root)
        except ValueError:
            rel = os.path.basename(npz_path)
    else:
        rel = os.path.basename(npz_path)
    stem = os.path.splitext(rel)[0]
    return stem.replace('\\', '/').replace('/', '__')


def _load_npz_traj(npz_path: str):
    data = np.load(npz_path)
    if 'imu' in data.files:
        return data['ts'], data['imu'], data['pos'], data['quat']
    if 'retargetted_imu' in data.files:
        return (
            data['retargetted_ts'], data['retargetted_imu'],
            data['retargetted_pos'], data['retargetted_quat'],
        )
    raise KeyError(f"missing imu/pos/quat keys in {npz_path}: {list(data.files)}")


def _prepare_net_input(imu_window, R_est_window, device):
    current_imu_seq = np.asarray(imu_window)
    current_R_seq = np.asarray(R_est_window)
    R_W_to_L0_start = Rotation.from_euler(
        'Z', -Rotation.from_matrix(current_R_seq[0]).as_euler('ZYX')[0], degrees=False,
    ).as_matrix()
    acc_L0_list = [
        (R_W_to_L0_start @ R_k @ current_imu_seq[k, 0:3].reshape(3, 1)).flatten()
        for k, R_k in enumerate(current_R_seq)
    ]
    gyro_L0_list = [
        (R_W_to_L0_start @ R_k @ current_imu_seq[k, 3:6].reshape(3, 1)).flatten()
        for k, R_k in enumerate(current_R_seq)
    ]
    net_in = (
        torch.FloatTensor(np.concatenate([acc_L0_list, gyro_L0_list], axis=-1))
        .unsqueeze(0).transpose(1, 2).to(device)
    )
    return net_in, R_W_to_L0_start


def detect_carrier_by_router(model, ts, imu, base_cfg, device, detect_carrier_cfg=None):
    """Propagate once (no velocity updates), average router weights, return (carrier, mean_w, expert_idx)."""
    SEQ_LEN = int(base_cfg['seq_len'])
    detect_cfg = detect_carrier_cfg or {}
    clone_interval, _ = _resolve_msckf_timing(detect_cfg, base_cfg)
    filter_kwargs = _build_filter_kwargs('car', detect_cfg)
    filt = ImuMSCKF(config=FilterConfig(**filter_kwargs))
    filt.initialize_with_state(
        int(ts[0] * 1e6),
        np.eye(3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
    )

    router_temp = float(_base_cfg_get(base_cfg, 'pretrain_eval_router_temperature'))
    imu_window_buf, R_est_window_buf = [], []
    weights = []
    num_samples = len(ts)

    for i in range(1, num_samples):
        t_us = int(ts[i] * 1e6)
        is_clone_frame = (i % clone_interval == 0)
        filt.propagate(imu[i, 0:3], imu[i, 3:6], t_us, t_augmentation_us=None)

        imu_window_buf.append(imu[i, 0:6])
        R_est_window_buf.append(filt.state.s_R.copy())
        if len(imu_window_buf) > SEQ_LEN:
            imu_window_buf.pop(0)
            R_est_window_buf.pop(0)

        if is_clone_frame and len(imu_window_buf) == SEQ_LEN:
            net_in, _ = _prepare_net_input(imu_window_buf, R_est_window_buf, device)
            with torch.no_grad():
                _mu, _var, pred_w, _orth = model(
                    net_in, temperature=router_temp, use_gumbel=False,
                )
            weights.append(pred_w[0].detach().cpu().numpy())

    if not weights and len(imu_window_buf) == SEQ_LEN:
        net_in, _ = _prepare_net_input(imu_window_buf, R_est_window_buf, device)
        with torch.no_grad():
            _mu, _var, pred_w, _orth = model(
                net_in, temperature=router_temp, use_gumbel=False,
            )
        weights.append(pred_w[0].detach().cpu().numpy())

    if not weights:
        raise RuntimeError(
            f"Router detection produced no windows "
            f"(N={num_samples}, SEQ_LEN={SEQ_LEN}, clone_interval={clone_interval})",
        )

    mean_w = np.mean(np.stack(weights, axis=0), axis=0)
    expert_idx = int(np.argmax(mean_w))
    carrier = ROUTER_EXPERT_TO_CARRIER[expert_idx]
    return carrier, mean_w, expert_idx


def _load_eval_model(mode, weights_path, seq_len, device):
    peeked_sd = None
    if os.path.exists(weights_path):
        peeked_sd, _meta = load_model_checkpoint(weights_path, device)

    base_model = CarrierConditionedMoE_Seq(num_experts=4, seq_len=seq_len).to(device)
    if mode == 'pretrain':
        model = base_model
    else:
        model = WeightTuneMoEFineTuner(base_model).to(device)

    if peeked_sd is not None:
        incompatible = model.load_state_dict(peeked_sd, strict=False)
        print(f"  [OK] Model weights loaded: {weights_path}")
        if incompatible.missing_keys:
            print(f"  [load] missing={incompatible.missing_keys[:8]}")
    else:
        print(f"  [WARN] Model weights not found: {weights_path}")
    model.eval()
    return model


def evaluate_single_carrier(
    carrier_name, carrier_cfg, base_cfg, device, mode,
    weights_path_override=None,
    *,
    npz_files=None,
    model=None,
    output_name_root=None,
):
    print("=" * 80)
    print(f"Evaluating carrier: [{carrier_name.upper()}] | mode: {mode.upper()}")
    if npz_files is None:
        print(f"Data path: {carrier_cfg.get('data_dir', '(from test_data_root)')}")
    else:
        print(f"Files: {len(npz_files)} (explicit list)")
    print("=" * 80)

    file_metrics = []

    SEQ_LEN, DT = base_cfg['seq_len'], base_cfg['dt']
    CLONE_INTERVAL, WINDOW_SIZE = _resolve_msckf_timing(carrier_cfg, base_cfg)
    carrier_eval_defaults = CARRIER_EVAL_DEFAULTS.get(carrier_name, {})
    BURN_IN_SEC = carrier_cfg.get(
        'burn_in_sec',
        carrier_eval_defaults.get('burn_in_sec', _base_cfg_get(base_cfg, 'burn_in_sec')),
    )

    START_FRAME = carrier_cfg.get('start_frame', 0)
    END_FRAME = carrier_cfg.get('end_frame', None)

    CALIB_SEC = carrier_cfg.get('calib_sec', 0.0)
    COV_SCALE_XY = carrier_cfg.get('cov_scale_xy', base_cfg.get('cov_scale_xy', 500.0))
    COV_SCALE_Z_MUL = carrier_cfg.get('cov_scale_z_mul', 0.1)

    output_dir = os.path.join(base_cfg['output_dir'], f"{mode}_{carrier_name}")
    os.makedirs(output_dir, exist_ok=True)

    if model is None:
        default_weights_path = base_cfg['weights'] if mode == 'pretrain' else base_cfg['weight_tune_weights']
        weights_path = (
            weights_path_override if weights_path_override is not None else default_weights_path
        )
        model = _load_eval_model(mode, weights_path, SEQ_LEN, device)

    if npz_files is None:
        test_root = base_cfg.get('_resolved_test_data_root')
        npz_files = _collect_dataset_npz(carrier_name, carrier_cfg, test_root)

    for file_idx, npz_file in enumerate(npz_files):
        base_name = _npz_output_stem(npz_file, output_name_root)
        try:
            raw_ts, raw_imu, raw_pos, raw_quat = _load_npz_traj(npz_file)
        except KeyError as exc:
            print(f"  [WARN] skip {npz_file}: {exc}")
            continue

        if CALIB_SEC > 0:
            stationary_frames = int(CALIB_SEC / DT)
            if len(raw_imu) > stationary_frames:
                gyro_bias_estimated = np.mean(raw_imu[:stationary_frames, 3:6], axis=0)
                print(f"  [CALIB] {base_name} initial {CALIB_SEC}s gyro bias: {gyro_bias_estimated}")
                raw_imu = raw_imu.copy()
                raw_imu[:, 3:6] -= gyro_bias_estimated
            else:
                print(f"  [WARN] Total length is shorter than the requested calibration window; skipping static bias init.")

        actual_start = START_FRAME if START_FRAME < len(raw_ts) else 0
        actual_end = END_FRAME if (END_FRAME is not None and END_FRAME <= len(raw_ts)) else len(raw_ts)
        if actual_end - actual_start < SEQ_LEN:
            continue

        ts = raw_ts[actual_start:actual_end]
        imu = raw_imu[actual_start:actual_end]
        pos = raw_pos[actual_start:actual_end]
        quat = raw_quat[actual_start:actual_end]
        num_samples = len(ts)

        filter_kwargs = _build_filter_kwargs(carrier_name, carrier_cfg)
        filter_cfg = FilterConfig(**filter_kwargs)
        filter = ImuMSCKF(config=filter_cfg)
        filter.initialize_with_state(
            int(ts[0] * 1e6), Rotation.from_quat(quat[0]).as_matrix(),
            ((pos[1] - pos[0]) / (ts[1] - ts[0])).reshape(3, 1), pos[0].reshape(3, 1),
            np.zeros((3, 1)), np.zeros((3, 1)),
        )

        imu_window_buf, R_est_window_buf = [], []
        est_positions = []
        net_vel_world_accum = np.zeros((num_samples, 3))
        net_vel_counts = np.zeros((num_samples, 1))

        for i in range(1, num_samples):
            t_us = int(ts[i] * 1e6)
            is_clone_frame = (i % CLONE_INTERVAL == 0)
            filter.propagate(imu[i, 0:3], imu[i, 3:6], t_us, t_augmentation_us=t_us if is_clone_frame else None)

            imu_window_buf.append(imu[i, 0:6])
            R_est_window_buf.append(filter.state.s_R.copy())
            if len(imu_window_buf) > SEQ_LEN:
                imu_window_buf.pop(0)
                R_est_window_buf.pop(0)

            if is_clone_frame and len(imu_window_buf) == SEQ_LEN:
                net_in, R_W_to_L0_start = _prepare_net_input(imu_window_buf, R_est_window_buf, device)

                with torch.no_grad():
                    if mode == 'pretrain':
                        _router_temp = float(_base_cfg_get(base_cfg, 'pretrain_eval_router_temperature'))
                        pred_mu_seq, pred_var_seq, _pred_w, _orth = model(
                            net_in, temperature=_router_temp, use_gumbel=False,
                        )
                    else:
                        _router_temp = float(_base_cfg_get(base_cfg, 'weight_tune_eval_router_temperature'))
                        pred_mu_seq, pred_var_seq, _pred_weights, _alphas, _, _, _ = model(
                            net_in, temperature=_router_temp,
                        )

                pred_mu_L0, pred_var_L0 = pred_mu_seq[0].cpu().numpy(), pred_var_seq[0].cpu().numpy()
                v_avg_L0_net = np.sum(pred_mu_L0 * DT, axis=0) / (SEQ_LEN * DT)
                net_vel_world_accum[i - SEQ_LEN + 1:i + 1] += (pred_mu_L0 @ R_W_to_L0_start)
                net_vel_counts[i - SEQ_LEN + 1:i + 1] += 1

                if filter.state.N > WINDOW_SIZE:
                    MEAS_COV_V = np.diag(np.sum(pred_var_L0, axis=0) / (SEQ_LEN ** 2)) * COV_SCALE_XY
                    MEAS_COV_V[2, 2] = MEAS_COV_V[2, 2] * COV_SCALE_Z_MUL

                    filter.update_velocity_with_history(
                        v_avg_L0_net, MEAS_COV_V,
                        filter.state.si_timestamps_us[-WINDOW_SIZE - 1], t_us,
                    )

            if is_clone_frame and filter.state.N > WINDOW_SIZE:
                filter.marginalize(0)
            est_positions.append(filter.state.s_p.flatten().copy())

        gt_positions, est_positions = pos[1:num_samples], np.array(est_positions)
        time_axis = ts[1:num_samples] - ts[0]

        valid_mask_full = net_vel_counts[:, 0] > 0
        net_vel_avg_full = np.zeros((num_samples, 3))
        net_vel_avg_full[valid_mask_full] = net_vel_world_accum[valid_mask_full] / net_vel_counts[valid_mask_full]
        net_vel_final = net_vel_avg_full[1:num_samples]

        net_pos_integrated = np.zeros_like(gt_positions)
        net_pos_integrated[0] = gt_positions[0]
        for k in range(1, len(net_pos_integrated)):
            net_pos_integrated[k] = net_pos_integrated[k - 1] + net_vel_final[k] * DT

        valid_idx = time_axis >= BURN_IN_SEC
        if np.any(valid_idx):
            gt_positions = gt_positions[valid_idx]
            raw_est, raw_net = est_positions[valid_idx], net_pos_integrated[valid_idx]
            est_positions = raw_est + (gt_positions[0] - raw_est[0])
            net_pos_integrated = raw_net + (gt_positions[0] - raw_net[0])

        dt_hz = 1.0 / DT
        ekf_ate_3d, ekf_rte_3d = compute_metrics(gt_positions, est_positions, dt_hz=dt_hz, rte_sec=10.0)
        net_ate_3d, net_rte_3d = compute_metrics(gt_positions, net_pos_integrated, dt_hz=dt_hz, rte_sec=10.0)
        ekf_ate_2d, ekf_rte_2d = compute_metrics(gt_positions[:, :2], est_positions[:, :2], dt_hz=dt_hz, rte_sec=10.0)
        net_ate_2d, net_rte_2d = compute_metrics(gt_positions[:, :2], net_pos_integrated[:, :2], dt_hz=dt_hz, rte_sec=10.0)

        out_base_name = base_name

        gt_path_m = gt_path_length_3d_m(gt_positions)

        file_metrics.append({
            "carrier": carrier_name,
            "ckpt": '-',
            "file": out_base_name,
            "gt_path_m": float(gt_path_m),
            "ekf_ate_3d": float(ekf_ate_3d), "ekf_rte_3d": float(ekf_rte_3d),
            "net_ate_3d": float(net_ate_3d), "net_rte_3d": float(net_rte_3d),
            "ekf_ate_2d": float(ekf_ate_2d), "ekf_rte_2d": float(ekf_rte_2d),
            "net_ate_2d": float(net_ate_2d), "net_rte_2d": float(net_rte_2d),
        })

        print_evaluation_metrics(
            gt_positions, ekf_ate_3d, ekf_rte_3d, net_ate_3d, net_rte_3d,
            ekf_ate_2d, ekf_rte_2d, net_ate_2d, net_rte_2d,
            DT, output_dir, out_base_name, actual_start, actual_end,
        )
        plot_trajectory_3d(
            gt_positions, est_positions, net_pos_integrated,
            output_dir, out_base_name, actual_start, actual_end,
        )

    return file_metrics


def evaluate_router_discovered(
    allowed_carriers, carriers_cfg, base_cfg, device, mode,
    *,
    test_data_root,
    weights_path_override=None,
):
    """Ignore folder names: classify each trajectory with the router, then use that carrier's config."""
    if mode != 'pretrain':
        print("[ERROR] --carrier-source router is only supported for --mode pretrain")
        return []

    npz_files = _collect_router_npz(test_data_root)
    if not npz_files:
        return []

    SEQ_LEN = int(base_cfg['seq_len'])
    default_weights = base_cfg['weights']
    weights_path = weights_path_override if weights_path_override is not None else default_weights
    model = _load_eval_model(mode, weights_path, SEQ_LEN, device)

    detect_cfg = _carrier_cfg_without_data_dir(carriers_cfg.get('car', {}))
    allowed = set(allowed_carriers)
    grouped = {c: [] for c in NATIVE_CARRIERS}
    detect_log = []

    print("=" * 80)
    print(f"Router carrier discovery | root={test_data_root} | files={len(npz_files)}")
    print("=" * 80)

    for npz_path in npz_files:
        stem = _npz_output_stem(npz_path, test_data_root)
        try:
            ts, imu, _pos, _quat = _load_npz_traj(npz_path)
        except KeyError as exc:
            print(f"  [WARN] skip {stem}: {exc}")
            continue
        if len(ts) < SEQ_LEN + 1:
            print(f"  [SKIP] {stem}: too short N={len(ts)}")
            continue
        try:
            carrier, mean_w, expert_idx = detect_carrier_by_router(
                model, ts, imu, base_cfg, device, detect_carrier_cfg=detect_cfg,
            )
        except RuntimeError as exc:
            print(f"  [SKIP] {stem}: {exc}")
            continue

        w_str = ', '.join(f"{NATIVE_CARRIERS[i]}={mean_w[i]:.3f}" for i in range(4))
        print(f"  [detect] {stem} -> {carrier} (expert={expert_idx}) | [{w_str}]")
        detect_log.append(f"{stem}\t{carrier}\texpert={expert_idx}\tmean_w={mean_w.tolist()}")

        if carrier not in allowed:
            print(f"  [SKIP] detected '{carrier}' not in requested carriers {sorted(allowed)}")
            continue
        if carrier not in carriers_cfg:
            print(f"  [SKIP] detected '{carrier}' missing from YAML carriers")
            continue
        grouped[carrier].append(npz_path)

    summary_path = os.path.join(base_cfg['output_dir'], 'router_detect_summary.txt')
    os.makedirs(base_cfg['output_dir'], exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(detect_log) + ('\n' if detect_log else ''))
    print(f"  [OK] router detect summary: {summary_path}")

    all_metrics = []
    for carrier in NATIVE_CARRIERS:
        files = grouped.get(carrier) or []
        if not files:
            continue
        carrier_cfg = _carrier_cfg_without_data_dir(carriers_cfg[carrier])
        metrics = evaluate_single_carrier(
            carrier,
            carrier_cfg,
            base_cfg,
            device,
            mode,
            weights_path_override=weights_path,
            npz_files=files,
            model=model,
            output_name_root=test_data_root,
        )
        if metrics:
            all_metrics.extend(metrics)
    return all_metrics


def _resolve_carriers(arg_carrier, mode: str, config: dict):
    tokens = [arg_carrier] if isinstance(arg_carrier, str) else list(arg_carrier)
    tokens = [t.strip() for t in tokens if t and str(t).strip()]
    all_carriers = list(config['carriers'].keys())

    if len(tokens) == 1 and tokens[0] == 'all':
        groups = config.get('mode_carrier_groups', {})
        carriers = groups.get(mode, all_carriers)
    else:
        carriers = []
        for token in tokens:
            if ',' in token:
                carriers.extend(c.strip() for c in token.split(',') if c.strip())
            else:
                carriers.append(token)

    valid, invalid = [], []
    for c in carriers:
        (valid if c in config['carriers'] else invalid).append(c)
    if invalid:
        print(f"[WARN] The following carriers are not in the YAML config and were ignored: {invalid}")
    return valid


def _print_summary(all_metrics, mode: str):
    if not all_metrics:
        print("\n[WARN] No metrics produced; skipping summary.")
        return

    ckpt_any = any(m.get('ckpt', '-') != '-' for m in all_metrics)

    print("\n" + "=" * 122)
    print(f"All-platform test summary | mode: {mode.upper()} | total {len(all_metrics)} entries")
    print("=" * 122)
    if ckpt_any:
        header = (
            f"{'Carrier':<12}{'Ckpt':<18}{'File':<40}"
            f"{'EKF ATE 3D':>12}{'EKF RTE 3D':>12}{'NET ATE 3D':>12}{'NET RTE 3D':>12}"
        )
        print(header)
        print("-" * 122)
        for m in all_metrics:
            ck = m.get('ckpt', '-')[:16]
            fname = (m['file'][:37] + '...') if len(m['file']) > 40 else m['file']
            print(
                f"{m['carrier']:<12}{ck:<18}{fname:<40}"
                f"{m['ekf_ate_3d']:>12.4f}{m['ekf_rte_3d']:>12.4f}"
                f"{m['net_ate_3d']:>12.4f}{m['net_rte_3d']:>12.4f}",
            )
    else:
        header = f"{'Carrier':<14}{'File':<48}{'EKF ATE 3D':>12}{'EKF RTE 3D':>12}{'NET ATE 3D':>12}{'NET RTE 3D':>12}"
        print(header)
        print("-" * 122)
        for m in all_metrics:
            fname = (m['file'][:45] + '...') if len(m['file']) > 48 else m['file']
            print(
                f"{m['carrier']:<14}{fname:<48}"
                f"{m['ekf_ate_3d']:>12.4f}{m['ekf_rte_3d']:>12.4f}"
                f"{m['net_ate_3d']:>12.4f}{m['net_rte_3d']:>12.4f}",
            )

    print("-" * 122)
    print(
        "Per-carrier summary (GT path-length weighted): "
        "m_bar = sum(w_i * m_i) / sum(w_i), w_i = burn-in GT 3D arc length (m)"
    )
    print("-" * 122)
    for cname, ck, ms in group_metrics_for_summary(all_metrics):
        avgs, total_path_m, n_seg = weighted_mean_by_gt_path(ms)
        if avgs is None:
            print(f"  [{cname}] ckpt={ck}: no valid GT path weights (skipped)")
            continue
        print(format_path_weighted_summary_line(
            cname, ck, avgs,
            total_path_m=total_path_m,
            n_segments=n_seg,
            ckpt_any=ckpt_any,
        ))
    print("=" * 122)


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    carriers = _resolve_carriers(args.carrier, args.mode, config)
    if not carriers:
        print(f"[ERROR] No carriers to evaluate (input: {args.carrier!r}, mode: {args.mode})")
        return

    all_metrics = []
    base_cfg = config['base']
    if args.output_dir:
        base_cfg['output_dir'] = args.output_dir
    carrier_source = _resolve_carrier_source(args.carrier_source, base_cfg)
    test_data_root = _infer_test_data_root(base_cfg, config['carriers'], args.data_root)
    base_cfg['_resolved_test_data_root'] = test_data_root

    print(
        f"Launching evaluation | mode: {args.mode.upper()} | device: {device} | "
        f"carrier_source: {carrier_source} | carriers: {carriers}"
    )
    if test_data_root:
        print(f"Test data root: {test_data_root}")
    if args.weights:
        print(f"Weights override: {args.weights}")
    print()

    if carrier_source == 'router':
        allowed = [c for c in carriers if c in NATIVE_CARRIERS]
        if not allowed:
            print("[ERROR] router carrier_source needs at least one native carrier in --carrier")
            return
        all_metrics = evaluate_router_discovered(
            allowed, config['carriers'], base_cfg, device, args.mode,
            test_data_root=test_data_root,
            weights_path_override=args.weights,
        )
        _print_summary(all_metrics, args.mode)
        return

    for carrier in carriers:
        carrier_metrics = evaluate_single_carrier(
            carrier,
            config['carriers'][carrier],
            base_cfg,
            device,
            args.mode,
            weights_path_override=args.weights,
        )
        if carrier_metrics:
            all_metrics.extend(carrier_metrics)

    _print_summary(all_metrics, args.mode)


if __name__ == "__main__":
    main()
