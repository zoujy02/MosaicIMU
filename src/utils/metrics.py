from collections import defaultdict

import numpy as np


class ExpertUsageAnalyzer:
    def __init__(self):
        self.vel_errors = []
        self.label_vel_errors = defaultdict(list)
        self.drift_errors = []
        self.label_drift_errors = defaultdict(list)
        self.all_weights = []
        self.label_weights = defaultdict(list)
        self.all_alphas = []
        self.label_alphas = defaultdict(list)

    def update(self, labels, pred_vel_seq, target_vel_seq, weights, alphas=None, dt=1.0 / 200.0):
        labels_np = labels.detach().cpu().numpy()
        p_vel = pred_vel_seq.detach().cpu().numpy()
        t_vel = target_vel_seq.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()

        v_errors = np.linalg.norm(p_vel - t_vel, axis=-1).mean(axis=-1)
        cum_vel_err = np.cumsum((p_vel - t_vel) * dt, axis=1)
        d_errors = np.linalg.norm(cum_vel_err[:, -1, :], axis=-1)

        self.vel_errors.extend(v_errors)
        self.drift_errors.extend(d_errors)
        self.all_weights.extend(weights_np)

        alphas_np = None
        if alphas is not None:
            alphas_np = alphas.detach().cpu().numpy()
            self.all_alphas.extend(alphas_np)

        for i, (l, v_err, d_err, w) in enumerate(zip(labels_np, v_errors, d_errors, weights_np)):
            self.label_vel_errors[l].append(v_err)
            self.label_drift_errors[l].append(d_err)
            self.label_weights[l].append(w)
            if alphas_np is not None:
                self.label_alphas[l].append(alphas_np[i])

    def compute_epoch_stats(self, epoch, class_weights=None):
        def safe_err_mean(label_idx, error_dict):
            vals = error_dict.get(label_idx, [])
            return float(np.mean(vals)) if len(vals) > 0 else 0.0

        def safe_weight_mean(label_idx):
            vals = self.label_weights.get(label_idx, [])
            if len(vals) == 0:
                return np.zeros(4, dtype=np.float64)
            return np.mean(np.array(vals), axis=0)

        def safe_alpha_mean(label_idx):
            vals = self.label_alphas.get(label_idx, [])
            return float(np.mean(vals)) if len(vals) > 0 else 0.0

        all_w_np = np.array(self.all_weights) if len(self.all_weights) > 0 else np.zeros((0, 4))
        all_a_np = np.array(self.all_alphas) if len(self.all_alphas) > 0 else np.zeros((0,))

        stats = {
            'epoch': epoch,
            'vel_err_mean': np.mean(self.vel_errors) if len(self.vel_errors) > 0 else 0.0,
            'drift_err_mean': np.mean(self.drift_errors) if len(self.drift_errors) > 0 else 0.0,
            'w_mean': np.mean(all_w_np, axis=0) if len(all_w_np) > 0 else np.zeros(4),
            'alpha_mean': np.mean(all_a_np) if len(all_a_np) > 0 else 0.0,
        }
        for i in range(5):
            stats[f'vel_err_{i}'] = safe_err_mean(i, self.label_vel_errors)
            stats[f'drift_err_{i}'] = safe_err_mean(i, self.label_drift_errors)
            stats[f'w_{i}'] = safe_weight_mean(i)
            stats[f'alpha_{i}'] = safe_alpha_mean(i)

        if class_weights is not None and isinstance(class_weights, dict) and len(class_weights) > 0:
            total_weight = 0.0
            weighted_sum = 0.0
            for label_idx, weight in class_weights.items():
                if len(self.label_drift_errors.get(label_idx, [])) > 0:
                    weighted_sum += weight * safe_err_mean(label_idx, self.label_drift_errors)
                    total_weight += weight
            stats['weighted_drift_err'] = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            stats['weighted_drift_err'] = stats['drift_err_mean']

        self.__init__()
        return stats

    def print_pretrain_report(
        self, stats, loss_task, loss_aux=None, loss_orth=None, loss_type=None, current_lr=None,
        *, compact=False, line_prefix="", carriers=None, show_router_weights=False,
    ):
        w_g = stats['w_mean']
        if compact:
            pre = f"{line_prefix} | " if line_prefix else ""
            if carriers is None:
                carriers = {0: 'car', 1: 'quadrupedal', 2: 'pedestrian', 3: 'drone'}
            vel_parts = " ".join(
                f"{name}:{stats[f'vel_err_{idx}']:.4f}"
                for idx, name in sorted(carriers.items())
            )
            drift = stats.get('weighted_drift_err', None)
            if drift is not None:
                drift_parts = " ".join(
                    f"{name}:{stats[f'drift_err_{idx}']:.4f}"
                    for idx, name in sorted(carriers.items())
                )
                print(
                    f"  {pre}task {loss_task:.4f} | vel {vel_parts} | "
                    f"drift {drift:.4f} m ({drift_parts})"
                )
            else:
                print(f"  {pre}task {loss_task:.4f} | vel {vel_parts}")
            if show_router_weights:
                print(f"  {pre}router global: [{w_g[0]:.3f}, {w_g[1]:.3f}, {w_g[2]:.3f}, {w_g[3]:.3f}]")
                for idx, name in sorted(carriers.items()):
                    w = stats[f'w_{idx}']
                    print(
                        f"  {pre}router {name}: "
                        f"[E0={w[0]:.3f}, E1={w[1]:.3f}, E2={w[2]:.3f}, E3={w[3]:.3f}]"
                    )
            return

        print(f"Epoch [{stats['epoch']}] mode: [{loss_type}] | LR: {current_lr:.6f}")
        print(f"  [Loss] Task(Vel/Disp): {loss_task:.4f} | L_car(KL): {loss_aux:.4f} | Orth: {loss_orth:.4f}")
        print(
            f"  [VelErr] mean: {stats['vel_err_mean']:.4f} | "
            f"Car: {stats['vel_err_0']:.4f} | Quad: {stats['vel_err_1']:.4f} | "
            f"Ped: {stats['vel_err_2']:.4f} | Drone: {stats['vel_err_3']:.4f}"
        )
        print(
            f"  [Drift] L0 weighted cum drift: {stats['weighted_drift_err']:.4f} m (eval metric) | "
            f"Car: {stats['drift_err_0']:.4f} | Quad: {stats['drift_err_1']:.4f} | "
            f"Ped: {stats['drift_err_2']:.4f} | Drone: {stats['drift_err_3']:.4f}"
        )
        print("  [Expert routing distribution]")
        print(f"   -> Global mean : [Exp 1: {w_g[0]:.2f} | Exp 2: {w_g[1]:.2f} | Exp 3: {w_g[2]:.2f} | Exp 4: {w_g[3]:.2f}]")
        for i, name in zip(range(4), ['Car', 'Quad', 'Ped', 'Drone']):
            w = stats[f'w_{i}']
            print(f"   -> {name:<5} : [Exp 1: {w[0]:.2f} | Exp 2: {w[1]:.2f} | Exp 3: {w[2]:.2f} | Exp 4: {w[3]:.2f}]")
        print("-" * 105)

    def print_finetune_report(self, stats, phase_name, avg_main_metric, emoji_map, idx_to_category):
        print(f"[{phase_name}] | main metric (Loss/Drift): {avg_main_metric:.4f} | global bypass alpha: {stats['alpha_mean']:.4f}")
        for idx, cat_name in idx_to_category.items():
            tag = emoji_map.get(cat_name, "[?]")
            drift = stats[f'drift_err_{idx}']
            alpha = stats[f'alpha_{idx}']
            w = stats[f'w_{idx}']

            if np.sum(w) == 0 and drift == 0.0:
                continue

            print(
                f"  {tag} {cat_name:<12} -> drift: {drift:.4f} m | alpha: {alpha:.4f} | "
                f"base router: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}]"
            )

        if 'weighted_drift_err' in stats and ("Val" in phase_name or "Validation" in phase_name):
            print(f"  [EarlyStop metric] weighted cumulative drift: {stats['weighted_drift_err']:.4f} m")
        print("-" * 90)


def print_phase_report(phase_name, avg_main_metric, avg_alpha, stats_dict, idx_to_category, emoji_map, is_val=False):
    metric_name = "Weighted Drift" if is_val else "Avg Loss"
    print(f"[{phase_name}]")
    print(f"  -> {metric_name}: {avg_main_metric:.4f} | overall alpha: {avg_alpha:.4f}")
    print("  [Per-carrier router & bypass-alpha breakdown]")

    if not stats_dict:
        print("   -> (no statistics available)")
        return

    all_routers = np.concatenate([s['router'] for s in stats_dict.values()], axis=0)
    g_router = np.mean(all_routers, axis=0)
    g_str = " | ".join([f"E{i+1}: {w:.2f}" for i, w in enumerate(g_router)])
    print(f"   -> Global mean : [{g_str}]")

    for lbl, stats in sorted(stats_dict.items()):
        name = idx_to_category.get(lbl, f"Label-{lbl}")
        tag = emoji_map.get(name, '[?]')
        avg_r = np.mean(stats['router'], axis=0)
        avg_a = np.mean(stats['alpha'])
        avg_d = np.mean(stats['drift'])

        r_str = " | ".join([f"E{i+1}: {w:.2f}" for i, w in enumerate(avg_r)])
        print(f"   -> {tag} {name.ljust(11)}: [{r_str}] | Alpha: {avg_a:.3f} | Drift: {avg_d:.4f} m")


def print_phase_report_compact(phase_name, avg_main_metric, avg_alpha, stats_dict, idx_to_category, emoji_map, is_val=False):
    metric_name = "w-drift (m)" if is_val else "loss"
    head = f"[{phase_name}] {metric_name}={avg_main_metric:.4f} | α={avg_alpha:.4f}"
    if not stats_dict:
        print(head + " | (no stats)")
        return
    parts = []
    for lbl, stats in sorted(stats_dict.items()):
        name = idx_to_category.get(lbl, f"L-{lbl}")
        avg_r = np.mean(stats['router'], axis=0)
        r_str = ",".join(f"{w:.2f}" for w in avg_r)
        parts.append(
            f"{name}:D={np.mean(stats['drift']):.3f},α={np.mean(stats['alpha']):.2f},w[{r_str}]",
        )
    print(head + " | " + " | ".join(parts))


def compute_metrics(gt, est, dt_hz=200.0, rte_sec=10.0):
    ate = np.sqrt(np.mean(np.sum((gt - est) ** 2, axis=1)))
    step = int(rte_sec * dt_hz)
    if len(gt) <= step:
        rte = np.nan
    else:
        delta_gt = gt[step:] - gt[:-step]
        delta_est = est[step:] - est[:-step]
        rte = np.sqrt(np.mean(np.sum((delta_gt - delta_est) ** 2, axis=1)))
    return ate, rte
