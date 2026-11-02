import torch

_ROUTER_PRIOR_EXPERT_TAGS = ('car', 'quadrupedal', 'pedestrian', 'drone')

_DEFAULT_CARRIER_SOFT_PRIORS = [
    [0.85, 0.05, 0.05, 0.05],
    [0.05, 0.85, 0.05, 0.05],
    [0.05, 0.05, 0.85, 0.05],
    [0.05, 0.05, 0.05, 0.85],
]


def resolve_router_prior_display_name(train_cfg, num_experts=4):
    explicit = train_cfg.get('router_prior_name')
    if explicit is not None:
        name = str(explicit).strip()
        if name:
            return name
    default = (0.8, 0.1, 0.05, 0.05)
    rp = train_cfg.get('router_prior', default)
    if isinstance(rp, (list, tuple)) and len(rp) == num_experts:
        idx = max(range(num_experts), key=lambda i: float(rp[i]))
        return f"q_{_ROUTER_PRIOR_EXPERT_TAGS[idx]}"
    return "q_prior"


def build_router_prior_tensor(train_cfg, num_experts=4):
    default = (0.8, 0.1, 0.05, 0.05)
    rp = train_cfg.get('router_prior', default)
    if not isinstance(rp, (list, tuple)) or len(rp) != num_experts:
        raise ValueError(
            f"router_prior in train_config must be a list of length {num_experts}, got {rp}"
        )
    t = torch.tensor([float(x) for x in rp], dtype=torch.float32)
    if (t <= 0).any():
        raise ValueError(f"router_prior values must be positive, got {rp}")
    return t


def build_carrier_soft_priors_tensor(loss_cfg, num_carriers=4, num_experts=4):
    lc = loss_cfg or {}
    raw = lc.get('carrier_soft_priors', _DEFAULT_CARRIER_SOFT_PRIORS)
    if not isinstance(raw, (list, tuple)) or len(raw) != num_carriers:
        raise ValueError(
            f"carrier_soft_priors must be a list of length {num_carriers}, got {raw}"
        )
    rows = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != num_experts:
            raise ValueError(
                f"each carrier_soft_priors row must have length {num_experts}, got {row}"
            )
        rows.append([float(x) for x in row])
    t = torch.tensor(rows, dtype=torch.float32)
    if (t <= 0).any():
        raise ValueError(f"carrier_soft_priors values must be positive, got {rows}")
    row_sums = t.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones(num_carriers), atol=0.0001):
        raise ValueError(
            f"each carrier_soft_priors row should sum to 1, got row sums {row_sums.tolist()}"
        )
    return t


def router_prior_warmup_weight(epoch, warmup_start, warmup_end):
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return 1.0
    return float(epoch - warmup_start) / float(warmup_end - warmup_start)


def heteroscedastic_nll_loss(pred_mu_seq, pred_var_seq, target_vel_seq):
    residual_sq = (target_vel_seq - pred_mu_seq) ** 2
    per_t = 0.5 * torch.sum(residual_sq / pred_var_seq + torch.log(pred_var_seq), dim=-1)
    return per_t.mean()


def velocity_mse_loss(pred_mu_seq, target_vel_seq):
    per_t = torch.sum((pred_mu_seq - target_vel_seq) ** 2, dim=-1)
    return per_t.mean()


def carrier_prior_kl_loss(weights, label, carrier_priors, router_log_eps=1e-8):
    q_c = carrier_priors[label]
    log_w = torch.log(weights + router_log_eps)
    log_q = torch.log(q_c + router_log_eps)
    return torch.sum(weights * (log_w - log_q), dim=-1).mean()


carrier_prior_guidance_loss = carrier_prior_kl_loss


def compute_moe_loss_l0(
    pred_mu_seq, pred_var_seq, weights, orth_loss,
    target_vel_seq, label, epoch,
    lambda_ce=0.05, alpha_orth=0.1,
    lambda_vel=1.0, lambda_nll=1.0, lambda_disp=10.0, dt=1.0 / 200.0,
    stage_switch_epoch=50,
    variance_detach_epochs=10,
    router_log_eps=1e-8,
    soft_priors=None,
):
    if epoch < variance_detach_epochs:
        pred_var_seq = pred_var_seq.detach()

    l_nll = heteroscedastic_nll_loss(pred_mu_seq, pred_var_seq, target_vel_seq)
    l_vel = velocity_mse_loss(pred_mu_seq, target_vel_seq)

    if soft_priors is not None:
        l_car = carrier_prior_kl_loss(
            weights, label, soft_priors, router_log_eps=router_log_eps,
        )
        loss_type = "L0_NLL+Vel+KL_car"
    else:
        num_experts = weights.shape[-1]
        one_hot_priors = torch.eye(num_experts, device=weights.device, dtype=weights.dtype)
        l_car = carrier_prior_kl_loss(
            weights, label, one_hot_priors, router_log_eps=router_log_eps,
        )
        loss_type = "L0_NLL+Vel+KL_car(onehot)"

    total_loss = (
        (lambda_nll * l_nll)
        + (lambda_vel * l_vel)
        + (lambda_ce * l_car)
        + (alpha_orth * orth_loss)
    )

    l_disp = torch.tensor(0.0, device=target_vel_seq.device)
    if epoch >= stage_switch_epoch:
        vel_err = (pred_mu_seq - target_vel_seq) * dt
        cum_vel_err = torch.cumsum(vel_err, dim=1)
        sq_norm = torch.sum(cum_vel_err ** 2, dim=-1)
        l_disp = torch.mean(sq_norm)

        total_loss = total_loss + lambda_disp * l_disp
        loss_type = loss_type.replace("Vel", "Vel+Disp")

    return total_loss, l_vel, l_nll, l_car, l_disp, loss_type


def compute_finetune_loss(
    pred_mu_seq,
    pred_var_seq,
    target_vel_seq,
    dt=1.0 / 200.0,
    lambda_int=10.0,
):
    l_nll = heteroscedastic_nll_loss(pred_mu_seq, pred_var_seq, target_vel_seq)
    l_vel = velocity_mse_loss(pred_mu_seq, target_vel_seq)

    l_int = torch.mean(torch.sum(torch.cumsum((pred_mu_seq - target_vel_seq) * dt, dim=1) ** 2, dim=-1))

    total_loss = l_nll + l_vel + (lambda_int * l_int)
    return total_loss, l_vel, l_nll, l_int


def compute_finetune_total_loss(
    pred_mu_seq,
    pred_var_seq,
    target_vel_seq,
    l_prior,
    lambda_prior,
    *,
    dt=1.0 / 200.0,
    lambda_int=10.0,
):
    loss_task, l_vel, l_nll, l_int = compute_finetune_loss(
        pred_mu_seq,
        pred_var_seq,
        target_vel_seq,
        dt=dt,
        lambda_int=lambda_int,
    )
    total = loss_task + lambda_prior * l_prior
    return total, loss_task, l_vel, l_nll, l_int
