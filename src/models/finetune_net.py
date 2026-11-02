import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv1D
from .moe_net import CarrierConditionedMoE_Seq


class NewSeqExpert(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.local_cnn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels), nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels), nn.GELU(),
        )
        self.lstm = nn.LSTM(in_channels, in_channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.local_cnn(x).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out.permute(0, 2, 1)


class ResidualUNetDecoder(nn.Module):
    def __init__(self, in_channels=256, expert_dim=64, dropout_p=0.1):
        super().__init__()
        self.up1 = nn.ConvTranspose1d(in_channels, 128, 3, 2, 1, 0)
        self.fuse1 = DoubleConv1D(128 + 64, 128, dropout_p)
        self.up2 = nn.ConvTranspose1d(128, 64, 2, 2)
        self.fuse2 = DoubleConv1D(64 + 64, 64, dropout_p)
        self.up3 = nn.ConvTranspose1d(64, expert_dim, 2, 2)
        self.fuse3 = DoubleConv1D(expert_dim, expert_dim, dropout_p)

        self.head_residual = nn.Conv1d(expert_dim, 6, 1)
        nn.init.constant_(self.head_residual.weight, 0)
        nn.init.constant_(self.head_residual.bias, 0)

    def forward(self, fused, s1, s2):
        x = self.fuse1(torch.cat([self.up1(fused), s2], dim=1))
        x = self.fuse2(torch.cat([self.up2(x), s1], dim=1))
        x = self.fuse3(self.up3(x))
        res = self.head_residual(x)
        return res[:, :3, :].transpose(1, 2), res[:, 3:, :].transpose(1, 2)


class RouterAdapter(nn.Module):
    def __init__(self, h_dim=128, num_experts=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_experts),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h):
        return self.mlp(h)


class WeightTuneMoEFineTuner(nn.Module):
    def __init__(self, base_model, q_car=None):
        super().__init__()
        self.base_model = base_model

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.new_expert = NewSeqExpert(128)
        self.new_decoder = ResidualUNetDecoder(256)

        self.gate = nn.Linear(128, 1)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, -2.0)

        self.router_adapter = RouterAdapter(h_dim=128, num_experts=4)
        self.beta = nn.Parameter(torch.tensor(0.1))

        if q_car is None:
            self.q_car = torch.tensor([0.8, 0.1, 0.05, 0.05], dtype=torch.float32)
        else:
            self.q_car = q_car
        self.register_buffer("q_car_buf", self.q_car)

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, x, label=None, temperature=4.0):
        self.base_model.eval()

        with torch.no_grad():
            F_seq, h, skip1, skip2 = self.base_model.encoder(x)
            z = self.base_model.router.proj(h)
            z_norm = F.normalize(z, p=2, dim=-1)
            p_norm = F.normalize(self.base_model.router.prototypes, p=2, dim=-1)
            logits_old = torch.matmul(z_norm, p_norm.t()) * self.base_model.router.scale
            stacked_old_features = torch.stack([expert(F_seq) for expert in self.base_model.experts], dim=1)

        delta = self.router_adapter(h)
        beta = torch.clamp(self.beta, 0.0, 1.0)
        logits_calib = logits_old + beta * delta
        w = F.softmax(logits_calib / temperature, dim=-1)

        mu_old, var_old = self.base_model.decoder(stacked_old_features, w, skip1, skip2)
        f_old_context = (stacked_old_features * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        f_new = self.new_expert(F_seq)
        mu_residual, var_residual_raw = self.new_decoder(
            torch.cat([f_old_context, f_new], dim=1), skip1, skip2,
        )

        alpha = torch.sigmoid(self.gate(h)).unsqueeze(-1)

        l_prior = torch.tensor(0.0, device=x.device)
        l_alpha_penalty = torch.tensor(0.0, device=x.device)
        l_alpha_reward = torch.tensor(0.0, device=x.device)

        eps = 1e-8
        if label is not None:
            mask_new = (label == 4)
            mask_old = ~mask_new
            if mask_new.any():
                w_new = w[mask_new]
                l_prior = l_prior + torch.sum(
                    w_new * (torch.log(w_new + eps) - torch.log(self.q_car_buf.unsqueeze(0) + eps)),
                    dim=-1,
                ).mean()

            if mask_old.any():
                w_base_old = F.softmax(logits_old[mask_old], dim=-1).detach()
                w_calib_old = w[mask_old]
                l_prior = l_prior + torch.sum(
                    w_base_old * (torch.log(w_base_old + eps) - torch.log(w_calib_old + eps)),
                    dim=-1,
                ).mean()

        mu = mu_old + alpha * mu_residual
        var = var_old + alpha.detach() * F.softplus(var_residual_raw)

        return mu, var, w, alpha.squeeze(-1), l_prior, l_alpha_penalty, l_alpha_reward
