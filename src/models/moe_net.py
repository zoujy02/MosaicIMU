import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv1D, DecoupledBasicBlock1D, UpgradedResBlock1D

class DecoupledDeepIMUEncoder(nn.Module):
    def __init__(self, in_channels=6, context_dim=128, seq_len=500):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = 64  
        dw_channels = in_channels * 16
        
        self.stem_conv = nn.Sequential(
            nn.Conv1d(in_channels, dw_channels, kernel_size=7, stride=2, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm1d(dw_channels),
            nn.GELU(),
            nn.Conv1d(dw_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.GELU()
        )
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(DecoupledBasicBlock1D, 64, 3, stride=1)  
        self.layer2 = self._make_layer(DecoupledBasicBlock1D, 128, 4, stride=2) 
        self.layer3 = self._make_layer(DecoupledBasicBlock1D, 256, 6, stride=1) 
        
        self.proj = nn.Sequential(
            nn.Conv1d(256, context_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(context_dim),
            nn.GELU()
        )
        
        self.attention_pool = nn.Sequential(
            nn.Conv1d(context_dim, context_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(context_dim // 2, 1, kernel_size=1) 
        )

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip1 = self.stem_conv(x)       
        x = self.stem_pool(skip1)       
        skip2 = self.layer1(x)          
        x = self.layer2(skip2)          
        x = self.layer3(x)              
        x = self.proj(x)                
        F_seq = x
        
        attn_logits = self.attention_pool(F_seq)         
        attn_weights = F.softmax(attn_logits, dim=2)      
        h = torch.sum(F_seq * attn_weights, dim=2)        
        
        return F_seq, h, skip1, skip2

class PrototypeRouter(nn.Module):
    def __init__(self, input_dim=128, proj_dim=64, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(num_experts, proj_dim))
        self.scale = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, h, temperature=1.0, use_gumbel=True):
        z = self.proj(h)
        z_norm = F.normalize(z, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.t())
        logits = sim * self.scale
        if use_gumbel:
            weights = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        else:
            weights = F.softmax(logits / temperature, dim=-1)
        proto_sim_matrix = torch.matmul(p_norm, p_norm.t())
        mask = 1 - torch.eye(self.num_experts, device=h.device)
        orth_loss = (torch.abs(proto_sim_matrix) * mask).sum() / (self.num_experts * (self.num_experts - 1))
        return weights, orth_loss

class ModularExpert_Seq_Lite(nn.Module):
    def __init__(self, in_channels=128, dropout_p=0.1):
        super().__init__()
        self.ds_bn = nn.BatchNorm1d(in_channels)
        self.res_group = nn.Sequential(
            UpgradedResBlock1D(in_channels, dilation=1, dropout_p=dropout_p),
            UpgradedResBlock1D(in_channels, dilation=4, dropout_p=dropout_p),
            UpgradedResBlock1D(in_channels, dilation=12, dropout_p=dropout_p),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )
        self.temporal = nn.LSTM(
            input_size=in_channels, hidden_size=in_channels // 2, 
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ds_bn(x)
        x = self.res_group(x)
        x = x.permute(0, 2, 1) 
        x, _ = self.temporal(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  
        return x


class TemporalVelHead(nn.Module):
    """Per-frame velocity head with dilated Conv1d stack + BiLSTM."""

    def __init__(self, in_channels=64, hidden=64, dropout_p=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=16, dilation=16, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )
        self.lstm = nn.LSTM(
            hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True,
        )
        self.head = nn.Conv1d(hidden, 3, kernel_size=1)

    def forward(self, feat):
        x = F.gelu(self.bn0(self.proj(feat)))
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        x_t, _ = self.lstm(x.transpose(1, 2))
        return self.head(x_t.transpose(1, 2)).transpose(1, 2)


class TemporalVarHead(nn.Module):
    """Per-frame variance head; same temporal stack as TemporalVelHead."""

    def __init__(self, in_channels=64, hidden=64, dropout_p=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_channels + 3, hidden, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=16, dilation=16, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )
        self.lstm = nn.LSTM(
            hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True,
        )
        self.head = nn.Conv1d(hidden, 3, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, feat, mu_seq):
        mu_abs = mu_seq.detach().abs().transpose(1, 2)
        x = torch.cat([feat, mu_abs], dim=1)
        x = F.gelu(self.bn0(self.proj(x)))
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        x_t, _ = self.lstm(x.transpose(1, 2))
        logits = self.head(x_t.transpose(1, 2))
        return F.softplus(logits).transpose(1, 2) + 1e-6


class SharedUNetDecoder(nn.Module):
    def __init__(self, in_channels=128, expert_dim=64, dropout_p=0.1):
        super().__init__()
        self.up1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.fuse1 = DoubleConv1D(in_channels + 64, in_channels, dropout_p)
        self.up2 = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.fuse2 = DoubleConv1D((in_channels // 2) + 64, in_channels // 2, dropout_p)
        self.up3 = nn.ConvTranspose1d(in_channels // 2, expert_dim, kernel_size=2, stride=2)
        self.fuse3 = DoubleConv1D(expert_dim, expert_dim, dropout_p)

        self.head_mu = TemporalVelHead(in_channels=expert_dim, hidden=expert_dim, dropout_p=dropout_p)
        self.head_var = TemporalVarHead(in_channels=expert_dim, hidden=expert_dim, dropout_p=dropout_p)

    def forward(self, expert_features_seq, router_weights, skip1, skip2):
        weights_expanded = router_weights.unsqueeze(-1).unsqueeze(-1)
        fused_latent = (expert_features_seq * weights_expanded).sum(dim=1) 

        x = self.up1(fused_latent)
        x = torch.cat([x, skip2], dim=1)
        x = self.fuse1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse2(x)
        
        x = self.up3(x)
        x = self.fuse3(x)
        
        mu_seq = self.head_mu(x)
        var_seq = self.head_var(x, mu_seq)
        return mu_seq, var_seq


class CarrierConditionedMoE_Seq(nn.Module):
    def __init__(
        self, in_channels=6, context_dim=128, expert_dim=64, num_experts=4,
        seq_len=500, dropout_p=0.1,
    ):
        super().__init__()
        self.encoder = DecoupledDeepIMUEncoder(in_channels, context_dim, seq_len)
        self.router = PrototypeRouter(context_dim, proj_dim=expert_dim, num_experts=num_experts) 
        self.experts = nn.ModuleList([ModularExpert_Seq_Lite(context_dim, dropout_p=dropout_p) for _ in range(num_experts)])
        self.decoder = SharedUNetDecoder(context_dim, expert_dim, dropout_p=dropout_p)
        
    def forward(self, x, temperature=1.0, use_gumbel=True):
        F_seq, h, skip1, skip2 = self.encoder(x)
        router_weights, orth_loss = self.router(h, temperature, use_gumbel=use_gumbel)
        
        expert_outputs = [expert(F_seq) for expert in self.experts]
        stacked_expert_features = torch.stack(expert_outputs, dim=1)
        mu_seq, var_seq = self.decoder(stacked_expert_features, router_weights, skip1, skip2)
        
        return mu_seq, var_seq, router_weights, orth_loss
