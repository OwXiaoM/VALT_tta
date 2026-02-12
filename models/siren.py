import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SineLayer(nn.Module):
    # ... (保持原有的 SineLayer 代码不变) ...
    def __init__(self, in_feat, lat_feat, out_feat, bias=True, is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_feat
        self.out_features = out_feat
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.linear_lats = nn.Linear(lat_feat, out_feat * 2, bias=bias) if lat_feat > 0 else None
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                            np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        """
        input: (coords, latent_vec)
        """
        intermed = self.linear(input[0])
        if self.linear_lats is not None:
            lats = self.linear_lats(input[1])
            # input[1] is the latent vector
            out = torch.sin((self.omega * intermed * lats[..., :self.out_features]) + lats[..., self.out_features:])
        else:
            out = torch.sin(self.omega * intermed)
        # 必须返回 tuple 以保持与 Siren 结构的兼容性
        return out, input[1]

# [新增] MoE Layer
class MoESineLayer(nn.Module):
    def __init__(self, in_feat, lat_feat, out_feat, num_experts=4, k=2, omega=30):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        # 门控网络：输入坐标特征 -> 输出专家权重
        self.gate = nn.Linear(in_feat, num_experts)
        
        # 专家列表：每个专家是一个 SineLayer
        # 注意：专家通常不是第一层 (is_first=False)
        self.experts = nn.ModuleList([
            SineLayer(in_feat, lat_feat, out_feat, is_first=False, omega=omega)
            for _ in range(num_experts)
        ])
        self.latest_loss = 0.0

    def forward(self, input_tuple):
        """
        input_tuple: (coords, latent_vec)
        """
        x, latents = input_tuple
        
        # 1. 计算门控 logits
        gate_logits = self.gate(x) # (N, num_experts)
        
        # 2. Top-K 选择
        weights, indices = torch.topk(gate_logits, self.k, dim=-1) # (N, k)
        weights = F.softmax(weights, dim=-1) # 归一化权重
        
        # 3. 计算辅助损失 (Load Balancing Loss)
        # 简单版：Importance Loss + Load Loss (CV squared)
        # 这里使用 differentiable load balancing loss
        soft_gates = F.softmax(gate_logits, dim=-1)
        importance = soft_gates.sum(0)
        # 为了避免计算过于复杂，这里计算变异系数的平方作为负载均衡损失
        # 目标是让每个专家的 importance 接近 uniform
        target = x.shape[0] / self.num_experts
        loss = torch.sum((importance - target)**2) / (target**2)
        self.latest_loss = loss
        
        # 4. 执行专家推理 (加权求和)
        # 注意：为了实现简单，这里使用类似 Soft-MoE 的加权方式，
        # 在实际大规模 MoE 中会使用稀疏索引 (sparse scatter/gather)
        final_output = torch.zeros(x.shape[0], self.experts[0].out_features, device=x.device)
        
        for i in range(self.k):
            idx = indices[:, i] # (N,)
            w = weights[:, i].unsqueeze(1) # (N, 1)
            
            # 这是一个简化的实现：迭代每个专家。
            # 只有当 k << num_experts 时，真正的稀疏实现才有性能优势。
            # 在全连接层中，如果不写 CUDA kernel，循环掩码是常见做法。
            for e_idx in range(self.num_experts):
                # 找出当前 Top-k 选择中选中专家 e_idx 的样本掩码
                mask = (idx == e_idx)
                if mask.any():
                    # 只计算选中的样本
                    masked_x = x[mask]
                    masked_latents = latents[mask] if latents is not None else None
                    
                    # 专家前向传播
                    expert_out, _ = self.experts[e_idx]((masked_x, masked_latents))
                    
                    # 累加结果
                    final_output[mask] += w[mask] * expert_out
                    
        return final_output, latents

class Siren(nn.Module):
    def __init__(self, in_size, lat_size, out_size, hidden_size, num_layers, f_om, h_om,
                 outermost_linear, modulated_layers, use_moe=False, num_experts=4, moe_k=2):
        super().__init__()
        l_in_mod = 0 in modulated_layers
        
        # 第一层通常不设为 MoE，用于提取基础特征
        self.net = [SineLayer(in_size, lat_size * l_in_mod, hidden_size, is_first=True, omega=f_om)]
        self.hidden_size = hidden_size
        
        for i in range(num_layers):
            l_in_mod = (i+1) in modulated_layers
            
            # [新增] 决定是否使用 MoE
            # 例如：我们可以在中间层使用 MoE
            is_moe_layer = use_moe and (i > 0 and i < num_layers - 1) 
            
            if is_moe_layer:
                self.net.append(MoESineLayer(hidden_size, lat_size * l_in_mod, hidden_size, 
                                             num_experts=num_experts, k=moe_k, omega=h_om))
            else:
                self.net.append(SineLayer(hidden_size, lat_size * l_in_mod, hidden_size, is_first=False, omega=h_om))

        if outermost_linear:
            self.final_linear = nn.Linear(hidden_size+lat_size, out_size, bias=True)
            with torch.no_grad():
                self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / h_om,
                                                np.sqrt(6 / hidden_size) / h_om)
        else:
            self.final_linear = SineLayer(hidden_size, 0, out_size, is_first=False, omega=h_om)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        # x is tuple (coords, latents)
        x = self.net(x)
        # x[0] is features, x[1] is latents passed through
        return self.final_linear(torch.cat([x[0], x[1]], dim=-1))

    # [新增] 获取 MoE 辅助损失的方法
    def get_moe_loss(self):
        loss = 0.0
        count = 0
        for module in self.net:
            if isinstance(module, MoESineLayer):
                loss += module.latest_loss
                count += 1
        return loss / max(count, 1) # 返回平均 MoE 损失