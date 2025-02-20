import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class RoPE(nn.Module):
    """
    旋转位置编码
    Args:
        d_model: 模型的维度
        theta: 旋转的角度
    INPUT:
       x: 输入张量，形状为 [..., seq_len, d_model]
    OUTPUT:
        旋转后的张量，形状与输入相同    
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 正确计算theta (10000^{-2i/d_model})
        theta = 10000 ** (-torch.arange(0, d_model//2, dtype=torch.float) / (d_model//2))
        self.register_buffer("theta", theta)  # [d_model//2]

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将输入的后半部分与前半部分交换并取反后拼接"""
        d = x.size(-1)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor,start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 [..., seq_len, d_model]
        Returns:
            旋转后的张量，形状与输入相同
        """
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model, f"特征维度应为 {self.d_model}，但输入为 {d_model}"
        
        # 生成位置索引 [seq_len]
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        
        # 计算位置与频率的外积 [seq_len, d_model//2]
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        
        # 扩展为完整维度 [seq_len, d_model]
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        
        # 计算旋转分量 [seq_len, d_model]
        cos = idx_theta2.cos()
        sin = idx_theta2.sin()
        
        # 自动广播到输入维度 [..., seq_len, d_model]
        # 例如输入形状为 [batch, heads, seq, dim] 时会自动适配
        return x * cos + self.rotate_half(x) * sin

@dataclass
class Config:
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    dropout: float
    max_bsz: int
    max_seq_len: int

class MLA(nn.Module):
    """
    Args:
       config (Config): 配置参数
           - dim (int): 模型维度
           - n_heads (int): 头数
           - q_lora_rank (int): 查询矩阵低秩分解的秩数
           - kv_lora_rank (int): 键值矩阵低秩分解的秩数
           - qk_nope_head_dim (int): 查询键矩阵不使用rope的维度数
           - qk_rope_head_dim (int): 查询键矩阵使用rope的维度数
           - qk_head_dim (int): 查询键矩阵的维度数
           - v_head_dim (int): 值矩阵的维度数
           - dropout (float): dropout比例
    INPUT:
        x (torch.Tensor): 输入张量, 形状为 [batch, seq_len, dim]
    OUTPUT:
        torch.Tensor: 输出张量, 形状为 [batch, seq_len, dim]
    """
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.dropout = config.dropout
        self.max_seq_len = config.max_seq_len
        self.max_bsz = config.max_bsz
        # 查询矩阵q低秩分解
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)# 低秩矩阵A
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, (self.qk_nope_head_dim+self.qk_rope_head_dim) * self.n_heads, bias=False)# 低秩矩阵B
        self.q_rope = RoPE(self.qk_rope_head_dim)
        # k,v矩阵低秩分解
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False) # 低秩矩阵A
        self.wkv_b = nn.Linear(self.kv_lora_rank, (self.qk_nope_head_dim+ self.v_head_dim)* self.n_heads, bias=False) # 低秩矩阵B
        self.k_rope = RoPE(self.qk_rope_head_dim)
        # 输出
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, bias=False)
        self.dropout = nn.Dropout(self.dropout)
        self.eps =1e-6
        self.register_buffer("k_cache", torch.zeros(self.max_bsz, self.max_seq_len, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim))
        self.register_buffer("v_cache", torch.zeros(self.max_bsz, self.max_seq_len, self.n_heads, self.v_head_dim))
   
    def forward(self,x: torch.Tensor,start_pos: int) -> torch.Tensor:
        bsz,seq_len,_ = x.size()
        end_pos = start_pos + seq_len
        # 计算q
        q=self.wq_a(x)
        q=self.q_norm(q)
        q=self.wq_b(q)
   
        q=q.view(bsz,seq_len,self.n_heads,self.qk_rope_head_dim+self.qk_nope_head_dim)
        q_nope,q_rope = torch.split(q, [self.qk_nope_head_dim,self.qk_rope_head_dim], dim=-1)
        # [bsz,seq_len,n_heads,qk_rope_head_dim]
        q_rope = q_rope.view(bsz,self.n_heads,seq_len,self.qk_rope_head_dim)
        q_rope = self.q_rope(q_rope,start_pos)# [bsz,n_heads,seq_len,qk_rope_head_dim]
        q_rope = q_rope.view(bsz,seq_len,self.n_heads,self.qk_rope_head_dim)
        # [bsz,seq_len,n_heads,qk_rope_head_dim]
        q = torch.cat([q_nope, q_rope.expand(-1,-1,self.n_heads,-1)], dim=-1)
        # [bsz,seq_len,n_heads,qk_rope_head_dim+qk_nope_head_dim]
        
        # 计算k,v,k_pe
        kv=self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2) # [bsz,seq_len,1,qk_rope_head_dim]
        # kv：[bsz,seq_len,kv_lora_rank]
        k_pe = self.k_rope(k_pe,start_pos)
        kv = self.wkv_b(F.rms_norm(kv,[kv.size(-1)],weight=nn.Parameter(torch.ones(self.kv_lora_rank)),eps=1e-6))
        kv = kv.view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope,v=torch.split(kv,[self.qk_nope_head_dim,self.v_head_dim],dim=-1)
        # k_nope:[bsz,seq_len,n_heads,qk_nope_head_dim]
        # v:[bsz,seq_len,n_heads,v_head_dim]
        k=torch.concat([k_nope,k_pe.expand(-1,-1,self.n_heads,-1)],dim=-1)
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        
        scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.qk_rope_head_dim+self.qk_nope_head_dim)
        # [bsz,seq_len,n_heads,seq_len]
        # if mask is not None:
        #     scores = scores + mask
        attn=F.softmax(scores,dim=-1)# [bsz,seq_len,n_heads,seq_len]
        context=torch.matmul(attn,v)# [bsz,seq_len,n_heads,v_head_dim]
        x=self.wo(context.flatten(2))# [batch_size, seq_len, n_local_heads * v_head_dim]
        return x
        """
        这里的 flatten(2) 表示从第3个维度（Python索引从0开始，
        dim=2 对应第3个维度）开始展平，将多头维度和值向量维度合并，为后续线性投影做准备。
        """
# 测试代码
def test_mla():
    config = Config(
        dim=512,
        n_heads=8,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        qk_head_dim=64,
        v_head_dim=64,
        dropout=0.1,
        max_bsz=32,
        max_seq_len=2048
    )
    model = MLA(config)
    
    # 测试正常前向传播
    x = torch.randn(2, 10, 512)# [bsz,seq_len,dim]
    output = model(x,start_pos=0)
    assert output.shape == (2, 10, 512), f"输出形状错误: {output.shape}"
    
    # 测试增量推理
    output = model(x[:, :1, :], start_pos=0)
    assert output.shape == (2, 1, 512), f"增量推理形状错误: {output.shape}"
    
    print("测试通过！")

test_mla()               
    
    
    
  