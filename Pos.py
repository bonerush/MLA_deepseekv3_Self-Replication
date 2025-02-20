import torch
import math
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
# 绝对位置编码
class PosEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PosEncoding, self).__init__()
        self.d_model = d_model # 词嵌入维度
        self.max_len = max_len # 最大序列长度
        self.pe = torch.zeros(self.max_len, self.d_model)# 初始化位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 创建位置索引i，并将其扩展为二维张量，形状为 [max_len, 1]
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model)) # 计算位置编码的分母部分,[d_model/2]
        # torch,arange(start,end,step)上面的代码生成出来的数组就是(0,2,4,...)-->这里就代表着2i的含义
        self.pe[:,0::2] = torch.sin(position*div_term)# 偶数列使用sin函数计算位置编码
        # NOTE 这里的0::2代表着从0开始到最后面，每隔2取一个数字，其实就是取偶数列
        self.pe[:,1::2] = torch.cos(position*div_term)# 奇数列使用cos函数计算位置编码
        # NOTE 这里的1::2代表着从1开始到最后面，每隔2取一个数字，其实就是取奇数列
        with torch.no_grad():
            self.pe = self.pe.view(1,max_len,d_model)# [max_len,d_model] --> [1,max_len,d_model]
    def forward(self,x):
        return x + self.pe[:,:x.size(1),:]# [bsz,seq_len,d_model]
    # NOTE 这里的x.size(0)其实就是seq_len,所以self.pe[:x.size(1),:]其实就是self.pe[:seq_len,:]

# 旋转位置编码1
class RoPE(nn.Module): 
    def __init__(self,d_model,seq_len):
        super(RoPE,self).__init__()
        theta = 10000.**(-torch.arange(0,d_model//2).float()/(d_model//2))# [d_model//2]
        seq_idx = torch.arange(seq_len,dtype=torch.float)# [seq_len] 
        idx_theta = torch.einsum('n,d->nd',seq_idx,theta)# [seq_len,d_model//2]
        idx_theta2 = torch.cat([idx_theta,idx_theta],dim=1)# [seq_len,d_model]
        idx_theta2 = idx_theta2.unsqueeze(0)# [bsz,seq_len,d_model]
        self.cos_cache = idx_theta2.cos()[:,:,:]# [bsz,seq_len,d_model]
        self.sin_cache = idx_theta2.sin()[:,:,:]# [bsz,seq_len,d_model]
    def rotate_half(self,x): 
        x1 = x[...,:x.size(-1)//2] #x:[bsz,seq_len,d_model] 
        # 取前一半的维度:x.size(-1)//2
        x2 = x[...,x.size(-1)//2:] #x2:[bsz,seq_len,d_model//2]    
        # 取后一半的维度x.shape(-1)//2:
        x = torch.cat([-x2,x1],dim=-1) #x:[bsz,seq_len,d_model]
        return x
    def forward(self,x):
        seq_len = x.size(1)
        cos = self.cos_cache[:,:seq_len,:]# [bsz,seq_len,d_model]
        sin = self.sin_cache[:,:seq_len,:]# [bsz,seq_len,d_model]
        x_rot = x*cos+self.rotate_half(x)*sin# [bsz,seq_len,d_model]
        return x_rot
    
# TODO 旋转位置编码2(未研究)
class RoPE2(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=x.device)
        
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

# 测试程序
if __name__ == "__main__":
    # 设置参数
    d_model = 64  # 词嵌入维度
    # max_len = 100  # 最大序列长度
    seq_len = 10  # 测试序列长度
    bsz = 32  # 批量大小

    # 创建 PosEncoding 实例
    # pos_encoder = PosEncoding(d_model, max_len)
    # 创建 RoPE 实例
    pos_encoder = RoPE2(d_model)
    # 创建随机输入张量 [bsz,seq_len, d_model]
    x = torch.randn(bsz,seq_len, d_model)

    # 打印输入张量
    print("输入张量:")
    print(x)

    # 应用位置编码
    output = pos_encoder(x)

    # 打印输出张量
    print("\n带有位置编码的输出张量:")
    print(output)

    # # 打印位置编码矩阵
    # print("\n位置编码矩阵:")
    # print(pos_encoder.pe[:seq_len, :])
