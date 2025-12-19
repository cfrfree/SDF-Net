import torch
import torch.nn as nn
import torch.nn.functional as F


class DEE_Module(nn.Module):
    """
    [源自 DEEN 论文] 多样化嵌入扩展模块
    作用：将融合后的特征扩展为多个互不相关的子特征，挖掘细节信息。
    """

    def __init__(self, in_dim, num_branches=3, dropout=0.5):
        super(DEE_Module, self).__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()

        # 构建多分支结构，每个分支通过不同的FC层学习不同的特征子空间
        for _ in range(num_branches):
            self.branches.append(
                nn.Sequential(
                    nn.Linear(in_dim, in_dim),
                    nn.BatchNorm1d(in_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_dim, in_dim),
                )
            )

    def forward(self, x):
        # x: [B, Dim]
        outputs = [branch(x) for branch in self.branches]
        return outputs
