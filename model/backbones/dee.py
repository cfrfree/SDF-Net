import torch
import torch.nn as nn
import torch.nn.functional as F


class DEE_Module(nn.Module):
    def __init__(self, in_dim, num_branches=3, dropout=0.5):
        super(DEE_Module, self).__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()

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
        outputs = [branch(x) for branch in self.branches]
        return outputs
