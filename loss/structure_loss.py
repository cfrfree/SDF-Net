import torch
import torch.nn as nn


class StructureConsistencyLoss(nn.Module):
    def __init__(self):
        super(StructureConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, f_struct, pids, camids):
        if f_struct is None:
            return torch.tensor(0.0, device=camids.device)

        loss = 0.0
        count = 0

        unique_pids = torch.unique(pids)

        for pid in unique_pids:
            idxs = pids == pid

            idx_v = idxs & (camids == 0)
            idx_t = idxs & (camids == 1)

            if idx_v.sum() > 0 and idx_t.sum() > 0:
                feat_v = f_struct[idx_v]
                feat_t = f_struct[idx_t]

                center_v = feat_v.mean(dim=0)
                center_t = feat_t.mean(dim=0)

                loss += self.mse(center_v, center_t)
                count += 1

        if count > 0:
            loss = loss / count
        else:
            loss = torch.tensor(0.0, device=camids.device)
        return loss
