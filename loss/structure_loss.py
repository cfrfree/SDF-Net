import torch
import torch.nn as nn

class StructureConsistencyLoss(nn.Module):
    def __init__(self):
        super(StructureConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, f_struct, pids, camids):
        """
        f_struct: [B, C] 结构特征
        pids: [B] 身份标签
        camids: [B] 相机标签 (0=Visible, 1=Thermal/SAR)
        """
        if f_struct is None:
            return torch.tensor(0.0, device=camids.device)
        
        loss = 0.0
        count = 0
        
        # 获取当前 Batch 中所有唯一的 PID
        unique_pids = torch.unique(pids)
        
        for pid in unique_pids:
            # 找到当前 PID 的所有样本索引
            idxs = (pids == pid)
            
            # 分离两个模态的索引
            # 假设 camid 0 是 Visible/Optical, 1 是 SAR/Infrared
            idx_v = idxs & (camids == 0)
            idx_t = idxs & (camids == 1) # 或者 >0
            
            # 只有当该 ID 同时存在于两个模态时才计算一致性
            if idx_v.sum() > 0 and idx_t.sum() > 0:
                # 提取特征
                feat_v = f_struct[idx_v]
                feat_t = f_struct[idx_t]
                
                # 计算中心 (Centroid)
                center_v = feat_v.mean(dim=0)
                center_t = feat_t.mean(dim=0)
                
                # 约束两个模态的结构中心一致
                loss += self.mse(center_v, center_t)
                count += 1
                
        if count > 0:
            loss = loss / count
        else:
            # 如果 Batch 里没有跨模态对，返回 0
            loss = torch.tensor(0.0, device=camids.device)
        return loss

