import torch
import torch.nn as nn
import torch.nn.functional as F


class InnovationLoss(nn.Module):
    """
    [融合创新] 综合损失函数
    包含：
    1. MSEL (from PMT): 约束 Shared 特征的模态不变性
    2. Orthogonality (from DEEN): 约束 DEE 特征的多样性
    3. Consistency (from CMT/FACENet): 约束跨模态预测一致性
    """

    def __init__(self, msel_weight=0.5, orth_weight=0.1, cons_weight=0.1):
        super(InnovationLoss, self).__init__()
        self.msel_w = msel_weight
        self.orth_w = orth_weight
        self.cons_w = cons_weight

    def forward(self, feat_shared, dee_feats, logits_list, labels, modal_ids):
        """
        feat_shared: 共享特征 [B, D] (可能为 None)
        dee_feats: DEE扩展出的特征列表 [f1, f2, f3] (可能为 None)
        logits_list: 对应DEE特征的分类logits [l1, l2, l3] (可能为 None)
        labels: 身份标签
        modal_ids: 模态标签 (0/1)
        """
        loss_total = 0.0

        # --- 1. MSEL Loss (PMT) ---
        # 目标：同ID内，RGB-RGB距离 应等于 RGB-SAR距离
        # [修改点] 增加非空判断 feat_shared is not None
        if self.msel_w > 0 and feat_shared is not None:
            loss_msel = self._compute_msel(feat_shared, labels, modal_ids)
            loss_total += self.msel_w * loss_msel

        # --- 2. Orthogonality Loss (DEEN) ---
        # 目标：DEE生成的不同特征之间余弦相似度趋近于0
        if self.orth_w > 0 and dee_feats is not None:
            loss_orth = self._compute_orth(dee_feats)
            loss_total += self.orth_w * loss_orth

        # --- 3. Consistency Loss (CMT/FACENet) ---
        # 目标：对于同一批数据，RGB样本的平均预测分布应与SAR样本的平均预测分布相似
        if self.cons_w > 0 and logits_list is not None:
            # 使用第一个主分支的logits计算
            loss_cons = self._compute_consistency(logits_list[0], modal_ids)
            loss_total += self.cons_w * loss_cons

        return loss_total

    def _compute_msel(self, features, labels, modal_ids):
        # 归一化
        features = F.normalize(features, p=2, dim=1)
        unique_labels = labels.unique()
        loss = 0.0
        count = 0

        for pid in unique_labels:
            idxs = (labels == pid).nonzero(as_tuple=True)[0]
            if len(idxs) < 2:
                continue

            f_rgb = features[idxs[modal_ids[idxs] == 0]]
            f_sar = features[idxs[modal_ids[idxs] == 1]]

            if len(f_rgb) > 0 and len(f_sar) > 0:
                # 跨模态中心距离
                center_rgb = f_rgb.mean(dim=0)
                center_sar = f_sar.mean(dim=0)
                dist_cross = 1.0 - torch.sum(center_rgb * center_sar)

                # 同模态内部平均距离 (简化版：以RGB内部为例)
                if len(f_rgb) > 1:
                    dist_intra = 1.0 - torch.mm(f_rgb, f_rgb.t()).mean()
                    # MSEL核心：惩罚 (跨模态距离 - 同模态距离) 的差异
                    loss += (dist_cross - dist_intra).pow(2)
                    count += 1

        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0).to(features.device)

    def _compute_orth(self, feats):
        loss = 0.0
        n = len(feats)
        norm_feats = [F.normalize(f, p=2, dim=1) for f in feats]
        for i in range(n):
            for j in range(i + 1, n):
                sim = (norm_feats[i] * norm_feats[j]).sum(dim=1).abs().mean()
                loss += sim
        return loss

    def _compute_consistency(self, logits, modal_ids):
        # 基于分布对齐的简化一致性损失
        rgb_logits = logits[modal_ids == 0]
        sar_logits = logits[modal_ids == 1]

        # 增加判断，防止只有单模态数据时报错
        if len(rgb_logits) == 0 or len(sar_logits) == 0:
            return torch.tensor(0.0).to(logits.device)

        prob_rgb = F.softmax(rgb_logits, dim=1).mean(dim=0)
        prob_sar = F.softmax(sar_logits, dim=1).mean(dim=0)

        # JS散度或对称KL散度
        loss = 0.5 * (
            F.kl_div(prob_rgb.log(), prob_sar, reduction="sum")
            + F.kl_div(prob_sar.log(), prob_rgb, reduction="sum")
        )
        return loss
