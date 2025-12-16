import torch.nn.functional as F
import torch


def orthogonality_loss(f_shared, f_spec):
    # 归一化
    f_shared = F.normalize(f_shared, p=2, dim=1)
    f_spec = F.normalize(f_spec, p=2, dim=1)
    # 计算余弦相似度矩阵
    # 我们希望同一个样本的 shared 和 spec 向量正交 (点积接近 0)
    cosine = torch.sum(f_shared * f_spec, dim=1)
    loss = torch.mean(torch.abs(cosine))
    return loss
