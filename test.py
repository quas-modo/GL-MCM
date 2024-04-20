import torch
import torch.nn.functional as F
import numpy as np

def cal_loss_auroc(logits):
    to_np = lambda x: x.data.cpu().numpy()
    sample_num, cate_num = logits.shape
    
    pos_cate_num = cate_num // 2
    neg_cate_num = cate_num // 2

    logits /= 100.0
    logits = to_np(F.softmax(logits, dim=1))
    pos_half = np.max(logits[:, :pos_cate_num], axis=1)
    neg_half = np.max(logits[:, pos_cate_num:], axis=1)

    condition = pos_half < neg_half
    indices = np.where(condition)[0]
    print(indices)
    p = torch.tensor(neg_half[indices])
    print(p.shape)
    if p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5)), 0)

if __name__ == "__main__":
    random_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)
    print(cal_loss_auroc(random_tensor))


