import torch
import torch.nn.functional as F

class CrossModalContrastiveLoss(torch.nn.Module):
    """
    跨模态对比损失（NT-Xent），支持单向或双向对比。
    - f_rgb: [B, D]，RGB 模态特征
    - f_event: [B, D]，Event 模态特征
    """
    def __init__(self, temperature=0.1, symmetric=True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, f_rgb, f_event):
        f_rgb = F.normalize(f_rgb, dim=1)
        f_event = F.normalize(f_event, dim=1)

        logits = torch.mm(f_rgb, f_event.T) / self.temperature
        labels = torch.arange(f_rgb.size(0)).to(f_rgb.device)

        loss_rgb2event = F.cross_entropy(logits, labels)

        if self.symmetric:
            logits_T = torch.mm(f_event, f_rgb.T) / self.temperature
            loss_event2rgb = F.cross_entropy(logits_T, labels)
            return (loss_rgb2event + loss_event2rgb) / 2
        else:
            return loss_rgb2event


class CosineDecorrelateLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f2, f3):
        f2 = F.normalize(f2, dim=1)
        f3 = F.normalize(f3, dim=1)
        sim = torch.sum(f2 * f3, dim=1)
        return torch.mean(sim ** 2)