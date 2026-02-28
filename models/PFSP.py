import torch
import torch.nn as nn
import torch.nn.functional as F




class PFSP(nn.Module):
    def __init__(self, dim=2048, part_num=6, height=16, width=8, time_len=8, fuse='mean'):
        super().__init__()
        self.part_num = part_num
        self.fuse = fuse
        self.prototypes = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, 2048)))
        self.pos_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(16 * 8, dim)))

        if fuse == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim * part_num, nhead=4, batch_first=True)
            self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # self.norm = nn.LayerNorm(dim)

    def forward(self, x_rgb, x_event):
        """
        x_rgb:   [B, C, T, H, W]
        x_event: [B, C, T, H, W]
        """
        B, T, C, H, W = x_rgb.shape
        P = self.part_num
        HW = H * W

        feat_list = []
        attn = []
        for t in range(T):
            rgb_t = x_rgb[:, t].view(B, C, -1).permute(0, 2, 1)    # [B, HW, C]
            event_t = x_event[:, t].view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

            rgb_t = rgb_t + self.pos_embedding  # 加 spatial 位置
            event_t = event_t + self.pos_embedding

            attn_t = torch.einsum('pc,bkc->bpk', self.prototypes, event_t)  # [B, P, HW]
            attn_t = F.softmax(attn_t, dim=-1)
            fused_t = torch.einsum('bpk,bkc->bpc', attn_t, rgb_t)  # [B, P, C]

            feat_list.append(fused_t)  # 每一帧引导后的 RGB 特征
            attn.append(attn_t)
        feat_seq = torch.stack(feat_list, dim=1)  # [B, T, P, C]
        attn_seq = torch.stack(attn, dim = 1) # [B, T , P, HW]
        attn_out = attn_seq.mean(dim =1)
        if self.fuse == 'mean':
            fused_out = feat_seq.mean(dim=1)  # [B, P, C]
        elif self.fuse == 'transformer':
            fused_out = self.temporal_transformer(feat_seq.view(B, T, -1))  # [B, T, PC]
            fused_out = fused_out.mean(dim=1).view(B, P, C)
        # fused_out = self.norm(fused_out)
        fused_out = fused_out.view(B,-1)
        return fused_out, attn_out  # [B, P, C]