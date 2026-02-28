import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from models.utils import inflate
from models.utils import c3d_blocks
from models.utils import nonlocal_blocks
import torch
from models.snn_model import SNNModule
import torch.utils.model_zoo as model_zoo
from models.resnet18 import *
from models.PFSP import PFSP
from torchvision import models
import math
from torchvision.transforms.functional import _get_inverse_affine_matrix
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

class STE(nn.Module):

    def __init__(self, seq_len=8):
        super(STE, self).__init__()
        # self.an_model = ANmodel()
        # self.in_planes = 2048
        self.base = ResNet()

        init_pretrained_weight(self.base, model_urls['resnet18'])
        print('Loading pretrained ImageNet model ......')

        # self.seq_len = seq_len
        # self.num_classes = num_classes
        # self.plances = 1024
        # self.mid_channel = 256
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x, pids=None, camid=None):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        # x = x.view(b * t, c, w, h)
        x = self.base(x)  # (b * t, c, 16, 8)
        # w = feat_map.size(2)
        # h = feat_map.size(3)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.mean(1)
        f = self.bn(x)
        return f

        






__all__ = ['AP3DResNet50', 'AP3DNLResNet50', 'NLResNet50', 'C2DResNet50', 
           'I3DResNet50', 
          ] 


class Bottleneck3D(nn.Module):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super().__init__()
        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet503D(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        # self.conv1 = inflate.inflate_conv(nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), time_dim=1) 
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        # spatial max pooling
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        # temporal avg pooling
        x = x.mean(1)
        f = self.bn(x)

        return f




class ResNet503D_event(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 
        
        # self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        
        self.conv1 = inflate.inflate_conv(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), time_dim=1) 
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        # spatial max pooling
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        # temporal avg pooling
        x = x.mean(1)
        f = self.bn(x)

        return f






class ResNet503D_RGB(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 
        
        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        
        # self.conv1 = inflate.inflate_conv(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), time_dim=1) 
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        # self.bn = nn.BatchNorm1d(2048)
        # init.normal_(self.bn.weight.data, 1.0, 0.02)
        # init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        return x





class ResNet503D_RGB_seg(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 
        self.seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
        for param in self.seg_model.parameters():
            param.requires_grad = False
        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        
        # self.conv1 = inflate.inflate_conv(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), time_dim=1) 
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)
        # self.mask_conv1 = nn.Sequential(
        #     nn.Conv3d(1, 256//8, kernel_size=1, bias=False),
        #     nn.BatchNorm3d(256//8),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(256//8, 256, kernel_size=1, bias=False),
        #     nn.Sigmoid()
        # )
        self.mask_conv2 = nn.Sequential(
            nn.Conv3d(1, 512//16, kernel_size=1, bias=False),
            nn.BatchNorm3d(512//16),
            nn.ReLU(inplace=True),
            nn.Conv3d(512//16, 512, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # masks = []
        # for t in range(T):
        #     frame = x[:, :, t, :, :]   
        #     out_t = self.seg_model(frame)['out']   
        #     prob_t = torch.softmax(out_t, dim=1) 
        #     mask_t = prob_t[:, 15:16, :, :] 
        #     masks.append(mask_t.unsqueeze(2))    
        # mask_seq = torch.cat(masks, dim=2)   
        with torch.no_grad():
            x_reshape = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, 3, H, W]
            out = self.seg_model(x_reshape)['out']  # [B*T, 21, H, W]
            prob = torch.softmax(out, dim=1)
            mask = prob[:,15:16]         # [B*T, 1, H, W]

            mask_seq = mask.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        _, _, Tm, Hm, Wm = mask_seq.shape
        mask_bt = mask_seq.permute(0, 2, 1, 3, 4).reshape(B*Tm, 1, Hm, Wm)
        mask_bt = mask_bt.half()  
        
        # target_h1, target_w1 = x.shape[-2:]
        # mask_resized1 = F.interpolate(mask_bt,  size=(target_h1, target_w1), mode='bilinear', align_corners=False)
        # mask_resized1 = mask_resized1.to(dtype=x.dtype)
        # mask_resized1 = mask_resized1.view(B, Tm, 1, target_h1, target_w1).permute(0, 2, 1, 3, 4)
        
        # x = x *  mask_resized * 0.2  + x 
        # mask_resized1 = self.mask_conv1(mask_resized1)
        # x = x * mask_resized1 + x  
        x = self.layer2(x)
        # B, C, T, H, W = mask_seq.shape
        # mask_seq_reshaped2 = mask_seq.permute(0, 2, 1, 3, 4).reshape(B*T, 1, H, W)
        target_h2, target_w2 = x.shape[-2:]
        mask_resized2 = F.interpolate(mask_bt,  size=(target_h2, target_w2), mode='bilinear', align_corners=False)
        mask_resized2 = mask_resized2.to(dtype=x.dtype)
        mask_resized2 = mask_resized2.view(B, Tm, 1, target_h2, target_w2).permute(0, 2, 1, 3, 4)
        
        
        # x = x *  mask_resized * 0.2  + x 
        mask_resized2 = self.mask_conv2(mask_resized2)
        x = x * mask_resized2 + x   
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        # spatial max pooling
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        # temporal avg pooling
        x = x.mean(1)
        f = self.bn(x)
        return f 







class EventEdgeExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(EventEdgeExtractor, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, event):
        B, C, T, H, W = event.shape
        edge_maps = []
        for t in range(T):
            x_t = event[:, :, t, :, :]  # [B, C, H, W]
            edge_t = self.edge_conv(x_t)  # [B, 1, H, W]
            edge_maps.append(edge_t.unsqueeze(2))  # → [B, 1, 1, H, W]

        edge_out = torch.cat(edge_maps, dim=2)  # [B, 1, T, H, W]
        return edge_out
        # return self.edge_conv(event)  # [B, 1, H, W]





class EventGuidedEnhance(nn.Module):
    def __init__(self, channel):
        super(EventGuidedEnhance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel),
            nn.Sigmoid()
        )

    def forward(self, feat_rgb, edge_att):
        B, C, T, H, W = feat_rgb.shape
        if edge_att.dim() == 4:
            edge_att = edge_att.unsqueeze(2)  # → [B, 1, 1, H, W]
            edge_att = edge_att.expand(-1, -1, T, -1, -1)  # → [B, 1, T, H, W]
        elif edge_att.dim() == 5 and edge_att.shape[2] != T:
            raise ValueError(f"edge_att 时间维度 {edge_att.shape[2]} 与 feat_rgb 不匹配：{T}")
        if edge_att.shape[-2:] != (H, W):
            edge_att = edge_att.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 1, *edge_att.shape[-2:])
            edge_att = F.interpolate(edge_att, size=(H, W), mode='bilinear', align_corners=False)
            edge_att = edge_att.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4).contiguous()
        # if feat_rgb.size() != edge_att.size():
        #     edge_att = F.interpolate(edge_att, size=feat_rgb.shape[2:], mode='trilinear', align_corners=False)
        x = feat_rgb * edge_att + feat_rgb
        x = self.conv(x)
        b, c, t, h, w = x.shape
        wei = self.avg_pool(x).view(b, c)
        wei = self.fc(wei).view(b, c, 1, 1, 1)
        return x * wei





class ModalEmbedding(nn.Module):
    def __init__(self, channels):
        """
        Modal Embedding: 注入模态感知信息。
        适用于共享backbone场景下区分RGB和Event输入。
        参数：
            channels: 特征图的通道数（如conv1输出64）
        """
        super(ModalEmbedding, self).__init__()
        self.token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))   # [1, C, T, H, W]
        # self.event_token = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        nn.init.kaiming_normal_(self.token, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.event_token, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: 输入特征 [B, C, T, H, W]
        mode: 'rgb' 或 'event'
        """
        return x + self.token
        # elif mode == 'event':
        #     return x + self.event_token
        # else:
        #     raise ValueError("mode must be 'rgb' or 'event'")






class ResNet503D_EVENT(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 
        
        # self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        
        self.conv1 = inflate.inflate_conv(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), time_dim=1) 
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
         
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        # self.bn = nn.BatchNorm1d(2048)
        # init.normal_(self.bn.weight.data, 1.0, 0.02)
        # init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        return x











class ResNet503D_event_(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 
        

        self.event_conv0 = inflate.inflate_conv(nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True), time_dim=1)
        # self.conv1 = inflate.inflate_conv(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), time_dim=1) 
        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x_rgb = x[:, :3, :, :, :]     # [B, 3, 8, H, W]
        x_event = x[:, 3:, :, :, :]   # [B, 2, 8, H, W]
        x_event = self.event_conv0(x_event) # [B, 3, 8, H, W]
        x = torch.cat((x_rgb, x_event), dim=2) 
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        # spatial max pooling
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        # temporal avg pooling
        x = x.mean(1)
        f = self.bn(x)
        return f





















def C2DResNet50(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def AP3DResNet50(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def I3DResNet50(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(config, c3d_blocks.I3D, c3d_idx, nl_idx, **kwargs)


def AP3DNLResNet50(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def NLResNet50(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)









class DualStreamNet(nn.Module):
    def __init__(self, config, rgb_ckpt=True, event_ckpt=True):
        super(DualStreamNet, self).__init__()
        # 两个独立的 3D ResNet 编码器（分别处理 RGB 和事件）

        self.event_skel = EventStructureExtractor(K=17, feat_dim=256)
        self.fc_gate    = nn.Sequential(nn.Linear(256, 512),
                                        nn.SiLU())        
        self.rgb_encoder =  rgb_exactor(config)
        self.event_encoder = eve_exactor(config)
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.beta_s = nn.Parameter(torch.tensor(0.05))
        self.spa_tau = nn.Parameter(torch.tensor(0.15))

        for name, param in self.rgb_encoder.named_parameters():
            if not any(k in name for k in [ 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                param.requires_grad = False

        for name, param in self.event_encoder.named_parameters():
            if not any(k in name for k in [ 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                param.requires_grad = False
        self.rgb_encoder_low = nn.Sequential(
            self.rgb_encoder.conv1,
            self.rgb_encoder.bn1,
            self.rgb_encoder.relu,
            self.rgb_encoder.maxpool,
            self.rgb_encoder.layer1
        )
        self.rgb_encoder_mid = self.rgb_encoder.layer2
        self.rgb_encoder_high = nn.Sequential(
            self.rgb_encoder.layer3,
            self.rgb_encoder.layer4
        )
        
        self.event_encoder_low = nn.Sequential(
            self.event_encoder.conv1,
            self.event_encoder.bn1,
            self.event_encoder.relu,
            self.event_encoder.maxpool,
            self.event_encoder.layer1
        )
        self.event_encoder_mid = self.event_encoder.layer2
        self.event_encoder_high = nn.Sequential(
            self.event_encoder.layer3,
            self.event_encoder.layer4
        )

        self.bn_final = DualBNNeck(2048 * 8)
        self.bn_final_rgb = DualBNNeck(2048 )
        self.bn_final_event = DualBNNeck(2048 )     

        self.safl = PFSP()


     
    def forward(self, x):
        # x: [B, C=5, T=16, H, W]

        
        x_rgb = x[:, :3, :, :, :]     # [B, 3, 8, H, W]
        x_event = x[:, 3:, :, :, :]   # [B, 2, 8, H, W]
        z_s, h_map ,coords   = self.event_skel(x_event) 
        x_rgb_low = self.rgb_encoder_low(x_rgb)
        x_rgb_mid = self.rgb_encoder_mid(x_rgb_low)
        x_event_low = self.event_encoder_low(x_event)
        x_event_mid = self.event_encoder_mid(x_event_low)
        # gate = torch.sigmoid(self.fc_gate(z_s)).view(-1, 512, 1, 1, 1)
        gate = torch.sigmoid(self.fc_gate(z_s)).view(-1, 512, 1, 1, 1)
        x_rgb_mid = x_rgb_mid * (1 + self.beta * gate) 

        # x_rgb_mid = x_rgb_mid * gate 
        B, _, Tm, Hm, Wm = x_rgb_mid.shape
        K = h_map.size(1)
        h_up = F.interpolate(h_map, size=(Tm, Hm, Wm), mode='trilinear', align_corners=False)
        spa_logits = h_up.sum(dim=1, keepdim=True)   
        spa_flat   = spa_logits.flatten(start_dim=3) 
        tau = torch.clamp(self.spa_tau.abs(), min=1e-3)
        spa_norm   = torch.softmax(spa_flat / tau, dim=-1)
        A_spa      = spa_norm.view(B, 1, Tm, Hm, Wm)    
        x_rgb_mid = x_rgb_mid * (1 + self.beta_s * A_spa)
        f_rgb = self.rgb_encoder_high(x_rgb_mid)

        f_event = self.event_encoder_high(x_event_mid)

        b, c, t, h, w = f_rgb.size()
        f_rgb = f_rgb.permute(0, 2, 1, 3, 4).contiguous()

        f_event = f_event.permute(0, 2, 1, 3, 4).contiguous()

        feat_rgb, attn_rgb = self.safl(f_rgb, f_event)

        f_rgb = torch.mean(f_rgb,1)
        f_rgb = f_rgb.view(b, c, h, w)
        f_event = torch.mean(f_event,1)
        f_event = f_event.view(b, c, h, w)        
     
        global_rgb =  f_rgb.mean(dim=(2,3))
 
        global_event =  f_event.mean(dim=(2,3))
        feats_rgb = torch.cat([feat_rgb, global_event, global_rgb], dim=1)

        if self.training:
            masks_rgb = attn_rgb.view(b, 6, w*h)
            loss_dp_rgb = 0 
            for i in range(6):
                for j in range(i+1, 6):
                    loss_dp_rgb += ((((masks_rgb[:, i] - masks_rgb[:, j]) ** 2).sum(dim=1) /(16 * 8)) ** 0.5).sum()
            loss_dp_rgb = - loss_dp_rgb / (b * 6 * (6 - 1) / 2)
            loss_dp_rgb *= 1.0
            loss_edge = edge_alignment_loss(h_map, x_event, weight=0.01)
            # loss_aff= affine_consistency_loss(h_map, soft_argmax, weight=0.01,
            #                         max_deg=10.0, scale_range=(0.9,1.1), max_translate=0.05)

        if not self.training:
            feats = self.bn_final(feats_rgb)
            torch.cuda.empty_cache()
            return feats
        else:
            feats = self.bn_final(feats_rgb)
            feats_rgb = self.bn_final_rgb(global_rgb)
            feats_event = self.bn_final_event(global_event)  
            torch.cuda.empty_cache()
            # return feats, feats_rgb, feats_event, loss_dp_rgb, loss_edge, loss_aff
            return feats, feats_rgb, feats_event, loss_dp_rgb, loss_edge
            # return feats, feats_rgb, feats_event, loss_dp_rgb



    

    def _load_weights(self, model, ckpt_path):
        print(f"正在加载预训练权重：{ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # 1. 判断是否是完整模型（包含 'model_state_dict'）
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # 否则就是纯 state_dict

        # 2. 去除多卡保存时的 module. 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "")
            new_state_dict[k_clean] = v

        # 3. 加载参数（建议 strict=False 更稳妥）
        model.load_state_dict(new_state_dict, strict=False)
        return model




class EdgeConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, depth_mul: int = 1):
        super().__init__()


        self.dw = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch * depth_mul,
            kernel_size=(1,3,3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            groups=in_ch,        # ★ 修正：真正的depthwise
            bias=False)
        self.pw  = nn.Conv3d(
            in_channels=in_ch * depth_mul,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,   
            padding=0,         # ★ 修正：PW不下采样
            bias=False)
        

        # self.dw = inflate.inflate_conv(conv2d_dw ,  time_dim=1) 
        # self.pw = inflate.inflate_conv(conv2d_pw, time_dim=1)
        self.bn1 = nn.BatchNorm3d(in_ch * depth_mul)
        
        # ② Pointwise 1×1×1

        # self.pw = nn.Conv3d(in_ch * depth_mul, out_ch,
        #                     kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True)

        self.use_res = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        res = x
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        if self.use_res:
            x = x + res
        return x



class EventStructureExtractor(nn.Module):
    def __init__(self, K=17, feat_dim=256):
        super().__init__()
        self.backbone = EdgeConv3D(in_ch=2, out_ch=64)

        conv2d_hmap_head = nn.Conv2d(
            in_channels=64,
            out_channels=K,      # depth multiplier
            kernel_size=1,
               )
        self.hmap_head = inflate.inflate_conv(conv2d_hmap_head,  time_dim=1) 
        # self.hmap_head = nn.Conv3d(64, K, 1)
        # self.gcn       = SkeletonGCN(K, feat_dim)
        self.gcn = DynAdjLiteGCN(K, feat_dim, topk=4, alpha_I=0.2)

    def forward(self, ev):                          # [B,2,T,H,W]
        f = self.backbone(ev)                       # [B,64,T',H',W']
        h = self.hmap_head(f)                       # [B,K,T',H',W']
        h = spatial_softmax(h)                      # 每通道一峰
        coords = soft_argmax(h)                     # [B,T',K,2]
        z_s = self.gcn(coords)                      # [B,feat_dim]
        return z_s, h, coords

def spatial_softmax(h):
    # h: [B,K,T,H,W] → flatten spatial dims
    B,K,T,H,W = h.shape
    h = h.reshape(B,K,T,-1)
    h = torch.softmax(h, dim=-1)
    return h.view(B,K,T,H,W)



def soft_argmax(h):
    # h 已做 softmax，返回 (x,y) ∈ [0,1] 坐标
    B,K,T,H,W = h.shape
    h = h.permute(0,2,1,3,4)      # [B,T,K,H,W]
    grids_y, grids_x = torch.meshgrid(
        torch.linspace(0,1,H,device=h.device),
        torch.linspace(0,1,W,device=h.device),
        indexing='ij'
    )
    coords_x = (h*grids_x).sum([-2,-1])   # [B,T,K]
    coords_y = (h*grids_y).sum([-2,-1])
    coords   = torch.stack([coords_x, coords_y], dim=-1)  # [B,T,K,2]
    return coords




class DynAdjLiteGCN(nn.Module):
    def __init__(self, K=17, out_dim=256, hid=128, topk=4, alpha_I=0.2,
                 tau_min=0.05, tau_max=0.5):
        super().__init__()
        self.topk, self.alpha_I = topk, alpha_I
        self.tau_raw = nn.Parameter(torch.tensor(0.0))
        self.tau_min, self.tau_max = tau_min, tau_max
        self.phi  = nn.Linear(2, hid, bias=False)   # 节点编码
        self.proj = nn.Linear(2, hid, bias=False)   # 残差分支
        self.head = nn.Linear(2*hid, out_dim)       # mean+max 读出

    def _tau(self):
        s = torch.sigmoid(self.tau_raw)
        return self.tau_min + (self.tau_max - self.tau_min) * s

    def forward(self, coords):                # [B,T,K,2] ∈ [0,1]
        x = coords.mean(1)                    # ★ 保留旧版：时序平均
        # 平移/尺度归一，避免相机/场景尺度影响
        mu  = x.mean(1, keepdim=True)
        std = x.std (1, keepdim=True) + 1e-6
        x_n = (x - mu) / std                  # [B,K,2]

        tau = self._tau()
        d2  = torch.cdist(x_n, x_n, p=2.0) ** 2     # [B,K,K]
        A   = torch.softmax(-d2 / tau, dim=-1)      # RBF 相似度

        # Top‑k 行稀疏
        if self.topk is not None and self.topk < x.size(1):
            v = A.topk(self.topk, dim=-1).values[..., -1:].detach()
            mask = (A >= v)
            A = torch.softmax(A.masked_fill(~mask, float('-inf')), dim=-1)

        # 与 I 混合，防止过平滑
        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A = self.alpha_I * I + (1 - self.alpha_I) * A

        h0 = self.phi(x_n)                    # [B,K,H]
        z  = torch.relu(A @ h0) + self.proj(x_n)     # 单次传播 + 残差

        readout = torch.cat([z.mean(1), z.max(1).values], dim=-1)
        return self.head(readout)             # [B,out_dim]





def sobel_edge_batch(x_event: torch.Tensor) -> torch.Tensor:
    """
    x_event : [B, 2, T, H, W]  → edge map [B, 1, T, H, W]  ∈ [0,1]
    先把正负极性累加成单通道，再做 2D Sobel。
    """
    B, C, T, H, W = x_event.shape
    # 1) 合并极性通道，保留时序
    x = x_event.sum(1)                           # [B,T,H,W]
    x = x.reshape(B*T, 1, H, W)

    # 2) Sobel kernel（固定参数的卷积）
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                           dtype=x.dtype, device=x.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)
    edge_x  = F.conv2d(x, sobel_x, padding=1)
    edge_y  = F.conv2d(x, sobel_y, padding=1)
    edge    = torch.sqrt(edge_x**2 + edge_y**2) # 梯度幅值

    # 3) 归一化到 0-1
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-6)
    edge = edge.view(B, 1, T, H, W)
    return edge.detach()                         # 不回传梯度





def edge_alignment_loss(h_map: torch.Tensor,
                        x_event: torch.Tensor,
                        weight: float = 1.0,
                        detach_event: bool = True) -> torch.Tensor:
    """
    h_map:   [B,K,T,H,W]  (softmax 后的热图, 每 (t,k) 在 HxW 上求和为1)
    x_event: [B,2,T,H0,W0]  (事件正/负极性)
    return:  标量损失（已乘以 weight）
    目标：最大化热图与事件边缘重合 → 用负号最小化
    """
    assert h_map.dim() == 5 and x_event.dim() == 5
    B, K, T, H, W = h_map.shape

    # --- 事件边缘图 E: [B,1,T,H0,W0] ∈ [0,1] ---
    # 你项目中已有 sobel_edge_batch(x_event)；这里假设它存在并返回 [B,1,T,H0,W0]
    E = sobel_edge_batch(x_event)  # 无需梯度
    if detach_event:
        E = E.detach()

    # 下采样/对齐到热图分辨率
    # trilinear 不会引入莫名的数值爆；align_corners=False 更稳
    E = F.interpolate(E, size=(T, H, W), mode='trilinear', align_corners=False)

    # 广播到 K 通道，与 h_map 点乘；取负号最小化 => 等价于最大化重合
    loss = -(h_map * E).mean()
    return weight * loss




def _build_affine_matrices(B: int,
                           max_deg: float = 10.0,
                           scale_range=(0.9, 1.1),
                           max_translate: float = 0.05,
                           device=None, dtype=None):
    """
    返回前向与逆向二维仿射矩阵（在归一化坐标系 [-1,1] 下）：
    A_fwd: [B, 2, 3]   - 用于把坐标点从原图映射到变换后
    A_inv: [B, 2, 3]   - A_fwd 的逆，用于 grid_sample 采样热图
    """
    device = device or torch.device('cpu')
    dtype = dtype or torch.float32

    # 采样旋转、缩放、平移（归一化坐标：±1 覆盖整幅图）
    theta = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 2 * (max_deg * math.pi / 180.0)
    scale = torch.empty(B, device=device, dtype=dtype).uniform_(scale_range[0], scale_range[1])
    tx = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 2 * max_translate
    ty = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 2 * max_translate

    cos_t = torch.cos(theta) * scale
    sin_t = torch.sin(theta) * scale

    # 前向矩阵（把原坐标映射到新坐标）：[ [a, b, tx], [c, d, ty] ]
    # 注意这是在 [-1,1] 坐标系中定义的刚性+缩放+平移
    a = cos_t; b = -sin_t
    c = sin_t; d =  cos_t
    A_fwd = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    A_fwd[:, 0, 0] = a
    A_fwd[:, 0, 1] = b
    A_fwd[:, 1, 0] = c
    A_fwd[:, 1, 1] = d
    A_fwd[:, 0, 2] = tx
    A_fwd[:, 1, 2] = ty

    # 求逆（转成 3x3 齐次再求逆）
    A33 = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    A33[:, :2, :3] = A_fwd
    A33[:, 2, 2] = 1.0
    A33_inv = torch.inverse(A33)
    A_inv = A33_inv[:, :2, :3]  # grid_sample 使用逆变换

    return A_fwd, A_inv


def affine_consistency_loss(h_map: torch.Tensor,
                            soft_argmax_fn,
                            weight: float = 1.0,
                            max_deg: float = 10.0,
                            scale_range=(0.9, 1.1),
                            max_translate: float = 0.05) -> torch.Tensor:
    """
    h_map:         [B,K,T,H,W] (softmax 后)
    soft_argmax_fn: 可调用对象，接受 h:[B,K,T,H,W]，返回 coords:[B,T,K,2] (∈[0,1])
    return:        标量损失（已乘以 weight）
    目标：对小幅二维仿射等变（equivariance）
    """
    assert h_map.dim() == 5
    B, K, T, H, W = h_map.shape
    device, dtype = h_map.device, h_map.dtype

    # ---- 1) 生成前/逆仿射（在归一化坐标）----
    A_fwd, A_inv = _build_affine_matrices(
        B, max_deg=max_deg, scale_range=scale_range, max_translate=max_translate,
        device=device, dtype=torch.float32  # 用 FP32 生成更稳
    )
    A_inv = A_inv.to(dtype)

    # ---- 2) 用逆仿射采样热图 → h_warp: [B,K,T,H,W] ----
    # grid_sample 支持 4D/5D。我们只做空间仿射（T 不变），所以把 B*T 视作 batch。
    h_bt = h_map.reshape(B*T, K, H, W)
    # 构建网格（每个样本的逆仿射），注意：grid_sample 的网格是 [N,H,W,2]
    grid_list = []
    for b in range(B):
        # 对每个样本，构建 HxW 的采样网格
        theta = A_inv[b:b+1]  # [1,2,3]
        grid = F.affine_grid(theta, size=(1, K, H, W), align_corners=False)  # [1,H,W,2]
        grid_list.append(grid)
    grid = torch.cat(grid_list, dim=0)                      # [B,H,W,2]
    grid = grid.unsqueeze(1).repeat(1, T, 1, 1, 1)          # [B,T,H,W,2]
    grid = grid.view(B*T, H, W, 2)                          # [B*T,H,W,2]

    h_warp_bt = F.grid_sample(h_bt, grid, mode='bilinear', padding_mode='zeros',
                              align_corners=False)
    h_warp = h_warp_bt.view(B, K, T, H, W)

    # ---- 3) 软坐标：原始 coords1 与 变换后 coords2 ----
    coords1 = soft_argmax_fn(h_map)    # [B,T,K,2]  in [0,1]
    coords2 = soft_argmax_fn(h_warp)   # [B,T,K,2]  in [0,1]

    # ---- 4) 用 "前向仿射" 变换 coords1 到新坐标系，与 coords2 对齐 ----
    # 坐标从 [0,1] 映射到 [-1,1]：u = 2x-1, v = 2y-1
    uv1 = coords1.clone()
    uv1 = uv1 * 2.0 - 1.0   # [B,T,K,2] in [-1,1]

    # 批量仿射： [B,2,3] @ [B,T,K,3,1]
    ones = torch.ones(B, T, K, 1, device=device, dtype=uv1.dtype)
    uv1_h = torch.cat([uv1, ones], dim=-1)           # [B,T,K,3]
    # 展开成矩阵乘法
    A = A_fwd.to(uv1.dtype).unsqueeze(1).unsqueeze(2)  # [B,1,1,2,3]
    uv1_new = torch.matmul(A, uv1_h.unsqueeze(-1)).squeeze(-1)  # [B,T,K,2]

    # 回到 [0,1]
    coords1_warped = (uv1_new + 1.0) * 0.5            # [B,T,K,2]

    # ---- 5) L1 一致性 ----
    loss = F.l1_loss(coords1_warped, coords2)
    return weight * loss







def DualStream(config):
    return DualStreamNet(config)



def C2DResNet50_event(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D_event_(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def C2DResNet50_event1(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D_event(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)

def resnet18_event():
    return ResNet503D_RGB_seg(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)
def rgb_exactor_seg(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D_RGB_seg(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)

def eve_exactor(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D_EVENT(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def rgb_exactor(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D_RGB(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def rgb_exactor_seg(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D_RGB_seg(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)








class DualBNNeck(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.bn_neck_i = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn_neck_i.bias, 0) 
        self.bn_neck_i.bias.requires_grad_(False)

    def forward(self, x):


        x= self.bn_neck_i(x)

        return x
