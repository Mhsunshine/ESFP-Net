from models.vis_heat import DualStreamNet
import argparse
from ./configs.default_vid import get_vid_config
from ./configs.default_img import get_img_config
import ./data.spatial_transforms as ST
import matplotlib.pyplot as plt
import numpy as np
import cv2
VID_DATASET = ['ccvid']


def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config

config = parse_option()
pretrained_dict = torch.load('/root/mxh/result1/experiment/2025-6-23/ccvid/c2dres50-ce-cal/best_model.pth.tar', map_location='cpu')
model = DualStreamNet(config, rgb_ckpt=True, event_ckpt=True)

model.load_state_dict(pretrained_dict['model_state_dict'], strict=False)
model.eval().cuda()


file_paths = ['/root/mxh/data/CCVID/CCVID/session1/001_01/00030.jpg', '/root/mxh/data/CCVID/CCVID/session1/001_01/00031.jpg', '/root/mxh/data/CCVID/CCVID/session1/001_01/00032.jpg', '/root/mxh/data/CCVID/CCVID/session1/001_01/00033.jpg', '/root/mxh/data/CCVID/CCVID/session1/001_01/00034.jpg','/root/mxh/data/CCVID/CCVID/session1/001_01/00035.jpg','/root/mxh/event_CCVID/CCVID/session1/001_01/00030.npy','/root/mxh/event_CCVID/CCVID/session1/001_01/00031.npy','/root/mxh/event_CCVID/CCVID/session1/001_01/00032.npy','/root/mxh/event_CCVID/CCVID/session1/001_01/00033.npy','/root/mxh/event_CCVID/CCVID/session1/001_01/00034.npy','/root/mxh/event_CCVID/CCVID/session1/001_01/00035.npy']
spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


spatial_transform_event = ST.Compose([
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

images = []
for path in file_paths[:6]:  # 前三个是图片
    img = Image.open(path).convert('RGB')  # 确保为RGB三通道
    # img_array = np.array(img)  # 形状为 (H, W, 3) [channels_last]
    images.append(img)
clip_rgb = [spatial_transform_test(img) for img in images]
clip_rgb = torch.stack(clip_rgb, 0).permute(1, 0, 2, 3)

npy_data = []
for path in file_paths[6:]:  # 后两个是npy
    data = np.load(path)      # 假设形状为 (H, W, C) 或 (C, H, W)
    npy_data.append(data)
clip_event = [spatial_transform_event(event) for event in npy_data] 
clip_event = torch.stack(clip_event, 0).permute(1, 0, 2, 3)



clip = torch.cat((clip_rgb, clip_event), dim=0) 

inputs = clip.unsqueeze(0)    

h_map = model(inputs)


B,K,T,H,W = h_map.shape
b, t = 0, 0                  # 选第 b 个样本，第 t 帧
heat = h_map[b,:,t].sum(0)   # K 通道叠加 → [H,W]
heat = (heat / heat.max()).cpu().numpy()

# 事件帧灰度底图（取正极通道）
base = imgs[b,0,t].cpu().numpy()   # [H,W]

# 生成彩色热图
heat_color = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

# α 混合
alpha = 0.5
overlay = (base[...,None]*255*(1-alpha) + heat_color*alpha).astype(np.uint8)

plt.figure(figsize=(6,3))
plt.subplot(1,3,1); plt.imshow(base, cmap='gray'); plt.axis('off'); plt.title('event gray')
plt.subplot(1,3,2); plt.imshow(heat, cmap='jet');   plt.axis('off'); plt.title('heat raw')
plt.subplot(1,3,3); plt.imshow(overlay);            plt.axis('off'); plt.title('overlay')
plt.tight_layout(); plt.show()