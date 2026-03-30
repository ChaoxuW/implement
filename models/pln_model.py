import torch
import torch.nn as nn
from .backbone import ResNet18Backbone
from .pln_head import PLNHead

class PLNModel(nn.Module):
    def __init__(self, backbone_pretrained=False, in_channels=3328, 
                 conv_out_channels=1536, head_out_channels=204):
        super(PLNModel, self).__init__()
        
        # 1. Backbone: ResNet18 + Projection (输出 3328 通道)
        self.backbone = ResNet18Backbone(pretrained=backbone_pretrained)
        
        # 2. 中间共享卷积层: 进一步融合特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. 四个功能分支 (保持原来的命名顺序)
        self.branch_names = ['left_top', 'right_top', 'left_bot', 'right_bot']
        self.branches = nn.ModuleList([
            PLNHead(in_channels=conv_out_channels, out_channels=head_out_channels) 
            for _ in range(4)
        ])

    def forward(self, x):
        backbone_features = self.backbone(x)
        shared_features = self.conv(backbone_features)
        
        # 构造 train.py 期望的字典结构
        branch_features = {
            'left_top': self.branches[0](shared_features),
            'right_top': self.branches[1](shared_features),
            'left_bot': self.branches[2](shared_features),
            'right_bot': self.branches[3](shared_features)
        }
        
        return {
            'backbone_features': backbone_features,
            'branch_features': branch_features
        }

def build_model(backbone_pretrained=False):
    return PLNModel(backbone_pretrained=backbone_pretrained)
