import torch
import torch.nn as nn
from .backbone import InceptionResNetV2Backbone
from .pln_head import PLNHead

class PLNModel(nn.Module):
    def __init__(self, backbone_pretrained=False, conv_out_channels=1536, head_out_channels=204):
        super(PLNModel, self).__init__()
        
        # 1. Backbone
        self.backbone = InceptionResNetV2Backbone(pretrained=backbone_pretrained)
        
        # 2. 四个功能分支
        # Inception的末端已经输出了 1536 维度的特征，因此直接作为输入
        self.branch_names = ['left_top', 'right_top', 'left_bot', 'right_bot']
        self.branches = nn.ModuleList([
            PLNHead(in_channels=conv_out_channels, out_channels=head_out_channels) 
            for _ in range(4)
        ])

    def forward(self, x):
        backbone_features = self.backbone(x)
        shared_features = backbone_features
        
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