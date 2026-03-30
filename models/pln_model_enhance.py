import torch
import torch.nn as nn
import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .backbone import ResNet18Backbone
    from .pln_head import PLNHead
except ImportError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.backbone import ResNet18Backbone
    from models.pln_head import PLNHead


class PLNModel(nn.Module):
    """
    PLN (Parsing-based Location Network) Model.
    
    Architecture:
    Input (B, 3, 448, 448) 
        → ResNet18 Backbone (with projection) → (B, 3328, 14, 14)
        → Conv Layer (1x1, 3x3, 3x3 stack) → (B, 1536, 14, 14)
        → 4 Branches (left-top, right-top, left-bot, right-bot)
        → Center Feature Synchronization (Feature Consensus)
        → Output
    """
    
    def __init__(self, backbone_pretrained=False, in_channels=3328, 
                 conv_out_channels=1536, head_out_channels=204):
        super(PLNModel, self).__init__()
        
        # Backbone
        self.backbone = ResNet18Backbone(pretrained=backbone_pretrained)
        
        # Intermediate conv layer (before branches)
        # Match paper: 3328 -> 1536 via 1x1, 3x3, 3x3 convs
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4 branches
        self.branch_left_top = PLNHead(in_channels=conv_out_channels)
        self.branch_right_top = PLNHead(in_channels=conv_out_channels)
        self.branch_left_bot = PLNHead(in_channels=conv_out_channels)
        self.branch_right_bot = PLNHead(in_channels=conv_out_channels)
        
        self.branch_names = ['left_top', 'right_top', 'left_bot', 'right_bot']

        # 定义需要进行“四分支同步(平均)”的通道索引
        # 预设条件：前102维为两个中心点，每个点51维
        # 点1 (0-50): 0(置信度), 1-2(坐标), 3-30(Link), 31-50(类别)
        # 点2 (51-101): 51(置信度), 52-53(坐标), 54-81(Link), 82-101(类别)
        self.sync_channels = (
            list(range(0, 3)) +     # 点1: 置信度 + 坐标 (3个通道)
            list(range(31, 51)) +   # 点1: 类别 (20个通道)
            list(range(51, 54)) +   # 点2: 置信度 + 坐标 (3个通道)
            list(range(82, 102))    # 点2: 类别 (20个通道)
        )
        # 共计同步 46 个通道，剩下 56 个通道 (两个点的Link) 保持独立
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, 448, 448)
        
        Returns:
            Dictionary containing:
                - backbone_features: (B, 512, 14, 14)
                - branch_features: Dict with (B, 204, 14, 14) features from each branch
        """
        # Backbone & Intermediate
        backbone_features = self.backbone(x)  # (B, 512, 14, 14)
        x = self.conv(backbone_features)      # (B, 338, 14, 14)
        
        # 独立获取四个分支的初步输出 (B, 204, 14, 14)
        out_lt = self.branch_left_top(x)
        out_rt = self.branch_right_top(x)
        out_lb = self.branch_left_bot(x)
        out_rb = self.branch_right_bot(x)
        

        # 1. 堆叠四个分支的输出，形状: (4, B, 204, 14, 14)
        stacked_outs = torch.stack([out_lt, out_rt, out_lb, out_rb], dim=0)
        
        # 2. 仅在指定的 sync_channels 上跨分支求平均
        # mean_features 形状: (B, 46, 14, 14)
        mean_features = stacked_outs[:, :, self.sync_channels, :, :].mean(dim=0)
        
        # 3. 使用 clone() 防止 In-place 报错，将同步后的特征填回各个张量
        out_lt_synced = out_lt.clone()
        out_rt_synced = out_rt.clone()
        out_lb_synced = out_lb.clone()
        out_rb_synced = out_rb.clone()
        
        out_lt_synced[:, self.sync_channels, :, :] = mean_features
        out_rt_synced[:, self.sync_channels, :, :] = mean_features
        out_lb_synced[:, self.sync_channels, :, :] = mean_features
        out_rb_synced[:, self.sync_channels, :, :] = mean_features
        
        branch_features = {
            'left_top': out_lt_synced,
            'right_top': out_rt_synced,
            'left_bot': out_lb_synced,
            'right_bot': out_rb_synced
        }
        
        return {
            'backbone_features': backbone_features,
            'branch_features': branch_features
        }


def build_model(backbone_pretrained=False):
    model = PLNModel(backbone_pretrained=backbone_pretrained)
    return model


if __name__ == '__main__':
    # Test the model
    model = build_model()
    dummy_input = torch.randn(2, 3, 448, 448)
    output = model(dummy_input)
    
    print("✓ Model output structure:")
    print(f"  backbone_features: {output['backbone_features'].shape}")
    print(f"  branch_features:")
    for branch_name, features in output['branch_features'].items():
        print(f"    {branch_name}: {features.shape}")