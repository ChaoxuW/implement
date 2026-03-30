import torch
import torch.nn as nn
from .backbone import InceptionResNetV2Backbone
from .pln_head import PLNHead

class PLNModel(nn.Module):
    """
    改进版 PLN 模型 (InceptionResNetV2 + 特征同步)
    
    改进点：
    在 forward 过程中，对四个分支输出的“中心点”相关通道进行跨分支平均（Mean Pooling），
    强制模型在中心点的预测上达成共识，从而增强坐标稳定性。
    """
    def __init__(self, backbone_pretrained=False, conv_out_channels=1536, head_out_channels=204):
        super(PLNModel, self).__init__()
        
        # 1. Backbone: 使用强大的 InceptionResNetV2
        self.backbone = InceptionResNetV2Backbone(pretrained=backbone_pretrained)
        
        # 2. 四个功能分支
        # Inception 的输出通道数通常为 1536
        self.branch_names = ['left_top', 'right_top', 'left_bot', 'right_bot']
        self.branches = nn.ModuleList([
            PLNHead(in_channels=conv_out_channels, out_channels=head_out_channels) 
            for _ in range(4)
        ])

        # 3. 定义需要同步的通道索引 (共 46 个通道)
        # 点1 (0-50): 0(置信度), 1-2(坐标), 31-50(类别) -> 共 23 维
        # 点2 (51-101): 51(置信度), 52-53(坐标), 82-101(类别) -> 共 23 维
        sync_idx = (
            list(range(0, 3)) +      # Point 1: Conf + Coord
            list(range(31, 51)) +    # Point 1: Classes
            list(range(51, 54)) +    # Point 2: Conf + Coord
            list(range(82, 102))     # Point 2: Classes
        )
        
        self.register_buffer('sync_idx', torch.tensor(sync_idx, dtype=torch.long))

    def forward(self, x):
        # 提取特征
        backbone_features = self.backbone(x)
        shared_features = backbone_features
        
        # 1. 获取四个分支的原始输出
        # 每个输出形状: (B, 204, 14, 14)
        outs = [branch(shared_features) for branch in self.branches]
        
        # 2. 堆叠分支进行跨分支计算: (4, B, 204, H, W)
        stacked_outs = torch.stack(outs, dim=0)
        
        # 3. 计算 4 个分支在中心点共享通道上的平均值
        # mean_shared 形状: (B, 46, H, W)
        mean_shared = stacked_outs[:, :, self.sync_idx, :, :].mean(dim=0)
        
        # 4. 将平均后的特征同步回四个分支
        # 使用 clone() 确保不会发生梯度覆盖错误
        synced_outs = stacked_outs.clone()
        
        # 利用 PyTorch 的高级索引将均值广播到所有 4 个分支
        # synced_outs[:, ...] 的第一个维度是 4，mean_shared 只有 B，会自动广播
        synced_outs[:, :, self.sync_idx, :, :] = mean_shared
        
        # 5. 构造输出字典
        branch_features = {
            'left_top': synced_outs[0],
            'right_top': synced_outs[1],
            'left_bot': synced_outs[2],
            'right_bot': synced_outs[3]
        }
        
        return {
            'backbone_features': backbone_features,
            'branch_features': branch_features
        }

def build_model(backbone_pretrained=False):
    return PLNModel(backbone_pretrained=backbone_pretrained)

if __name__ == '__main__':
    # 快速测试代码
    model = build_model(backbone_pretrained=False)
    test_input = torch.randn(1, 3, 448, 448)
    output = model(test_input)
    
    print("✓ 改进版 Inception 模型输出结构:")
    print(f"  Backbone features shape: {output['backbone_features'].shape}")
    for name, feat in output['branch_features'].items():
        print(f"  Branch {name} shape: {feat.shape}")