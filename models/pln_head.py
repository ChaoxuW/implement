import torch
import torch.nn as nn
import torch.nn.functional as F


class PLNHead(nn.Module):
    """
    Single branch head for PLN model.
    Input: (B, 1536, 14, 14) from shared conv layer
    
    Structure:
    Match `net.py` branch and expand_Cov:
    Conv (1536→1536) → Conv (1536→204) → Sequential Dilation Convs (rates: 2, 2, 4, 8, 16, 1, 1)
    Output: (B, 204, 14, 14)
    """
    
    def __init__(self, in_channels=1536, out_channels=204):
        super(PLNHead, self).__init__()
        
        self.branch = nn.Sequential(
            # Branch initial convs: 1536 → 1536
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # 1536 → 204
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # expand_Cov sequence
            # 1. Dilation 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 2. Dilation 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 3. Dilation 4
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 4. Dilation 8
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 5. Dilation 16
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=16, dilation=16, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 6. Dilation 1 (Normal Conv)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 7. Final Conv (Logits output, so bias=True, no BN, no ReLU)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        # 获取原始logits输出 (B, 204, 14, 14)
        raw_output = self.branch(x)
        
        # 将输出在通道维度上拆分为四个点，每个点51维
        # 每个点的51维结构: [1(存在) + 2(坐标) + 14(行链接) + 14(列链接) + 20(类别)]
        B, C, H, W = raw_output.shape
        # 重塑为 (B, 4, 51, H, W) 以便按点处理
        raw_output_reshaped = raw_output.view(B, 4, 51, H, W)
        
        # 对每个点的特定维度应用sigmoid
        # 索引说明:
        # 0: 存在概率 (1维) -> sigmoid
        # 1:3: 坐标偏移 (2维) -> sigmoid
        # 3:31: 链接预测 (28维) -> 保持线性，后续用softmax
        # 31:51: 类别概率 (20维) -> sigmoid
        processed_output = raw_output_reshaped.clone()
        
        # 对存在概率(第0维)应用sigmoid
        processed_output[:, :, 0:1, :, :] = torch.sigmoid(raw_output_reshaped[:, :, 0:1, :, :])

        # 对坐标偏移(第1-2维)应用sigmoid
        processed_output[:, :, 1:3, :, :] = torch.sigmoid(raw_output_reshaped[:, :, 1:3, :, :])

        # 对链接预测分别在行(3:17)和列(17:31)的特征维度(dim=2)应用softmax
        processed_output[:, :, 3:17, :, :] = torch.softmax(raw_output_reshaped[:, :, 3:17, :, :], dim=2)
        processed_output[:, :, 17:31, :, :] = torch.softmax(raw_output_reshaped[:, :, 17:31, :, :], dim=2)

        # 对类别概率(第31-51维)应用sigmoid
        processed_output[:, :, 31:51, :, :] = torch.sigmoid(raw_output_reshaped[:, :, 31:51, :, :])
        # 重塑回原始形状 (B, 204, H, W)
        final_output = processed_output.view(B, C, H, W)
        
        return final_output