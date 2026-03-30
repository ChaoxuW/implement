import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    """
    ResNet18 Backbone for feature extraction.
    Input: (B, 3, 448, 448)
    Output: (B, 512, 14, 14)
    """
    
    def __init__(self, pretrained=False):
        super(ResNet18Backbone, self).__init__()
        # Load ResNet18
        try:
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None
            resnet18 = models.resnet18(weights=weights)
        except Exception:
            resnet18 = models.resnet18(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # Smooth transition from 512 to 3328
        # Using a sequence of convolutions to gradually increase channels
        # and provide richer feature fusion before hitting the heavy 3328 dimension.
        self.projection = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2048, 3328, kernel_size=1, bias=False),
            nn.BatchNorm2d(3328),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, 448, 448)
        
        Returns:
            features: Feature tensor of shape (B, 3328, 14, 14)
        """
        x = self.backbone(x)
        return self.projection(x)


class InceptionResNetV2Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(InceptionResNetV2Backbone, self).__init__()
        # Stem 448,448,3 --> 53,53,192
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(*[Block35(scale=0.17) for _ in range(10)])
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(*[Block17(scale=0.10) for _ in range(20)])
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(*[Block8(scale=0.20) for _ in range(9)])
        self.block8 = Block8(noReLU=True)
        
        # 维度变换层
        self.conv2d_7b_yy = BasicConv2d(2080, 3328, kernel_size=1, stride=1, padding=1) # 14x14x3328
        self.avgpool_1a = nn.AvgPool2d(1, count_include_pad=False)
        self.conv1 = BasicConv2d(3328, 1536, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1)

        # 如果设置为 True，则自动从官方 URL 下载并加载权重
        if pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        # Cadene 提供的官方 InceptionResNetV2 权重 URL
        url = 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
        print("正在下载 InceptionResNetV2 预训练权重")
        try:
            checkpoint = torch.hub.load_state_dict_from_url(url, progress=True)
            # 过滤掉不匹配的权重（只保留名字对得上的卷积层权重）
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("✓ InceptionResNetV2 官方预训练权重加载成功！")
        except Exception as e:
            print(f"✗ 预训练权重加载失败: {e}")

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b_yy(x)
        x = self.avgpool_1a(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


        
def expand_Cov():
    return nn.Sequential(
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=4, dilation=4),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=8, dilation=8),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=16, dilation=16),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=1),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=1),
    )


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out