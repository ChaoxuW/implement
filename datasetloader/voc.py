import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class PLNDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=448, S=14, num_classes=20):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.S = S
        self.num_classes = num_classes
        
        # 获取所有标签文件，并对应找到图片，并按字母顺序排序，确保 eval 时的稳定性
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        
        # 图像预处理：只需要 Resize 和 ToTensor，Label因为已经是归一化的，不需要变
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        # 1. 读取 Label
        label_file = self.label_files[idx]
        label_path = os.path.join(self.label_dir, label_file)
        
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    # 格式:   (已归一化到 0~1)
                    bboxes.append([float(x) for x in parts])
        
        # 2. 读取 Image
        # 假设图片格式为 .jpg
        img_name = label_file.replace('.txt', '.jpg') 
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # 3. 生成 4 个分支的 Target [14, 14, 204]
        target_lt = self._build_branch_target(bboxes, branch_type='left_top')
        target_rt = self._build_branch_target(bboxes, branch_type='right_top')
        target_lb = self._build_branch_target(bboxes, branch_type='left_bot')
        target_rb = self._build_branch_target(bboxes, branch_type='right_bot')

        #将真实的 bboxes 填充到固定大小 (例如最大 50 个框)，防止 DataLoader 报错
        max_boxes = 50
        gt_boxes = torch.zeros((max_boxes, 5), dtype=torch.float32)
        num_boxes = min(len(bboxes), max_boxes)
        if num_boxes > 0:
            gt_boxes[:num_boxes] = torch.tensor(bboxes[:num_boxes], dtype=torch.float32)
            
        # 返回值增加 gt_boxes
        return img, (target_lt, target_rt, target_lb, target_rb),gt_boxes

    def _build_branch_target(self, bboxes, branch_type):
        """
        204 = 4个点 (2中心 + 2角点) * 51维
        51维 = 1(存在) + 2(xy偏移) + 14(行链接) + 14(列链接) + 20(类别)
        """
        target = torch.zeros((self.S, self.S, 204), dtype=torch.float32)
        
        # 记录每个 grid 已经填了几个中心点和角点（最多填2个）
        center_counts = torch.zeros((self.S, self.S), dtype=torch.long)
        corner_counts = torch.zeros((self.S, self.S), dtype=torch.long)

        for box in bboxes:
            cls_id = int(box[0])
            # 防止原始数据出现 x_min > x_max 的脏数据，做个保护
            xmin, ymin = min(box[1], box[3]), min(box[2], box[4])
            xmax, ymax = max(box[1], box[3]), max(box[2], box[4])

            # 计算归一化中心点
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

            # 根据分支类型，确定当前需要的角点归一化坐标
            if branch_type == 'left_top':
                cor_x, cor_y = xmin, ymin
            elif branch_type == 'right_top':
                cor_x, cor_y = xmax, ymin
            elif branch_type == 'left_bot':
                cor_x, cor_y = xmin, ymax
            elif branch_type == 'right_bot':
                cor_x, cor_y = xmax, ymax

            # 映射到 S*S 的网格上 (乘以 14)
            cx_grid_float, cy_grid_float = cx * self.S, cy * self.S
            cor_x_grid_float, cor_y_grid_float = cor_x * self.S, cor_y * self.S

            # 获取网格索引 (整数部分)
            cx_g, cy_g = int(cx_grid_float), int(cy_grid_float)
            corx_g, cory_g = int(cor_x_grid_float), int(cor_y_grid_float)

            # 越界保护
            cx_g = max(0, min(cx_g, self.S - 1))
            cy_g = max(0, min(cy_g, self.S - 1))
            corx_g = max(0, min(corx_g, self.S - 1))
            cory_g = max(0, min(cory_g, self.S - 1))

            # 获取网格内偏移量 (小数部分)
            cx_off, cy_off = cx_grid_float - cx_g, cy_grid_float - cy_g
            corx_off, cory_off = cor_x_grid_float - corx_g, cor_y_grid_float - cory_g

            # 获取网格内偏移量 (小数部分)
            cx_off, cy_off = cx_grid_float - cx_g, cy_grid_float - cy_g
            corx_off, cory_off = cor_x_grid_float - corx_g, cor_y_grid_float - cory_g

            # 同步填充中心点和角点
            
            b = max(center_counts[cy_g, cx_g].item(), corner_counts[cory_g, corx_g].item())

            if b < 2:  # B=2，说明这个 grid 还有空余槽位
                # --- 填充中心点 ---
                base_c = b * 51
                target[cy_g, cx_g, base_c + 0] = 1.0                # 0: 是否存在点
                target[cy_g, cx_g, base_c + 1] = cx_off             # 1: x_offset
                target[cy_g, cx_g, base_c + 2] = cy_off             # 2: y_offset
                target[cy_g, cx_g, base_c + 3 + cory_g] = 1.0       # 行链接 (指向角点行)
                target[cy_g, cx_g, base_c + 3 + self.S + corx_g] = 1.0 # 列链接 (指向角点列)
                target[cy_g, cx_g, base_c + 31 + cls_id] = 1.0      # 类别
                
                # --- 填充角点 ---
                base_cor = (2 + b) * 51
                target[cory_g, corx_g, base_cor + 0] = 1.0            # 0: 是否存在点
                target[cory_g, corx_g, base_cor + 1] = corx_off       # 1: x_offset
                target[cory_g, corx_g, base_cor + 2] = cory_off       # 2: y_offset
                target[cory_g, corx_g, base_cor + 3 + cy_g] = 1.0     # 行链接 (指向中心点行)
                target[cory_g, corx_g, base_cor + 3 + self.S + cx_g] = 1.0 # 列链接 (指向中心点列)
                target[cory_g, corx_g, base_cor + 31 + cls_id] = 1.0  # 类别
                
                # 同步更新计数，保证下次找空槽时两者一致
                center_counts[cy_g, cx_g] = b + 1
                corner_counts[cory_g, corx_g] = b + 1

        return target

if __name__ == "__main__":
    img_dir = "./datasets/VOC/images/train2012"
    label_dir = "./datasets/VOC/labels/train2012"
    
    # 实例化 Dataset
    dataset = PLNDataset(img_dir, label_dir, img_size=448, S=14)
    
    # 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 获取一个 Batch 的数据进行验证
    for imgs, targets in dataloader:
        print(f"输入图片 Batch 尺寸: {imgs.shape}")
        
        target_lt, target_rt, target_lb, target_rb = targets
        print(f"Left-Top 分支标签尺寸: {target_lt.shape}") # 预期 [Batch, 14, 14, 204]
        
        # 简单验证一下维度是否正确
        assert target_lt.shape[-1] == 204
        print("DataLoader 测试通过，维度构建对齐！")
        break
