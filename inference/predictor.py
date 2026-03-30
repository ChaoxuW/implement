"""
predictor.py
PLN模型推理器，负责将模型的四个分支输出解码为边界框、类别和置信度。
实现论文第3.3节“Inference”中的算法。
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional

class PLNInference:
    """
    Point Linking Network 推理器。
    输入：模型输出的四个分支特征 (B, 204, S, S)
    输出：每个图像的边界框列表 [xmin, ymin, xmax, ymax], 类别ID, 置信度分数
    """
    def __init__(self, S=14, B=2, num_classes=20, conf_thresh=0.01, nms_thresh=0.5):
        """
        初始化推理参数。

        Args:
            S (int): 特征图网格大小 (默认14)。
            B (int): 每个网格预测的点对数量 (默认2，对应两个中心点槽位和两个角点槽位)。
            num_classes (int): 数据集类别数 (VOC为20)。
            conf_thresh (float): 置信度阈值，低于此值的预测将被过滤。
            nms_thresh (float): 非极大值抑制(NMS)的IoU阈值。
        """
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # 根据论文，每个点的特征维度为 1(存在) + 2(坐标) + S(行链接) + S(列链接) + C(类别)
        # 代码中已固定为 51 = 1 + 2 + 14 + 14 + 20
        self.pointsize = 51
        assert self.pointsize == 1 + 2 + S + S + num_classes, \
            f"特征维度{self.pointsize}与配置S={S}, C={num_classes}不匹配"
        
        # 四个分支对应的角点类型
        self.branch_corner_mapping = {
            'left_top': (0, 0),      # 预测 (中心点, 左上角点)
            'right_top': (1, 0),     # 预测 (中心点, 右上角点)
            'left_bot': (0, 1),      # 预测 (中心点, 左下角点)
            'right_bot': (1, 1),     # 预测 (中心点, 右下角点)
        }
    
    @torch.no_grad()
    def decode_branch(self, branch_output: torch.Tensor, corner_type: str) -> List[Dict]:
        """
        解码单个分支的输出。
        
        模型输出 `branch_output` 的形状为 (B, 204, S, S)。
        204 = 4个点 * 51维/点。
        前102维 (2 * 51) 对应两个中心点，后102维对应两个角点。

        根据论文公式(5)计算每个候选点对(中心点-角点)成为物体的概率。

        Args:
            branch_output: 单个分支的模型输出，形状 (1, 204, S, S)。
            corner_type: 分支类型，['left_top', 'right_top', 'left_bot', 'right_bot']。

        Returns:
            detections: 该分支检测到的物体列表，每个元素是包含边界框、类别、分数的字典。
        """
        device = branch_output.device
        batch_size = branch_output.shape[0]
        assert batch_size == 1, "解码器目前仅支持batch_size=1"
        
        # 1. 调整维度顺序 -> (S, S, 204)
        pred = branch_output[0].permute(1, 2, 0).contiguous()  # (S, S, 204)
        
        all_detections = []
        
        # 2. 遍历所有网格(S*S)和所有点对槽位(B=2)
        for cy in range(self.S):  # 中心点所在行
            for cx in range(self.S):  # 中心点所在列
                for b in range(self.B):  # 遍历中心点槽位
                    # 中心点特征起始索引
                    c_base = b * self.pointsize
                    
                    # 2.1 获取中心点存在概率 P(O)_ij
                    p_center = pred[cy, cx, c_base + 0]  # 已经是概率值
                    if p_center < self.conf_thresh:
                        continue
                    
                    # 2.2 获取中心点坐标偏移 (x_offset, y_offset)
                    center_x_offset = pred[cy, cx, c_base + 1] 
                    center_y_offset = pred[cy, cx, c_base + 2]  
                    
                    # 计算中心点绝对坐标 (归一化到0~1)
                    center_x_abs = (cx + center_x_offset.item()) / self.S
                    center_y_abs = (cy + center_y_offset.item()) / self.S
                    
                    # 2.3 获取中心点链接概率分布 L^x, L^y (长度各为S)
                    # 第3~16维是行链接(指向角点行)，第17~30维是列链接(指向角点列)
                    # 【注意】链接预测需要softmax归一化
                    center_link_row = pred[cy, cx, c_base + 3 : c_base + 3 + self.S]
                    center_link_col = pred[cy, cx, c_base + 3 + self.S : c_base + 3 + 2*self.S]
                    # 预测链接到的角点网格 (行, 列)
                    corner_gy = torch.argmax(center_link_row).item()  # 角点所在行索引
                    corner_gx = torch.argmax(center_link_col).item()  # 角点所在列索引
                    link_prob = center_link_row[corner_gy].item() * center_link_col[corner_gx].item()
                    
                    # 2.4 找到对应的角点 (根据论文，中心点槽位b链接到角点槽位b)
                    # 角点特征起始索引: 前两个点(0,1)是中心点，后两个点(2,3)是角点
                    corner_base = (self.B + b) * self.pointsize
                    
                    # 获取角点存在概率 P(C)_st
                    # 【修改】删除 torch.sigmoid，直接使用模型输出值
                    p_corner = pred[corner_gy, corner_gx, corner_base + 0]  # 已经是概率值
                    if p_corner < self.conf_thresh:
                        continue
                    
                    # 获取角点坐标偏移
                    # 【修改】删除 torch.sigmoid，直接使用模型输出值
                    corner_x_offset = pred[corner_gy, corner_gx, corner_base + 1]  # 已经是(0,1)内的偏移量
                    corner_y_offset = pred[corner_gy, corner_gx, corner_base + 2]  # 已经是(0,1)内的偏移量
                    
                    # 计算角点绝对坐标
                    corner_x_abs = (corner_gx + corner_x_offset.item()) / self.S
                    corner_y_abs = (corner_gy + corner_y_offset.item()) / self.S
                    
                    # 获取角点链接概率分布 (应指回中心点，用于验证)
                    # 【注意】链接预测需要softmax归一化
                    corner_link_row = pred[corner_gy, corner_gx, corner_base + 3 : corner_base + 3 + self.S]
                    corner_link_col = pred[corner_gy, corner_gx, corner_base + 3 + self.S : corner_base + 3 + 2*self.S]
                    link_prob_reverse = corner_link_row[cy].item() * corner_link_col[cx].item()
                    
                    # 2.5 获取类别概率分布 Q(n)
                    # 特征第31~50维是类别one-hot (20类)
                    # 【修改】删除 torch.sigmoid，直接使用模型输出值
                    center_cls_probs = pred[cy, cx, c_base + 31 : c_base + 51]  # 已经是概率值
                    corner_cls_probs = pred[corner_gy, corner_gx, corner_base + 31 : corner_base + 51]  # 已经是概率值
                    
                    # 2.6 根据论文公式(5)计算点对成为物体的概率
                    # P_obj = P_center * P_corner * (Q_center * Q_corner) * (Link_center->corner + Link_corner->center)/2
                    for cls_id in range(self.num_classes):
                        cls_prob = center_cls_probs[cls_id].item() * corner_cls_probs[cls_id].item()
                        if cls_prob < 1e-3:  # 类别概率过低则跳过
                            continue
                        
                        # 点对成为第cls_id类物体的总置信度
                        obj_score = p_center.item() * p_corner.item() * cls_prob * (link_prob + link_prob_reverse) / 2.0
                        
                        if obj_score < self.conf_thresh:
                            continue
                        
                        # 2.7 根据中心点和角点坐标，以及分支类型，计算边界框
                        # 分支类型决定了这个角点是哪个角（左上、右上、左下、右下）
                        if corner_type == 'left_top':
                            xmin = corner_x_abs
                            ymin = corner_y_abs
                            xmax = 2 * center_x_abs - corner_x_abs  # 对称得到右下角
                            ymax = 2 * center_y_abs - corner_y_abs
                        elif corner_type == 'right_top':
                            xmin = 2 * center_x_abs - corner_x_abs
                            ymin = corner_y_abs
                            xmax = corner_x_abs
                            ymax = 2 * center_y_abs - corner_y_abs
                        elif corner_type == 'left_bot':
                            xmin = corner_x_abs
                            ymin = 2 * center_y_abs - corner_y_abs
                            xmax = 2 * center_x_abs - corner_x_abs
                            ymax = corner_y_abs
                        elif corner_type == 'right_bot':
                            xmin = 2 * center_x_abs - corner_x_abs
                            ymin = 2 * center_y_abs - corner_y_abs
                            xmax = corner_x_abs
                            ymax = corner_y_abs
                        else:
                            raise ValueError(f"未知的分支类型: {corner_type}")
                        
                        # 确保边界框坐标在[0,1]范围内且有效
                        xmin, xmax = sorted([max(0.0, xmin), min(1.0, xmax)])
                        ymin, ymax = sorted([max(0.0, ymin), min(1.0, ymax)])
                        
                        if (xmax - xmin) < 0.001 or (ymax - ymin) < 0.001:  # 过滤无效框
                            continue
                        
                        detection = {
                            'bbox': [xmin, ymin, xmax, ymax],  # 归一化坐标
                            'class_id': cls_id,
                            'score': obj_score,
                            'branch': corner_type
                        }
                        all_detections.append(detection)
        
        return all_detections
    
    def nms(self, detections: List[Dict]) -> List[Dict]:
        """
        对解码后的所有检测框进行非极大值抑制(NMS)。

        Args:
            detections: decode_branch返回的检测框列表。

        Returns:
            经过NMS过滤后的检测框列表。
        """
        if not detections:
            return []
        
        # 按分数降序排序
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算IoU
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        classes = np.array([d['class_id'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU低于阈值的框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    @torch.no_grad()
    def __call__(self, branch_features: Dict[str, torch.Tensor]) -> List[Dict]:
        """
        主调用函数，整合四个分支的输出，进行解码和NMS，返回最终检测结果。

        Args:
            branch_features: 模型输出的分支特征字典，包含四个键：'left_top', 'right_top', 'left_bot', 'right_bot'。
                            每个值形状为 (1, 204, 14, 14)。

        Returns:
            最终检测结果列表，每个元素为字典，包含：
                - 'bbox': [xmin, ymin, xmax, ymax] (归一化坐标)
                - 'class_id': 整数类别ID (0~19)
                - 'score': 置信度分数
        """
        all_branch_detections = []
        
        # 1. 分别解码四个分支
        for branch_name, features in branch_features.items():
            branch_dets = self.decode_branch(features, branch_name)
            all_branch_detections.extend(branch_dets)
        
        # 2. 合并所有分支的检测结果，然后进行NMS
        final_detections = self.nms(all_branch_detections)
        
        return final_detections


def test_predictor():
    """测试推理器功能"""
    # 模拟模型输出
    dummy_branch_features = {
        'left_top': torch.randn(1, 204, 14, 14),
        'right_top': torch.randn(1, 204, 14, 14),
        'left_bot': torch.randn(1, 204, 14, 14),
        'right_bot': torch.randn(1, 204, 14, 14)
    }
    
    # 初始化推理器
    predictor = PLNInference(S=14, B=2, num_classes=20, conf_thresh=0.1, nms_thresh=0.5)
    
    # 执行推理
    detections = predictor(dummy_branch_features)
    
    print(f"✓ Predictor 测试通过!")
    print(f"检测到 {len(detections)} 个物体")
    if detections:
        for i, det in enumerate(detections[:3]):  # 打印前3个
            print(f"  物体{i}: 类别{det['class_id']}, 分数{det['score']:.3f}, 框{det['bbox']}")
    
    return detections


if __name__ == "__main__":
    test_predictor()
