import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Union


class PLNVisualizer:
    """
    PLN 模型可视化工具
    支持绘制真实标签、预测框、类别标签等
    """
    
    def __init__(self, class_names, class_colors=None):
        """
        Args:
            class_names: List of class names (len=20 for VOC)
            class_colors: Dict mapping class_id to BGR color, None则随机生成
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        if class_colors is None:
            # 随机生成颜色
            np.random.seed(42)
            self.class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) 
                                for i in range(self.num_classes)}
        else:
            self.class_colors = class_colors

    def draw_boxes(self, image, boxes, labels=None, scores=None, 
                   box_type='pred', thickness=2, font_scale=0.5):
        """
        在图像上绘制框
        Args:
            image: np.ndarray, BGR 图像，形状 [H, W, 3]
            boxes: np.ndarray 或 List，形状 [N, 4] -> [x1, y1, x2, y2] (归一化或像素坐标)
            labels: List[int] or np.ndarray，形状 [N]，类别ID
            scores: List[float] or np.ndarray，形状 [N]，置信度
            box_type: 'pred' (绿色) 或 'gt' (红色)
            thickness: 框线条粗度
            font_scale: 字体大小
        Returns:
            image: 绘制后的图像
        """
        if image is None or boxes is None or len(boxes) == 0:
            return image
        
        image = image.copy()
        h, w = image.shape[:2]
        
        # 转换为 numpy 数组
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # 颜色定义
        color_map = {
            'pred': (0, 255, 0),    # 绿色（预测）
            'gt': (0, 0, 255),      # 红色（真实）
        }
        color = color_map.get(box_type, (255, 0, 0))
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # 检查是否为归一化坐标（0~1）或像素坐标
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签和分数
            label_text = ""
            if labels is not None:
                cls_id = int(labels[i]) if i < len(labels) else 0
                if 0 <= cls_id < self.num_classes:
                    label_text = self.class_names[cls_id]
            
            if scores is not None and i < len(scores):
                label_text = f"{label_text} {scores[i]:.2f}" if label_text else f"{scores[i]:.2f}"
            
            if label_text:
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                              font_scale, 1)
                text_x = x1
                text_y = max(y1 - 5, text_size[1])
                
                # 背景框
                cv2.rectangle(image, 
                            (text_x, text_y - text_size[1] - 2),
                            (text_x + text_size[0], text_y + 2),
                            color, -1)
                
                # 文字
                cv2.putText(image, label_text, 
                          (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, (255, 255, 255), 1)
        
        return image

    def draw_gts(self, image, gt_boxes, gt_labels, thickness=2, font_scale=0.5):
        """
        绘制真实标签框 (GT)
        Args:
            image: np.ndarray, BGR 图像
            gt_boxes: np.ndarray 或 List，形状 [M, 4] -> [x1, y1, x2, y2]
            gt_labels: np.ndarray 或 List，形状 [M]，类别ID
        Returns:
            image: 绘制后的图像
        """
        return self.draw_boxes(image, gt_boxes, labels=gt_labels, 
                             box_type='gt', thickness=thickness, font_scale=font_scale)

    def draw_predictions(self, image, pred_boxes, pred_labels, pred_scores, 
                        thickness=2, font_scale=0.5):
        """
        绘制预测框
        Args:
            image: np.ndarray, BGR 图像
            pred_boxes: np.ndarray 或 List，形状 [N, 4] -> [x1, y1, x2, y2]
            pred_labels: np.ndarray 或 List，形状 [N]，类别ID
            pred_scores: np.ndarray 或 List，形状 [N]，置信度
        Returns:
            image: 绘制后的图像
        """
        return self.draw_boxes(image, pred_boxes, labels=pred_labels, 
                             scores=pred_scores, box_type='pred', 
                             thickness=thickness, font_scale=font_scale)

    def draw_both(self, image, gt_boxes, gt_labels, pred_boxes, pred_labels, 
                 pred_scores, thickness=1, font_scale=0.4):
        """
        在同一图像上同时绘制 GT（红色）和预测框（绿色）
        Args:
            image: np.ndarray, BGR 图像
            gt_boxes: np.ndarray，形状 [M, 4]
            gt_labels: np.ndarray，形状 [M]
            pred_boxes: np.ndarray，形状 [N, 4]
            pred_labels: np.ndarray，形状 [N]
            pred_scores: np.ndarray，形状 [N]
        Returns:
            image: 绘制后的图像
        """
        image = self.draw_gts(image, gt_boxes, gt_labels, thickness=thickness, 
                            font_scale=font_scale)
        image = self.draw_predictions(image, pred_boxes, pred_labels, pred_scores, 
                                     thickness=thickness, font_scale=font_scale)
        return image

    def save_image(self, image, save_path):
        """保存图像到文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)
        print(f"✓ Image saved to: {save_path}")

    def display_image(self, image, window_name="PLN Detection", wait_time=1000):
        """显示图像"""
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()

    @staticmethod
    def load_image(image_path):
        """加载 BGR 图像"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    @staticmethod
    def denormalize_boxes(boxes, image_height, image_width):
        """
        将归一化坐标转转换为像素坐标
        Args:
            boxes: np.ndarray，形状 [N, 4], 范围 [0, 1]
            image_height: int，图像高度
            image_width: int，图像宽度
        Returns:
            boxes: np.ndarray，形状 [N, 4], 单位为像素
        """
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        boxes = boxes.copy()
        boxes[:, 0] *= image_width   # x1
        boxes[:, 1] *= image_height  # y1
        boxes[:, 2] *= image_width   # x2
        boxes[:, 3] *= image_height  # y2
        return boxes

    @staticmethod
    def normalize_boxes(boxes, image_height, image_width):
        """
        将像素坐标转换为归一化坐标
        Args:
            boxes: np.ndarray，形状 [N, 4], 单位为像素
            image_height: int，图像高度
            image_width: int，图像宽度
        Returns:
            boxes: np.ndarray，形状 [N, 4], 范围 [0, 1]
        """
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        boxes = boxes.copy()
        boxes[:, 0] /= image_width   # x1
        boxes[:, 1] /= image_height  # y1
        boxes[:, 2] /= image_width   # x2
        boxes[:, 3] /= image_height  # y2
        return boxes


# ================= 使用示例 =================
if __name__ == "__main__":
    # VOC 类别名
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 初始化可视化工具
    visualizer = PLNVisualizer(VOC_CLASSES)
    
    # 示例：绘制一个简单的测试
    test_image = np.ones((448, 448, 3), dtype=np.uint8) * 255
    
    # 真实框 (归一化坐标)
    gt_boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]])
    gt_labels = np.array([0, 14])  # person 和 aeroplane
    
    # 预测框
    pred_boxes = np.array([[0.12, 0.11, 0.31, 0.29], [0.51, 0.51, 0.79, 0.81]])
    pred_labels = np.array([0, 14])
    pred_scores = np.array([0.95, 0.87])
    
    # 绘制
    result = visualizer.draw_both(test_image, gt_boxes, gt_labels, 
                                 pred_boxes, pred_labels, pred_scores)
    
    print("✓ Visualization test passed!")
