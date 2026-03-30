"""
evaluator.py
PLN模型评估器，计算mAP等指标。
参考PASCAL VOC和COCO评估标准。
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import json
from pathlib import Path


class PLNEvaluator:
    """
    PLN模型评估器，计算mAP (mean Average Precision)。
    
    评估流程：
    1. 收集所有图片的预测结果和真实标签。
    2. 对每个类别单独计算精度-召回率(Precision-Recall)曲线。
    3. 计算每个类别的AP (Average Precision)。
    4. 计算所有类别的mAP。
    """
    
    def __init__(self, num_classes=20, iou_threshold=0.5):
        """
        初始化评估器。

        Args:
            num_classes (int): 类别数量，VOC为20。
            iou_threshold (float): 判断预测框是否正确的IoU阈值 (PASCAL VOC标准为0.5)。
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        
        # 存储所有图片的预测和真实标签
        self._predictions = []  # 元素: (image_id, bbox, score, class_id)
        self._ground_truths = []  # 元素: (image_id, bbox, class_id, difficult)
        self.image_ids = set()
        
    def add_single_prediction(self, image_id: str, pred_boxes: List, pred_scores: List, pred_classes: List):
        """
        添加单张图片的预测结果。

        Args:
            image_id: 图片标识符。
            pred_boxes: 预测框列表，每个框为 [xmin, ymin, xmax, ymax] (归一化或像素坐标)。
            pred_scores: 预测分数列表，与pred_boxes一一对应。
            pred_classes: 预测类别列表，与pred_boxes一一对应。
        """
        assert len(pred_boxes) == len(pred_scores) == len(pred_classes)
        
        for box, score, cls_id in zip(pred_boxes, pred_scores, pred_classes):
            self._predictions.append({
                'image_id': image_id,
                'bbox': box,
                'score': score,
                'class_id': int(cls_id)
            })
        
        self.image_ids.add(image_id)
    
    def add_single_ground_truth(self, image_id: str, gt_boxes: List, gt_classes: List, difficult_flags=None):
        """
        添加单张图片的真实标签。

        Args:
            image_id: 图片标识符。
            gt_boxes: 真实框列表，每个框为 [xmin, ymin, xmax, ymax]。
            gt_classes: 真实类别列表，与gt_boxes一一对应。
            difficult_flags: 困难样本标记列表 (VOC数据集)，与gt_boxes一一对应。如果为None则全为0。
        """
        assert len(gt_boxes) == len(gt_classes)
        
        if difficult_flags is None:
            difficult_flags = [0] * len(gt_boxes)
        else:
            assert len(gt_boxes) == len(difficult_flags)
        
        for box, cls_id, difficult in zip(gt_boxes, gt_classes, difficult_flags):
            self._ground_truths.append({
                'image_id': image_id,
                'bbox': box,
                'class_id': int(cls_id),
                'difficult': int(difficult)
            })
        
        self.image_ids.add(image_id)
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        计算两个边界框的IoU (Intersection over Union)。

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU值
        """
        # 确保坐标是浮点数
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]
        
        # 计算交集区域
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算各自面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return iou
    
    def evaluate_class(self, class_id: int):
        """
        评估单个类别的AP。

        Args:
            class_id: 要评估的类别ID。

        Returns:
            ap: 该类别的平均精度(AP)。
            precision: 精度数组。
            recall: 召回率数组。
        """
        # 筛选该类别的预测和真实标签
        class_preds = [p for p in self._predictions if p['class_id'] == class_id]
        class_gts = [g for g in self._ground_truths if g['class_id'] == class_id]
        
        # 按分数降序排序预测
        class_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # 统计图片中该类别的真实框数量 (忽略困难样本)
        gt_count_per_image = defaultdict(int)
        for gt in class_gts:
            if gt['difficult'] == 0:  # 非困难样本才计数
                gt_count_per_image[gt['image_id']] += 1
        
        total_gt = sum(gt_count_per_image.values())
        if total_gt == 0:
            return 0.0, np.array([0]), np.array([0])
        
        # 初始化匹配状态
        gt_matched = {gt_idx: False for gt_idx, _ in enumerate(class_gts)}
        
        tp = np.zeros(len(class_preds))  # 真正例
        fp = np.zeros(len(class_preds))  # 假正例
        
        # 遍历每个预测
        for pred_idx, pred in enumerate(class_preds):
            image_id = pred['image_id']
            pred_bbox = pred['bbox']
            
            # 找到同图片同类别的真实框
            image_gt_indices = [
                gt_idx for gt_idx, gt in enumerate(class_gts) 
                if gt['image_id'] == image_id and gt['difficult'] == 0
            ]
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx in image_gt_indices:
                if gt_matched[gt_idx]:  # 这个真实框已经被匹配过了
                    continue
                
                gt_bbox = class_gts[gt_idx]['bbox']
                iou = self.calculate_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 判断是否匹配成功
            if best_iou >= self.iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # 计算累积的TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精度和召回率
        eps = np.finfo(np.float32).eps
        recalls = tp_cumsum / (total_gt + eps)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        
        # 计算AP (Average Precision) - 使用PASCAL VOC 2012的11点插值法
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                precision_at_t = np.max(precisions[mask])
            else:
                precision_at_t = 0.0
            ap += precision_at_t / 11.0
        
        return ap, precisions, recalls
    
    def evaluate(self):
        """
        评估所有类别，计算mAP。

        Returns:
            eval_results: 包含各类别AP和总体mAP的字典。
        """
        aps = []
        class_results = {}
        
        print(f"\n{'='*60}")
        print("开始评估...")
        print(f"总图片数: {len(self.image_ids)}")
        print(f"总预测框数: {len(self._predictions)}")
        print(f"总真实框数: {len(self._ground_truths)}")
        print(f"IoU阈值: {self.iou_threshold}")
        print(f"{'='*60}")
        
        for class_id in range(self.num_classes):
            ap, precisions, recalls = self.evaluate_class(class_id)
            aps.append(ap)
            class_results[class_id] = {
                'ap': float(ap),
                'num_predictions': len([p for p in self._predictions if p['class_id'] == class_id]),
                'num_gts': len([g for g in self._ground_truths if g['class_id'] == class_id])
            }
            
            print(f"类别 {class_id:2d}: AP = {ap:.4f}")
        
        mAP = np.mean(aps) if aps else 0.0
        
        print(f"{'='*60}")
        print(f"mAP@{int(self.iou_threshold*100)}: {mAP:.4f}")
        print(f"{'='*60}")
        
        eval_results = {
            'mAP': float(mAP),
            'class_APs': class_results,
            'config': {
                'num_classes': self.num_classes,
                'iou_threshold': self.iou_threshold,
                'num_images': len(self.image_ids)
            }
        }
        
        return eval_results
    
    def reset(self):
        """重置评估器状态"""
        self._predictions.clear()
        self._ground_truths.clear()
        self.image_ids.clear()
    
    def save_results(self, filepath: str, eval_results: Dict):
        """保存评估结果到JSON文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 评估结果已保存到: {filepath}")


def test_evaluator():
    """测试评估器功能"""
    evaluator = PLNEvaluator(num_classes=3, iou_threshold=0.5)
    
    # 添加测试数据
    # 图片1: 2个真实框
    evaluator.add_single_ground_truth(
        image_id='img1',
        gt_boxes=[[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]],
        gt_classes=[0, 1],
        difficult_flags=[0, 0]
    )
    
    # 图片1: 3个预测框
    evaluator.add_single_prediction(
        image_id='img1',
        pred_boxes=[[0.12, 0.11, 0.31, 0.29], [0.52, 0.51, 0.71, 0.69], [0.8, 0.8, 0.9, 0.9]],
        pred_scores=[0.95, 0.87, 0.65],
        pred_classes=[0, 1, 2]
    )
    
    # 评估
    results = evaluator.evaluate()
    
    print(f"\n测试结果:")
    print(f"mAP: {results['mAP']:.4f}")
    for cls_id, cls_result in results['class_APs'].items():
        print(f"  类别{cls_id}: AP={cls_result['ap']:.4f}")
    
    return results


if __name__ == "__main__":
    test_evaluator()