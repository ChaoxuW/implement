"""
test.py
PLN模型测试端入口。
加载训练好的模型，在测试集上运行预测和评估，计算mAP，可选可视化。
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pln_model_inc_im import build_model
from dataloader.voc import PLNDataset
from inference.predictor import PLNInference
from eval import PLNEvaluator
from utils.visualize import PLNVisualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PLN模型测试脚本')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型权重路径 (.pth文件)')
    parser.add_argument('--data_dir', type=str, default='./datasets/VOC',
                       help='VOC数据集根目录')
    parser.add_argument('--split', type=str, default='train2007',
                       choices=['test2007', 'val2007', 'val2012','train2007'],
                       help='要测试的数据集划分')
    parser.add_argument('--img_size', type=int, default=448,
                       help='输入图像大小')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='测试批次大小 (目前仅支持1)')
    parser.add_argument('--conf_thresh', type=float, default=0.01,
                       help='置信度阈值')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                       help='NMS的IoU阈值')
    parser.add_argument('--eval_iou', type=float, default=0.5,
                       help='评估用的IoU阈值 (PASCAL VOC标准为0.5)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='运行设备: cuda 或 cpu')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化预测结果')
    parser.add_argument('--visualize_num', type=int, default=10,
                       help='要可视化的图片数量')
    parser.add_argument('--visualize_dir', type=str, default='./visualizations_inc_im',
                       help='可视化结果保存目录')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存评估结果到JSON文件')
    parser.add_argument('--results_dir', type=str, default='./results_inc_im',
                       help='评估结果保存目录')
    
    return parser.parse_args()


def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"\n加载模型: {model_path}")
    
    # 构建模型
    model = build_model(backbone_pretrained=False)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功，设备: {device}")
    return model


def load_test_dataset(data_dir, split, img_size):
    """加载测试数据集"""
    print(f"\n加载数据集: {split}")
    
    # 根据split确定路径
    img_dir = os.path.join(data_dir, 'images', split)
    label_dir = os.path.join(data_dir, 'labels', split)
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    
    # 创建数据集
    dataset = PLNDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        S=14,
        num_classes=20
    )
    
    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # 测试时batch_size固定为1
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ 数据集加载成功: {len(dataset)} 张图片")
    return dataset, dataloader


def test_model(args):
    """主测试函数"""
    # 创建输出目录
    Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # VOC类别名
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 1. 加载模型
    model = load_model(args.model_path, args.device)
    
    # 2. 加载数据集
    dataset, dataloader = load_test_dataset(args.data_dir, args.split, args.img_size)
    
    # 3. 初始化推理器和评估器
    predictor = PLNInference(
        S=14, 
        B=2, 
        num_classes=20,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh
    )
    
    evaluator = PLNEvaluator(
        num_classes=20,
        iou_threshold=args.eval_iou
    )
    
    # 4. 初始化可视化工具
    visualizer = PLNVisualizer(VOC_CLASSES) if args.visualize else None
    
    # 5. 遍历测试集进行推理和评估
    print(f"\n开始测试 {args.split} 数据集...")
    print("=" * 70)
    
    for batch_idx, (images, targets, gt_boxes_info) in enumerate(tqdm(dataloader, desc="Testing")):
        image_id = dataset.label_files[batch_idx].replace('.txt', '')
        
        # 数据移到设备
        images = images.to(args.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(images)
        
        # 使用predictor解码四个分支的输出
        detections = predictor(outputs['branch_features'])
        
        # 提取预测结果
        pred_boxes = [det['bbox'] for det in detections]
        pred_scores = [det['score'] for det in detections]
        pred_classes = [det['class_id'] for det in detections]
        
        # 提取真实标签
        # targets包含四个分支的target，但真实框信息在gt_boxes_info中
        # gt_boxes_info形状: (1, max_boxes, 5) -> [class_id, x1, y1, x2, y2]
        gt_boxes_info_np = gt_boxes_info[0].numpy()  # (max_boxes, 5)
        
        # 过滤掉填充的零框
        valid_mask = np.any(gt_boxes_info_np[:, 1:] != 0, axis=1)  # 检查坐标是否全零
        valid_gt_boxes_info = gt_boxes_info_np[valid_mask]
        
        if len(valid_gt_boxes_info) > 0:
            gt_boxes = valid_gt_boxes_info[:, 1:5].tolist()  # [x1, y1, x2, y2]
            gt_classes = valid_gt_boxes_info[:, 0].astype(int).tolist()
        else:
            gt_boxes = []
            gt_classes = []
        
        # 添加到评估器
        evaluator.add_single_prediction(image_id, pred_boxes, pred_scores, pred_classes)
        evaluator.add_single_ground_truth(image_id, gt_boxes, gt_classes)
        
        # 可视化 (如果启用)
        if args.visualize and batch_idx < args.visualize_num:
            # 加载原始图像
            img_path = os.path.join(dataset.img_dir, f"{image_id}.jpg")
            if os.path.exists(img_path):
                img_bgr = visualizer.load_image(img_path)
                img_h, img_w = img_bgr.shape[:2]
                
                # 将归一化坐标转换为像素坐标
                if pred_boxes:
                    pred_boxes_pixel = visualizer.denormalize_boxes(
                        np.array(pred_boxes), img_h, img_w
                    )
                else:
                    pred_boxes_pixel = np.array([])
                    
                if gt_boxes:
                    gt_boxes_pixel = visualizer.denormalize_boxes(
                        np.array(gt_boxes), img_h, img_w
                    )
                else:
                    gt_boxes_pixel = np.array([])
                
                # 绘制预测和真实框
                if len(pred_boxes_pixel) > 0 or len(gt_boxes_pixel) > 0:
                    result_img = visualizer.draw_both(
                        img_bgr,
                        gt_boxes_pixel, gt_classes,
                        pred_boxes_pixel, pred_classes, pred_scores,
                        thickness=2,
                        font_scale=0.6
                    )
                    
                    # 保存可视化结果
                    save_path = os.path.join(args.visualize_dir, f"{image_id}.jpg")
                    visualizer.save_image(result_img, save_path)
        
        # 可选：打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  已处理 {batch_idx + 1} / {len(dataloader)} 张图片")
    
    # 6. 计算并输出评估结果
    print(f"\n{'='*70}")
    print(f"测试完成! 正在计算评估指标...")
    print(f"{'='*70}")
    
    eval_results = evaluator.evaluate()
    
    # 7. 保存评估结果
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            args.results_dir, 
            f"eval_{args.split}_{timestamp}.json"
        )
        evaluator.save_results(results_path, eval_results)
    
    print(f"\n✓ 测试完成!")
    
    return eval_results


if __name__ == "__main__":
    from datetime import datetime
    
    args = parse_args()
    
    # 参数验证
    if args.batch_size != 1:
        print("警告: 当前推理器仅支持batch_size=1，将自动调整为1")
        args.batch_size = 1
    
    # 运行测试
    try:
        results = test_model(args)
        
        # 打印汇总结果
        print(f"\n{'='*60}")
        print(f"测试汇总:")
        print(f"  数据集: {args.split}")
        print(f"  模型: {args.model_path}")
        print(f"  mAP@{int(args.eval_iou*100)}: {results['mAP']:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)