"""
PLN (Point Linking Network) 训练脚本
整合多个VOC数据集进行训练
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from models.pln_model_inc_im import build_model
from dataloader.voc import PLNDataset
from losses.pln_loss import plnLoss

# ================= 配置参数 =================
class Config:
    """训练超参数配置"""
    # 数据集配置
    img_size = 448
    S = 14  # 网格大小
    num_classes = 20
    
    # 数据集路径
    train_datasets = {
        'train2007': './datasets/VOC/images/train2007',
        'train2012': './datasets/VOC/images/train2012',
        'val2007': './datasets/VOC/images/val2007',
        'val2012': './datasets/VOC/images/val2012',
    }
    
    label_dir_template = './datasets/VOC/labels/{}'
    
    # 训练配置
    num_epochs = 120
    batch_size = 56
    learning_rate = 1e-4
    weight_decay = 1e-4
    momentum = 0.9
    
    # 损失权重
    w_coord = 2.0   # 坐标损失权重
    w_class = 1   # 类别损失权重
    w_link = 1   # 链接损失权重
    noobscale=0.04
    
    # 优化器配置
    optimizer_type = 'Adam'  # 'sgd' 或 'adam'
    lr_scheduler = 'cosine'  # 'cosine', 'step', 或 'none'
    
    # 设备和日志
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 8
    save_interval = 20 # 每 N 个 epoch 保存一次模型
    eval_interval = 5   # 每 N 个 epoch 进行一次评估
    
    # 输出目录
    output_dir = './checkpoint_refined'
    log_dir = './logs_refined'
    
    def __str__(self):
        """打印配置"""
        lines = ["=" * 50]
        lines.append("Training Configuration")
        lines.append("=" * 50)
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                lines.append(f"{key:<20}: {value}")
        lines.append("=" * 50)
        return "\n".join(lines)

# ================= 数据加载 =================
def create_data_loaders(config):
    """
    创建整合所有训练集的 DataLoader
    """
    print("\n正在加载数据集...")
    
    all_datasets = []
    dataset_names = []
    
    for dataset_name, img_dir in config.train_datasets.items():
        label_dir = config.label_dir_template.format(dataset_name)
        
        print(f"  Loading {dataset_name}...")
        dataset = PLNDataset(
            img_dir=img_dir,
            label_dir=label_dir,
            img_size=config.img_size,
            S=config.S,
            num_classes=config.num_classes
        )
        
        all_datasets.append(dataset)
        dataset_names.append(dataset_name)
        print(f"    ✓ {dataset_name}: {len(dataset)} images")
    
    # 合并所有数据集
    combined_dataset = ConcatDataset(all_datasets)
    total_samples = len(combined_dataset)
    print(f"\n总训练样本数: {total_samples}")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == 'cuda')
    )
    
    return train_loader

# ================= 训练函数 =================
def train_epoch(model, train_loader, loss_fn, optimizer, config, epoch):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    loss_dict_sum = {
        'p_loss': 0.0,
        'coord_loss': 0.0,
        'link_loss': 0.0,
        'class_loss': 0.0,
        'noobj_loss': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, (images, targets,_) in enumerate(pbar):
        # 数据移到设备
        images = images.to(config.device)
        target_lt, target_rt, target_lb, target_rb = targets
        target_lt = target_lt.to(config.device)
        target_rt = target_rt.to(config.device)
        target_lb = target_lb.to(config.device)
        target_rb = target_rb.to(config.device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(images)
        
        pred_lt = output['branch_features']['left_top']
        pred_rt = output['branch_features']['right_top']
        pred_lb = output['branch_features']['left_bot']
        pred_rb = output['branch_features']['right_bot']
        
        # 计算损失 (4 个分支分别计算，再相加)
        loss_lt, losses_lt = loss_fn(pred_lt, target_lt)
        loss_rt, losses_rt = loss_fn(pred_rt, target_rt)
        loss_lb, losses_lb = loss_fn(pred_lb, target_lb)
        loss_rb, losses_rb = loss_fn(pred_rb, target_rb)
        
        loss = loss_lt + loss_rt + loss_lb + loss_rb
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        for key in loss_dict_sum:
            loss_dict_sum[key] += (losses_lt[key] + losses_rt[key] + 
                                   losses_lb[key] + losses_rb[key]).item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_losses = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_losses

def train(config=None):
    """
    完整的训练函数
    
    Args:
        config: Config 对象，如果为 None 则使用默认配置
    
    Returns:
        trained_model: 训练完成的模型
    """
    if config is None:
        config = Config()
    
    print(config)
    
    # 设置随机种子
    torch.manual_seed(42)
    if config.device == 'cuda':
        torch.cuda.manual_seed(42)
    
    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化模型、损失函数、优化器
    print(f"\n初始化模型（设备: {config.device}）...")
    model = build_model(backbone_pretrained=True)
    model = model.to(config.device)
    
    loss_fn = plnLoss(
        S=config.S,
        B=2,
        w_class=config.w_class,
        w_coord=config.w_coord,
        w_link=config.w_link,
        noob_scale=config.noobscale
    )
    
    # 优化器
    if config.optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:  # adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    # 学习率调度器
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # 加载数据
    train_loader = create_data_loaders(config)
    
    # 训练日志
    train_log = {
        'config': config.__dict__,
        'epochs': []
    }
    
    best_loss = float('inf')
    
    # 训练循环
    print(f"\n开始训练，共 {config.num_epochs} 个 epoch...")
    print("=" * 70)
    
    for epoch in range(config.num_epochs):
        # 训练
        avg_loss, avg_losses = train_epoch(model, train_loader, loss_fn, optimizer, config, epoch)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录日志
        epoch_log = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'losses': avg_losses,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_log['epochs'].append(epoch_log)
        
        # 打印信息
        print(f"\nEpoch {epoch+1:3d}/{config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  p_loss: {avg_losses['p_loss']:.4f}, "
              f"coord_loss: {avg_losses['coord_loss']:.4f}, "
              f"link_loss: {avg_losses['link_loss']:.4f}, "
              f"class_loss: {avg_losses['class_loss']:.4f}, "
              f"noobj_loss: {avg_losses['noobj_loss']:.4f}")
        
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(config.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ 保存最优模型到: {best_model_path}")
        
        # 定期保存模型
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.output_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ 保存检查点到: {checkpoint_path}")
    
    # 保存训练日志
    log_path = os.path.join(config.log_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    print(f"\n✓ 训练日志保存到: {log_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ 最终模型保存到: {final_model_path}")
    
    return model, config

# ================= 测试函数 =================
if __name__ == "__main__":
    config = Config()
    model, config = train(config)
    print("\n✓ 增强模型训练完成！")