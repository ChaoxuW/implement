import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

CLASS_NUM=20

class plnLoss(nn.Module):
    """
    点链接网络(Point Linking Network)的损失函数
    
    对于204维的特征，其包含四个点的信息
    每个点51维（1是否存在点，2坐标，28行列链接，20类别）
    四个点包含两个中心点，两个边点
    一共五类损失：
    点是否存在
    点坐标预测
    点的行列链接
    物体类别
    负样本损失
    """
    def __init__(self,S,B,w_class,w_coord,w_link,noob_scale):
        super(plnLoss,self).__init__()
        self.S=S
        self.B=B
        self.w_class=w_class
        self.w_link= w_link
        self.w_coord=w_coord
        self.classes=CLASS_NUM
        self.noob_scale=noob_scale

        self.pointsize=51
        self.numpoint=4
    
    def forward(self,pred,target):
        """""
        预测的张量形状：(B,204,14,14) 【注意：此张量值已在(0,1)范围内，由模型Sigmoid产生】
        目标张量形状：（B,14,14,204）
        返回总loss以及各个loss，以便后续记录
        """""
        device=pred.device
        batch_size=pred.size(0)

        #调整顺序对齐
        pred_re=pred.permute(0,2,3,1)

        p_loss=0
        coord_loss=0
        link_loss=0
        class_loss=0
        noob_loss=0

        #对四个点遍历
        for i in range(4):
            start=i*self.pointsize
            end=(i+1)*self.pointsize
            pred_now=pred_re[:,:,:,start:end]
            target_now=target[:,:,:,start:end]

            #对是否存在点进行处理
            pred_I=pred_now[:,:,:,0]           # 已经是概率值
            target_I=target_now[:,:,:,0]
            #查看是否是负样本
            if_neg=target_I==0

            if if_neg.any():
                sq_err=(pred_I-target_I)**2
                neg_err=sq_err*if_neg.float()
                noob_loss+=neg_err.sum()
            
            if_pos=~if_neg

            if if_pos.any():
                pred_I_pos=pred_I[if_pos]
                target_I_pos=target_I[if_pos]

                p_loss+=F.mse_loss(pred_I_pos,target_I_pos,reduction='sum')

                # 坐标损失（第 1-2 维，共 2 维）
                pred_coord=pred_now[:,:,:,1:3]          # 已经是(0,1)内的偏移量
                target_coord=target_now[:,:,:,1:3]
                pred_coord_pos=pred_coord[if_pos]
                target_coord_pos=target_coord[if_pos]
                coord_loss+=F.mse_loss(pred_coord_pos,target_coord_pos,reduction='sum')

                # 链接损失（第 3-30 维，共 28 维）
                pred_link=pred_now[:,:,:,3:31]          # 链接预测值
                target_link=target_now[:,:,:,3:31]
                pred_link_pos=pred_link[if_pos]
                target_link_pos=target_link[if_pos]
                link_loss+=F.mse_loss(pred_link_pos,target_link_pos,reduction='sum')
                
                # 类别损失（第 31-50 维，共 20 维）
                pred_class=pred_now[:,:,:,31:51]        # 已经是概率值
                target_class=target_now[:,:,:,31:51]
                pred_class_pos=pred_class[if_pos]
                target_class_pos=target_class[if_pos]
                class_loss+=F.mse_loss(pred_class_pos,target_class_pos,reduction='sum')


        if p_loss==0 and coord_loss==0 and link_loss==0 and class_loss==0:
            # 不能直接返回0！即使没有正样本，也要计算背景负样本(noob_loss)的梯度
            zero_tensor = torch.tensor(0.0, device=device)
            total_loss = self.noob_scale * noob_loss / batch_size
            loss_dict = {
                'p_loss': zero_tensor,
                'coord_loss': zero_tensor,
                'link_loss': zero_tensor,
                'class_loss': zero_tensor,
                'noobj_loss': noob_loss / batch_size
            }
            return total_loss, loss_dict
        
        total_loss=(
            p_loss+
            self.w_coord * coord_loss+
            self.w_class * class_loss+
            self.w_link * link_loss+
            self.noob_scale*noob_loss
        )
        

        total_loss = total_loss / batch_size

        # 返回总损失和各个子损失字典
        losses_dict = {
            'p_loss': p_loss / batch_size,
            'coord_loss': coord_loss / batch_size,
            'link_loss': link_loss / batch_size,
            'class_loss': class_loss / batch_size,
            'noobj_loss': noob_loss / batch_size,
        }
        
        return total_loss, losses_dict

