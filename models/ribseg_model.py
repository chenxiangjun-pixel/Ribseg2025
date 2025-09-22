import torch.nn as nn
import torch
import torch.nn.functional as F
from models.model_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class RibSegModel(nn.Module):
    """
    专门为肋骨分割设计的PointNet++模型
    针对二分类任务（背景+肋骨）进行优化
    """
    def __init__(self, num_classes=2, normal_channel=False, use_attention=True):
        super(RibSegModel, self).__init__()
        self.num_classes = num_classes
        self.normal_channel = normal_channel
        self.use_attention = use_attention
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
            
        # Set Abstraction layers - 多尺度特征提取
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512, 
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[32, 64, 128], 
            in_channel=3 + additional_channel, 
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128, 
            radius_list=[0.4, 0.8], 
            nsample_list=[64, 128], 
            in_channel=128 + 128 + 64, 
            mlp_list=[[128, 128, 256], [128, 196, 256]]
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=512 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True
        )
        
        # Feature Propagation layers - 特征上采样
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        
        # 肋骨分割专用特征融合层
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6 + additional_channel,  # 移除了类别标签维度
            mlp=[128, 128]
        )
        
        # 注意力机制（可选）
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv1d(128, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.Sigmoid()
            )
        
        # 分类头 - 针对二分类优化
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        
        # 二分类输出层
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
        # 额外的特征增强层
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, xyz):
        """
        前向传播
        Args:
            xyz: 输入点云 [B, C, N] 其中C=3或6（包含法向量）
        Returns:
            seg_pred: 分割预测 [B, N, num_classes]
            global_feat: 全局特征 [B, 1024]
        """
        B, C, N = xyz.shape
        
        # 处理输入
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        # Set Abstraction - 下采样和特征提取
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation - 上采样和特征融合
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # 肋骨分割专用特征融合（不包含类别标签）
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        
        # 特征增强
        enhanced_features = self.feature_enhance(l0_points)
        
        # 注意力机制（可选）
        if self.use_attention:
            attention_weights = self.attention(enhanced_features)
            enhanced_features = enhanced_features * attention_weights
        
        # 分类头
        feat = F.relu(self.bn1(self.conv1(enhanced_features)))
        feat = self.drop1(feat)
        
        # 二分类输出
        seg_pred = self.conv2(feat)
        
        # 输出格式调整
        seg_pred = seg_pred.permute(0, 2, 1)  # [B, N, num_classes]
        
        return seg_pred, l3_points


class RibSegLoss(nn.Module):
    """
    肋骨分割专用损失函数
    结合交叉熵损失和Dice损失，处理类别不平衡问题
    """
    def __init__(self, ce_weight=0.3, dice_weight=0.7, class_weights=None, smooth=1e-6):
        super(RibSegLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # 类别权重
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
            
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def dice_loss(self, pred_logits, target):
        """计算Dice损失
        支持输入：
        - pred_logits: [B, N, C] 或 [N, C]
        - target: [B, N] 或 [N]
        """
        if pred_logits.dim() == 3:
            # [B, N, C] -> [B*N, C]
            probs = F.softmax(pred_logits, dim=-1)  # [B, N, C]
            pred_foreground = probs[..., 1].contiguous().view(-1)
        else:
            # [N, C] -> [N, C]
            probs = F.softmax(pred_logits, dim=-1)  # [N, C]
            pred_foreground = probs[..., 1].contiguous().view(-1)
        
        target_foreground = target.float().contiguous().view(-1)
        intersection = (pred_foreground * target_foreground).sum()
        dice = (2. * intersection + self.smooth) / (pred_foreground.sum() + target_foreground.sum() + self.smooth)
        return 1 - dice
    
    def forward(self, pred, target, trans_feat=None):
        """
        计算组合损失
        Args:
            pred: 预测结果 [B*N, num_classes] 或 [B, N, num_classes]
            target: 真实标签 [B*N] 或 [B, N]
            trans_feat: 变换特征（PointNet++中的正则化项，这里不使用）
        """
        # 确保CE权重与输入在同一设备
        if hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            if self.ce_loss.weight.device != pred.device:
                self.ce_loss.weight = self.ce_loss.weight.to(pred.device)
        
        # 处理不同的输入维度
        if pred.dim() == 2:
            # pred: [B*N, C], target: [N] (经过view(-1, 1)[:, 0]处理)
            ce_logits = pred
            target_flat = target
            # 为了Dice损失，直接使用2维输入
            pred_dice = pred
            target_dice = target
        else:
            # pred: [B, N, C], target: [B, N]
            B, N, C = pred.shape
            ce_logits = pred.contiguous().view(-1, C)
            target_flat = target.contiguous().view(-1)
            pred_dice = pred
            target_dice = target
        
        # 交叉熵损失
        ce_loss = self.ce_loss(ce_logits, target_flat)
        
        # Dice损失
        dice_loss = self.dice_loss(pred_dice, target_dice)
        
        # 组合损失
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss


def get_model(num_classes=2, normal_channel=False, use_attention=True):
    """
    获取肋骨分割模型
    Args:
        num_classes: 分割类别数，默认为2（背景+肋骨）
        normal_channel: 是否使用法向量信息
        use_attention: 是否使用注意力机制
    """
    return RibSegModel(num_classes=num_classes, normal_channel=normal_channel, use_attention=use_attention)


def get_loss(ce_weight=0.3, dice_weight=0.7, class_weights=None):
    """
    获取肋骨分割损失函数
    Args:
        ce_weight: 交叉熵损失权重
        dice_weight: Dice损失权重
        class_weights: 类别权重
    """
    return RibSegLoss(ce_weight=ce_weight, dice_weight=dice_weight, class_weights=class_weights)


# 为了保持兼容性，保留原始接口
class get_model_legacy(nn.Module):
    """
    保持与原始PointNet++接口兼容的模型
    """
    def __init__(self, num_part, num_category, normal_channel=False):
        super(get_model_legacy, self).__init__()
        # 使用新的肋骨分割模型
        self.rib_seg_model = RibSegModel(num_classes=num_part, normal_channel=normal_channel)
        
    def forward(self, xyz, cls_label=None):
        """
        兼容原始接口的前向传播
        Args:
            xyz: 输入点云
            cls_label: 类别标签（肋骨分割中不使用，但保持接口兼容）
        """
        return self.rib_seg_model(xyz)


class get_loss_legacy(nn.Module):
    """
    保持与原始PointNet++接口兼容的损失函数
    """
    def __init__(self):
        super(get_loss_legacy, self).__init__()
        self.rib_seg_loss = RibSegLoss()
        
    def forward(self, pred, target, trans_feat=None):
        return self.rib_seg_loss(pred, target, trans_feat)
