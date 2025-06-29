import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import time
import json
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import pandas as pd
import copy
import math
from scipy.optimize import linear_sum_assignment


# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_theme()

torch.backends.cudnn.benchmark = True


# 位置编码实现
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 应该是 hidden_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # 确保输入掩码是3维 [batch_size, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # 改为增加通道维度而不是批次维度

        not_mask = ~mask.squeeze(1)  # 移除通道维度
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # 修正位置编码计算
        pos_x = torch.stack((
            pos_x[:, :, :, 0::2].sin(),
            pos_x[:, :, :, 1::2].cos()
        ), dim=4).flatten(3)

        pos_y = torch.stack((
            pos_y[:, :, :, 0::2].sin(),
            pos_y[:, :, :, 1::2].cos()
        ), dim=4).flatten(3)

        # 合并位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# Transformer中的多头注意力机制
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True):
        batch_size = query.size(0)

        # 移除冗余的调试信息
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.out_linear(attn_output)

        return output, attn_weights if need_weights else None


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 确保d_model能被nhead整除
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead = nhead

        # 创建编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer)
                                      for _ in range(num_encoder_layers)])

        # 创建解码器层
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer)
                                      for _ in range(num_decoder_layers)])

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0)

    def forward(self, src, query_embed, pos_embed):
        print(f"\nTransformer input shapes:")
        print(f"src shape: {src.shape}")
        print(f"query_embed shape: {query_embed.shape}")
        print(f"pos_embed shape: {pos_embed.shape}")

        bs, c, h, w = src.shape

        # 重塑输入
        src = src.flatten(2).permute(0, 2, 1)  # (bs, h*w, c)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # (bs, h*w, c)
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)  # (bs, num_queries, c)
        tgt = torch.zeros_like(query_embed)

        print(f"\nReshaped dimensions:")
        print(f"src: {src.shape}")
        print(f"pos_embed: {pos_embed.shape}")
        print(f"query_embed: {query_embed.shape}")
        print(f"tgt: {tgt.shape}")

        # 编码器处理
        memory = src
        for layer in self.encoder:
            memory = layer(memory + pos_embed)

        # 解码器处理
        output = tgt
        for layer in self.decoder:
            output = layer(output + query_embed, memory + pos_embed)

        return output, memory


# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2, _ = self.self_attn(src, src, src,
                                 key_padding_mask=src_key_padding_mask,
                                 need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# Transformer解码器层
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.multihead_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 自注意力
        tgt2, _ = self.self_attn(tgt, tgt, tgt,
                                 key_padding_mask=tgt_key_padding_mask,
                                 need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力
        tgt2, _ = self.multihead_attn(tgt, memory, memory,
                                      key_padding_mask=memory_key_padding_mask,
                                      need_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# 匈牙利匹配器
class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器的实现
    将预测框与目标框进行最优匹配
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        初始化匹配器的参数
        Args:
            cost_class: 类别损失的权重
            cost_bbox: 边界框L1损失的权重
            cost_giou: GIoU损失的权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "所有成本都不能为0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        执行匈牙利匹配
        Args:
            outputs: 模型的输出字典，包含'pred_logits'和'pred_boxes'
            targets: 目标列表，每个目标包含'boxes'和'labels'
        Returns:
            列表[tuple]，包含每个图像的(pred_idx, tgt_idx)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 同样展平目标
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算类别成本
        cost_class = -out_prob[:, tgt_ids]

        # 计算L1成本
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算GIoU成本
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def box_cxcywh_to_xyxy(x):
    """
    将中心点-宽高格式的边界框转换为左上角-右下角格式
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    计算两组框之间的广义IoU
    """
    # 获取框的坐标
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    # 计算外接矩形的坐标
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_area(boxes):
    """
    计算一组框的面积
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # 保持与预训练权重一致的结构
        self.backbone = nn.Sequential(
            nn.ModuleList([
                torchvision.models.resnet50(pretrained=True)
            ])
        )
        # 移除fc层
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 移除最后的fc和avgpool层

        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # 修改transformer结构以匹配预训练权重
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4
        )

        # 其他组件保持不变
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(100, hidden_dim)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # 修改损失权重
        self.weight_dict = {
            'loss_ce': 1,
            'loss_bbox': 5,  # 降低边界框损失的权重
            'loss_giou': 2
        }

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 特别初始化bbox_embed
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes + 1) * bias_value

        # 初始化bbox_embed为小值
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def compute_loss(self, outputs, targets):
        """修改损失计算"""
        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        idx = self._get_src_permutation_idx(indices)

        # 分类损失
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

        # 边界框损失 - 添加值范围检查
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 确保预测框和目标框的值在[0,1]范围内
        src_boxes = torch.clamp(src_boxes, 0, 1)
        target_boxes = torch.clamp(target_boxes, 0, 1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.mean()

        losses = {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }

        # 应用权重
        return {k: v * self.weight_dict[k] for k, v in losses.items()}

    def _get_src_permutation_idx(self, indices):
        """辅助函数：获取源序列的排列索引"""
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

# MLP网络
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CustomDetrFeatureExtractor:
    """自定义的DETR特征提取器"""

    def __init__(self, target_size=(800, 800)):
        self.target_size = target_size  # 固定目标尺寸
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def __call__(self, images, annotations=None):
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        processed_annotations = []

        for img in images:
            # 获取原始尺寸
            width, height = img.size
            orig_size = torch.tensor([height, width])

            # 计算缩放比例
            scale_w = self.target_size[0] / width
            scale_h = self.target_size[1] / height

            # 调整图像大小到固定尺寸
            img = img.resize(self.target_size, Image.BILINEAR)

            # 转换为张量
            img = transforms.ToTensor()(img)

            # 标准化
            img = transforms.Normalize(mean=self.image_mean, std=self.image_std)(img)

            # 不要在这里堆叠图像
            processed_images.append(img)

            if annotations is not None:
                # 处理标注
                boxes = annotations['boxes']
                scaled_boxes = boxes.clone()
                scaled_boxes[:, [0, 2]] *= scale_w
                scaled_boxes[:, [1, 3]] *= scale_h

                processed_annotation = {
                    'boxes': scaled_boxes,
                    'labels': annotations['labels'],
                    'image_id': annotations.get('image_id', torch.tensor([0])),
                    'orig_size': orig_size,
                    'size': torch.tensor([self.target_size[1], self.target_size[0]])
                }

                processed_annotations.append(processed_annotation)

        # 如果只有一个图像，直接返回处理后的张量
        if len(processed_images) == 1:
            batch = {
                'pixel_values': processed_images[0],  # 这里不需要stack
                'labels': processed_annotations[0] if annotations is not None else None
            }
        else:
            # 多个图像时才进行堆叠
            batch = {
                'pixel_values': torch.stack(processed_images),
                'labels': processed_annotations if annotations is not None else None
            }

        return batch


class VOCDataset(Dataset):
    def __init__(self, root, year='2012', image_set='train', transform=None):
        """
        初始化VOC数据集
        Args:
            root (str): 数据集根目录，包含VOCdevkit文件夹
            year (str): VOC数据集年份，默认为'2012'
            image_set (str): 数据集类型，'train'或'val'
            transform (callable, optional): 可选的图像转换
        """
        super(VOCDataset, self).__init__()

        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform

        # 使用固定尺寸的特征提取器
        self.feature_extractor = CustomDetrFeatureExtractor(target_size=(800, 800))

        # VOC类别
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 获取图像列表
        self.images = []
        self.annotations = []

        # 构建文件路径
        image_sets_file = os.path.join(
            self.root, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt'
        )

        # 检查文件是否存在
        if not os.path.exists(image_sets_file):
            raise RuntimeError(f'Dataset not found or corrupted: {image_sets_file}')

        # 读取图像列表
        with open(image_sets_file) as f:
            for line in f:
                image_id = line.strip()

                # 构建图像和标注文件的完整路径
                image_path = os.path.join(
                    self.root, 'VOC' + year, 'JPEGImages', image_id + '.jpg'
                )
                annotation_path = os.path.join(
                    self.root, 'VOC' + year, 'Annotations', image_id + '.xml'
                )

                # 只添加同时存在图像和标注的样本
                if os.path.exists(image_path) and os.path.exists(annotation_path):
                    self.images.append(image_path)
                    self.annotations.append(annotation_path)
                else:
                    print(f"警告：跳过缺失文件的样本 {image_id}")

        print(f"加载了 {len(self.images)} 个 {image_set} 样本")

        # 限制数据集大小（用于测试）
        if image_set == 'train':
            self.images = self.images[:1000]  # 只用前1000张训练图像
            self.annotations = self.annotations[:1000]
        elif image_set == 'val':
            self.images = self.images[:200]  # 只用前200张验证图像
            self.annotations = self.annotations[:200]

        if len(self.images) == 0:
            raise RuntimeError(f'No valid images found in {self.root}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            # 加载图像
            img_path = self.images[idx]
            if not os.path.exists(img_path):
                print(f"警告：图像文件不存在: {img_path}")
                return self._get_empty_item()

            img = Image.open(img_path)
            if img is None:
                print(f"警告：无法加载图像: {img_path}")
                return self._get_empty_item()

            img = img.convert("RGB")
            width, height = img.size

            # 解析XML标注文件
            ann_path = self.annotations[idx]
            if not os.path.exists(ann_path):
                print(f"警告：标注文件不存在: {ann_path}")
                return self._get_empty_item()

            tree = ET.parse(ann_path)
            root = tree.getroot()

            boxes = []
            labels = []
            areas = []

            # 处理每个目标
            for obj in root.iter('object'):
                try:
                    difficult = int(obj.find('difficult').text)
                    if difficult:  # 跳过difficult的样本
                        continue

                    name = obj.find('name').text
                    if name not in self.class_to_idx:  # 跳过未知类别
                        continue

                    label = self.class_to_idx[name]

                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # 确保边界框坐标在图像范围内
                    xmin = max(0, min(xmin, width))
                    ymin = max(0, min(ymin, height))
                    xmax = max(0, min(xmax, width))
                    ymax = max(0, min(ymax, height))

                    # 确保边界框有效
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    # 计算面积
                    area = (xmax - xmin) * (ymax - ymin)

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
                    areas.append(area)
                except Exception as e:
                    print(f"警告：处理目标时出错: {str(e)}")
                    continue

            # 处理无目标的情况
            if len(boxes) == 0:
                # 创建一个虚拟的背景目标
                boxes = torch.zeros((1, 4), dtype=torch.float32)
                labels = torch.zeros(1, dtype=torch.int64)
                areas = torch.zeros(1, dtype=torch.float32)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                areas = torch.as_tensor(areas, dtype=torch.float32)

            # 准备目标字典
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': areas,
                'orig_size': torch.tensor([height, width]),
                'size': torch.tensor([height, width])
            }

            # 使用特征提取器处理图像和标注
            try:
                encodings = self.feature_extractor(img, target)
                if encodings is None:
                    print(f"警告：特征提取器返回None: {img_path}")
                    return self._get_empty_item()
                return encodings
            except Exception as e:
                print(f"警告：特征提取失败: {str(e)}")
                return self._get_empty_item()

        except Exception as e:
            print(f"警告：处理样本 {idx} 时出错: {str(e)}")
            return self._get_empty_item()

    def _get_empty_item(self):
        """返回一个有效的空样本"""
        empty_image = torch.zeros((3, 800, 800), dtype=torch.float32)
        empty_boxes = torch.zeros((1, 4), dtype=torch.float32)
        empty_labels = torch.zeros(1, dtype=torch.int64)

        return {
            'pixel_values': empty_image,
            'labels': {
                'boxes': empty_boxes,
                'labels': empty_labels,
                'image_id': torch.tensor([0]),
                'area': torch.zeros(1, dtype=torch.float32),
                'orig_size': torch.tensor([800, 800]),
                'size': torch.tensor([800, 800])
            }
        }




class DETRDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 20  # VOC数据集的类别数
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'VOCdevkit')
        self.weights_dir = os.path.join(self.base_dir, 'weights')
        self.weights_path = os.path.join(self.weights_dir, 'detr-r50-e632da11.pth')
        self.model = None
        self.train_loader = None
        self.val_loader = None

        # 扩展metrics字典
        self.metrics = {
            'train_loss': [],
            'val_map': [],
            'inference_time': [],
            'per_class_ap': {},
            'per_class_precision': {},
            'per_class_recall': {},
            'epoch_times': [],
            'learning_rates': [],
            'total_parameters': 0,
            'trainable_parameters': 0,
            'per_epoch_class_ap': {},
            'per_epoch_class_precision': {},
            'per_epoch_class_recall': {}
        }

        # 添加结果保存目录
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        self.experiment_timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Data directory: {self.data_dir}")
        print(f"Weights directory: {self.weights_dir}")

    def _build_model(self):
        """构建DETR模型"""
        print("\n=== 构建DETR模型 ===")
        print(f"当前时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"用户: chanyeol88")

        try:
            # 创建模型实例
            model = DETR(num_classes=self.num_classes)
            print("DETR模型实例化成功")

            # 加载预训练权重
            if os.path.exists(self.weights_path):
                print(f"\n开始加载预训练权重: {self.weights_path}")
                try:
                    # 加载权重文件
                    state_dict = torch.load(self.weights_path, map_location=self.device)

                    # 处理权重字典
                    if 'model' in state_dict:
                        state_dict = state_dict['model']

                    # 创建新的状态字典
                    new_state_dict = {}

                    # 处理键名不匹配
                    for k, v in state_dict.items():
                        # 处理backbone部分
                        if k.startswith('backbone.0.body'):
                            new_k = k.replace('backbone.0.body', 'backbone.0')
                            new_state_dict[new_k] = v
                        # 处理transformer编码器部分
                        elif k.startswith('transformer.encoder.layers'):
                            new_k = k.replace('layers', '')
                            new_state_dict[new_k] = v
                        # 处理transformer解码器部分
                        elif k.startswith('transformer.decoder.layers'):
                            new_k = k.replace('layers', '')
                            new_state_dict[new_k] = v
                        else:
                            new_state_dict[k] = v

                    # 加载处理后的权重
                    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

                    if len(missing_keys) > 0:
                        print("\n警告：以下键在预训练权重中缺失:")
                        for key in missing_keys:
                            print(f"  {key}")

                    if len(unexpected_keys) > 0:
                        print("\n警告：发现意外的键:")
                        for key in unexpected_keys:
                            print(f"  {key}")

                    print("\n预训练权重加载成功！")

                except Exception as e:
                    print(f"\n加载预训练权重时出错: {str(e)}")
                    print("将使用随机初始化的权重继续")
            else:
                print(f"\n未找到预训练权重文件: {self.weights_path}")
                print("使用随机初始化的权重")

            # 将模型移到指定设备
            model = model.to(self.device)

            # 计算并打印模型参数信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n模型参数统计:")
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")

            # 保存到指标字典
            self.metrics['total_parameters'] = total_params
            self.metrics['trainable_parameters'] = trainable_params

            print("\n模型构建成功！")

            return model

        except Exception as e:
            print(f"\n构建模型时出错: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            raise

    def prepare_data(self):
        """准备数据加载器"""
        print("\n准备数据加载器...")

        try:
            train_dataset = VOCDataset(
                root=self.data_dir,
                year='2012',
                image_set='train'
            )

            val_dataset = VOCDataset(
                root=self.data_dir,
                year='2012',
                image_set='val'
            )

            print("\n创建数据加载器...")
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                collate_fn=self._collate_fn
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=self._collate_fn
            )

            print(f"训练样本数: {len(train_dataset)}")
            print(f"验证样本数: {len(val_dataset)}")
            print("数据加载器创建成功！")

        except Exception as e:
            print(f"准备数据时出错: {str(e)}")
            print("请确保数据集路径正确，且数据集完整。")
            raise

    def _collate_fn(self, batch):
        try:
            # 过滤掉无效的样本
            valid_batch = [item for item in batch if item is not None and
                           'pixel_values' in item and 'labels' in item]

            if not valid_batch:
                print("警告：批次中没有有效样本")
                return self._get_empty_batch()

            # 确保所有图像都是3D张量 (C, H, W)
            pixel_values = [item['pixel_values'] for item in valid_batch]

            # 检查维度
            for i, pv in enumerate(pixel_values):
                if pv.dim() != 3:
                    print(f"警告：第{i}个图像维度错误，维度为{pv.dim()}")
                    continue

            # 堆叠图像
            pixel_values = torch.stack(pixel_values)

            # 收集标签
            labels = [item['labels'] for item in valid_batch]

            return {
                'pixel_values': pixel_values,
                'labels': labels
            }
        except Exception as e:
            print(f"警告：批处理失败: {str(e)}")
            return self._get_empty_batch()

    def _get_empty_batch(self):
        """返回一个有效的空批次"""
        return {
            'pixel_values': torch.zeros((1, 3, 800, 800), dtype=torch.float32),
            'labels': [{
                'boxes': torch.zeros((1, 4), dtype=torch.float32),
                'labels': torch.zeros(1, dtype=torch.int64),
                'image_id': torch.tensor([0]),
                'area': torch.zeros(1, dtype=torch.float32),
                'orig_size': torch.tensor([800, 800]),
                'size': torch.tensor([800, 800])
            }]
        }

    def verify_dataset(self):
        """验证数据集是否存在且完整"""
        print("\n验证 VOC2012 数据集...")

        # 检查必要的目录和文件是否存在
        required_paths = [
            os.path.join(self.data_dir, 'VOC2012', 'ImageSets', 'Main', 'train.txt'),
            os.path.join(self.data_dir, 'VOC2012', 'ImageSets', 'Main', 'val.txt'),
            os.path.join(self.data_dir, 'VOC2012', 'JPEGImages'),
            os.path.join(self.data_dir, 'VOC2012', 'Annotations')
        ]

        for path in required_paths:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"必需的路径不存在: {path}\n"
                    "请确保您已正确放置 VOC2012 数据集在 VOCdevkit 目录中。"
                )

        # 计算数据集大小
        train_file = os.path.join(self.data_dir, 'VOC2012', 'ImageSets', 'Main', 'train.txt')
        val_file = os.path.join(self.data_dir, 'VOC2012', 'ImageSets', 'Main', 'val.txt')

        try:
            with open(train_file) as f:
                train_samples = len(f.readlines())
            with open(val_file) as f:
                val_samples = len(f.readlines())

            print(f"数据集验证完成！")
            print(f"找到 {train_samples} 个训练样本")
            print(f"找到 {val_samples} 个验证样本")

            # 验证权重文件
            if os.path.exists(self.weights_path):
                print(f"找到预训练权重文件: {self.weights_path}")
            else:
                print(f"警告：预训练权重文件不存在: {self.weights_path}")
                print("模型将使用随机初始化的权重")

            return True

        except Exception as e:
            print(f"验证数据集时出错: {str(e)}")
            raise

    def train(self, num_epochs=10):
        """训练模型"""
        print("\n=== 训练开始 ===")
        print(f"当前时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"用户: chanyeol88")
        print(f"总轮次: {num_epochs}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")

        if self.model is None:
            self.model = self._build_model()

        # 设置优化器
        param_dicts = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_map = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\n=== Epoch [{epoch + 1}/{num_epochs}] ===")
            epoch_start_time = time.time()
            self.model.train()
            total_loss = 0
            valid_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"训练")

            for i, batch in enumerate(progress_bar):
                try:
                    if batch is None or 'pixel_values' not in batch or 'labels' not in batch:
                        continue

                    images = batch['pixel_values'].to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

                    # 维度检查
                    if images.dim() != 4:
                        print(f"\n跳过批次 {i}: 图像维度错误 ({images.dim()})")
                        continue

                    # 前向传播和损失计算
                    outputs, loss_dict = self.model(images, targets)
                    losses = sum(loss_dict.values())

                    if not torch.isfinite(losses):
                        print(f"\n跳过批次 {i}: 损失值无效 ({losses.item()})")
                        continue

                    # 反向传播和优化
                    optimizer.zero_grad()
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    optimizer.step()

                    total_loss += losses.item()
                    valid_batches += 1

                    # 更新进度条信息
                    progress_bar.set_postfix({
                        'loss': f"{losses.item():.4f}",
                        'avg_loss': f"{total_loss / valid_batches:.4f}",
                        'valid_batches': f"{valid_batches}/{i + 1}"
                    })

                    # 详细损失信息
                    if (i + 1) % 50 == 0:
                        print("\n=== 批次损失详情 ===")
                        print(f"Batch [{i + 1}/{len(self.train_loader)}]")
                        print(f"当前损失: {losses.item():.4f}")
                        print(f"平均损失: {total_loss / valid_batches:.4f}")
                        print(f"有效批次: {valid_batches}/{i + 1}")
                        print("各项损失:")
                        for k, v in loss_dict.items():
                            print(f"  {k}: {v.item():.4f}")

                except RuntimeError as e:
                    print(f"\n处理批次 {i} 时出错: {str(e)}")
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    continue

            # 每个epoch结束后的处理
            print("\n=== Epoch 总结 ===")
            epoch_time = time.time() - epoch_start_time
            epoch_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')

            # 更新指标
            self.metrics['train_loss'].append(epoch_loss)
            self.metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
            self.metrics['epoch_times'].append(epoch_time)

            print(f"Epoch耗时: {epoch_time:.2f}秒")
            print(f"平均损失: {epoch_loss:.4f}")
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 更新学习率
            lr_scheduler.step()

            # GPU清理和冷却
            print("\n执行GPU清理和冷却...")
            torch.cuda.empty_cache()
            time.sleep(30)

            # 验证
            if (epoch + 1) % 2 == 0:
                print("\n=== 执行验证 ===")
                results, map_score = self.evaluate_detailed(epoch)
                self.metrics['val_map'].append(map_score)

                # 保存最佳模型
                if map_score > best_map:
                    best_map = map_score
                    best_model_path = os.path.join(
                        self.weights_dir,
                        f'best_detr_model_{self.experiment_timestamp}.pth'
                    )
                    self.save_checkpoint(best_model_path)
                    print(f"\n发现新的最佳模型！")
                    print(f"最佳mAP: {best_map:.4f}")
                    print(f"模型已保存至: {best_model_path}")

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.weights_dir,
                    f'detr_checkpoint_epoch_{epoch + 1}_{self.experiment_timestamp}.pth'
                )
                self.save_checkpoint(checkpoint_path)
                print(f"\n保存检查点: {checkpoint_path}")

            # 绘制并保存指标曲线
            self.plot_metrics(epoch + 1)

        # 训练结束总结
        total_time = time.time() - start_time
        print("\n=== 训练完成 ===")
        print(f"总训练时间: {total_time / 3600:.2f} 小时")
        print(f"最终最佳mAP: {best_map:.4f}")
        print(f"结束时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

        # 保存最终指标
        self.save_metrics()

def main():
    """主函数"""
    print("初始化 DETR 检测系统...")
    print(f"当前时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"用户: chanyeol88")

    detector = DETRDetector()

    try:
        # 验证数据集
        detector.verify_dataset()

        # 准备数据加载器
        detector.prepare_data()

        # 训练模型
        detector.train(num_epochs=10)

        print("\n完整流程执行成功！")

    except Exception as e:
        print(f"\n执行过程中出错: {str(e)}")
        raise


if __name__ == '__main__':
    main()