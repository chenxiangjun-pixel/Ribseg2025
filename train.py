import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_utils.RibFracDataLoader_1cls import PartNormalDataset
from data_utils.endpoint_utils import detect_endpoints_from_points, detect_endpoints_per_component, detect_24x2_endpoints
sys.path.append(os.path.dirname(os.path.abspath(__file__)))    # 添加项目根目录到路径


def calculate_dice_coefficient(pred, target, smooth=1e-6):
	"""计算Dice系数（基于二分类标签）"""
	pred = torch.softmax(pred, dim=-1)
	pred = pred.argmax(dim=-1)
	pred_flat = pred.contiguous().view(-1)
	target_flat = target.contiguous().view(-1)
	intersection = (pred_flat * target_flat).sum()
	dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
	return dice.item()


def calculate_iou(pred, target, smooth=1e-6):
	"""计算IoU（基于二分类标签）"""
	pred = torch.softmax(pred, dim=-1)
	pred = pred.argmax(dim=-1)
	pred_flat = pred.contiguous().view(-1)
	target_flat = target.contiguous().view(-1)
	intersection = (pred_flat * target_flat).sum()
	union = pred_flat.sum() + target_flat.sum() - intersection
	iou = (intersection + smooth) / (union + smooth)
	return iou.item()


def calculate_metrics(pred, target, num_classes=2) -> Dict[str, float]:
	"""计算其余评估指标（Accuracy/Precision/Recall/Specificity/F1-score）"""
	pred = torch.softmax(pred, dim=-1)
	pred_labels = pred.argmax(dim=-1)
	pred_flat = pred_labels.contiguous().view(-1)
	target_flat = target.contiguous().view(-1)

	tp = ((pred_flat == 1) & (target_flat == 1)).sum().float()
	tn = ((pred_flat == 0) & (target_flat == 0)).sum().float()
	fp = ((pred_flat == 1) & (target_flat == 0)).sum().float()
	fn = ((pred_flat == 0) & (target_flat == 1)).sum().float()

	accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
	precision = tp / (tp + fp + 1e-6)
	recall = tp / (tp + fn + 1e-6)  # Sensitivity
	specificity = tn / (tn + fp + 1e-6)
	f1_score = 2 * precision * recall / (precision + recall + 1e-6)

	dice = calculate_dice_coefficient(pred, target)
	iou = calculate_iou(pred, target)

	return {
		'accuracy': float(accuracy.item()),
		'precision': float(precision.item()),
		'recall': float(recall.item()),
		'sensitivity': float(recall.item()),
		'specificity': float(specificity.item()),
		'f1_score': float(f1_score.item()),
		'dice': float(dice),
		'iou': float(iou),
	}


class EarlyStopping:
	"""早停机制"""
	def __init__(self, patience=10, min_delta=0.001):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_score = None
		self.early_stop = False

	def __call__(self, val_score):
		if self.best_score is None:
			self.best_score = val_score
		elif val_score < self.best_score + self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = val_score
			self.counter = 0


def load_config(config_path: str) -> Dict[str, Any]:
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	return config


def log_string(str):
	try:
		logger.info(str)
	except:
		print(str)


def main():
	parser = argparse.ArgumentParser(description='肋骨分割训练')
	parser.add_argument('--config', type=str, default='config/train_config.yaml', help='配置文件路径')
	parser.add_argument('--batch_size', type=int, help='批次大小')
	parser.add_argument('--epochs', type=int, help='训练轮数')
	parser.add_argument('--lr', type=float, help='学习率')
	parser.add_argument('--device', type=str, default='auto', help='设备')
	args = parser.parse_args()

	# 加载配置
	config = load_config(args.config)

	# 命令行覆盖
	if args.batch_size is not None:
		config['data']['batch_size'] = args.batch_size
	if args.epochs is not None:
		config['training']['epochs'] = args.epochs
	if args.lr is not None:
		config['training']['learning_rate'] = args.lr

	# 设备
	if args.device == 'auto':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device(args.device)
	log_string(f'Using device: {device}')

	# 固定实验目录命名（优先配置）
	experiment_name = (
		config.get('experiment', {}).get('name')
		if isinstance(config.get('experiment'), dict) else None
	)
	if not experiment_name or not str(experiment_name).strip():
		experiment_name = 'ribseg_exp'  # 默认固定名称

	experiment_dir = Path('log/part_seg') / experiment_name
	experiment_dir.mkdir(parents=True, exist_ok=True)

	# 保存配置快照
	with open(experiment_dir / 'config.yaml', 'w', encoding='utf-8') as f:
		yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

	# 日志
	global logger
	import logging
	logger = logging.getLogger('rib_seg')
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(experiment_dir / 'training.log', encoding='utf-8')
	file_handler.setLevel(logging.INFO)
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)
	logger.handlers = []
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# TensorBoard
	tb_dir = experiment_dir / 'tensorboard'
	tb_dir.mkdir(exist_ok=True)
	writer = SummaryWriter(log_dir=str(tb_dir))

	log_string('CONFIGURATION ...')
	log_string(f"Model: {config['model']['name']}")
	log_string(f"Batch size: {config['data']['batch_size']}")
	log_string(f"Epochs: {config['training']['epochs']}")
	log_string(f"Learning rate: {config['training']['learning_rate']}")
	log_string(f"Optimizer: {config['training']['optimizer']}")
	log_string(f"Use attention: {config['model']['use_attention']}")
	log_string(f"Experiment dir: {experiment_dir}")

	# 数据集
	train_dataset = PartNormalDataset(
		root=config['data']['root'],
		npoints=config['data']['npoint'],
		split='train',
		normal_channel=config['model']['normal_channel']
	)
	val_dataset = PartNormalDataset(
		root=config['data']['root'],
		npoints=config['data']['npoint'],
		split='val',
		normal_channel=config['model']['normal_channel']
	)
	log_string(f"The number of training data is: {len(train_dataset)}")
	log_string(f"The number of test data is: {len(val_dataset)}")

	# DataLoader（兼容Windows）
	try:
		if os.name == 'nt':
			num_workers = 0
			pin_memory = False
		else:
			num_workers = config['data']['num_workers']
			pin_memory = config['data']['pin_memory']
		train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=config['data']['batch_size'],
			shuffle=True,
			num_workers=num_workers,
			pin_memory=pin_memory,
			drop_last=True,
		)
		val_loader = torch.utils.data.DataLoader(
			val_dataset,
			batch_size=config['data']['batch_size'],
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
			drop_last=False,
		)
	except Exception as e:
		log_string(f"DataLoader initialization failed: {e}, using fallback settings")
		train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=config['data']['batch_size'],
			shuffle=True,
			num_workers=0,
			pin_memory=False,
			drop_last=True,
		)
		val_loader = torch.utils.data.DataLoader(
			val_dataset,
			batch_size=config['data']['batch_size'],
			shuffle=False,
			num_workers=0,
			pin_memory=False,
			drop_last=False,
		)

	# 模型
	try:
		if config['model']['name'] == 'pointnet2_rib_seg':
			from models.ribseg_model import get_model, get_loss
			classifier = get_model(
				num_classes=config['model']['num_part'],
				normal_channel=config['model']['normal_channel'],
				use_attention=config['model']['use_attention'],
			)
		else:
			from models.pointnet2_part_seg_msg import get_model, get_loss
			classifier = get_model(
				num_part=config['model']['num_part'],
				num_category=config['model']['num_category'],
				normal_channel=config['model']['normal_channel'],
			)
		criterion = get_loss()
		log_string(f"Model loaded: {config['model']['name']}")
	except Exception as e:
		log_string(f"Model loading failed: {e}")
		writer.close()
		return

	total_params = sum(p.numel() for p in classifier.parameters())
	log_string(f"Total parameters: {total_params:,}")

	classifier = classifier.to(device)
	criterion = criterion.to(device)

	# 优化器
	if config['training']['optimizer'] == 'Adam':
		optimizer = torch.optim.Adam(
			classifier.parameters(),
			lr=float(config['training']['learning_rate']),
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=float(config['training']['weight_decay']),
		)
	elif config['training']['optimizer'] == 'AdamW':
		optimizer = torch.optim.AdamW(
			classifier.parameters(),
			lr=float(config['training']['learning_rate']),
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=float(config['training']['weight_decay']),
		)
	else:
		optimizer = torch.optim.SGD(
			classifier.parameters(),
			lr=float(config['training']['learning_rate']),
			momentum=0.9,
			weight_decay=float(config['training']['weight_decay']),
		)

	# 学习率调度
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer,
		step_size=int(config['training']['step_size']),
		gamma=float(config['training']['lr_decay']),
	)

	# 早停
	early_stopping = EarlyStopping(
		patience=config['early_stopping']['patience'],
		min_delta=config['early_stopping']['min_delta'],
	)

	# 断点续训（固定目录便于复用）
	best_model_path = experiment_dir / 'best_model.pth'
	start_epoch = 0
	best_val_dice = 0.0
	if best_model_path.exists():
		log_string(f"Loading existing model from {best_model_path}")
		checkpoint = torch.load(best_model_path, map_location=device)
		classifier.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		best_val_dice = checkpoint.get('best_val_dice', 0.0)
		log_string(f"Resumed from epoch {start_epoch}, best_val_dice: {best_val_dice:.4f}")
	else:
		log_string("No existing model, starting training from scratch...")

	log_string("Starting training...")

	for epoch in range(start_epoch, config['training']['epochs']):
		log_string(f'Epoch {epoch + 1} ({epoch + 1}/{config["training"]["epochs"]}):')
		log_string(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

		# BN momentum（简单固定）
		for m in classifier.modules():
			if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
				m.momentum = 0.1
		log_string(f'BN momentum updated to: {0.1:.6f}')

		# 训练
		classifier.train()
		train_metrics = {
			'loss': 0.0, 'accuracy': 0.0, 'dice': 0.0, 'iou': 0.0,
			'precision': 0.0, 'recall': 0.0, 'sensitivity': 0.0,
			'specificity': 0.0, 'f1_score': 0.0,
		}
		train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
		for batch_id, (points, cls, target) in enumerate(train_pbar):
			points, target = points.to(device), target.to(device)
			points = points.transpose(2, 1)  # [B, 3, N]

			optimizer.zero_grad()

			# 数据增强（全Torch张量，GPU安全）
			if config['training'].get('use_augmentation', True):
				if torch.rand(1).item() > 0.5:
					scale = torch.rand(1).item() * 0.4 + 0.8
					points[:, :3, :] *= scale
				if torch.rand(1).item() > 0.5:
					shift = torch.rand(1, 3, 1, device=points.device) * 0.2 - 0.1
					points[:, :3, :] += shift
				if torch.rand(1).item() > 0.5:
					jitter = torch.randn_like(points) * 0.01
					points += jitter

			seg_pred, trans_feat = classifier(points)
			seg_pred = seg_pred.contiguous().view(-1, config['model']['num_part'])

			if target.dim() > 1:
				target = target.view(-1)
			target = target.long()

			loss = criterion(seg_pred, target, trans_feat)
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				metrics = calculate_metrics(seg_pred, target, config['model']['num_part'])
				for key in metrics:
					if key in train_metrics:
						train_metrics[key] += metrics[key]
				train_metrics['loss'] += float(loss.item())

			train_pbar.set_postfix({
				'Loss': f'{loss.item():.4f}',
				'Dice': f'{metrics["dice"]:.4f}',
				'Acc': f'{metrics["accuracy"]:.4f}',
			})

		# 训练均值
		for key in train_metrics:
			train_metrics[key] /= max(len(train_loader), 1)

		# 验证
		classifier.eval()
		val_metrics = {
			'loss': 0.0, 'accuracy': 0.0, 'dice': 0.0, 'iou': 0.0,
			'precision': 0.0, 'recall': 0.0, 'sensitivity': 0.0,
			'specificity': 0.0, 'f1_score': 0.0,
		}
		endpoints_dump = []

		with torch.no_grad():
			val_pbar = tqdm(val_loader, desc='Validation')
			for batch_id, (points, cls, target) in enumerate(val_pbar):
				points, target = points.to(device), target.to(device)
				points = points.transpose(2, 1)

				seg_pred, trans_feat = classifier(points)
				seg_pred_flat = seg_pred.contiguous().view(-1, config['model']['num_part'])

				if target.dim() > 1:
					target_flat = target.view(-1)
				else:
					target_flat = target
				target_flat = target_flat.long()

				loss = criterion(seg_pred_flat, target_flat, trans_feat)

				metrics = calculate_metrics(seg_pred_flat, target_flat, config['model']['num_part'])
				for key in metrics:
					if key in val_metrics:
						val_metrics[key] += metrics[key]
				val_metrics['loss'] += float(loss.item())

				# 端点检测
				try:
					points_np = points.transpose(2, 1).detach().cpu().numpy()  # [B,N,3]
					pred_choice = seg_pred.argmax(dim=-1).detach().cpu().numpy()  # [B,N]
					for bi in range(points.shape[0]):
						xyz = points_np[bi]
						lbl = pred_choice[bi]
						endpoints_global = detect_endpoints_from_points(xyz, lbl, min_positive_points=200)
						endpoints_components = detect_endpoints_per_component(
							xyz, lbl, voxel_size=0.02, min_points_component=150, keep_topk_per_comp=2,
						)
						endpoints_24x2_result = detect_24x2_endpoints(
							xyz, lbl, voxel_size=0.02, min_points_component=150,
						)
						record = {
							'sample': f'{batch_id}:{bi}',
							'global_endpoints': endpoints_global,
							'components': endpoints_components,
							'endpoints_24x2': endpoints_24x2_result,
						}
						endpoints_dump.append(record)
				except Exception:
					pass

				val_pbar.set_postfix({
					'Loss': f'{loss.item():.4f}',
					'Dice': f'{metrics["dice"]:.4f}',
					'Acc': f'{metrics["accuracy"]:.4f}',
				})

		# 验证均值
		for key in val_metrics:
			val_metrics[key] /= max(len(val_loader), 1)

		# 导出端点 JSON
		try:
			export_dir = experiment_dir / 'endpoints'
			export_dir.mkdir(exist_ok=True)
			with open(export_dir / f'val_endpoints_epoch_{epoch + 1}.json', 'w', encoding='utf-8') as f:
				json.dump(endpoints_dump, f, ensure_ascii=False, indent=2)
			# 统计
			total_samples = len(endpoints_dump)
			successful_24x2 = sum(1 for r in endpoints_dump if r['endpoints_24x2']['success_count'] > 0)
			avg_components = sum(r['endpoints_24x2']['components_used'] for r in endpoints_dump) / max(total_samples, 1)
			avg_success = sum(r['endpoints_24x2']['success_count'] for r in endpoints_dump) / max(total_samples, 1)
			log_string(f'Val endpoints: {successful_24x2}/{total_samples} samples, avg {avg_components:.1f} components, avg {avg_success:.1f}/24 ribs')
		except Exception as e:
			log_string(f'Write endpoints JSON failed: {e}')

		# 学习率步进
		scheduler.step()

		# 日志
		log_string(
			f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.4f}, '
			f'Dice: {train_metrics["dice"]:.4f}, IoU: {train_metrics["iou"]:.4f}, '
			f'Prec: {train_metrics["precision"]:.4f}, Rec/Sens: {train_metrics["recall"]:.4f}, '
			f'Spec: {train_metrics["specificity"]:.4f}, F1: {train_metrics["f1_score"]:.4f}'
		)
		log_string(
			f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.4f}, '
			f'Dice: {val_metrics["dice"]:.4f}, IoU: {val_metrics["iou"]:.4f}, '
			f'Prec: {val_metrics["precision"]:.4f}, Rec/Sens: {val_metrics["recall"]:.4f}, '
			f'Spec: {val_metrics["specificity"]:.4f}, F1: {val_metrics["f1_score"]:.4f}'
		)

		# TensorBoard记录（标量）
		writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
		for k, v in train_metrics.items():
			writer.add_scalar(f'train/{k}', v, epoch + 1)
		for k, v in val_metrics.items():
			writer.add_scalar(f'val/{k}', v, epoch + 1)

		# 保存 best
		if val_metrics['dice'] > best_val_dice:
			best_val_dice = val_metrics['dice']
			torch.save({
				'epoch': epoch + 1,
				'model_state_dict': classifier.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'best_val_dice': best_val_dice,
				'val_metrics': val_metrics,
			}, best_model_path)
			log_string(f'New best model saved with Dice: {best_val_dice:.4f}')

		# 早停
		early_stopping(val_metrics['dice'])
		if early_stopping.early_stop:
			log_string(f'Early stopping triggered at epoch {epoch + 1}')
			break

	log_string('Training completed!')
	log_string(f'Best validation Dice: {best_val_dice:.4f}')
	writer.close()


if __name__ == '__main__':
	main()
