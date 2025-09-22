import argparse
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
from data_utils.RibFracDataLoader_1cls import PartNormalDataset
from models.ribseg_model import get_model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))

def parse_args():
    parser = argparse.ArgumentParser('Rib Segmentation Test')
    parser.add_argument('--device', type=str, default='auto', help='cuda/cpu/auto')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in testing')
    parser.add_argument('--num_point', type=int, default=30000, help='point number')
    parser.add_argument('--data_root', type=str, default='./data/pn', help='dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split')
    parser.add_argument('--experiment_name', type=str, default='ribseg_experiment', help='experiment dir name under log/part_seg')
    parser.add_argument('--weights', type=str, default='', help='override weights path .pth')
    parser.add_argument('--normal_channel', action='store_true', default=False, help='use normal information')
    parser.add_argument('--use_attention', action='store_true', default=True, help='use attention in model')
    return parser.parse_args()


def find_weights(weights_arg: str, experiment_name: str) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        if p.exists():
            return p
    p = Path('log/part_seg') / experiment_name / 'best_model.pth'
    if p.exists():
        return p
    return Path('log/best_model.pth')


def dice_coeff_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = torch.sum(pred * target)
    return float((2 * intersection + 1.0) / (torch.sum(pred) + torch.sum(target) + 1.0))


def main():
    args = parse_args()

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # 数据
    test_dataset = PartNormalDataset(
        root=args.data_root,
        npoints=args.num_point,
        split=args.split,
        normal_channel=args.normal_channel,
    )
    # Windows 兼容：num_workers=0
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'The number of {args.split} data is: {len(test_dataset)}')

    # 模型
    num_part = 2
    classifier = get_model(num_classes=num_part, normal_channel=args.normal_channel, use_attention=args.use_attention).to(device)

    # 权重
    weights_path = find_weights(args.weights, args.experiment_name)
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights not found at: {weights_path}')
    checkpoint = torch.load(str(weights_path), map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # 统计
    dice_sum = 0.0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    test_bar = tqdm(test_loader, desc='Test2.0')
    with torch.no_grad():
        for points, label, target in test_bar:
            points = points.float().to(device)  # [B, N, C]
            target = target.long().to(device)   # [B, N]
            points = points.transpose(2, 1)     # [B, C, N]

            seg_pred, _ = classifier(points)    # [B, N, 2]
            B, N, C = seg_pred.shape
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target_flat = target.view(-1)
            pred_choice = seg_pred.argmax(dim=1)

            # Dice（点级）
            dice = dice_coeff_binary(pred_choice, target_flat)
            dice_sum += dice

            # 二分类混淆矩阵
            tp += int(((pred_choice == 1) & (target_flat == 1)).sum().item())
            tn += int(((pred_choice == 0) & (target_flat == 0)).sum().item())
            fp += int(((pred_choice == 1) & (target_flat == 0)).sum().item())
            fn += int(((pred_choice == 0) & (target_flat == 1)).sum().item())

            test_bar.set_postfix(dice=f'{dice:.4f}')

    num_batches = max(len(test_loader), 1)
    test_dice = dice_sum / num_batches
    test_iou = tp / (tp + fp + fn + 1e-6)
    test_sens = tp / (tp + fn + 1e-6)  # Sensitivity/Recall
    test_spec = tn / (tn + fp + 1e-6)
    test_prec = tp / (tp + fp + 1e-6)
    test_f1 = 2 * test_prec * test_sens / (test_prec + test_sens + 1e-6)
    test_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    print(
        f'Test2.0 | Acc {test_acc:.4f}, Dice {test_dice:.4f}, IoU {test_iou:.4f}, '
        f'Precision {test_prec:.4f}, Sensitivity {test_sens:.4f}, Specificity {test_spec:.4f}, F1 {test_f1:.4f}'
    )


if __name__ == '__main__':
    main()
