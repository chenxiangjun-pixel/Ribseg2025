import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
from models.ribseg_model import get_model


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """将Nx3点标准化为单位球体"""
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc_centered = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_centered ** 2, axis=1)))
    return pc_centered / (m + 1e-8)


def parse_args():
    p = argparse.ArgumentParser("Inference for rib segmentation")
    p.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto")
    p.add_argument("--num_point", type=int, default=30000, help="用于推理的采样点数量")
    p.add_argument("--experiment_name", type=str, default="ribseg_experiment", help="保存实验数据的目录")
    p.add_argument("--weights", type=str, default="", help="覆盖权重路径 .pth")
    p.add_argument("--data_root", type=str, default="./data/pn", help="包含data_pn和label_pn的路径")
    p.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="数据选择")
    p.add_argument("--output_dir", type=str, default="./inference_results", help="推理结果输出目录")
    p.add_argument("--normal_channel", action="store_true", default=False, help="是否使用normal_channel")
    p.add_argument("--use_attention", action="store_true", default=True, help="是否使用注意力模块")
    p.add_argument("--seed", type=int, default=42, help="再现性；<0禁用确定性采样")
    p.add_argument("--disable_dice", action="store_true", help="跳过dce系数计算（没有GT可用时）")
    return p.parse_args()


def find_weights(weights_arg: str, experiment_name: str) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        if p.exists():
            return p
    cand = [Path("log/part_seg") / experiment_name / "best_model.pth",Path("log") / "best_model.pth", Path("best_model.pth")]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到权重. 请检查: {[str(x) for x in cand]} and --weights={weights_arg!r}")


def load_case_list(data_root: str, split: str) -> List[str]:
    data_dir = Path(data_root) / "data_pn" / split
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到推理所用数据集: {data_dir}")
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"在{data_dir}中没找到.npy文件")
    return files


def try_load_gt(label_dir: Path, fname: str) -> Optional[np.ndarray]:
    f = label_dir / fname
    if f.exists():
        seg = np.load(str(f)).astype(np.int32)
        seg[seg != 0] = 1
        return seg
    return None


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and device.type == "cpu":
        print("[WARN] CUDA不可用；在CPU上运行。")

    num_part = 2  # 肋骨和背景(只分两类)
    model = get_model(num_classes=num_part, normal_channel=args.normal_channel, use_attention=args.use_attention).to(device)
    model.eval()
    weights_path = find_weights(args.weights, args.experiment_name)
    ckpt = torch.load(str(weights_path), map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    data_dir = Path(args.data_root) / "data_pn" / args.split
    label_dir = Path(args.data_root) / "label_pn" / args.split
    out_point_dir = Path(args.output_dir) / "point"
    out_label_dir = Path(args.output_dir) / "label"
    out_point_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    case_list = load_case_list(args.data_root, args.split)
    total_time = 0.0
    dice_sum = 0.0
    n_dice = 0

    with torch.no_grad():
        for fn in tqdm(case_list, desc=f"Inference ({args.split})"):
            arr = np.load(str(data_dir / fn)).astype(np.float32)
            pts_idx = arr[:, :3].astype(np.int32)  # 用于重建的原始整数体素索引
            gt = None
            if not args.disable_dice:
                gt = try_load_gt(label_dir, fn)
            N = pts_idx.shape[0]
            if args.num_point <= 0 or args.num_point > N:
                choice = np.random.choice(N, N, replace=False)
            else:
                choice = np.random.choice(N, args.num_point, replace=(args.num_point > N))
            pts_idx = pts_idx[choice, :]
            if gt is not None:
                gt = gt[choice]
            # 保存采样的原始索引以供后续处理
            case_id = Path(fn).stem  # e.g., RibFrac101
            np.save(str(out_point_dir / case_id), pts_idx.astype(np.int32))
            # 归一化模型副本
            pts_float = pts_idx.astype(np.float32)
            pts_float = pc_normalize(pts_float)
            pts_t = torch.from_numpy(pts_float).unsqueeze(0).to(device)  # [1, N, 3]
            pts_t = pts_t.transpose(2, 1)  # [1, 3, N]
            t0 = time.perf_counter()
            logits, _ = model(pts_t)  # [B, num_part, N]
            total_time += (time.perf_counter() - t0)
            logits = logits.contiguous().view(-1, num_part)  # [N, num_part]
            pred_choice = logits.argmax(dim=1)               # [N] in {0,1}
            pred_bin = (pred_choice > 0).long()
            # Dice系数计算
            if gt is not None:
                gt_t = torch.from_numpy(gt).long().to(device)
                inter = (pred_bin & gt_t).sum().float()
                union = pred_bin.sum().float() + gt_t.sum().float()
                dice = (2.0 * inter + 1.0) / (union + 1.0)   # smooth=1
                dice_sum += float(dice.item())
                n_dice += 1

            # 保存预测的标签（int8）
            np.save(str(out_label_dir / case_id), pred_choice.detach().cpu().numpy().astype(np.int8))

    avg_time = total_time / max(len(case_list), 1)
    print(f"[INFO] 每例数据平均推理时间: {avg_time:.4f}s")
    if n_dice > 0:
        print(f"[INFO] 平均Dice系数: {dice_sum / n_dice:.6f}")
    else:
        print("[INFO] 跳过Dice系数计算(没找到ground truth).")


if __name__ == "__main__":
    main()