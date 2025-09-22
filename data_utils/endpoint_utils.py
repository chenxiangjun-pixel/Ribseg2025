import os
from typing import List, Tuple, Dict

import numpy as np


def pca_endpoints(points_xyz: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    基于主成分分析(PCA)在第一主轴上取投影最小/最大点，作为两端极点。
    输入: points_xyz [N,3]
    输出: [(x1,y1,z1), (x2,y2,z2)] 或空列表
    """
    if points_xyz.size == 0:
        return []
    pts = points_xyz.astype(np.float64)
    mean = pts.mean(axis=0, keepdims=True)
    pts_c = pts - mean
    # SVD on covariance (equivalent to PCA first component)
    # [N,3] -> [3,3]
    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    pc1 = Vt[0]  # [3]
    proj = pts_c @ pc1  # [N]
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    return [tuple(pts[i_min]), tuple(pts[i_max])]


def detect_endpoints_from_points(points_xyz: np.ndarray, pred_labels: np.ndarray,
                                 min_positive_points: int = 200) -> List[Tuple[float, float, float]]:
    """
    在验证阶段直接对点云预测结果提取端点（不进行体素化）。
    简化策略：对所有预测为肋骨(1)的点做PCA，取第一主轴两端作为端点。
    """
    mask = (pred_labels > 0)
    pos_pts = points_xyz[mask]
    if pos_pts.shape[0] < min_positive_points:
        return []
    return pca_endpoints(pos_pts)


def _quantize_points(points_xyz: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """将连续坐标量化到体素索引。
    返回: voxel_idx [N,3] 的 int 索引，以及平移 offset（浮点，用于可选还原）。
    注意：这里只用于连通性聚类，不做体素占据图。
    """
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros(3, dtype=np.float32)
    pts = points_xyz.astype(np.float32)
    offset = pts.min(axis=0)
    shifted = pts - offset
    idx = np.floor(shifted / max(voxel_size, 1e-6)).astype(np.int32)
    return idx, offset


def _connected_components_voxels(voxel_idx: np.ndarray) -> List[np.ndarray]:
    """对体素索引做26邻域连通域分解，返回各连通域中点的索引集合(list of arrays of point indices)。"""
    if voxel_idx.shape[0] == 0:
        return []
    # 建立从体素坐标到点索引列表的映射
    voxel_to_points: Dict[Tuple[int, int, int], List[int]] = {}
    for i, (ix, iy, iz) in enumerate(voxel_idx):
        key = (int(ix), int(iy), int(iz))
        voxel_to_points.setdefault(key, []).append(i)
    voxels = set(voxel_to_points.keys())
    visited = set()
    components: List[List[int]] = []
    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]
    for v in voxels:
        if v in visited:
            continue
        stack = [v]
        visited.add(v)
        comp_points: List[int] = []
        while stack:
            cv = stack.pop()
            comp_points.extend(voxel_to_points.get(cv, []))
            cx, cy, cz = cv
            for dx, dy, dz in neighbors:
                nv = (cx + dx, cy + dy, cz + dz)
                if nv in voxels and nv not in visited:
                    visited.add(nv)
                    stack.append(nv)
        if comp_points:
            components.append(comp_points)
    return [np.array(c, dtype=np.int32) for c in components]


def detect_endpoints_per_component(points_xyz: np.ndarray,
                                   pred_labels: np.ndarray,
                                   voxel_size: float = 0.02,
                                   min_points_component: int = 150,
                                   keep_topk_per_comp: int = 2) -> List[Dict]:
    """
    按连通域（近似每根肋骨）分别输出端点：
    1) 选取预测为肋骨的点
    2) 体素量化 + 26邻域连通域
    3) 对每个连通域做PCA，输出两端极点（或不足时跳过）
    返回: list[ { 'component_id': int, 'num_points': int, 'endpoints': [(x,y,z),(x,y,z)] } ]
    """
    mask = (pred_labels > 0)
    pos_pts = points_xyz[mask]
    if pos_pts.shape[0] < min_points_component:
        return []
    vox_idx, _ = _quantize_points(pos_pts, voxel_size)
    comps = _connected_components_voxels(vox_idx)
    results: List[Dict] = []
    for cid, comp_indices in enumerate(comps):
        if comp_indices.shape[0] < min_points_component:
            continue
        comp_pts = pos_pts[comp_indices]
        endpoints = pca_endpoints(comp_pts)
        if len(endpoints) == 2:
            results.append({
                'component_id': cid,
                'num_points': int(comp_pts.shape[0]),
                'endpoints': [(float(e[0]), float(e[1]), float(e[2])) for e in endpoints]
            })
    return results


def detect_24x2_endpoints(points_xyz: np.ndarray, pred_labels: np.ndarray,
                          voxel_size: float = 0.02, min_points_component: int = 150) -> Dict:
    """
    强制输出24×2=48个端点（每根肋骨2个端点）。
    
    策略：
    1) 连通域分解得到组件
    2) 若组件>24：按点数排序取前24
    3) 若组件<24：对最大组件沿PCA主轴二分（递归）直至达到24个或无可分
    4) 对每个组件做PCA，取两端点
    
    返回: {
        'endpoints_24x2': List[Tuple[float,float,float]|None],  # 固定48个，不足则None
        'components_used': int,  # 实际使用的组件数
        'split_count': int,      # 二分次数
        'success_count': int     # 成功提取的端点对数
    }
    """
    mask = (pred_labels > 0)
    pos_pts = points_xyz[mask]
    if pos_pts.shape[0] < min_points_component:
        return {
            'endpoints_24x2': [None] * 48,
            'components_used': 0,
            'split_count': 0,
            'success_count': 0
        }
    
    # 1. 连通域分解
    vox_idx, _ = _quantize_points(pos_pts, voxel_size)
    comps = _connected_components_voxels(vox_idx)
    
    # 过滤小组件
    valid_comps = []
    for comp_indices in comps:
        if comp_indices.shape[0] >= min_points_component:
            comp_pts = pos_pts[comp_indices]
            valid_comps.append(comp_pts)
    
    if not valid_comps:
        return {
            'endpoints_24x2': [None] * 48,
            'components_used': 0,
            'split_count': 0,
            'success_count': 0
        }
    
    # 2. 调整到24个组件
    split_count = 0
    while len(valid_comps) < 24 and valid_comps:
        # 找最大组件进行二分
        max_comp_idx = max(range(len(valid_comps)), key=lambda i: valid_comps[i].shape[0])
        max_comp = valid_comps[max_comp_idx]
        
        # 沿PCA主轴二分
        if max_comp.shape[0] < min_points_component * 2:
            break  # 无法再分
        
        endpoints = pca_endpoints(max_comp)
        if len(endpoints) != 2:
            break  # 无法获取主轴
        
        # 计算投影并二分
        mean = max_comp.mean(axis=0, keepdims=True)
        pts_c = max_comp - mean
        U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
        pc1 = Vt[0]
        proj = pts_c @ pc1
        median_proj = np.median(proj)
        
        mask1 = proj <= median_proj
        mask2 = proj > median_proj
        
        comp1 = max_comp[mask1]
        comp2 = max_comp[mask2]
        
        if comp1.shape[0] >= min_points_component and comp2.shape[0] >= min_points_component:
            valid_comps[max_comp_idx] = comp1
            valid_comps.append(comp2)
            split_count += 1
        else:
            break
    
    # 3. 如果还是超过24个，按点数排序取前24
    if len(valid_comps) > 24:
        comp_sizes = [(i, comp.shape[0]) for i, comp in enumerate(valid_comps)]
        comp_sizes.sort(key=lambda x: x[1], reverse=True)
        top24_indices = [i for i, _ in comp_sizes[:24]]
        valid_comps = [valid_comps[i] for i in top24_indices]
    
    # 4. 为每个组件提取两端点
    endpoints_24x2 = []
    success_count = 0
    
    for i in range(24):
        if i < len(valid_comps):
            comp_pts = valid_comps[i]
            endpoints = pca_endpoints(comp_pts)
            if len(endpoints) == 2:
                endpoints_24x2.extend([(float(e[0]), float(e[1]), float(e[2])) for e in endpoints])
                success_count += 1
            else:
                endpoints_24x2.extend([None, None])
        else:
            endpoints_24x2.extend([None, None])
    
    return {
        'endpoints_24x2': endpoints_24x2,
        'components_used': len(valid_comps),
        'split_count': split_count,
        'success_count': success_count
    }



