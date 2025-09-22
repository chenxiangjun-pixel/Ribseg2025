import numpy as np
import nibabel as nib
import skimage
from skimage.measure import label, regionprops
import os
from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings('ignore')

# --- 路径配置 ---
source_data_dir = 'F:/Data_Collection/Ribfrac/image_test/'   # 替换为你自己的原始CT数据路径
dis_c2_point_dir = "./inference_results/point/"
dis_c2_label_dir = "./inference_results/label/"
output_dir = "./inference_results/nii/"
centreline_dir = "./inference_results/centerline/"

# 最优参数配置
OPTIMAL_PARAMS = {
    "dilation_radius": [5, 5, 5],
    "use_closing": True,
    "closing_radius": 3,
    "use_erosion": True,
    "max_ribs": 20,
    "fill_holes": True
}


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    """使用SimpleITK细化从 mask中提取1体素宽的肋骨中心线骨架.
    Args:
        mask: 3D二值numpy array (0/1)
    Returns:
        具有中心线体素为1的相同形状的3D二进制numpy数组
    """
    itk_img = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
    skeleton = sitk.BinaryThinning(itk_img)
    return sitk.GetArrayFromImage(skeleton)


def post_proc(s_i, loc, label, params):
    try:
        mask_res = np.zeros(s_i.shape)
        # 创建点云mask
        for i in range(loc.shape[0]):
            index = loc[i]
            x, y, z = index[0], index[1], index[2]
            if 0 <= x < s_i.shape[0] and 0 <= y < s_i.shape[1] and 0 <= z < s_i.shape[2]:
                mask_res[x][y][z] = label[i]
        # 转换为SimpleITK图像
        image_array = sitk.GetImageFromArray(mask_res.astype('int8'))
        # 1. 形态学闭运算
        if params.get('use_closing', False):
            print("    应用形态学闭运算...")
            closed = sitk.BinaryMorphologicalClosing(image_array,params.get('closing_radius',3),sitk.sitkBall)
            processed = closed
        else:
            processed = image_array
        # 2. 膨胀操作
        print(f"    应用半径扩张 {params.get('dilation_radius', [5, 5, 5])}...")
        dilated = sitk.BinaryDilate(processed, params.get('dilation_radius', [5, 5, 5]), sitk.sitkBall)
        # 3. 腐蚀操作
        if params.get('use_erosion', False):
            print("    应用腐蚀操作...")
            # 如果没有提供具体 erosion_radius，使用 2
            erosion_radius = params.get('erosion_radius', 2)
            eroded = sitk.BinaryErode(dilated, erosion_radius, sitk.sitkBall)
            processed = eroded
        else:
            processed = dilated
        # 4. 填补空洞
        if params.get('fill_holes', True):
            print("    填补空洞...")
            holesfilled = sitk.BinaryFillhole(processed, fullyConnected=True)
            processed = holesfilled
        result = sitk.GetArrayFromImage(processed)
        res = np.multiply(s_i, result)
        # 连通域分析
        print("    执行连通域分析...")
        res1 = skimage.measure.label(res, connectivity=1)
        rib_p = regionprops(res1)
        # 按面积排序，保留最大的N个
        rib_p.sort(key=lambda x: x.area, reverse=True)
        max_ribs = min(params.get('max_ribs', 20), len(rib_p))
        print(f"    从{len(rib_p)}总组件中保留最大的{max_ribs}组件")
        im = np.in1d(res1, [x.label for x in rib_p[:max_ribs]]).reshape(res1.shape)
        return im.astype('int8'), res1  # Return both final mask and labeled components

    except Exception as e:
        print(f"    Error in post_proc: {e}")
        return np.zeros(s_i.shape, dtype='int8'), np.zeros(s_i.shape, dtype='int8')


def process_all_files():
    """处理所有文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(centreline_dir, exist_ok=True)
    ct_files = [f for f in os.listdir(source_data_dir) if f.endswith('.nii.gz')]
    print(f"找到{len(ct_files)}个待处理CT数据")
    for ct_file in tqdm(ct_files, desc="Processing files"):
        try:
            print(f"\nProcessing: {ct_file}")
            # 加载CT数据
            s_i_nib = nib.load(os.path.join(source_data_dir, ct_file))
            s_i = s_i_nib.get_fdata()
            s_i[s_i != 0] = 1  # 转换为二值mask
            s_i = s_i.astype('int8')
            # 使用稳健的basename规则，避免依赖固定长度切片
            if ct_file.endswith('-image.nii.gz'):
                basename = ct_file.replace('-image.nii.gz', '')
            else:
                # 兜底：去掉扩展名
                basename = os.path.splitext(os.path.splitext(ct_file)[0])[0]
            # 加载点和标签数据
            point_file = basename + '.npy'
            label_file = basename + '.npy'
            point_path = os.path.join(dis_c2_point_dir, point_file)
            label_path = os.path.join(dis_c2_label_dir, label_file)
            if not os.path.exists(point_path):
                print(f"    Warning: Point文件没找到: {point_path}")
                continue
            if not os.path.exists(label_path):
                print(f"    Warning: Label文件没找到: {label_path}")
                continue
            loc = np.load(point_path)
            label = np.load(label_path)
            print(f"    加载带有标签的{len(loc)}个点")
            result, labeled_components = post_proc(s_i, loc, label, OPTIMAL_PARAMS)
            # 保存结果为NIfTI文件
            output_filename = basename + '-mask.nii'
            output_path = os.path.join(output_dir, output_filename)
            # 使用原始CT的仿射变换和头信息
            new_nii = nib.Nifti1Image(result, s_i_nib.affine, s_i_nib.header)
            nib.save(new_nii, output_path)
            print(f"    处理结果保存至: {output_dir, output_filename}")
            # --- 提取并保存中心线 ---
            try:
                print("    开始提取中心线...")
                skeleton = extract_centerline(result)
                skeleton = (skeleton > 0).astype('uint8')
                centre_filename = basename + '-centraline.nii'
                centre_path = os.path.join(centreline_dir, centre_filename)
                centre_nii = nib.Nifti1Image(skeleton, s_i_nib.affine, s_i_nib.header)
                nib.save(centre_nii, centre_path)
                print(f"    中心线保存至: {centreline_dir, centre_filename}")
            except Exception as e:
                print(f"    Warning: {ct_file}文件的中心线提取失败: {e}")

        except Exception as e:
            print(f"    {ct_file}文件处理出错: {e}")
            continue


if __name__ == '__main__':
    process_all_files()
