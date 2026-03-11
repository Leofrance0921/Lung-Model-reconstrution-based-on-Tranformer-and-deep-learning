import os
import sys
import numpy as np
import SimpleITK as sitk
import trimesh
from totalsegmentator.python_api import totalsegmentator
from skimage import measure
import multiprocessing

# 1. 核心补丁：解决 pydicom 路径兼容性与多进程支持
try:
    import pydicom.pixels
except ImportError:
    import pydicom.pixel_data_handlers
    sys.modules['pydicom.pixels'] = pydicom.pixel_data_handlers

def create_mesh_from_mask(mask_img, label_value=None):
    """从 NIfTI 掩码对象生成变换后的 Trimesh 对象"""
    data = mask_img.get_fdata()
    affine = mask_img.affine
    
    # 提取特定标签或所有非零区域
    mask = (data == label_value) if label_value else (data > 0)
    
    if not np.any(mask):
        return None

    # Marching Cubes 提取表面
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5)
    
    # 应用 Affine 矩阵：坐标对齐
    verts = verts @ affine[:3, :3].T + affine[:3, 3]
    return trimesh.Trimesh(vertices=verts, faces=faces)

def run_full_lung_pipeline(dicom_dir, output_folder="./full_lung_model"):
    # 创建输出目录
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    temp_nifti = os.path.join(output_folder, "temp_input.nii.gz")
    
    # --- 步骤 1: DICOM 转 NIfTI ---
    print(f"🔄 1/5: 正在读取 DICOM 序列...")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        print("❌ 错误：在指定路径下未找到 DICOM 文件！")
        return
    reader.SetFileNames(dicom_names)
    sitk.WriteImage(reader.Execute(), temp_nifti)
    
    # --- 步骤 2: 提取肺部外壳 ---
    print("🫁 2/5: 正在提取肺部外壳轮廓...")
    lung_roi = ["lung_upper_lobe_left", "lung_lower_lobe_left", 
                "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    # 使用 total 任务提取指定子集以避免 API 报错
    lung_mask = totalsegmentator(temp_nifti, None, task="total", ml=True, roi_subset=lung_roi)
    lung_mesh = create_mesh_from_mask(lung_mask)
    
    # --- 步骤 3: 提取精细肺血管 ---
    print("🧠 3/5: 正在提取精细肺血管 (高精度模式)...")
    vessel_mask = totalsegmentator(temp_nifti, None, task="lung_vessels", ml=True)
    vessel_mesh = create_mesh_from_mask(vessel_mask)

    # --- 步骤 4: 几何平滑 (不进行减面，保留全部顶点) ---
    print("✨ 4/5: 正在执行平滑优化...")
    if lung_mesh:
        # 使用拉普拉斯平滑消除外壳阶梯感
        trimesh.smoothing.filter_laplacian(lung_mesh, iterations=10)
        lung_mesh.visual.face_colors = [200, 200, 200, 100] 
        
    if vessel_mesh:
        # 使用 Taubin 平滑保护血管细节
        trimesh.smoothing.filter_taubin(vessel_mesh, iterations=30)
        vessel_mesh.visual.face_colors = [180, 0, 0, 255]

    # --- 步骤 5: 导出总体合并模型 ---
    print("💾 5/5: 正在导出高精度模型...")
    
    # 导出独立的最高精度文件
    lung_mesh.export(os.path.join(output_folder, "lung_shell_raw.stl"))
    vessel_mesh.export(os.path.join(output_folder, "lung_vessels_raw.stl"))
    
    # 合并为一个总体的 STL 模型
    combined_mesh = trimesh.util.concatenate([lung_mesh, vessel_mesh])
    total_stl_path = os.path.join(output_folder, "total_lung_with_vessels_high_res.stl")
    combined_mesh.export(total_stl_path)
    
    # 导出 OBJ 场景 (带材质信息)
    scene = trimesh.Scene([lung_mesh, vessel_mesh])
    scene.export(os.path.join(output_folder, "full_lung_visual.obj"))
    
    print("-" * 30)
    print(f"✅ 高精度模型生成完毕！")
    print(f"🔹 总体模型路径: {total_stl_path}")
    print("-" * 30)

    try:
        scene.show(title="High Resolution Lung View")
    except Exception as e:
        print(f"💡 预览窗口跳过: {e}")

if __name__ == "__main__":
    # 必须加保护，防止 Windows 递归生成子进程
    multiprocessing.freeze_support()
    
    # 替换为你的路径
    my_dicom_path = r"C:\Users\14033\Desktop\demo1.1\Zhao\3"
    run_full_lung_pipeline(my_dicom_path)