import numpy as np
import matplotlib.pyplot as plt
import os

def verify_npz_results(npz_path):
    if not os.path.exists(npz_path):
        print(f"❌ 找不到文件: {npz_path}")
        return

    # 1. 加载数据
    print(f"📂 正在加载: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    ct_array = data['ct_array']
    lung_mask = data['lung_mask']

    # 2. 取中间的一层 (Mid-slice)
    mid_z = ct_array.shape[0] // 2
    ct_slice = ct_array[mid_z]
    mask_slice = lung_mask[mid_z]

    # 3. 绘图预览
    plt.figure(figsize=(12, 6))

    # 左图：原始 CT (设置窗宽窗位，适合看肺部)
    plt.subplot(1, 2, 1)
    plt.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=200)
    plt.title(f"Original CT (Slice {mid_z})")
    plt.axis('off')

    # 右图：CT + 掩码叠加 (红色半透明)
    plt.subplot(1, 2, 2)
    plt.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=200)
    # 将 mask 中为 1 的部分显示为红色
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    plt.imshow(masked, cmap='autumn', alpha=0.4) 
    plt.title("Segmentation Overlay")
    plt.axis('off')

    plt.tight_layout()
    print("🎨 正在弹出预览窗口...")
    plt.show()

if __name__ == "__main__":
    # 指向你生成的 npz 路径
    target_npz = r"./segmentation_output/segmentation_results.npz"
    verify_npz_results(target_npz)