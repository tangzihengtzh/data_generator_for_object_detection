# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# 从一张带透明背景的 PNG 中，自动把所有互不重叠的目标
# 按最小外接矩形裁剪出来，并以 PNG（含 Alpha 通道）保存。
#
# 使用方法：
# 1. 把脚本保存为 extract_objects.py
# 2. 修改 INPUT_PATH、OUTPUT_DIR 变量
# 3. 运行: python extract_objects.py
# """
#
# import os
# from pathlib import Path
#
# import numpy as np
# from PIL import Image
# from scipy import ndimage
#
#
# # ========= 仅改动这两行 =========
# INPUT_PATH = r"E:\python_prj\data_gen_for_VIT_SUNET\image\new_input3.png"         # 输入 PNG 文件路径
# OUTPUT_DIR = r"E:\python_prj\data_gen_for_VIT_SUNET\real_lib\c3"      # 输出文件夹路径
# # =================================
#
#
# def extract_objects(img_path: str, out_dir: str, connectivity: int = 2) -> None:
#     """核心处理流程：读取 PNG → 连通域分割 → 裁剪保存"""
#     # 读取并转为 RGBA
#     img = Image.open(img_path).convert("RGBA")
#     alpha = np.array(img.split()[-1])      # 取 Alpha 通道
#     mask = alpha > 0                       # 二值掩膜：True=目标像素
#
#     # 连通域标记：返回 labels 与目标数量 num
#     structure = np.ones((3, 3)) if connectivity == 2 else None
#     labels, num = ndimage.label(mask, structure=structure)
#
#     if num == 0:
#         print("⚠️  没有检测到任何目标。")
#         return
#
#     os.makedirs(out_dir, exist_ok=True)
#     print(f"检测到 {num} 个目标，开始裁剪并保存…")
#
#     for idx in range(1, num + 1):
#         coords = np.argwhere(labels == idx)          # [[y, x], ...]
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0) + 1        # +1 使切片包含 max
#
#         crop = img.crop((x_min, y_min, x_max, y_max))
#         out_path = Path(out_dir) / f"object_{idx:03d}.png"
#         crop.save(out_path)
#         print(f"✅  已保存 {out_path}")
#
#     print("全部目标处理完毕！")
#
#
# if __name__ == "__main__":
#     extract_objects(INPUT_PATH, OUTPUT_DIR, connectivity=2)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从一张带透明背景的 PNG 中，自动把所有互不重叠的目标
按最小外接矩形裁剪出来，并以 PNG（含 Alpha 通道）保存。

使用方法：
1. 把脚本保存为 extract_objects.py
2. 修改 INPUT_PATH、OUTPUT_DIR 变量
3. 运行: python extract_objects.py
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# ========= 仅改动这两行 =========
INPUT_PATH = r"E:\python_prj\data_gen_for_VIT_SUNET\image\new_input4.png"         # 输入 PNG 文件路径
OUTPUT_DIR = r"E:\python_prj\data_gen_for_VIT_SUNET\real_lib\c4"      # 输出文件夹路径
# =================================


def extract_objects(img_path: str, out_dir: str, connectivity: int = 2) -> None:
    """核心处理流程：读取 PNG → 连通域分割 → 裁剪保存（仅保留对应目标的像素）"""
    # 读取并转为 RGBA
    img = Image.open(img_path).convert("RGBA")
    alpha = np.array(img.split()[-1])      # 取 Alpha 通道
    mask = alpha > 0                       # 二值掩膜：True=目标像素

    # 连通域标记：返回 labels 与目标数量 num
    structure = np.ones((3, 3)) if connectivity == 2 else None
    labels, num = ndimage.label(mask, structure=structure)

    if num == 0:
        print("⚠️  没有检测到任何目标。")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"检测到 {num} 个目标，开始裁剪并保存…")

    for idx in range(1, num + 1):
        # 找到当前目标像素的坐标范围
        coords = np.argwhere(labels == idx)          # [[y, x], ...]
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1        # +1 使切片包含 max

        # 裁剪原图
        crop = img.crop((x_min, y_min, x_max, y_max))

        # 获取属于当前目标的局部掩膜
        region_mask = (labels[y_min:y_max, x_min:x_max] == idx)

        # 将裁剪图转换为数组，去除其他对象像素（设为全透明）
        crop_arr = np.array(crop)
        # 对所有通道设为0，也可仅对 alpha 通道设为0：crop_arr[~region_mask, 3] = 0
        crop_arr[~region_mask, :] = 0

        # 转回图像并保存
        crop_clean = Image.fromarray(crop_arr)
        out_path = Path(out_dir) / f"object_{idx:03d}.png"
        crop_clean.save(out_path)
        print(f"✅  已保存 {out_path}")

    print("全部目标处理完毕！")


if __name__ == "__main__":
    extract_objects(INPUT_PATH, OUTPUT_DIR, connectivity=2)
