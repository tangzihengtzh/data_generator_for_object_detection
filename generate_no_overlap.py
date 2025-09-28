# import os
# import random
# import numpy as np
# import cv2
# from PIL import Image, ImageEnhance
#
# # ==== 参数设置 ====
# lib_root = "real_lib"
# canvas_size = (448, 448)
# # 麦 红豆 米 枣
# # 绿豆 红豆 麦粒 红枣
# instance_ranges = [(38, 62), (26, 40), (136, 346), (4, 10)]
# # instance_ranges = [(1, 2), (1, 2), (1, 2), (1, 2)]
# scale_factors = [0.1, 0.1, 0.1, 0.1]
# brightness = [1.0, 1.0, 1.0, 1.0]
# gaussian_sigmas = [6.0, 6.0, 3.0, 12.0]  # 每类单独设置扩散范围
# # 3,5,5,3,12
#
# # 加载素材
# def load_random_png_from(class_dir):
#     files = os.listdir(class_dir)
#     selected = random.choice(files)
#     return Image.open(os.path.join(class_dir, selected)).convert("RGBA")
#
# # 生成高斯图
# def generate_gaussian_map(shape, center, sigma):
#     h, w = shape
#     x0, y0 = center
#     y, x = np.ogrid[:h, :w]
#     g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#     return g
#
# DESIRED_MARGIN = -80  # 最小间，小于则允许重合
# def check_overlap(mask, mask_patch, x1, y1, des=0):
#     canvas_h, canvas_w = canvas_size
#     tmp = np.zeros_like(mask)          # 跟 mask_total 同尺寸
#     h, w = mask_patch.shape
#     # 将 patch 放到 tmp，上下左右做 clip 防止越界
#     x0, y0 = max(x1, 0), max(y1, 0)
#     x_end = min(x1 + w, canvas_w)
#     y_end = min(y1 + h, canvas_h)
#     tmp[y0:y_end, x0:x_end] = mask_patch[
#         y0 - y1 : y_end - y1,
#         x0 - x1 : x_end - x1
#     ]
#
#     if des > 0:
#         from scipy.ndimage import binary_dilation
#         struct = np.ones((2*des+1, 2*des+1), np.uint8)
#         tmp = binary_dilation(tmp, structure=struct)
#
#     return np.any(mask & tmp)
#
#
#
#
# # 主函数
#
# def generate_sample_no_overlap(output_path: str):
#     os.makedirs(output_path, exist_ok=True)
#
#     bg_color = (0, 0, 255, 0)
#     canvas = Image.new("RGBA", canvas_size, bg_color)
#     density_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]
#     mask_total = np.zeros(canvas_size, dtype=np.uint8)
#
#     # 优先顺序：红枣放在最前面，index = 3
#     order = [3, 1, 0, 2]
#
#     for cls_index in order:
#         cls_dir = os.path.join(lib_root, f"c{cls_index+1}")
#         base_scale = scale_factors[cls_index]
#         min_n, max_n = instance_ranges[cls_index]
#         n_instances = random.randint(min_n, max_n)
#         sigma = gaussian_sigmas[cls_index]
#
#         attempts = 0
#         placed = 0
#         while placed < n_instances and attempts < n_instances * 10:
#             inst = load_random_png_from(cls_dir)
#             angle = random.randint(0, 360)
#             scale = base_scale * random.uniform(0.9, 1.1)
#
#             w, h = inst.size
#             inst = inst.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)
#             inst = inst.rotate(angle, expand=True)
#
#             enhancer = ImageEnhance.Brightness(inst)
#             inst = enhancer.enhance(brightness[cls_index])
#
#             iw, ih = inst.size
#             if iw >= canvas_size[1] or ih >= canvas_size[0]:
#                 attempts += 1
#                 continue
#
#             px = random.randint(0, canvas_size[1] - iw)
#             py = random.randint(0, canvas_size[0] - ih)
#
#             alpha = np.array(inst)[:, :, 3]
#             mask_patch = (alpha > 0).astype(np.uint8)
#
#             if check_overlap(mask_total, mask_patch, px, py, des=DESIRED_MARGIN):
#                 attempts += 1
#                 continue
#
#             canvas.paste(inst, (px, py), inst)
#             mask_total[py:py+ih, px:px+iw] = np.logical_or(mask_total[py:py+ih, px:px+iw], mask_patch)
#
#             center_x = px + iw // 2
#             center_y = py + ih // 2
#             gaussian = generate_gaussian_map(canvas_size, (center_x, center_y), sigma)
#             density_maps[cls_index] += gaussian
#
#             attempts += 1
#             placed += 1
#
#     canvas_rgb = canvas.convert("RGB")
#     canvas_rgb.save(os.path.join(output_path, "image.png"))
#
#     for i, dmap in enumerate(density_maps):
#         np.save(os.path.join(output_path, f"density_c{i+1}.npy"), dmap.astype(np.float32))
#         normed = (dmap / (dmap.max() + 1e-6) * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(output_path, f"density_c{i+1}.png"), normed)


import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from scipy.ndimage import binary_dilation, binary_erosion

# ==== 参数设置 ====
lib_root = "real_lib"
canvas_size = (448, 448)
# 麦 红豆 米 枣
# 绿豆 红豆 麦粒 红枣
instance_ranges = [(38, 62), (26, 40), (136, 346), (4, 10)]
# instance_ranges = [(1, 2), (1, 2), (1, 2), (1, 2)]
scale_factors = [0.1, 0.1, 0.1, 0.1]
brightness = [1.0, 1.0, 1.0, 1.0]
gaussian_sigmas = [6.0, 6.0, 3.0, 12.0]  # 每类单独设置扩散范围
# 3,5,5,3,12

# 允许的最小间距（正值：禁止到此距离；负值：允许此像素重叠）
DESIRED_MARGIN = -2  # 负值表示允许重叠位移最多DESIRED_MARGIN像素

def load_random_png_from(class_dir):
    files = os.listdir(class_dir)
    selected = random.choice(files)
    return Image.open(os.path.join(class_dir, selected)).convert("RGBA")

# 生成高斯图
def generate_gaussian_map(shape, center, sigma):
    h, w = shape
    x0, y0 = center
    y, x = np.ogrid[:h, :w]
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g

# 检查与现有mask是否在割除禁止区域内重叠
# des > 0: 在原mask基础上膨胀，禁止到膨胀区域内重叠
# des < 0: 在原mask基础上腐蚀，允许边缘内重叠，只有深度超过abs(des)的部分才算冲突
# des == 0: 精确无重叠

def check_overlap(mask, mask_patch, x1, y1, des=0):
    canvas_h, canvas_w = canvas_size
    tmp = np.zeros_like(mask)
    h, w = mask_patch.shape
    x0, y0 = max(x1, 0), max(y1, 0)
    x_end = min(x1 + w, canvas_w)
    y_end = min(y1 + h, canvas_h)
    tmp[y0:y_end, x0:x_end] = mask_patch[y0 - y1 : y_end - y1, x0 - x1 : x_end - x1]

    if des > 0:
        struct = np.ones((2 * des + 1, 2 * des + 1), dtype=np.uint8)
        mask_check = binary_dilation(mask, structure=struct)
    elif des < 0:
        abs_des = abs(des)
        struct = np.ones((2 * abs_des + 1, 2 * abs_des + 1), dtype=np.uint8)
        mask_check = binary_erosion(mask, structure=struct)
    else:
        mask_check = mask

    return np.any(mask_check & tmp)

# 主生成流程
def generate_sample_with_allowed_overlap(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    bg_color = (0, 0, 255, 0)
    canvas = Image.new("RGBA", canvas_size, bg_color)
    density_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]
    mask_total = np.zeros(canvas_size, dtype=np.uint8)

    # 放置顺序：红枣最先放
    order = [3, 1, 0, 2]

    for cls_index in order:
        cls_dir = os.path.join(lib_root, f"c{cls_index+1}")
        base_scale = scale_factors[cls_index]
        min_n, max_n = instance_ranges[cls_index]
        n_instances = random.randint(min_n, max_n)
        sigma = gaussian_sigmas[cls_index]

        attempts = 0
        placed = 0
        while placed < n_instances and attempts < n_instances * 10:
            inst = load_random_png_from(cls_dir)
            angle = random.randint(0, 360)
            scale = base_scale * random.uniform(0.9, 1.1)

            w, h = inst.size
            inst = inst.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)
            inst = inst.rotate(angle, expand=True)
            inst = ImageEnhance.Brightness(inst).enhance(brightness[cls_index])

            iw, ih = inst.size
            if iw >= canvas_size[1] or ih >= canvas_size[0]:
                attempts += 1
                continue

            px = random.randint(0, canvas_size[1] - iw)
            py = random.randint(0, canvas_size[0] - ih)

            alpha = np.array(inst)[:, :, 3]
            mask_patch = (alpha > 0).astype(np.uint8)

            if check_overlap(mask_total, mask_patch, px, py, des=DESIRED_MARGIN):
                attempts += 1
                continue

            canvas.paste(inst, (px, py), inst)
            mask_total[py:py+ih, px:px+iw] = np.logical_or(mask_total[py:py+ih, px:px+iw], mask_patch)

            center_x = px + iw // 2
            center_y = py + ih // 2
            density_maps[cls_index] += generate_gaussian_map(canvas_size, (center_x, center_y), sigma)

            placed += 1
            attempts += 1

    # 保存结果
    canvas.convert("RGB").save(os.path.join(output_path, "image.png"))
    for i, dmap in enumerate(density_maps):
        np.save(os.path.join(output_path, f"density_c{i+1}.npy"), dmap.astype(np.float32))
        normed = (dmap / (dmap.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"density_c{i+1}.png"), normed)

if __name__ == "__main__":
    generate_sample_with_allowed_overlap("output_samples")
