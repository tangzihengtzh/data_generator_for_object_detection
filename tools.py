import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm  # 进度条库

# ==== 参数设置 ====
lib_root = "real_lib"
canvas_size = (448, 448)
# 各类实例数量范围：绿豆、红豆、麦粒、红枣
instance_ranges = [(38, 62), (26, 40), (136, 346), (4, 10)]
scale_factors = [0.1, 0.1, 0.1, 0.1]
brightness = [1.0, 1.0, 1.0, 1.0]
gaussian_sigmas = [6.0, 6.0, 3.0, 12.0]
# 允许的最小间距（正值禁止重叠，负值允许边缘重叠）
DESIRED_MARGIN = -2

# 预加载每类文件列表，避免循环内频繁IO
class_files = [os.listdir(os.path.join(lib_root, f"c{i+1}")) for i in range(4)]
# 预计算结构元素
dilate_struct = np.ones((2 * DESIRED_MARGIN + 1, 2 * DESIRED_MARGIN + 1), dtype=np.uint8) if DESIRED_MARGIN > 0 else None
erode_struct = np.ones((2 * abs(DESIRED_MARGIN) + 1, 2 * abs(DESIRED_MARGIN) + 1), dtype=np.uint8) if DESIRED_MARGIN < 0 else None

# 加载素材
def load_random_png_from_list(idx):
    files = class_files[idx]
    selected = random.choice(files)
    path = os.path.join(lib_root, f"c{idx+1}", selected)
    return Image.open(path).convert("RGBA")

# 生成高斯图
def generate_gaussian_map(shape, center, sigma):
    h, w = shape
    x0, y0 = center
    y, x = np.ogrid[:h, :w]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

# 简易的矩形相交检测
def boxes_intersect(b1, b2, margin=0):
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    return not (x1_max + margin < x2_min or x2_max + margin < x1_min or
                y1_max + margin < y2_min or y2_max + margin < y1_min)

# 检查重叠：先BB快速检测，若可能重叠再做精确mask检测
def check_overlap(mask, mask_patch, x1, y1, placed_boxes):
    # 计算当前实例BBox
    h, w = mask_patch.shape
    cur_box = (x1, y1, x1 + w, y1 + h)
    # 轮询已放置BBox
    for bx in placed_boxes:
        if boxes_intersect(bx, cur_box, margin=DESIRED_MARGIN):
            # 可能冲突，做精确mask check
            canvas_h, canvas_w = canvas_size
            tmp = np.zeros_like(mask)
            x0, y0 = max(x1, 0), max(y1, 0)
            x_end = min(x1 + w, canvas_w)
            y_end = min(y1 + h, canvas_h)
            tmp[y0:y_end, x0:x_end] = mask_patch[y0 - y1:y_end - y1, x0 - x1:x_end - x1]
            # 根据margin处理mask
            if DESIRED_MARGIN > 0:
                mask_check = binary_dilation(mask, structure=dilate_struct)
            elif DESIRED_MARGIN < 0:
                mask_check = binary_erosion(mask, structure=erode_struct)
            else:
                mask_check = mask
            if np.any(mask_check & tmp):
                return True
    return False

# 主生成流程：按轮询交替放置，附带BBox优化和进度条
def generate_sample_interleaved(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    canvas = Image.new("RGBA", canvas_size, (0,0,255,0))
    mask_total = np.zeros(canvas_size, dtype=np.uint8)
    density_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]
    placed_boxes = []  # 存放已放置实例的BBox

    # 随机确定每类要放置的数量
    targets = [random.randint(r[0], r[1]) for r in instance_ranges]
    placed = [0] * 4
    total_to_place = sum(targets)
    max_attempts = total_to_place * 10
    attempts = 0

    # 轮询顺序（可随机打乱），红枣最先尝试
    order = [3, 1, 0, 2]

    # 进度条：总进度按要放置的实例总数
    pbar = tqdm(total=total_to_place, desc="放置进度")

    # 放置循环
    while sum(placed) < total_to_place and attempts < max_attempts:
        for cls_index in order:
            if placed[cls_index] >= targets[cls_index]:
                continue
            inst = load_random_png_from_list(cls_index)
            # 随机变换
            scale = scale_factors[cls_index] * random.uniform(0.9, 1.1)
            inst = inst.resize((int(inst.width*scale), int(inst.height*scale)), Image.Resampling.LANCZOS)
            inst = inst.rotate(random.randint(0,360), expand=True)
            inst = ImageEnhance.Brightness(inst).enhance(brightness[cls_index])
            iw, ih = inst.size
            if iw >= canvas_size[1] or ih >= canvas_size[0]:
                attempts += 1
                continue
            px = random.randint(0, canvas_size[1]-iw)
            py = random.randint(0, canvas_size[0]-ih)
            alpha = np.array(inst)[:,:,3]
            mask_patch = (alpha>0).astype(np.uint8)
            # 优化后的重叠检测
            if check_overlap(mask_total, mask_patch, px, py, placed_boxes):
                attempts += 1
                continue
            # 粘贴并更新掩膜、BBox
            canvas.paste(inst, (px,py), inst)
            mask_total[py:py+ih, px:px+iw] = np.logical_or(mask_total[py:py+ih, px:px+iw], mask_patch)
            placed_boxes.append((px, py, px+iw, py+ih))
            # 更新 density map
            center = (px+iw//2, py+ih//2)
            density_maps[cls_index] += generate_gaussian_map(canvas_size, center, gaussian_sigmas[cls_index])
            placed[cls_index] += 1
            pbar.update(1)  # 更新进度条
            attempts += 1
            if sum(placed) >= total_to_place:
                break
        else:
            continue
        break

    pbar.close()

    # 保存结果
    canvas.convert("RGB").save(os.path.join(output_path, "image.png"))
    for i, dmap in enumerate(density_maps):
        np.save(os.path.join(output_path, f"density_c{i+1}.npy"), dmap.astype(np.float32))
        normed = ((dmap/dmap.max())*255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"density_c{i+1}.png"), normed)

if __name__ == "__main__":
    generate_sample_interleaved("output_samples")
