import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw

# ==== 参数设置 ====
lib_root = "lib"
output_dir = "output"
canvas_size = (384, 384)  # (H, W)
from PIL import ImageEnhance


# 每类目标数量范围（最小值, 最大值）
instance_ranges = [
    (24, 32),   # c1
    (24, 32),   # c2
    (14, 22),  # c3
    (3, 5),    # c4
]

# 每类缩放因子（基础比例）
scale_factors = [0.2, 0.2, 0.2, 0.2]

# 高斯核参数
gaussian_sigma = 2
gaussian_radius = gaussian_sigma * 1  # 半径 = 3σ，直径 = 6σ

# ==== 准备输出文件夹 ====
os.makedirs(output_dir, exist_ok=True)

# ==== 工具函数 ====
def load_random_png_from(class_dir):
    files = os.listdir(class_dir)
    selected = random.choice(files)
    return Image.open(os.path.join(class_dir, selected)).convert("RGBA")

def generate_gaussian_map(shape, center, sigma):
    """生成单个高斯响应图"""
    h, w = shape
    x0, y0 = center
    y, x = np.ogrid[:h, :w]
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g

# ==== 主函数 ====
def generate_sample():
    bg_color = tuple([random.randint(180, 220)] * 3 + [255])
    canvas = Image.new("RGBA", canvas_size, bg_color)
    density_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]

    object_centers_all_classes = [[] for _ in range(4)]  # 记录中心点以防重叠

    for cls_index in range(4):
        cls_dir = os.path.join(lib_root, f"c{cls_index+1}")
        base_scale = scale_factors[cls_index]
        min_n, max_n = instance_ranges[cls_index]
        n_instances = random.randint(min_n, max_n)

        object_centers = []

        attempts = 0
        while len(object_centers) < n_instances and attempts < n_instances * 10:
            inst = load_random_png_from(cls_dir)
            angle = random.randint(0, 360)
            scale = base_scale * random.uniform(0.9, 1.1)

            w, h = inst.size
            inst = inst.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)
            inst = inst.rotate(angle, expand=True)

            # 调整亮度
            enhancer = ImageEnhance.Brightness(inst)
            inst = enhancer.enhance(brightness[cls_index])

            iw, ih = inst.size

            if iw >= canvas_size[1] or ih >= canvas_size[0]:
                attempts += 1
                continue

            px = random.randint(0, canvas_size[1] - iw)
            py = random.randint(0, canvas_size[0] - ih)
            center_x = px + iw // 2
            center_y = py + ih // 2

            # 检查是否和已有目标重叠（与其他中心点最小距离限制）
            min_dist = 2 * gaussian_radius
            too_close = any(np.hypot(center_x - cx, center_y - cy) < min_dist for cx, cy in object_centers)
            if too_close:
                attempts += 1
                continue

            # 记录中心，粘贴图像
            object_centers.append((center_x, center_y))
            object_centers_all_classes[cls_index].append((center_x, center_y))
            canvas.paste(inst, (px, py), inst)

            # 添加高斯峰
            gaussian = generate_gaussian_map(canvas_size, (center_x, center_y), gaussian_sigma)
            density_maps[cls_index] += gaussian

            attempts += 1

    # ==== 保存合成图像 ====
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(os.path.join(output_dir, "image.png"))

    # ==== 保存密度图 ====
    for i, dmap in enumerate(density_maps):
        # 保存 npy 格式
        np.save(os.path.join(output_dir, f"density_c{i+1}.npy"), dmap.astype(np.float32))

        # 保存 png 格式（仅用于可视化）
        normed = (dmap / (dmap.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"density_c{i+1}.png"), normed)

# === 保留原先的导入和函数不动 ===
# 每类缩放因子（基础比例）
scale_factors = [0.2, 0.2, 0.2, 0.2]
# 亮度调节
brightness = [0.8,1,1,1]
def generate_sample_to(output_path: str):
    # 保证输出文件夹存在
    os.makedirs(output_path, exist_ok=True)

    bg_color = tuple([random.randint(180, 220)] * 3 + [255])
    canvas = Image.new("RGBA", canvas_size, bg_color)
    density_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]

    object_centers_all_classes = [[] for _ in range(4)]  # 记录中心点以防重叠

    for cls_index in range(4):
        cls_dir = os.path.join(lib_root, f"c{cls_index+1}")
        base_scale = scale_factors[cls_index]
        min_n, max_n = instance_ranges[cls_index]
        n_instances = random.randint(min_n, max_n)

        object_centers = []

        attempts = 0
        while len(object_centers) < n_instances and attempts < n_instances * 10:
            inst = load_random_png_from(cls_dir)
            angle = random.randint(0, 360)
            scale = base_scale * random.uniform(0.9, 1.1)

            w, h = inst.size
            inst = inst.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)
            inst = inst.rotate(angle, expand=True)

            # 调整亮度
            enhancer = ImageEnhance.Brightness(inst)
            inst = enhancer.enhance(brightness[cls_index])

            iw, ih = inst.size

            if iw >= canvas_size[1] or ih >= canvas_size[0]:
                attempts += 1
                continue

            px = random.randint(0, canvas_size[1] - iw)
            py = random.randint(0, canvas_size[0] - ih)
            center_x = px + iw // 2
            center_y = py + ih // 2

            # 检查距离
            min_dist = 2 * gaussian_radius
            too_close = any(np.hypot(center_x - cx, center_y - cy) < min_dist for cx, cy in object_centers)
            if too_close:
                attempts += 1
                continue

            object_centers.append((center_x, center_y))
            object_centers_all_classes[cls_index].append((center_x, center_y))
            canvas.paste(inst, (px, py), inst)

            gaussian = generate_gaussian_map(canvas_size, (center_x, center_y), gaussian_sigma)
            density_maps[cls_index] += gaussian

            attempts += 1

    # === 保存合成图像 ===
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(os.path.join(output_path, "image.png"))

    # === 保存密度图 ===
    for i, dmap in enumerate(density_maps):
        np.save(os.path.join(output_path, f"density_c{i+1}.npy"), dmap.astype(np.float32))
        normed = (dmap / (dmap.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"density_c{i+1}.png"), normed)


def generate_sample_to_mask(output_path: str):
    from PIL import ImageEnhance

    os.makedirs(output_path, exist_ok=True)

    bg_color = tuple([random.randint(180, 220)] * 3 + [255])
    canvas = Image.new("RGBA", canvas_size, bg_color)
    density_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]

    object_centers_all_classes = [[] for _ in range(4)]

    for cls_index in range(4):
        cls_dir = os.path.join(lib_root, f"c{cls_index+1}")
        base_scale = scale_factors[cls_index]
        min_n, max_n = instance_ranges[cls_index]
        n_instances = random.randint(min_n, max_n)

        object_centers = []

        attempts = 0
        while len(object_centers) < n_instances and attempts < n_instances * 10:
            inst = load_random_png_from(cls_dir)
            angle = random.randint(0, 360)
            scale = base_scale * random.uniform(0.9, 1.1)

            w, h = inst.size
            inst = inst.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)
            inst = inst.rotate(angle, expand=True)

            # 调整亮度
            enhancer = ImageEnhance.Brightness(inst)
            inst = enhancer.enhance(brightness[cls_index])

            iw, ih = inst.size
            if iw >= canvas_size[1] or ih >= canvas_size[0]:
                attempts += 1
                continue

            px = random.randint(0, canvas_size[1] - iw)
            py = random.randint(0, canvas_size[0] - ih)
            center_x = px + iw // 2
            center_y = py + ih // 2

            # 检查距离（防止目标之间重叠）
            min_dist = 2 * gaussian_radius
            too_close = any(np.hypot(center_x - cx, center_y - cy) < min_dist for cx, cy in object_centers)
            if too_close:
                attempts += 1
                continue

            object_centers.append((center_x, center_y))
            object_centers_all_classes[cls_index].append((center_x, center_y))
            canvas.paste(inst, (px, py), inst)

            # === 替换为掩码叠加逻辑 ===
            alpha = np.array(inst)[:, :, 3]
            mask_patch = (alpha > 0).astype(np.uint8)

            x1, y1 = px, py
            x2, y2 = px + iw, py + ih
            if x2 > canvas_size[1] or y2 > canvas_size[0]:
                continue

            density_maps[cls_index][y1:y2, x1:x2] += mask_patch.astype(np.float32)

            attempts += 1

    # 保存合成图像
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(os.path.join(output_path, "image.png"))

    # 保存密度图（mask形式）
    for i, dmap in enumerate(density_maps):
        dmap = np.clip(dmap, 0, 1)  # 保证最大为1（不会因为重叠而累加）
        np.save(os.path.join(output_path, f"density_c{i+1}.npy"), dmap.astype(np.float32))
        normed = (dmap * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"density_c{i+1}.png"), normed)

def generate_sample_to_dual(output_path: str):
    from PIL import ImageEnhance

    os.makedirs(output_path, exist_ok=True)

    bg_color = tuple([random.randint(180, 220)] * 3 + [255])
    canvas = Image.new("RGBA", canvas_size, bg_color)
    dot_maps  = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]
    mask_maps = [np.zeros(canvas_size, dtype=np.float32) for _ in range(4)]

    object_centers_all_classes = [[] for _ in range(4)]

    for cls_index in range(4):
        cls_dir = os.path.join(lib_root, f"c{cls_index+1}")
        base_scale = scale_factors[cls_index]
        min_n, max_n = instance_ranges[cls_index]
        n_instances = random.randint(min_n, max_n)

        object_centers = []

        attempts = 0
        while len(object_centers) < n_instances and attempts < n_instances * 10:
            inst = load_random_png_from(cls_dir)
            angle = random.randint(0, 360)
            scale = base_scale * random.uniform(0.9, 1.1)

            w, h = inst.size
            inst = inst.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)
            inst = inst.rotate(angle, expand=True)

            # 调整亮度
            enhancer = ImageEnhance.Brightness(inst)
            inst = enhancer.enhance(brightness[cls_index])

            iw, ih = inst.size
            if iw >= canvas_size[1] or ih >= canvas_size[0]:
                attempts += 1
                continue

            px = random.randint(0, canvas_size[1] - iw)
            py = random.randint(0, canvas_size[0] - ih)
            center_x = px + iw // 2
            center_y = py + ih // 2

            min_dist = 2 * gaussian_radius
            too_close = any(np.hypot(center_x - cx, center_y - cy) < min_dist for cx, cy in object_centers)
            if too_close:
                attempts += 1
                continue

            object_centers.append((center_x, center_y))
            object_centers_all_classes[cls_index].append((center_x, center_y))
            canvas.paste(inst, (px, py), inst)

            # === dot map 添加高斯峰 ===
            gaussian = generate_gaussian_map(canvas_size, (center_x, center_y), gaussian_sigma)
            dot_maps[cls_index] += gaussian

            # === mask map 添加掩码块 ===
            alpha = np.array(inst)[:, :, 3]
            mask_patch = (alpha > 0).astype(np.uint8)
            x1, y1 = px, py
            x2, y2 = px + iw, py + ih
            if x2 > canvas_size[1] or y2 > canvas_size[0]:
                continue
            mask_maps[cls_index][y1:y2, x1:x2] += mask_patch.astype(np.float32)

            attempts += 1

    # 保存图像
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(os.path.join(output_path, "image.png"))

    # 保存两种GT（.npy + .png）
    for i in range(4):
        # DOT
        dmap = dot_maps[i]
        np.save(os.path.join(output_path, f"density_c{i+1}_dot.npy"), dmap.astype(np.float32))
        normed = (dmap / (dmap.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"density_c{i+1}_dot.png"), normed)

        # MASK
        mmap = np.clip(mask_maps[i], 0, 1)
        np.save(os.path.join(output_path, f"density_c{i+1}_mask.npy"), mmap.astype(np.float32))
        cv2.imwrite(os.path.join(output_path, f"density_c{i+1}_mask.png"), (mmap * 255).astype(np.uint8))


# ==== 执行 ====
if __name__ == "__main__":
    generate_sample()
