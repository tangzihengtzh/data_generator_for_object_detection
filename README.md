# Small Object Detection Dataset Generator

## Introduction
This repository provides a lightweight toolkit for generating synthetic datasets for small object detection tasks.
By using only a small number of object samples (transparent PNGs), the tool automatically synthesizes large-scale training data with various transformations and ground-truth labels (density maps, masks, or dot annotations).

## Features
- Object extraction: Automatically crop individual objects from PNG images with transparent background.
- Random placement: Place objects with random scale, rotation, brightness, and position.
- Non-overlap control: Configurable overlap margin to prevent or allow object overlapping.
- Ground-truth generation: Support for Gaussian dot maps, binary masks, and dual annotations.
- Batch synthesis: Easily generate hundreds of dataset samples in one command.
- Density map for counting: Generate per-class density maps for object counting tasks.

## Project Structure
```
├── div_tar.py              # Extract objects from PNG
├── gen_one_item.py         # Generate one synthetic sample
├── gen_bat_item.py         # Batch sample generator
├── generate_no_overlap.py  # Non-overlap or controlled-overlap synthesis
├── tools.py                # Utility functions & optimized generator
├── lib/                    # Input object libraries (c1,c2,c3,c4…)
└── output/                 # Generated dataset outputs
```

## Usage
### 1. Prepare object library
Put cropped PNG objects (transparent background) into subfolders:
```
lib/
 ├── c1/
 ├── c2/
 ├── c3/
 └── c4/
```

### 2. Generate one sample
```
python gen_one_item.py
```
Output: `output/image.png`, density maps (`.npy` + `.png`).

### 3. Generate multiple samples
```
python gen_bat_item.py
```
You can modify `num_items` in the script.

### 4. Non-overlap / controlled overlap
```
python generate_no_overlap.py
```

## Applications
- Few-shot small object detection
- Object counting via density maps
- Aerial imagery or microscopic target simulation

## License
MIT License. Free for research and educational use.


# 小目标检测数据集生成工具

## 简介
本项目是一个轻量级的小目标检测数据集生成工具。
用户只需少量带透明背景的样本图片（PNG），即可自动生成大规模训练数据，包含多种增强与标签（密度图、掩码、点标注）。

## 特性
- 目标提取：自动从带透明背景的 PNG 中裁剪单个目标。
- 随机放置：随机缩放、旋转、亮度调整与位置放置。
- 重叠控制：可配置目标之间的最小间距，支持禁止或部分允许重叠。
- 标签生成：支持高斯点图、掩码图，以及两者并存的双标签。
- 批量合成：一条命令即可生成数百组样本数据。
- 计数任务支持：生成基于类别的密度图，可用于目标计数任务。

## 项目结构
```
├── div_tar.py              # 裁剪单个目标
├── gen_one_item.py         # 单样本合成
├── gen_bat_item.py         # 批量生成器
├── generate_no_overlap.py  # 重叠控制生成
├── tools.py                # 工具与优化版生成器
├── lib/                    # 输入素材库 (c1,c2,c3,c4…)
└── output/                 # 生成的数据集
```

## 使用方法
### 1. 准备素材库
将裁剪后的透明 PNG 素材放入以下目录：
```
lib/
 ├── c1/
 ├── c2/
 ├── c3/
 └── c4/
```

### 2. 生成单个样本
```
python gen_one_item.py
```
输出: `output/image.png`，以及密度图 (`.npy` + `.png`)。

### 3. 批量生成数据
```
python gen_bat_item.py
```
可在脚本中修改 `num_items` 控制生成数量。

### 4. 控制目标重叠
```
python generate_no_overlap.py
```

## 应用场景
- 少样本小目标检测
- 基于密度图的目标计数
- 航拍小目标/显微小目标模拟

单个数据集实例如下，包含多类别标签密度图
<img width="1830" height="233" alt="image" src="https://github.com/user-attachments/assets/41fda8e5-525e-4adb-9a5a-bdac90959ab6" />


## 许可证
MIT 协议，支持科研与教育使用。
