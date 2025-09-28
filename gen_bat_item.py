import os
from gen_one_item import generate_sample_to
from gen_one_item import generate_sample_to_mask
from gen_one_item import generate_sample_to_dual

output_root = "data_dual"
num_items = 200  # 要生成多少组数据

os.makedirs(output_root, exist_ok=True)

for i in range(1, num_items + 1):
    item_dir = os.path.join(output_root, f"item{i}")
    # generate_sample_to(item_dir)
    # generate_sample_to_mask(item_dir)
    generate_sample_to_dual(item_dir)
    print(f"[{i}/{num_items}] Generated: {item_dir}")
