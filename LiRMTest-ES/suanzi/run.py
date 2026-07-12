# 示例：对KITTI某帧点云应用七种算子
from suanzi.new.em_interference_effect import em_interference_effect
from suanzi.new.extreme_dry_effect import extreme_dry_effect
from suanzi.new.high_temp import high_temp_effect
from suanzi.new.humid_effect import humid_effect
from suanzi.new.light_effect import light_effect
from suanzi.new.low_temp import low_temp_effect
import os
import random


def process_files(input_dir, output_dir,label_dir ,mode, count):
    """
    批量处理点云文件
    :param input_dir: 原始点云文件所在目录
    :param output_dir: 处理后文件保存目录
    :param mode: 选取模式，'sequential'（顺序选取）或 'random'（随机选取）
    :param count: 处理文件数量，None表示处理所有文件
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有.bin文件
    bin_files = [f for f in os.listdir(input_dir) if f.endswith('.bin')]
    if not bin_files:
        print("未找到任何.bin文件")
        return

    # 根据模式筛选文件
    if mode == 'sequential':
        # 按文件名排序（假设文件名是数字编号，如000000.bin）
        bin_files.sort()
        selected_files = bin_files[:count] if count else bin_files
    elif mode == 'random':
        # 随机选取指定数量文件
        selected_count = count if count else len(bin_files)
        selected_files = random.sample(bin_files, min(selected_count, len(bin_files)))
    else:
        print("模式错误，仅支持 'sequential' 或 'random'")
        return

    # 批量处理选中的文件
    for bin_file in selected_files:
        base_name = os.path.splitext(bin_file)[0]
        kitti_bin = os.path.join(input_dir, bin_file)
        # 从手动指定的label_dir中获取label文件，不再自动替换路径
        kitti_label = os.path.join(label_dir, f"{base_name}.txt")

        # 检查label文件是否存在
        if not os.path.exists(kitti_label):
            print(f"警告：未找到{base_name}.txt，极端干燥算子可能无法正常运行")

        # # 1. 湿气算子
        # humid_effect(
        #     kitti_bin,
        #     os.path.join(output_dir, f"humid_{base_name}.bin"),
        #     humidity=90
        # )
        #
        # # 2. 高温算子
        # high_temp_effect(
        #     kitti_bin,
        #     os.path.join(output_dir, f"high_temp_{base_name}.bin"),
        #     temp=50
        # )
        #
        # 3. 低温算子
        # low_temp_effect(
        #     kitti_bin,
        #     os.path.join(output_dir, f"low_temp_{base_name}.bin"),
        #     temp=-10
        # )

        # # # 4. 极端干燥算子
        # extreme_dry_effect(
        #     kitti_bin,
        #     kitti_label,
        #     os.path.join(output_dir, f"dry_{base_name}.bin")
        # )

        # 5. 电磁干扰算子
        em_interference_effect(
            kitti_bin,
            os.path.join(output_dir, f"em_{base_name}.bin"),
            em_intensity=100
        )
        #
        # # # 6. 强光/逆光算子
        # light_effect(
        #     kitti_bin,
        #     os.path.join(output_dir, f"strong_light_{base_name}.bin"),
        #     light_type="strong"
        # )
        # light_effect(
        #     kitti_bin,
        #     os.path.join(output_dir, f"backlight_{base_name}.bin"),
        #     light_type="backlight"
        # )

        print(f"已处理：{bin_file}")
if __name__ == "__main__":
    # # KITTI数据路径（需替换为你的数据路径）
    # kitti_bin = "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/bin/000001.bin"
    # kitti_label = "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/txt/000001.txt"
    #
    # # 1. 湿气算子
    # humid_effect(kitti_bin, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/humid_000001.bin", humidity=90)
    # 2. 高温算子
    # high_temp_effect(kitti_bin, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/high_temp_000000.bin", temp=50)
    # # 3. 低温算子
    # low_temp_effect(kitti_bin, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/low_temp_000000.bin", temp=-10)
    # # 4. 极端干燥算子
    # extreme_dry_effect(kitti_bin, kitti_label, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/dry_000000.bin")
    # # 5. 电磁干扰算子
    # em_interference_effect(kitti_bin, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/em_000000.bin", em_intensity=100)
    # # 6. 强光/逆光算子
    # light_effect(kitti_bin, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/strong_light_000000.bin", light_type="strong")
    # light_effect(kitti_bin, "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/backlight_000000.bin", light_type="backlight")

    # 配置路径（请根据实际情况修改）
    input_bin_dir = "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/velodyne"  # 原始bin文件目录
    output_dir = "E:/数据/数据集A/电磁干扰算子/电磁干扰算子点云3.0" # 输出目录"E:/数据/数据集A/逆光算子/逆光算子点云"  # 输出目录
    label_dir = "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/label_2"

    # 处理配置
    process_mode = "sequential"  # 模式：'sequential'（顺序）或 'random'（随机）
    process_count = None  # 处理数量：None表示全部，数字表示指定数量

    # 执行批量处理
    process_files(input_bin_dir, output_dir, label_dir,process_mode, process_count)
    print("批量处理完成！")