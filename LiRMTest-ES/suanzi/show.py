# import numpy as np
# import open3d as o3d
#
#
# def read_kitti_bin(bin_path, downsample_rate=2):
#     """
#     读取KITTI格式点云（.bin），支持下采样（解决加载慢）
#     :param bin_path: 点云文件路径（.bin）
#     :param downsample_rate: 下采样率（2=每2个点取1个，默认2，卡顿可设为3）
#     :return: numpy格式点云（N,4：x,y,z,intensity）
#     """
#     # 读取KITTI点云（float32，x,y,z,intensity顺序）
#     pc_np = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#     # 格式校验（避免错误格式导致渲染失败）
#     if pc_np.shape[1] != 4:
#         raise ValueError(f"点云格式错误！需为4列（x,y,z,intensity），当前{pc_np.shape[1]}列")
#     # 下采样：减少点数，加快渲染速度（保留关键特征）
#     pc_np = pc_np[::downsample_rate]
#     print(f"成功读取点云：{bin_path}，下采样后点数：{len(pc_np)}（原始约{len(pc_np) * downsample_rate}个）")
#     return pc_np
#
#
# def visualize_single_pc(bin_path,
#                         color=[0, 1, 0],  # 点云颜色（默认绿色，可改为[1,0,0]红、[0,0,1]蓝等）
#                         point_size=2,  # 点大小（默认2，稀疏点云可设为3）
#                         background_color=[0.05, 0.05, 0.05]):  # 背景色（近黑色）
#     """
#     可视化单个点云文件
#     :param bin_path: 点云文件路径（.bin）
#     :param color: 点云RGB颜色（0-1范围，如[0,1,0]绿色）
#     :param point_size: 点云大小（1-5，根据点数调整）
#     :param background_color: 窗口背景色（0-1范围）
#     """
#     # 1. 读取点云
#     pc_np = read_kitti_bin(bin_path)
#
#     # 2. 转换为Open3D点云对象
#     pc_o3d = o3d.geometry.PointCloud()
#     pc_o3d.points = o3d.utility.Vector3dVector(pc_np[:, :3])  # 仅用x,y,z坐标
#     pc_o3d.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pc_np), 1)))  # 批量赋值颜色
#
#     # 3. 创建可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(
#         window_name=f"点云可视化 - {bin_path.split('/')[-1]}",
#         width=1000,
#         height=800
#     )
#
#     # 4. 添加点云并设置显示参数（优化渲染效果）
#     vis.add_geometry(pc_o3d)
#     opt = vis.get_render_option()
#     opt.background_color = np.asarray(background_color)  # 背景色
#     opt.show_coordinate_frame = True  # 显示坐标系（x红、y绿、z蓝，辅助定位）
#     opt.point_size = point_size  # 点大小（关键：避免过密或过疏）
#     opt.line_width = 1  # 坐标系线宽
#     opt.point_show_normal = False  # 关闭法向量显示（减少计算，加快加载）
#
#     # 5. 启动可视化（支持交互）
#     print("\n=== 可视化窗口已启动 ===")
#     print("交互操作：")
#     print("1. 鼠标左键拖动：旋转视角")
#     print("2. 鼠标滚轮：缩放点云（放大查看细节）")
#     print("3. 鼠标右键拖动：平移视野")
#     print("4. 按ESC键：关闭窗口")
#     vis.run()
#     vis.destroy_window()
#
#
# # --------------------------
# # 使用示例（替换为你的点云路径）
# # --------------------------
# if __name__ == "__main__":
#     # 替换为你的KITTI点云文件路径（.bin格式）
#     # pc_bin_path = "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/bin/000000.bin"
#     pc_bin_path = "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/dry_000000.bin"
#
#     # 可视化原始点云（绿色，点大小2）
#     visualize_single_pc(
#         bin_path=pc_bin_path,
#         color=[0, 1, 0],  # 绿色（原始点云）
#         point_size=2
#     )
#
#     # 若要可视化蜕变后点云（如高温算子生成的点云），只需修改路径和颜色：
#     # visualize_single_pc(
#     #     bin_path="output/high_temp_000000.bin",
#     #     color=[1, 0, 0],  # 红色（蜕变后点云）
#     #     point_size=2
#     # )

import numpy as np
import open3d as o3d


def read_kitti_bin(bin_path, downsample_rate=2):
    """
    读取KITTI点云（.bin），返回坐标+强度数据，支持下采样
    :param bin_path: KITTI点云路径
    :param downsample_rate: 下采样率（默认2，卡顿可设为3）
    :return: pc_np (N,4: x,y,z,intensity), total_original_points (原始点数)
    """
    pc_np = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    total_original_points = len(pc_np)

    # 格式校验
    if pc_np.shape[1] != 4:
        raise ValueError(f"点云格式错误！需为4列（x,y,z,intensity），当前{pc_np.shape[1]}列")

    # 下采样
    pc_np = pc_np[::downsample_rate]
    return pc_np, total_original_points


def intensity_to_color(intensity_norm):
    """强度映射为彩色（Jet色彩表：低→蓝，中→绿，高→红）"""
    n_points = len(intensity_norm)
    color = np.zeros((n_points, 3), dtype=np.float32)

    # 低强度（0-0.33）：蓝→青
    mask1 = intensity_norm < 0.33
    color[mask1, 0] = 0
    color[mask1, 1] = intensity_norm[mask1] / 0.33
    color[mask1, 2] = 1 - intensity_norm[mask1] / 0.33

    # 中强度（0.33-0.66）：青→黄
    mask2 = (intensity_norm >= 0.33) & (intensity_norm < 0.66)
    offset = (intensity_norm[mask2] - 0.33) / 0.33
    color[mask2, 0] = offset
    color[mask2, 1] = 1
    color[mask2, 2] = 0

    # 高强度（0.66-1）：黄→红
    mask3 = intensity_norm >= 0.66
    offset = (intensity_norm[mask3] - 0.66) / 0.34
    color[mask3, 0] = 1
    color[mask3, 1] = 1 - offset
    color[mask3, 2] = 0

    return color, mask1, mask2, mask3  # 返回mask用于统计点数


def count_intensity_distribution(intensity, mask1, mask2, mask3):
    """
    统计不同强度区间的点数
    :param intensity: 原始强度值（未归一化）
    :param mask1/mask2/mask3: 低/中/高强度区间的掩码
    :return: 强度区间统计结果（字典）
    """
    # 低强度区间（归一化0-0.33 → 原始强度按实际范围映射）
    low_intensity = intensity[mask1]
    # 中强度区间（0.33-0.66）
    mid_intensity = intensity[mask2]
    # 高强度区间（0.66-1）
    high_intensity = intensity[mask3]

    # 自定义弱对象强度区间（如原始强度≤50，可根据需求修改）
    weak_mask = intensity <= 50
    weak_intensity = intensity[weak_mask]

    return {
        "低强度（蓝）": {
            "点数": len(low_intensity),
            "占比": f"{len(low_intensity) / len(intensity) * 100:.2f}%",
            "强度范围": f"[{low_intensity.min():.2f}, {low_intensity.max():.2f}]"
        },
        "中强度（绿）": {
            "点数": len(mid_intensity),
            "占比": f"{len(mid_intensity) / len(intensity) * 100:.2f}%",
            "强度范围": f"[{mid_intensity.min():.2f}, {mid_intensity.max():.2f}]"
        },
        "高强度（红）": {
            "点数": len(high_intensity),
            "占比": f"{len(high_intensity) / len(intensity) * 100:.2f}%",
            "强度范围": f"[{high_intensity.min():.2f}, {high_intensity.max():.2f}]"
        },
        "弱对象候选（≤50）": {
            "点数": len(weak_intensity),
            "占比": f"{len(weak_intensity) / len(intensity) * 100:.2f}%",
            "强度范围": f"[{weak_intensity.min():.2f}, {weak_intensity.max():.2f}]"
        }
    }


def visualize_kitti_pc_with_intensity_count(bin_path, downsample_rate=2, point_size=2):
    """
    可视化KITTI点云（含总点数+强度分布统计+强度映射）
    :param bin_path: 点云文件路径（.bin）
    :param downsample_rate: 下采样率
    :param point_size: 点大小（默认2）
    """
    # 1. 读取点云数据
    pc_np, total_original = read_kitti_bin(bin_path, downsample_rate)
    x, y, z, intensity = pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], pc_np[:, 3]
    total_downsampled = len(pc_np)

    # 2. 强度值预处理（过滤异常值+归一化）
    intensity = np.clip(intensity, 0, 255)  # 过滤0-255外的异常值
    intensity_min = np.min(intensity)
    intensity_max = np.max(intensity)

    if intensity_max - intensity_min < 1e-6:
        intensity_norm = np.zeros_like(intensity)
    else:
        intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)

    # 3. 强度映射彩色+获取区间掩码
    pc_color, mask1, mask2, mask3 = intensity_to_color(intensity_norm)

    # 4. 统计不同强度区间点数
    intensity_stats = count_intensity_distribution(intensity, mask1, mask2, mask3)

    # 5. 转换为Open3D点云对象
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_np[:, :3])
    pc_o3d.colors = o3d.utility.Vector3dVector(pc_color)

    # 6. 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    window_name = f"KITTI点云可视化 - {bin_path.split('/')[-1]}"
    vis.create_window(window_name=window_name, width=1200, height=800)

    # 7. 配置显示参数
    vis.add_geometry(pc_o3d)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05])
    opt.show_coordinate_frame = True
    opt.point_size = point_size
    opt.line_width = 1
    opt.point_show_normal = False

    # 8. 打印详细统计信息
    print("=" * 60)
    print(f"文件路径：{bin_path}")
    print(f"原始点数：{total_original}")
    print(f"下采样后总点数：{total_downsampled}（采样率：1/{downsample_rate}）")
    print(f"整体强度值范围（过滤后）：[{intensity_min:.2f}, {intensity_max:.2f}]")
    print("\n📊 不同强度区间点数统计：")
    for key, value in intensity_stats.items():
        print(f"  {key}: {value['点数']} 个点（{value['占比']}），强度范围{value['强度范围']}")
    print("\n🎨 颜色含义：")
    print("  🔵 蓝色：低强度（0-0.33归一化强度）→ 弱对象候选")
    print("  🟢 绿色：中强度（0.33-0.66归一化强度）")
    print("  🔴 红色：高强度（0.66-1归一化强度）→ 近距离高反射目标")
    print("\n💡 交互操作：")
    print("  左键拖动：旋转视角 | 滚轮：缩放 | 右键拖动：平移 | ESC：关闭窗口")
    print("=" * 60)

    # 9. 启动可视化
    vis.run()
    vis.destroy_window()


# --------------------------
# 使用示例（替换为你的KITTI路径）
# --------------------------
if __name__ == "__main__":
    # 替换为你的KITTI点云文件路径（.bin格式）
    # kitti_bin_path = "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/bin/000000.bin"
    kitti_bin_path ="C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/rain_000000.bin"

    # 可视化原始点云
    visualize_kitti_pc_with_intensity_count(
        bin_path=kitti_bin_path,
        downsample_rate=2,
        point_size=2
    )

    # 可视化蜕变后点云（如高温算子生成的点云，对比强度分布变化）
    # visualize_kitti_pc_with_intensity_count(
    #     bin_path="output/high_temp_000000.bin",
    #     downsample_rate=2,
    #     point_size=2
    # )