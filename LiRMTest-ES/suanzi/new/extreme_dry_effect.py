import math
import numpy as np


def extreme_dry_effect(kitti_bin_path, kitti_label_path, save_path, dust_density=1.0):
    """
    极端干燥（尘埃）算子：贴合激光大气散射物理规律（比尔-朗伯定律+边缘模糊）
    :param kitti_bin_path: KITTI原始点云路径
    :param kitti_label_path: KITTI标签路径（.txt）
    :param save_path: 生成点云保存路径
    :param dust_density: 尘埃浓度系数（默认1.0，建议范围0.5~2.0，越大越干燥/尘埃越多）
    """
    # 1. 读取点云与标签
    pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
    x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    N = len(pc)
    if N == 0:
        raise ValueError("原始点云为空")

    # 2. 解析目标信息（修正tz未存储的bug，补充完整目标参数）
    targets = []
    with open(kitti_label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:  # 过滤无效标签行
                continue
            if parts[0] in ['Car', 'Pedestrian', 'Cyclist']:  # KITTI核心目标类型
                # 解析目标中心(x,y,z)、尺寸(l,w,h)、高度范围
                tx, ty, tz = float(parts[11]), float(parts[12]), float(parts[13])
                l, w, h = float(parts[8]), float(parts[9]), float(parts[10])
                R = math.sqrt((l / 2) ** 2 + (w / 2) ** 2)  # 目标水平外接圆半径
                z_min = tz - h / 2  # 目标下沿高度
                z_max = tz + h / 2  # 目标上沿高度
                targets.append({
                    'center': (tx, ty, tz),
                    'R': R,
                    'z_range': (z_min, z_max)
                })

    # 3. 核心物理参数（贴合尘埃散射的比尔-朗伯定律）
    # 尘埃衰减系数（经验值，基于大气光学实测，dust_density缩放）
    alpha = 0.01 * dust_density  # 距离衰减系数（m^-1）
    # 边缘增强衰减系数（尘埃导致边缘散射更显著）
    edge_alpha = 0.02 * dust_density
    # 边缘位置模糊噪声系数（尘埃导致的轻微位置偏移）
    edge_noise_sigma = 0.01 * dust_density

    # 4. 步骤1：全局距离相关的强度指数衰减（核心修正，符合比尔-朗伯定律）
    # 计算每个点到LiDAR的传输距离
    d = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    d = np.maximum(d, 1e-3)  # 避免距离为0
    # 指数衰减：I = I0 * exp(-alpha * d)
    I_dry = I0 * np.exp(-alpha * d)

    # 5. 步骤2：目标边缘点的增强衰减+位置模糊（贴合尘埃边缘散射特性）
    x_dry, y_dry, z_dry = x0.copy(), y0.copy(), z0.copy()
    for target in targets:
        tx, ty, tz = target['center']
        R = target['R']
        z_min, z_max = target['z_range']

        # 计算点到目标中心的水平距离
        r = np.sqrt((x0 - tx) ** 2 + (y0 - ty) ** 2)
        # 目标边缘点判定：
        # - 水平距离：0.7*R < r < 1.2*R（边缘范围扩大，更贴合实际）
        # - 高度范围：在目标上下沿内（修正原始tz未定义的bug）
        # - 距离LiDAR>5m（近距点尘埃影响可忽略）
        edge_mask = (r > 0.7 * R) & (r < 1.2 * R) & \
                    (z0 >= z_min) & (z0 <= z_max) & (d > 5)

        if np.sum(edge_mask) == 0:
            continue  # 无边缘点则跳过

        # 边缘点额外指数衰减（叠加全局衰减）
        edge_decay = np.exp(-edge_alpha * (r[edge_mask] - 0.7 * R))
        I_dry[edge_mask] *= edge_decay

        # 边缘点轻微位置模糊（尘埃散射导致的坐标噪声）
        noise_x = np.random.normal(0, edge_noise_sigma, np.sum(edge_mask))
        noise_y = np.random.normal(0, edge_noise_sigma, np.sum(edge_mask))
        noise_z = np.random.normal(0, edge_noise_sigma, np.sum(edge_mask))
        x_dry[edge_mask] += noise_x
        y_dry[edge_mask] += noise_y
        z_dry[edge_mask] += noise_z

    # 6. 物理约束：强度限制在0~255，坐标无极端偏移
    I_dry = np.clip(I_dry, 0, 255)
    # 限制位置噪声上限（避免边缘点偏移过大）
    max_offset = 0.1  # 最大偏移0.1m，符合尘埃散射的轻微影响
    x_dry = np.clip(x_dry, x0.min() - max_offset, x0.max() + max_offset)
    y_dry = np.clip(y_dry, y0.min() - max_offset, y0.max() + max_offset)
    z_dry = np.clip(z_dry, z0.min() - max_offset, z0.max() + max_offset)

    # 7. 组装并保存点云
    final_pc = np.column_stack([x_dry, y_dry, z_dry, I_dry]).astype(np.float32)
    final_pc.tofile(save_path)

    # 输出统计信息
    global_decay_ratio = np.mean(I_dry / (I0 + 1e-6))  # 全局平均衰减率
    edge_point_num = sum([np.sum((np.sqrt((x0 - tx) ** 2 + (y0 - ty) ** 2) > 0.7 * R) &
                                 (np.sqrt((x0 - tx) ** 2 + (y0 - ty) ** 2) < 1.2 * R) &
                                 (z0 >= z_min) & (z0 <= z_max))
                          for tx, ty, tz, R, (z_min, z_max) in
                          [(t['center'][0], t['center'][1], t['center'][2], t['R'], t['z_range']) for t in targets]])
    print(f"极端干燥场景点云已保存：{save_path}")
    print(f"全局强度平均衰减率：{global_decay_ratio:.2f}（尘埃浓度系数：{dust_density}）")
    print(f"目标边缘点数量：{edge_point_num}（已叠加增强衰减+位置模糊）")

# 示例调用
# extreme_dry_effect(
#     kitti_bin_path="data/000001.bin",
#     kitti_label_path="data/000001.txt",
#     save_path="data/000001_dry.bin",
#     dust_density=1.5  # 高尘埃浓度（极端干燥）
# )
# import math
#
# import numpy as np
#
#
# def extreme_dry_effect(kitti_bin_path, kitti_label_path, save_path):
#     """
#     极端干燥算子：模拟尘埃导致目标边缘点强度衰减
#     :param kitti_bin_path: KITTI原始点云路径
#     :param kitti_label_path: KITTI标签路径（.txt），用于识别目标边缘
#     :param save_path: 生成点云保存路径
#     """
#     # 1. 读取点云与标签
#     pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
#     x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
#     targets = []
#
#     with open(kitti_label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if parts[0] in ['Car', 'Pedestrian', 'Cyclist']:  # KITTI核心目标类型
#                 # 解析目标中心(x,y,z)与尺寸(l,w,h)，计算外接圆半径R
#                 tx, ty, tz = float(parts[11]), float(parts[12]), float(parts[13])
#                 l, w = float(parts[8]), float(parts[9])
#                 R = math.sqrt((l / 2) ** 2 + (w / 2) ** 2)  # 目标外接圆半径
#                 targets.append({'center': (tx, ty), 'R': R})
#
#     # 2. 识别边缘点并衰减强度
#     I_dry = I0.copy()
#     for target in targets:
#         tx, ty = target['center']
#         R = target['R']
#         # 计算每个点到目标中心的距离
#         r = np.sqrt((x0 - tx) ** 2 + (y0 - ty) ** 2)
#         # 边缘点：r > 0.8*R 且在目标高度范围内（z0接近tz）
#         edge_mask = (r > 0.8 * R) & (np.abs(z0 - tz) < 1.5)  # 高度容忍1.5m
#         # 边缘点强度衰减
#         decay_factor = 0.3 + 0.7 * (1 - (r[edge_mask] - 0.8 * R) / (0.2 * R))
#         I_dry[edge_mask] = I0[edge_mask] * decay_factor
#
#     pc[:, 3] = np.clip(I_dry, 0, 255)
#     pc.astype(np.float32).tofile(save_path)
#     print(f"极端干燥场景点云已保存：{save_path}，边缘点强度已衰减")