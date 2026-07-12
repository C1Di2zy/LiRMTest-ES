import numpy as np
import math


def low_temp_effect(kitti_bin_path, save_path, temp=-10, random_seed=42):
    """
    低温算子：模拟LiDAR电机扫描速度不稳定导致水平方向点云疏密不均（贴合物理特性）
    物理依据：
    1. 低温导致电机转速波动 → 水平角（方位角）扫描间隔周期性波动；
    2. 垂直角（俯仰角）由线束固定，距离不变，仅水平角影响x/y坐标；
    3. 波动幅度随温度降低增大（电机负温度系数），且有硬件上限；
    4. 波动频率匹配LiDAR水平扫描周期（0~2π覆盖一次完整扫描）。
    :param kitti_bin_path: KITTI原始点云路径（.bin）
    :param save_path: 生成点云保存路径（.bin）
    :param temp: 环境温度（℃），LiDAR典型工作范围：-40~60℃
    :param random_seed: 随机种子（保证波动结果可复现）
    """
    # ========== 1. 鲁棒性初始化与校验 ==========
    np.random.seed(random_seed)  # 波动结果可复现
    # 温度范围校验（LiDAR硬件工作范围）
    if not (-40 <= temp <= 60):
        raise ValueError(f"温度超出LiDAR典型工作范围（-40~60℃），当前输入：{temp}℃！")
    # 读取点云并校验
    pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
    if len(pc) == 0:
        raise ValueError("输入点云文件为空，请检查路径！")
    x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    original_num = len(pc)

    # ========== 2. 基础物理量计算（防呆+符合LiDAR特性） ==========
    # 距离（m）：加1e-6避免除零，低温不影响距离测量
    d = np.sqrt(x0 ** 2 + y0 ** 2 + (z0 + 1e-6) ** 2)
    # 原始水平角（方位角，弧度）：范围[-π, π]，按角度排序（模拟扫描顺序）
    theta0 = np.arctan2(y0, x0)
    # 垂直角（俯仰角，弧度）：低温不影响，固定不变
    phi = np.arcsin(z0 / d)  # 已防除零

    # ========== 3. 低温导致的水平扫描角波动（核心物理逻辑） ==========
    # 3.1 按原始水平角排序（模拟LiDAR从左到右/顺时针扫描顺序）
    sort_idx = np.argsort(theta0)
    theta_sorted = theta0[sort_idx]  # 排序后的水平角（从小到大）
    d_sorted = d[sort_idx]
    phi_sorted = phi[sort_idx]

    # 3.2 波动因子：贴合电机温度特性（负温度系数，有上限）
    # gamma：波动幅度（低温越大，波动越明显），上限0.2（硬件机械极限）
    gamma = 0.02 + 0.004 * (25 - temp)  # 25℃时基础波动0.02，-10℃时0.14，-40℃时0.2（上限）
    gamma = np.clip(gamma, 0.01, 0.2)  # 最小波动0.01，最大0.2（硬件保护）
    # omega：波动频率（匹配水平扫描周期，2π覆盖一次完整扫描）
    omega = 2 * math.pi / (theta_sorted.max() - theta_sorted.min() + 1e-6)  # 按实际角度范围适配

    # 3.3 计算波动后的水平角间隔（原始间隔0.1°，转弧度）
    delta_theta0 = math.radians(0.1)  # LiDAR原始水平角分辨率（0.1°）
    k_sorted = np.arange(len(theta_sorted))  # 按扫描顺序的索引
    # 正弦波动：模拟扫描速度时快时慢 → 角度间隔时大时小
    delta_theta_lowT = delta_theta0 * (1 + gamma * np.sin(omega * k_sorted))

    # 3.4 累积水平角（基于扫描顺序，而非全局点数）
    theta_lowT = theta_sorted[0] + np.cumsum(delta_theta_lowT)
    # 限制水平角范围（避免超出[-π, π]）
    theta_lowT = np.mod(theta_lowT + math.pi, 2 * math.pi) - math.pi

    # ========== 4. 坐标转换（仅水平角变化，垂直角/距离不变） ==========
    # 球坐标转笛卡尔坐标（仅theta变化，phi/d固定）
    x_lowT = d_sorted * np.cos(theta_lowT) * np.cos(phi_sorted)
    y_lowT = d_sorted * np.sin(theta_lowT) * np.cos(phi_sorted)
    z_lowT = d_sorted * np.sin(phi_sorted)  # 垂直角不变，z坐标不变

    # ========== 5. 恢复原始点云顺序并保存 ==========
    # 反向排序恢复原始点云顺序
    inv_sort_idx = np.argsort(sort_idx)
    pc[:, 0] = x_lowT[inv_sort_idx]
    pc[:, 1] = y_lowT[inv_sort_idx]
    pc[:, 2] = z_lowT[inv_sort_idx]  # 垂直角不变，z坐标与原始一致
    pc[:, 3] = I0  # 低温不影响强度（强度由回波决定）

    # 保存点云
    pc.astype(np.float32).tofile(save_path)

    # ========== 6. 量化日志输出 ==========
    print(f"低温场景点云已保存：{save_path}")
    print(f"环境温度：{temp}℃，波动幅度gamma：{gamma:.4f}")
    print(f"原始点数：{original_num}，处理后点数：{len(pc)}")
    print(f"水平角波动范围：±{gamma * 100:.2f}%（相对原始0.1°间隔）")

# 示例调用
# low_temp_effect("input/000000.bin", "output/low_temp_-10.bin", temp=-10, random_seed=42)
# low_temp_effect("input/000000.bin", "output/low_temp_-40.bin", temp=-40, random_seed=42)
# import numpy as np
#
# import math
#
#
# def low_temp_effect(kitti_bin_path, save_path, temp=-10):
#     """
#     低温算子：模拟扫描速度不稳定导致点云疏密不均
#     :param kitti_bin_path: KITTI原始点云路径
#     :param save_path: 生成点云保存路径
#     :param temp: 环境温度（℃），默认-10℃（极端低温）
#     """
#     # 1. 读取KITTI点云并转换为球坐标
#     pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
#     x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
#
#     d = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)  # 距离
#     theta0 = np.arctan2(y0, x0)  # 原始水平角（弧度）
#     phi = np.arcsin(z0 / (d + 1e-6))  # 垂直角（避免除零）
#     N = len(pc)
#
#     # 2. 计算波动后的水平角间隔
#     delta_theta0 = math.radians(0.1)  # 原始间隔0.1°
#     gamma = 0.05 + 0.005 * (25 - temp)
#     omega = math.pi / 10
#     k = np.arange(1, N + 1)
#     delta_theta_lowT = delta_theta0 * (1 + gamma * np.sin(omega * k))
#
#     # 3. 修正水平角并转换回笛卡尔坐标
#     theta_lowT = np.cumsum(delta_theta_lowT) + theta0[0]  # 累积水平角
#     x_lowT = d * np.cos(theta_lowT) * np.cos(phi)
#     y_lowT = d * np.sin(theta_lowT) * np.cos(phi)
#     z_lowT = d * np.sin(phi)
#
#     # 4. 更新点云并保存
#     pc[:, 0], pc[:, 1], pc[:, 2] = x_lowT, y_lowT, z_lowT
#     pc.astype(np.float32).tofile(save_path)
#     print(f"低温场景点云已保存：{save_path}，点数：{len(pc)}")