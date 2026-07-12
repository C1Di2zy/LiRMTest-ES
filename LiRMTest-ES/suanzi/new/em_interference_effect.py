import math
import numpy as np


def fix_invalid_pc(pc):
    """过滤点云异常值，保证坐标合法（解决检测时的shapely报错）"""
    # 1. 过滤NaN/Inf值（算子叠加噪声可能产生）
    mask = ~(np.isnan(pc).any(axis=1) | np.isinf(pc).any(axis=1))
    pc = pc[mask]

    # 2. 限制坐标范围（KITTI场景合理范围，避免极端偏移）
    # x: 前向（0~100m）, y: 左右（-50~50m）, z: 高低（-10~20m）
    x_valid = (pc[:, 0] >= 0) & (pc[:, 0] <= 100)
    y_valid = (pc[:, 1] >= -50) & (pc[:, 1] <= 50)
    z_valid = (pc[:, 2] >= -10) & (pc[:, 2] <= 20)
    pc = pc[x_valid & y_valid & z_valid]

    # 3. 过滤距离过近/过远的点（LiDAR有效探测范围）
    d = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    pc = pc[(d >= 0.5) & (d <= 100)]  # 0.5~100m为有效范围

    return pc


def em_interference_effect(kitti_bin_path, save_path, em_intensity=100):
    """
    高压电线电磁干扰算子：贴合LiDAR电磁干扰物理特性（50Hz工频干扰+距离相关噪声+强度波动）
    :param kitti_bin_path: KITTI原始点云路径
    :param save_path: 生成点云保存路径
    :param em_intensity: 电磁强度（V/m），默认100V/m（高压电线场景，建议范围50~200V/m）
    """
    # 1. 读取KITTI点云（LiDAR坐标系：x-前，y-左，z-上）
    pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
    x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    N = len(pc)
    if N == 0:
        raise ValueError("原始点云为空")

    # 2. 核心物理参数（贴合50Hz工频电磁干扰特性）
    f = 50.0  # 工频50Hz
    c = 3e8  # 光速（LiDAR测距依赖光速）
    # EMI强度系数（电磁学经验值，可根据实测调整）
    k_d = 1e-6 * math.sqrt(em_intensity)  # 测距波动系数
    k_noise = 5e-5 * math.sqrt(em_intensity)  # 坐标噪声系数（与距离正相关）
    k_I = 0.1 * (em_intensity / 100)  # 强度波动系数

    # 3. 计算原始距离（点到LiDAR原点）
    d0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    # 避免距离为0的极端情况
    d0 = np.maximum(d0, 1e-3)

    # 4. 工频干扰导致的测距值周期性波动（核心修正：干扰测距而非角度）
    # LiDAR扫描时序：假设点云按扫描顺序排列，时间戳与点索引线性相关（合理简化）
    # 时间步长：LiDAR典型扫描周期~0.1s，单帧点云时间跨度~0.1s，故dt=0.1/N
    dt = 0.1 / N
    t = np.arange(N) * dt  # 物理意义：单帧内每个点的扫描时间戳（s）
    # 测距波动：d = d0 + 波动值（与EM强度、距离、工频相关）
    d_em = d0 + k_d * d0 * np.sin(2 * math.pi * f * t)  # 波动幅度与距离正相关

    # 5. 球坐标转换（仅用于坐标重构，角度无偏移）
    theta0 = np.arctan2(y0, x0)  # 水平角（绕z轴）
    phi0 = np.arcsin(z0 / (d0 + 1e-6))  # 俯仰角（天顶角）
    # 基于干扰后的距离重构坐标（角度由扫描结构决定，无EMI偏移）
    x_em = d_em * np.cos(theta0) * np.cos(phi0)
    y_em = d_em * np.sin(theta0) * np.cos(phi0)
    z_em = d_em * np.sin(phi0)

    # 6. 距离相关的随机坐标噪声（核心修正：噪声强度与距离正相关）
    sigma = k_noise * d0  # 远距离点信号弱，干扰更明显
    noise_x = np.random.normal(0, sigma, N)
    noise_y = np.random.normal(0, sigma, N)
    noise_z = np.random.normal(0, sigma, N)
    # 叠加噪声（限制坐标噪声上限，避免极端值）
    max_noise = 0.5  # 最大坐标噪声（m），符合LiDAR实测干扰范围
    x_em += np.clip(noise_x, -max_noise, max_noise)
    y_em += np.clip(noise_y, -max_noise, max_noise)
    z_em += np.clip(noise_z, -max_noise, max_noise)

    # 7. 电磁干扰导致的强度波动（核心新增：符合探测器受干扰特性）
    I_em = I0 + np.random.normal(0, k_I * 255, N)  # 强度波动基于255满量程
    I_em = np.clip(I_em, 0, 255)  # 限制强度在LiDAR合法范围

    # 8. 物理约束的额外虚警噪声点（修正：数量/分布/强度符合EMI虚警特性）
    # 虚警点数：上限5%（实测EMI虚警率<5%），与EM强度非线性相关
    max_extra_ratio = 0.05  # 最大虚警比例
    extra_ratio = min(0.01 * (em_intensity / 100), max_extra_ratio)
    num_extra_noise = int(N * extra_ratio)

    final_pc = np.column_stack([x_em, y_em, z_em, I_em])
    if num_extra_noise > 0:
        # 虚警点空间约束：LiDAR有效视场（x:0~80m, y:-40~40m, z:-5~10m）
        # （KITTI场景LiDAR有效探测范围，符合物理）
        extra_x = np.random.uniform(0, 80, num_extra_noise)
        extra_y = np.random.uniform(-40, 40, num_extra_noise)
        extra_z = np.random.uniform(-5, 10, num_extra_noise)
        # 虚警点强度：随机分布（0~255），但以低强度为主（符合EMI虚警特性）
        extra_intensity = np.random.weibull(2, num_extra_noise) * 100  # 威布尔分布（低强度为主）
        extra_intensity = np.clip(extra_intensity, 0, 255)

        extra_noise = np.column_stack([extra_x, extra_y, extra_z, extra_intensity])
        final_pc = np.vstack([final_pc, extra_noise])

    # ========== 关键集成：保存前过滤所有异常点 ==========
    final_pc = fix_invalid_pc(final_pc)

    # 9. 保存KITTI .bin格式（保持数据类型正确）
    final_pc = final_pc.astype(np.float32)
    final_pc.tofile(save_path)
    print(f"电磁干扰场景点云已保存：{save_path}")
    print(f"原始点数：{N}，干扰后总点数：{len(final_pc)}（虚警噪声点：{num_extra_noise}）")
    # 新增：输出过滤后的统计信息，便于验证
    filtered_num = N + num_extra_noise - len(final_pc)
    print(f"过滤异常点数：{filtered_num}（NaN/Inf/极端坐标/无效距离）")

# import math
#
# import numpy as np
#
# def em_interference_effect(kitti_bin_path, save_path, em_intensity=100):
#     """
#     高压电线电磁干扰算子：模拟周期性条纹与随机噪声
#     :param kitti_bin_path: KITTI原始点云路径
#     :param save_path: 生成点云保存路径
#     :param em_intensity: 电磁强度（V/m），默认100V/m（高压电线场景）
#     """
#     # 1. 读取KITTI点云并转换为球坐标
#     pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
#     x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
#     N = len(pc)
#
#     d = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
#     theta0 = np.arctan2(y0, x0)
#     phi = np.arcsin(z0 / (d + 1e-6))
#
#     # 2. 周期性水平角偏移（条纹）
#     delta = 0.001 * math.sqrt(em_intensity)
#     f = 50  # 工频50Hz
#     t = (np.arange(N) - 1) * 0.1  # 时间戳
#     theta_em = theta0 + delta * np.sin(2 * math.pi * f * t)
#
#     # 3. 转换回笛卡尔坐标并添加随机噪声
#     x_em = d * np.cos(theta_em) * np.cos(phi)
#     y_em = d * np.sin(theta_em) * np.cos(phi)
#     z_em = d * np.sin(phi)
#
#     sigma_em = math.sqrt(0.0005 * em_intensity)
#     noise = np.random.normal(0, sigma_em, size=(N, 3))
#     x_em += noise[:, 0]
#     y_em += noise[:, 1]
#     z_em += noise[:, 2]
#
#     # 4. 添加额外噪声点
#     num_extra_noise = int(N * 0.02 * em_intensity / 100)
#     if num_extra_noise > 0:
#         # 噪声点坐标在原始点云范围内
#         min_x, max_x = np.min(x0), np.max(x0)
#         min_y, max_y = np.min(y0), np.max(y0)
#         min_z, max_z = np.min(z0), np.max(z0)
#
#         extra_x = np.random.uniform(min_x, max_x, num_extra_noise)
#         extra_y = np.random.uniform(min_y, max_y, num_extra_noise)
#         extra_z = np.random.uniform(min_z, max_z, num_extra_noise)
#         extra_intensity = np.random.uniform(0.1, 0.3, num_extra_noise)  # 低强度噪声
#
#         extra_noise = np.column_stack([extra_x, extra_y, extra_z, extra_intensity])
#         pc = np.vstack([np.column_stack([x_em, y_em, z_em, I0]), extra_noise])
#     else:
#         pc = np.column_stack([x_em, y_em, z_em, I0])
#
#     # 5. 保存KITTI .bin
#     pc.astype(np.float32).tofile(save_path)
#     print(f"电磁干扰场景点云已保存：{save_path}，总点数：{len(pc)}")