import numpy as np

def high_temp_effect(kitti_bin_path, save_path, temp=50, random_seed=42):
    """
    夏天正午高温算子：模拟高温导致远距离目标缺失、密度下降（符合LiDAR物理特性）
    物理依据：
    1. 大气消光衰减：比尔-朗伯定律（I = I0 * exp(-α*d)，α为消光系数，随温度/高度指数增长）
    2. 探测概率：随距离高斯衰减，高温下热噪声降低信噪比，进一步压低探测概率
    3. 点云密度：由探测概率自然筛选，无需二次独立剔除
    :param kitti_bin_path: KITTI原始点云路径
    :param save_path: 生成点云保存路径
    :param temp: 环境温度（℃），默认50℃（极端高温）
    :param random_seed: 随机种子（保证结果可复现）
    """
    # 设置随机种子（可复现性）
    np.random.seed(random_seed)

    # 1. 读取KITTI点云（x:前向，y:左向，z:高度，I:回波强度）
    pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
    if len(pc) == 0:
        raise ValueError("输入点云文件为空！")
    x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    original_num = len(pc)

    # 2. 计算目标距离d（避免除0，加极小值）
    d = np.sqrt(x0 ** 2 + y0 ** 2 + (z0 + 1e-8) ** 2)

    # 3. 回波强度衰减（比尔-朗伯定律，贴合大气物理）
    # 3.1 基础消光系数（25℃时的基准值，单位：m^-1）
    alpha_base = 0.0012  # 分子散射+气溶胶散射的基准消光系数
    # 3.2 温度修正：高温下大气消光系数指数增长（经验公式，贴合实测数据）
    temp_delta = max(0, temp - 25)  # 仅考虑25℃以上的高温影响
    alpha_temp = alpha_base * np.exp(0.02 * temp_delta)  # 指数型温度修正
    # 3.3 高度修正：高空大气稀薄，消光系数略降低（z轴正方向为上）
    alpha_height = alpha_temp * np.exp(-0.0001 * z0)  # 高度每上升100m，消光系数降1%
    # 3.4 回波强度衰减（裁剪到0-255，符合KITTI强度范围）
    I_highT = I0 * np.exp(-alpha_height * d)
    pc[:, 3] = np.clip(I_highT, 0, 255)

    # 4. 远距离点云探测概率（连续非线性模型，无维度错误）
    # 4.1 基准探测概率（25℃时，随距离高斯衰减）
    detect_prob_base = np.exp(-(d / 80) ** 2)  # 80m处探测概率~e^-1≈0.37，100m≈0.15（非0）
    # 4.2 高温修正：热噪声降低探测概率（温度越高，衰减越明显）
    temp_factor = 1 - 0.006 * temp_delta  # 50℃时，探测概率额外降低15%（0.006*25）
    temp_factor = np.clip(temp_factor, 0.5, 1.0)  # 避免极端温度导致概率为负
    detect_prob = detect_prob_base * temp_factor
    # 4.3 边界校验：近距离（d≤10m）探测概率≈1（不受高温影响）
    detect_prob[d <= 10] = 1.0

    # 5. 按探测概率筛选点云（密度衰减自然体现，无需二次剔除）
    r = np.random.uniform(0, 1, size=len(pc))
    pc = pc[r <= detect_prob]

    # 6. 保存KITTI .bin（保证float32格式）
    pc.astype(np.float32).tofile(save_path)
    print(f"高温场景点云已保存：{save_path}")
    print(f"原始点数：{original_num}，最终点数：{len(pc)}，密度衰减比：{len(pc)/original_num:.3f}")
# import numpy as np
#
# def high_temp_effect(kitti_bin_path, save_path, temp=50):
#     """
#     夏天正午高温算子：模拟高温导致远距离目标缺失、密度下降
#     :param kitti_bin_path: KITTI原始点云路径
#     :param save_path: 生成点云保存路径
#     :param temp: 环境温度（℃），默认50℃（极端高温）
#     """
#     # 1. 读取KITTI点云
#     pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
#     x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
#     original_num = len(pc)
#
#     # 2. 计算目标距离d
#     d = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
#
#     # 3. 回波强度衰减
#     beta = 0.0015 + 0.0001 * max(0, temp - 25)
#     I_highT = I0 * np.exp(-beta * d)
#     pc[:, 3] = np.clip(I_highT, 0, 255)
#
#     # 4. 远距离点云过滤（基于探测概率）
#     detect_prob = np.zeros_like(d)
#     detect_prob[d <= 50] = 0.95 - 0.01 * max(0, temp - 25)
#     detect_prob[(d > 50) & (d <= 100)] = 0.95 - 0.01 * max(0, temp - 25) - 0.005 * (d[(d > 50) & (d <= 100)] - 50)
#     detect_prob[d > 100] = 0
#
#     r = np.random.uniform(0, 1, size=len(pc))
#     pc = pc[r <= detect_prob]
#
#     # 5. 点云密度修正（额外剔除部分点）
#     target_density_ratio = 1 - 0.008 * max(0, temp - 25)
#     target_num = int(original_num * target_density_ratio)
#     if len(pc) > target_num:
#         pc = pc[np.random.choice(len(pc), target_num, replace=False)]
#
#     # 6. 保存KITTI .bin
#     pc.astype(np.float32).tofile(save_path)
#     print(f"高温场景点云已保存：{save_path}，最终点数：{len(pc)}（原始{original_num}）")