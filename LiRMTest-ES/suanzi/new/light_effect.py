import numpy as np


def light_effect(kitti_bin_path, save_path, light_type="strong", random_seed=42):
    """
    强光/逆光算子：模拟光照对LiDAR点云的物理影响（贴合实测规律）
    物理依据：
    1. 强光：环境光干扰与“距离成正相关”（远距离噪声大、近距离噪声小），噪声服从高斯分布；
    2. 逆光：衰减程度与“激光方向和光源方向的夹角”强相关，远距离点因SNR过低易丢失；
    3. 通用：LiDAR强度噪声幅度与回波信号强度成反比（回波越弱，噪声占比越高）。
    :param kitti_bin_path: KITTI原始点云路径（.bin）
    :param save_path: 生成点云保存路径（.bin）
    :param light_type: 光照类型（"strong"=强光，"backlight"=逆光）
    :param random_seed: 随机种子（保证噪声/探测概率结果可复现）
    """
    # ========== 1. 鲁棒性初始化 ==========
    np.random.seed(random_seed)  # 噪声/探测概率可复现
    # 光照类型参数校验
    if light_type not in ["strong", "backlight"]:
        raise ValueError("light_type必须为'strong'（强光）或'backlight'（逆光）！")
    # 读取点云并校验
    pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
    if len(pc) == 0:
        raise ValueError("输入点云文件为空，请检查路径！")
    x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    original_num = len(pc)

    # ========== 2. 基础物理量计算（防呆：加极小值避免除零） ==========
    # 距离（m）：加1e-8避免d=0时除零
    d = np.sqrt(x0 ** 2 + y0 ** 2 + (z0 + 1e-8) ** 2)
    # 激光方向单位向量（指向目标）
    dir_x, dir_y, dir_z = x0 / d, y0 / d, z0 / d
    # 光源（太阳）方向：假设逆光为“激光朝向+x方向（车头）”时正对光源（可按需调整）
    light_dir = np.array([1.0, 0.0, 0.0])  # 光源在+x方向（车头前方）

    # ========== 3. 光照效应核心逻辑 ==========
    if light_type == "strong":
        # ---------- 强光：环境光干扰（距离相关+高斯噪声） ----------
        # 噪声幅度：近距离（d≤10m）噪声小（回波强），远距离（d>10m）噪声随距离增长
        noise_scale = np.where(d <= 10, 3, 3 + 0.8 * np.log1p(d - 10))  # 对数增长避免无上限
        noise_scale = np.clip(noise_scale, 3, 15)  # 限制噪声范围3~15（与原代码±15兼容）
        # 高斯噪声（贴合自然噪声分布，原代码均匀分布不符合物理）
        delta_I = np.random.normal(0, noise_scale, size=len(pc))
        I_light = I0 + delta_I

    elif light_type == "backlight":
        # ---------- 逆光：角度依赖衰减 + 远距离点探测概率下降 ----------
        # 1. 逆光角度计算：激光方向与光源方向的夹角余弦值（越接近1，越正对光源，逆光越强）
        cos_theta = dir_x * light_dir[0] + dir_y * light_dir[1] + dir_z * light_dir[2]
        cos_theta = np.maximum(0, cos_theta)  # 仅保留“朝向光源”的方向（cosθ≥0）
        # 2. 强度衰减因子：逆光越强（cos_theta越大）、距离越远，衰减越严重
        decay_factor = np.exp(-0.8 * cos_theta * np.log1p(d / 10))  # 物理衰减规律：指数衰减
        decay_factor = np.clip(decay_factor, 0.2, 1.0)  # 最小保留20%强度，避免完全归零
        # 3. 逆光噪声：随衰减程度增强（衰减越严重，SNR越低，噪声越大）
        noise_scale = 5 * (1 - decay_factor) + 2  # 衰减越狠，噪声越大（2~7范围）
        delta_I = np.random.normal(0, noise_scale, size=len(pc))
        I_light = I0 * decay_factor + delta_I
        # 4. 逆光探测概率衰减（核心物理缺失补充）：远距离+强逆光点易丢失
        detect_prob = np.exp(-1.2 * cos_theta * (d / 50) ** 2)  # 50m处强逆光点探测概率≈0.1
        detect_prob = np.clip(detect_prob, 0.1, 1.0)  # 避免概率过低
        # 按探测概率筛选点云（体现逆光导致的点缺失）
        r = np.random.uniform(0, 1, size=len(pc))
        mask = r <= detect_prob
        pc = pc[mask]
        I_light = I_light[mask]  # 同步更新强度

    # ========== 4. 强度截断 + 保存点云 ==========
    pc[:, 3] = np.clip(I_light, 0, 255)  # 符合LiDAR强度0-255的物理范围
    pc.astype(np.float32).tofile(save_path)

    # ========== 5. 日志输出（量化分析） ==========
    final_num = len(pc)
    density_ratio = final_num / original_num
    print(f"{light_type}场景点云已保存：{save_path}")
    print(f"原始点数：{original_num}，最终点数：{final_num}，密度保留比：{density_ratio:.3f}")

# 示例调用（可直接运行）
# light_effect("input/000000.bin", "output/strong_light.bin", light_type="strong", random_seed=42)
# light_effect("input/000000.bin", "output/backlight.bin", light_type="backlight", random_seed=42)


# import numpy as np
#
# def light_effect(kitti_bin_path, save_path, light_type="strong"):
#     """
#     强光/逆光算子：区分两种光照场景
#     :param kitti_bin_path: KITTI原始点云路径
#     :param save_path: 生成点云保存路径
#     :param light_type: 光照类型（"strong"=强光，"backlight"=逆光）
#     """
#     # 1. 读取KITTI点云
#     pc = np.fromfile(kitti_bin_path, dtype=np.float32).reshape(-1, 4)
#     x0, y0, z0, I0 = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
#     d = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
#
#     if light_type == "strong":
#         # 强光：所有点添加±15强度噪声
#         delta_I = np.random.uniform(-15, 15, size=len(pc))
#         I_light = I0 + delta_I
#     elif light_type == "backlight":
#         # 逆光：仅x>0区域（朝前）衰减，cosα = max(0, -x0/d)
#         cos_alpha = np.maximum(0, -x0 / (d + 1e-6))
#         decay_factor = 0.4 + 0.6 * cos_alpha
#         delta_I = np.random.uniform(-20, 20, size=len(pc))
#         I_light = I0 * decay_factor + delta_I
#     else:
#         raise ValueError("light_type must be 'strong' or 'backlight'")
#
#     # 2. 强度值截断并保存
#     pc[:, 3] = np.clip(I_light, 0, 255)
#     pc.astype(np.float32).tofile(save_path)
#     print(f"{light_type}场景点云已保存：{save_path}，点数：{len(pc)}")