import numpy as np
import math
import os
from scipy.spatial import KDTree
import random


# 点云数据类（兼容KITTI格式）
class PointCloud:
    def __init__(self, points=None):
        self.points = points if points is not None else np.empty((0, 4), dtype=np.float32)  # x,y,z,intensity

    def load_from_bin(self, file_path):
        """从KITTI bin文件加载点云"""
        self.points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return self

    def save_to_bin(self, file_path):
        """保存点云到bin文件"""
        self.points.astype(np.float32).tofile(file_path)

    def get_min_max_3d(self):
        """获取x,y,z坐标的最大最小值"""
        return np.min(self.points[:, :3], axis=0), np.max(self.points[:, :3], axis=0)

    def get_center_3d(self):
        """获取点云中心坐标（用于天气变换的空间约束）"""
        min_p, max_p = self.get_min_max_3d()
        return (min_p + max_p) / 2


class LiDARTransforms:
    """实现符合物理特性的仿射变换和天气变换算子"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)  # 统一numpy随机种子（关键：保证可复现）
        self.PI = math.pi
        # 传感器物理参数（LiDAR雨/雾/雪衰减系数参考行业标准）
        self.RAIN_ATTENUATION_COEFF = 0.015  # 雨衰系数
        self.FOG_ATTENUATION_COEFF = 0.04    # 雾衰系数
        self.SNOW_REFLECTIVITY = 1.5         # 雪花反射率倍率（高于背景）

    # ------------------------------ 仿射变换（符合刚体物理特性） ------------------------------
    def symmetry_transform(self, pc):
        """对称变换：沿y轴翻转（符合KITTI右手系物理规则）"""
        new_points = pc.points.copy()
        new_points[:, 1] = -new_points[:, 1]  # y轴取反
        return PointCloud(new_points)

    def translation_transform(self, pc, tx=0.0, ty=0.0, tz=0.0):
        """平移变换：3D刚体平移（约束平移范围，避免超出物理场景）"""
        # 物理约束：平移量不超过点云最大尺寸的10%
        min_p, max_p = pc.get_min_max_3d()
        max_trans = (max_p - min_p).max() * 0.1
        tx = np.clip(tx, -max_trans, max_trans)
        ty = np.clip(ty, -max_trans, max_trans)
        tz = np.clip(tz, -max_trans, max_trans)

        new_points = pc.points.copy()
        new_points[:, 0] += tx
        new_points[:, 1] += ty
        new_points[:, 2] += tz
        return PointCloud(new_points)

    def rotation_transform(self, pc, ry=0.0):
        """旋转变换：绕z轴旋转（yaw角，保证旋转矩阵正交性）"""
        # 物理约束：旋转角范围±π
        ry = np.clip(ry, -self.PI, self.PI)
        cos_ry = math.cos(ry)
        sin_ry = math.sin(ry)

        # 验证旋转矩阵正交性（刚体变换核心）
        R = np.array([[cos_ry, -sin_ry, 0],
                      [sin_ry, cos_ry, 0],
                      [0, 0, 1]], dtype=np.float32)
        assert np.allclose(R.T @ R, np.eye(3)), "旋转矩阵非正交（违反刚体物理特性）"
        assert np.isclose(np.linalg.det(R), 1), "旋转矩阵行列式≠1（违反刚体物理特性）"

        new_points = pc.points.copy()
        x, y = new_points[:, 0], new_points[:, 1]
        new_points[:, 0] = x * cos_ry - y * sin_ry
        new_points[:, 1] = x * sin_ry + y * cos_ry
        return PointCloud(new_points)

    def scale_transform(self, pc, scale=1.0):
        """缩放变换：仅支持均匀缩放（符合刚体物理特性，非均匀缩放无物理意义）"""
        # 物理约束：缩放范围0.8~1.2（避免过度缩放）
        scale = np.clip(scale, 0.8, 1.2)
        new_points = pc.points.copy()
        new_points[:, :3] *= scale  # 均匀缩放x/y/z
        return PointCloud(new_points)

    # ------------------------------ 天气变换（符合LiDAR物理特性） ------------------------------
    def rain_effect(self, pc):
        """雨效变换：符合物理特性（远距离点丢失+雨滴噪声+强度衰减）"""
        new_points = pc.points.copy()
        distances = np.linalg.norm(new_points[:, :3], axis=1)

        # 1. 距离相关的点丢失（远距离更容易被雨遮挡）
        rain_quantity = random.choice([2, 4, 6, 8, 10])
        drop_prob = 1 - np.exp(-self.RAIN_ATTENUATION_COEFF * rain_quantity * (distances / 100))  # 距离归一化
        keep_mask = np.random.rand(len(new_points)) > drop_prob
        new_points = new_points[keep_mask].copy()

        # 2. 添加雨滴噪声点（近距离高反射）
        min_p, max_p = pc.get_min_max_3d()
        rain_count = int(len(new_points) * 0.05 * (rain_quantity / 10))  # 雨越大，噪声点越多
        if rain_count > 0:
            # 雨滴分布：0~50m范围内，z轴0~5m（贴近地面）
            rain_x = np.random.uniform(min_p[0], min(max_p[0], 50), rain_count)
            rain_y = np.random.uniform(min_p[1], min(max_p[1], 50), rain_count)
            rain_z = np.random.uniform(0, 5, rain_count)
            # 雨滴强度：高反射（基础强度+随机波动）
            rain_intensity = np.random.uniform(1.0, 2.0, rain_count)
            rain_points = np.column_stack((rain_x, rain_y, rain_z, rain_intensity))
            new_points = np.vstack((new_points, rain_points))

        # 3. 距离相关的强度衰减
        intensity_factor = np.exp(-self.RAIN_ATTENUATION_COEFF * rain_quantity * (distances[keep_mask] / 100))
        new_points[:len(intensity_factor), 3] *= intensity_factor

        return PointCloud(new_points)

    def snow_effect(self, pc):
        """雪效变换：符合物理特性（雪花噪声+强度衰减+近地面分布）"""
        new_points = pc.points.copy()
        distances = np.linalg.norm(new_points[:, :3], axis=1)
        snow_quantity = random.choice([2, 4, 6, 8, 10])

        # 1. 强度衰减（雪的散射导致）
        intensity_factor = np.exp(-0.01 * snow_quantity * (distances / 100))
        new_points[:, 3] *= intensity_factor

        # 2. 添加雪花噪声点（近地面+高反射）
        min_p, max_p = pc.get_min_max_3d()
        snow_count = int(len(new_points) * 0.08 * (snow_quantity / 10))
        if snow_count > 0:
            # 雪花分布：z轴0~3m（近地面），x/y在点云范围内
            snow_x = np.random.uniform(min_p[0], max_p[0], snow_count)
            snow_y = np.random.uniform(min_p[1], max_p[1], snow_count)
            snow_z = np.random.uniform(0, 3, snow_count)
            # 雪花强度：高于背景（乘反射率倍率）
            snow_points_3d = np.column_stack((snow_x, snow_y, snow_z))
            kdtree = KDTree(pc.points[:, :3])
            _, indices = kdtree.query(snow_points_3d, k=3)
            base_intensity = np.mean(pc.points[indices, 3], axis=1)
            snow_intensity = base_intensity * self.SNOW_REFLECTIVITY
            # 过滤无效强度
            snow_intensity = np.clip(snow_intensity, 0, 255)
            snow_points = np.column_stack((snow_x, snow_y, snow_z, snow_intensity))
            new_points = np.vstack((new_points, snow_points))

        return PointCloud(new_points)

    def fog_effect(self, pc):
        """雾效变换：符合物理特性（全局强度衰减+信噪比降低，无固定形状噪声）"""
        new_points = pc.points.copy()
        distances = np.linalg.norm(new_points[:, :3], axis=1)
        fog_quantity = random.choice([2, 4, 6, 8, 10])

        # 1. 距离相关的强衰减（雾的核心物理效应）
        fog_coeff = self.FOG_ATTENUATION_COEFF * fog_quantity
        intensity_factor = np.exp(-fog_coeff * (distances / 100))
        new_points[:, 3] *= intensity_factor

        # 2. 添加全局高斯噪声（模拟雾的散射噪声）
        noise = np.random.normal(0, 0.02 * fog_quantity, new_points.shape)
        new_points[:, :3] += noise[:, :3]  # 坐标微扰
        new_points[:, 3] += noise[:, 3]    # 强度噪声
        # 约束强度范围
        new_points[:, 3] = np.clip(new_points[:, 3], 0, 255)

        return PointCloud(new_points)


# 标签和校准文件处理（符合KITTI格式物理规则）
class LiDARLabelProcessor:
    @staticmethod
    def update_label(labin_path, labout_path, transform_type, **kwargs):
        """更新标签文件（仅仿射变换需要更新，天气变换无需更新）"""
        if not os.path.exists(labin_path):
            return
        with open(labin_path, 'r') as f_in, open(labout_path, 'w') as f_out:
            lines = f_in.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue
                parts = line.split()
                label_type = parts[0]
                if label_type == "DontCare":
                    f_out.write(line + '\n')
                    continue

                # 解析KITTI标签：type, truncation, occlusion, alpha, bbox(4), dimensions(3), location(3), rotation_y, score
                # 索引：2:truncation, 3:occlusion, 4:alpha, 5-8:bbox, 9:height,10:width,11:length, 12:x,13:y,14:z, 15:rotation_y
                lab = [0.0] * 16
                for i in range(2, min(16, len(parts)+1)):
                    lab[i] = float(parts[i-1]) if (i-1) < len(parts) else 0.0

                # 根据变换类型更新（仅处理仿射变换，天气变换跳过）
                if transform_type == "symmetry":
                    # 对称变换：y轴取反 + 旋转角取反（符合KITTI坐标系物理规则）
                    lab[13] = -lab[13]  # 物体中心y坐标取反
                    lab[15] = (-lab[15]) % (2 * math.pi)  # 旋转角取反（修正原错误）
                elif transform_type == "scale":
                    # 缩放变换：均匀缩放尺寸（符合刚体特性）
                    scale = kwargs.get('scale', 1.0)
                    lab[9] *= scale   # 高度（z）
                    lab[10] *= scale  # 宽度（y）
                    lab[11] *= scale  # 长度（x）
                elif transform_type == "rotation":
                    # 旋转变换：旋转角叠加（绕z轴）
                    ry = kwargs.get('ry', 0.0)
                    lab[15] = (lab[15] + ry) % (2 * math.pi)
                    # 旋转物体中心坐标（x/y）
                    cos_ry = math.cos(ry)
                    sin_ry = math.sin(ry)
                    x, y = lab[12], lab[13]
                    lab[12] = x * cos_ry - y * sin_ry
                    lab[13] = x * sin_ry + y * cos_ry
                elif transform_type == "translation":
                    # 平移变换：更新物体中心坐标
                    tx = kwargs.get('tx', 0.0)
                    ty = kwargs.get('ty', 0.0)
                    tz = kwargs.get('tz', 0.0)
                    lab[12] += tx  # x
                    lab[13] += ty  # y
                    lab[14] += tz  # z

                # 写回标签（保留原始格式）
                output_parts = [label_type] + parts[1:2]  # 保留truncation/occlusion等原始字段
                output_parts += [f"{lab[i]:.6f}" for i in range(2, 16)]
                # 补充score字段（如果有）
                if len(parts) > 15:
                    output_parts += parts[15:]
                f_out.write(' '.join(output_parts) + '\n')

    @staticmethod
    def update_calib(calib_in, calib_out, transform_type, **kwargs):
        """更新校准文件（仅仿射变换需要更新，天气变换无需更新）"""
        if not os.path.exists(calib_in):
            return
        with open(calib_in, 'r') as f_in, open(calib_out, 'w') as f_out:
            lines = f_in.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue
                parts = line.split()
                calib_type = parts[0]
                if calib_type != "Tr_velo_to_cam:":
                    f_out.write(line + '\n')
                    continue

                # 解析Tr_velo_to_cam矩阵（激光雷达→相机）
                tr_matrix = np.eye(4, dtype=np.float32)
                for i in range(3):
                    for j in range(4):
                        idx = 1 + i * 4 + j
                        tr_matrix[i, j] = float(parts[idx]) if idx < len(parts) else 0.0

                # 构建变换矩阵（逆变换，因为点云已变换，校准矩阵需反向补偿）
                change_matrix = np.eye(4, dtype=np.float32)
                if transform_type == "symmetry":
                    change_matrix[1, 1] = -1.0  # y轴取反
                elif transform_type == "translation":
                    # 平移逆变换（包含z轴，修正原错误）
                    tx = kwargs.get('tx', 0.0)
                    ty = kwargs.get('ty', 0.0)
                    tz = kwargs.get('tz', 0.0)
                    change_matrix[0, 3] = -tx
                    change_matrix[1, 3] = -ty
                    change_matrix[2, 3] = -tz
                elif transform_type == "scale":
                    # 缩放逆变换（仅均匀缩放）
                    scale = kwargs.get('scale', 1.0)
                    change_matrix[0, 0] = 1.0 / scale
                    change_matrix[1, 1] = 1.0 / scale
                    change_matrix[2, 2] = 1.0 / scale
                elif transform_type == "rotation":
                    # 旋转变换逆矩阵（绕z轴反向旋转）
                    ry = kwargs.get('ry', 0.0)
                    cos_ry = math.cos(-ry)
                    sin_ry = math.sin(-ry)
                    change_matrix[0, 0] = cos_ry
                    change_matrix[0, 1] = sin_ry
                    change_matrix[1, 0] = -sin_ry
                    change_matrix[1, 1] = cos_ry
                    # 验证正交性
                    R = change_matrix[:3, :3]
                    assert np.allclose(R.T @ R, np.eye(3)), "校准旋转矩阵非正交"

                # 矩阵乘法更新（激光雷达→相机 变换 = 原变换 × 补偿变换）
                result_matrix = tr_matrix @ change_matrix

                # 写回校准矩阵（保留原始格式）
                output_parts = [calib_type]
                for i in range(3):
                    for j in range(4):
                        output_parts.append(f"{result_matrix[i, j]:.8f}")
                f_out.write(' '.join(output_parts) + '\n')


def batch_process(input_config, output_root, mode, count):
    """
    批量处理点云文件及相关标签、校准文件（修正参数错误，符合物理逻辑）

    :param input_config: 输入配置字典，包含:
        - bin_dir: 点云bin文件目录
        - label_dir: 标签txt文件目录
        - calib_dir: 校准文件目录
    :param output_root: 输出根目录（自动创建bin、txt、calib子目录）
    :param mode: 选取模式，'sequential'（顺序）或 'random'（随机）
    :param count: 处理文件数量，None表示全部
    """
    # 初始化输出目录（修正原参数覆盖问题）
    output_dirs = {
        'bin': os.path.join(output_root, 'bin'),
        'txt': os.path.join(output_root, 'txt'),
        'calib': os.path.join(output_root, 'calib')
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 初始化处理器
    transforms = LiDARTransforms(seed=42)
    label_processor = LiDARLabelProcessor()

    # 获取所有bin文件并筛选
    bin_files = [f for f in os.listdir(input_config['bin_dir']) if f.endswith('.bin')]
    if not bin_files:
        print("未找到任何.bin文件")
        return

    # 按模式筛选文件
    if mode == 'sequential':
        bin_files.sort()
        selected_files = bin_files[:count] if count else bin_files
    elif mode == 'random':
        selected_count = count if count else len(bin_files)
        selected_files = random.sample(bin_files, min(selected_count, len(bin_files)))
    else:
        print("模式错误，仅支持 'sequential' 或 'random'")
        return

    # 定义变换列表（修正天气变换的标签/校准更新逻辑，符合物理特性）
    transform_list = [
        # (变换名, 变换函数, 额外参数, 是否更新标签, 是否更新校准)
        ('symmetry', transforms.symmetry_transform, {}, True, True),
        ('rain', transforms.rain_effect, {}, False, False),  # 天气变换不更新标签/校准
        ('snow', transforms.snow_effect, {}, False, False),
        ('fog', transforms.fog_effect, {}, False, False),
        ('rotation_90', transforms.rotation_transform, {'ry': math.radians(90)}, True, True),
        ('translation_0.5', transforms.translation_transform, {'tx': 0.5, 'ty': 0.5, 'tz': 0.0}, True, True),
        ('scale_1.2', transforms.scale_transform, {'scale': 1.2}, True, True)  # 仅均匀缩放
    ]

    # 批量处理每个文件
    for bin_file in selected_files:
        base_name = os.path.splitext(bin_file)[0]
        print(f"处理文件: {base_name}")

        # 输入文件路径
        input_paths = {
            'bin': os.path.join(input_config['bin_dir'], bin_file),
            'txt': os.path.join(input_config['label_dir'], f"{base_name}.txt"),
            'calib': os.path.join(input_config['calib_dir'], f"{base_name}.txt")
        }

        # 加载点云
        try:
            pc = PointCloud().load_from_bin(input_paths['bin'])
        except Exception as e:
            print(f"加载点云失败 {bin_file}: {e}")
            continue

        # 应用所有变换
        for transform_name, transform_func, params, update_label, update_calib in transform_list:
            # 执行变换
            try:
                transformed_pc = transform_func(pc, **params)
            except Exception as e:
                print(f"变换失败 {transform_name}_{base_name}: {e}")
                continue

            # 保存变换后的点云
            output_bin = os.path.join(output_dirs['bin'], f"{transform_name}_{base_name}.bin")
            transformed_pc.save_to_bin(output_bin)
            print(f"点云已保存至: {output_bin}")

            # 更新标签（仅仿射变换需要）
            if update_label and os.path.exists(input_paths['txt']):
                output_label = os.path.join(output_dirs['txt'], f"{transform_name}_{base_name}.txt")
                base_transform_type = transform_name.split('_')[0]  # 提取基础变换类型
                label_processor.update_label(
                    input_paths['txt'], output_label, base_transform_type, **params
                )

            # 更新校准文件（仅仿射变换需要）
            if update_calib and os.path.exists(input_paths['calib']):
                output_calib = os.path.join(output_dirs['calib'], f"{transform_name}_{base_name}.txt")
                base_transform_type = transform_name.split('_')[0]
                label_processor.update_calib(
                    input_paths['calib'], output_calib, base_transform_type, **params
                )

    print("批量处理完成!")


# 使用示例
if __name__ == "__main__":
    # 配置输入输出路径
    input_config = {
        'bin_dir': "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/velodyne",
        'label_dir': "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/txt",
        'calib_dir': "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/calib"
    }
    output_root = "E:/数据/数据集A/旧算子2.0"

    # 处理配置
    process_mode = "sequential"  # 顺序处理：'sequential'，随机处理：'random'
    process_count = None  # 处理数量：None表示全部，数字表示指定数量

    # 执行批量处理（修正参数传递错误）
    batch_process(input_config, output_root, process_mode, process_count)
# import numpy as np
# import math
# import os
# from scipy.spatial import KDTree
# import random
#
#
# # 点云数据类（兼容KITTI格式）
# class PointCloud:
#     def __init__(self, points=None):
#         self.points = points if points is not None else np.empty((0, 4), dtype=np.float32)  # x,y,z,intensity
#
#     def load_from_bin(self, file_path):
#         """从KITTI bin文件加载点云"""
#         self.points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
#         return self
#
#     def save_to_bin(self, file_path):
#         """保存点云到bin文件"""
#         self.points.astype(np.float32).tofile(file_path)
#
#     def get_min_max_3d(self):
#         """获取x,y,z坐标的最大最小值"""
#         return np.min(self.points[:, :3], axis=0), np.max(self.points[:, :3], axis=0)
#
#
# class LiDARTransforms:
#     """实现定义的仿射变换和天气变换算子"""
#
#     def __init__(self, seed=42):
#         random.seed(seed)
#         self.PI = math.pi
#
#     # ------------------------------ 仿射变换 ------------------------------
#     def symmetry_transform(self, pc):
#         """对称变换：沿y轴翻转（原代码one_pcd_augment.h实现）"""
#         new_points = pc.points.copy()
#         new_points[:, 1] = -new_points[:, 1]  # y坐标取反
#         return PointCloud(new_points)
#
#     def translation_transform(self, pc, tx=0.0, ty=0.0, tz=0.0):
#         """平移变换：沿x,y,z轴平移"""
#         new_points = pc.points.copy()
#         new_points[:, 0] += tx
#         new_points[:, 1] += ty
#         new_points[:, 2] += tz
#         return PointCloud(new_points)
#
#     def rotation_transform(self, pc, ry=0.0):
#         """旋转变换：绕z轴旋转（yaw角，单位弧度）"""
#         new_points = pc.points.copy()
#         cos_ry = math.cos(ry)
#         sin_ry = math.sin(ry)
#
#         # 旋转x,y坐标
#         x = new_points[:, 0]
#         y = new_points[:, 1]
#         new_points[:, 0] = x * cos_ry - y * sin_ry
#         new_points[:, 1] = x * sin_ry + y * cos_ry
#         return PointCloud(new_points)
#
#     def scale_transform(self, pc, sx=1.0, sy=1.0, sz=1.0):
#         """缩放变换：沿x,y,z轴缩放"""
#         new_points = pc.points.copy()
#         new_points[:, 0] *= sx
#         new_points[:, 1] *= sy
#         new_points[:, 2] *= sz
#         return PointCloud(new_points)
#
#     # ------------------------------ 天气变换 ------------------------------
#     def rain_effect(self, pc):
#         """雨效变换（原代码one_pcd_augment.h的rain实现）"""
#         # 随机选择参数（原代码quantity和changerate数组）
#         quantity = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#         changerate = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
#         ran = random.randint(0, 9)
#
#         # 随机丢弃点
#         keep_ratio = changerate[ran]
#         keep_mask = np.random.rand(pc.points.shape[0]) < keep_ratio
#         new_points = pc.points[keep_mask].copy()
#
#         # 距离相关的强度衰减
#         resparameter = pow(quantity[ran], 0.6)
#         distances = np.linalg.norm(new_points[:, :3], axis=1)
#         intensity_factor = np.exp(-0.02 * resparameter * distances)
#         new_points[:, 3] *= intensity_factor
#
#         return PointCloud(new_points)
#
#     def snow_effect(self, pc):
#         """雪效变换（原代码one_pcd_augment.h的snow实现）"""
#         # 随机选择参数
#         quantity = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#         changerate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#         ran = random.randint(0, 9)
#
#         # 计算需要添加的雪花点数
#         num_snow = int(pc.points.shape[0] * changerate[ran])
#         if num_snow == 0:
#             return PointCloud(pc.points.copy())
#
#         # 获取点云范围
#         min_p, max_p = pc.get_min_max_3d()
#
#         # 生成正态分布的雪花点坐标
#         rng = np.random.default_rng()
#         snow_x = rng.normal(0, pow(max(np.abs(min_p[0]), np.abs(max_p[0])) / 3, 2) / 8, num_snow)
#         snow_y = rng.normal(0, pow(max(np.abs(min_p[1]), np.abs(max_p[1])) / 3, 2) / 8, num_snow)
#         snow_z = rng.normal(0, pow(max(np.abs(min_p[2]), np.abs(max_p[2])) / 3, 2) / 8, num_snow)
#
#         # 过滤超出点云范围的雪花点
#         valid_mask = (snow_x >= min_p[0]) & (snow_x <= max_p[0]) & \
#                      (snow_y >= min_p[1]) & (snow_y <= max_p[1]) & \
#                      (snow_z >= 0) & (snow_z <= max_p[2])
#         snow_x = snow_x[valid_mask]
#         snow_y = snow_y[valid_mask]
#         snow_z = snow_z[valid_mask]
#         num_valid = len(snow_x)
#         if num_valid == 0:
#             return PointCloud(pc.points.copy())
#
#         # 使用KDTree计算邻近点强度的平均值
#         kdtree = KDTree(pc.points[:, :3])
#         snow_points = np.column_stack((snow_x, snow_y, snow_z))
#         _, indices = kdtree.query(snow_points, k=5)
#         snow_intensity = np.mean(pc.points[indices, 3], axis=1)
#
#         # 合并原始点云和雪花点
#         snow_points = np.column_stack((snow_x, snow_y, snow_z, snow_intensity))
#         new_points = np.vstack((pc.points, snow_points))
#
#         # 距离相关的强度衰减
#         resparameter = pow(quantity[ran], 0.5)
#         distances = np.linalg.norm(new_points[:, :3], axis=1)
#         intensity_factor = np.exp(-0.01 * resparameter * distances)
#         new_points[:, 3] *= intensity_factor
#
#         return PointCloud(new_points)
#
#     def fog_effect(self, pc):
#         """雾效变换（原代码one_pcd_augment.h的fog实现）"""
#         # 随机选择参数
#         quantity = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#         changerate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#         ran = random.randint(0, 9)
#
#         # 生成圆柱状分布的雾点
#         R = 1.35  # 圆柱半径
#         H = 3  # 圆柱高度
#         density = changerate[ran]
#         num_fog = int(360 * density)
#
#         # 生成角度和高度
#         angles = np.random.uniform(0, 2 * self.PI, num_fog)
#         heights = np.random.uniform(0, H, num_fog)
#
#         # 计算雾点坐标
#         fog_x = R * np.cos(angles)
#         fog_y = R * np.sin(angles)
#         fog_z = heights
#
#         # 使用KDTree计算邻近点强度的平均值
#         kdtree = KDTree(pc.points[:, :3])
#         fog_points = np.column_stack((fog_x, fog_y, fog_z))
#         _, indices = kdtree.query(fog_points, k=5)
#         fog_intensity = np.mean(pc.points[indices, 3], axis=1)
#
#         # 合并原始点云和雾点
#         fog_points = np.column_stack((fog_x, fog_y, fog_z, fog_intensity))
#         new_points = np.vstack((pc.points, fog_points))
#
#         # 距离相关的强度衰减
#         resparameter = pow(quantity[ran], 0.7)
#         distances = np.linalg.norm(new_points[:, :3], axis=1)
#         intensity_factor = np.exp(-0.03 * resparameter * distances)
#         new_points[:, 3] *= intensity_factor
#
#         return PointCloud(new_points)
#
#
# # 标签和校准文件处理（对应原代码changelabel.h和changecalib.h）
# class LiDARLabelProcessor:
#     @staticmethod
#     def update_label(labin_path, labout_path, transform_type, **kwargs):
#         """更新标签文件（对应changelabel.h逻辑）"""
#         with open(labin_path, 'r') as f_in, open(labout_path, 'w') as f_out:
#             lines = f_in.readlines()
#             for idx, line in enumerate(lines):
#                 if not line.strip():
#                     continue
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#
#                 label_type = parts[0]
#                 if label_type == "DontCare":
#                     f_out.write(line)
#                     continue
#
#                 # 解析标签字段（KITTI格式）
#                 lab = [0.0] * 16  # 索引2-15存储标签数据
#                 for i in range(2, 16):
#                     lab[i] = float(parts[i - 1]) if (i - 1) < len(parts) else 0.0
#
#                 # 根据变换类型更新标签
#                 if transform_type == "symmetry":
#                     # 对称变换：调整旋转角
#                     lab[15] = (math.pi - lab[15]) % (2 * math.pi)
#                 elif transform_type == "scale":
#                     # 缩放变换：调整尺寸
#                     h, w, l = kwargs.get('h', 1.0), kwargs.get('w', 1.0), kwargs.get('l', 1.0)
#                     lab[9] *= h  # 高度
#                     lab[10] *= w  # 宽度
#                     lab[11] *= l  # 长度
#                 elif transform_type == "rotation":
#                     # 旋转变换：调整旋转角
#                     ry = kwargs.get('ry', 0.0)
#                     lab[15] = (lab[15] + ry) % (2 * math.pi)
#
#                 # 写回更新后的标签
#                 output_parts = [label_type] + [f"{lab[i]:.6f}" for i in range(2, 16)]
#                 f_out.write(' '.join(output_parts) + '\n')
#
#     @staticmethod
#     def update_calib(calib_in, calib_out, transform_type, **kwargs):
#         """更新校准文件（对应changecalib.h逻辑）"""
#         with open(calib_in, 'r') as f_in, open(calib_out, 'w') as f_out:
#             lines = f_in.readlines()
#             for line in lines:
#                 parts = line.strip().split()
#                 if not parts:
#                     f_out.write(line)
#                     continue
#
#                 calib_type = parts[0]
#                 if calib_type != "Tr_velo_to_cam:":
#                     f_out.write(line)
#                     continue
#
#                 # 解析Tr_velo_to_cam矩阵
#                 tr_matrix = np.eye(4, dtype=np.float32)
#                 for i in range(3):
#                     for j in range(4):
#                         tr_matrix[i, j] = float(parts[1 + i * 4 + j])
#
#                 # 构建变换矩阵
#                 change_matrix = np.eye(4, dtype=np.float32)
#                 if transform_type == "symmetry":
#                     change_matrix[1, 1] = -1.0
#                 elif transform_type == "translation":
#                     tx, ty = kwargs.get('tx', 0.0), kwargs.get('ty', 0.0)
#                     change_matrix[0, 3] = -tx
#                     change_matrix[1, 3] = -ty
#                 elif transform_type == "scale":
#                     sx, sy, sz = kwargs.get('sx', 1.0), kwargs.get('sy', 1.0), kwargs.get('sz', 1.0)
#                     change_matrix[0, 0] = 1.0 / sx
#                     change_matrix[1, 1] = 1.0 / sy
#                     change_matrix[2, 2] = 1.0 / sz
#                 elif transform_type == "rotation":
#                     ry = kwargs.get('ry', 0.0)
#                     change_matrix[0, 0] = math.cos(-ry)
#                     change_matrix[0, 1] = math.sin(-ry)
#                     change_matrix[1, 0] = -math.sin(-ry)
#                     change_matrix[1, 1] = math.cos(-ry)
#
#                 # 矩阵乘法更新校准矩阵
#                 result_matrix = tr_matrix @ change_matrix
#
#                 # 写回更新后的校准数据
#                 output_parts = [calib_type]
#                 for i in range(3):
#                     for j in range(4):
#                         output_parts.append(f"{result_matrix[i, j]:.8f}")
#                 f_out.write(' '.join(output_parts) + '\n')
#
#
# def batch_process(input_config,  output_bin_path,    output_label_path,  output_calib_path, mode, count):
#     """
#     批量处理点云文件及相关标签、校准文件
#
#     :param input_config: 输入配置字典，包含:
#         - bin_dir: 点云bin文件目录
#         - label_dir: 标签txt文件目录
#         - calib_dir: 校准文件目录
#     :param output_root: 输出根目录（会自动创建bin、txt、calib子目录）
#     :param mode: 选取模式，'sequential'（顺序）或 'random'（随机）
#     :param count: 处理文件数量，None表示全部
#     """
#     # 初始化输出目录
#     output_dirs = {
#         'bin': os.path.join(output_root, 'bin'),
#         'txt': os.path.join(output_root, 'txt'),
#         'calib': os.path.join(output_root, 'calib')
#     }
#     for dir_path in output_dirs.values():
#         os.makedirs(dir_path, exist_ok=True)
#
#     # 初始化处理器
#     transforms = LiDARTransforms()
#     label_processor = LiDARLabelProcessor()
#
#     # 获取所有bin文件并筛选
#     bin_files = [f for f in os.listdir(input_config['bin_dir']) if f.endswith('.bin')]
#     if not bin_files:
#         print("未找到任何.bin文件")
#         return
#
#     # 按模式筛选文件
#     if mode == 'sequential':
#         bin_files.sort()  # 按文件名排序（假设是数字编号）
#         selected_files = bin_files[:count] if count else bin_files
#     elif mode == 'random':
#         selected_count = count if count else len(bin_files)
#         selected_files = random.sample(bin_files, min(selected_count, len(bin_files)))
#     else:
#         print("模式错误，仅支持 'sequential' 或 'random'")
#         return
#
#     # 定义要执行的变换列表 (变换名, 变换函数, 额外参数, 是否需要更新标签, 是否需要更新校准)
#     transform_list = [
#         ('symmetry', transforms.symmetry_transform, {}, True, True),
#         ('rain', transforms.rain_effect, {}, True, False),
#         ('snow', transforms.snow_effect, {}, True, False),
#         ('fog', transforms.fog_effect, {}, True, False),
#         ('rotation_90', transforms.rotation_transform, {'ry': math.radians(90)}, True, True),
#         ('translation_0.5', transforms.translation_transform, {'tx': 0.5, 'ty': 0.5}, False, True),
#         ('scale_1.2', transforms.scale_transform, {'sx': 1.2, 'sy': 1.2, 'sz': 1.2}, True, True)
#     ]
#
#     # 批量处理每个文件
#     for bin_file in selected_files:
#         base_name = os.path.splitext(bin_file)[0]
#         print(f"处理文件: {base_name}")
#
#         # 输入文件路径
#         input_paths = {
#             'bin': os.path.join(input_config['bin_dir'], bin_file),
#             'txt': os.path.join(input_config['label_dir'], f"{base_name}.txt"),
#             'calib': os.path.join(input_config['calib_dir'], f"{base_name}.txt")
#         }
#
#         # 加载点云
#         pc = PointCloud().load_from_bin(input_paths['bin'])
#
#         # 应用所有变换
#         for transform_name, transform_func, params, update_label, update_calib in transform_list:
#             # 执行变换
#             transformed_pc = transform_func(pc, **params)
#
#             # 保存变换后的点云
#             output_bin_path = os.path.join(output_dirs['bin'], f"{transform_name}_{base_name}.bin")
#             transformed_pc.save_to_bin(output_bin_path)
#             print(f"点云已保存至: {output_bin_path}")
#
#             # 更新标签（如果需要）
#             if update_label and os.path.exists(input_paths['txt']):
#                 output_label_path = os.path.join(output_dirs['txt'], f"{transform_name}_{base_name}.txt")
#                 label_processor.update_label(
#                     input_paths['txt'],
#                     output_label_path,
#                     transform_name.split('_')[0],  # 取基础变换类型（如rotation_90 -> rotation）
#                     **params
#                 )
#
#             # 更新校准文件（如果需要）
#             if update_calib and os.path.exists(input_paths['calib']):
#                 output_calib_path = os.path.join(output_dirs['calib'], f"{transform_name}_{base_name}.txt")
#                 label_processor.update_calib(
#                     input_paths['calib'],
#                     output_calib_path,
#                     transform_name.split('_')[0], **params
#                 )
#
#     print("批量处理完成!")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置输入输出路径
#     input_config = {
#         'bin_dir': "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/velodyne",
#         'label_dir': "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/txt",
#         'calib_dir': "C:/Users/Lzd/新建文件夹/原始数据集/KITTI/training/calib"
#     }
#     output_root = "E:/数据/数据集A/旧算子"
#
#     # 自定义输出路径（可随意修改为你想要的路径）
#     output_bin_path = "E:/数据/数据集A/旧算子/bin"  # 变换后点云保存路径
#     output_label_path = "E:/数据/数据集A/旧算子/label" # 更新后标签保存路径
#     output_calib_path = "E:/数据/数据集A/旧算子/calib"  # 更新后校准文件保存路径
#
#     # 处理配置
#     process_mode = "sequential"  # 顺序处理：'sequential'，随机处理：'random'
#     process_count = 2000  # 处理数量：None表示全部，数字表示指定数量
#
#     # 执行批量处理
#     batch_process(input_config,  output_bin_path,output_label_path,output_calib_path, process_mode, process_count)
#     # # 初始化变换器和处理器
#     # transforms = LiDARTransforms()
#     # label_processor = LiDARLabelProcessor()
#     #
#     # # 加载示例点云
#     # pc = PointCloud().load_from_bin("C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/bin/000000.bin")
#     #
#     # # 1. 对称变换示例
#     # # symmetric_pc = transforms.symmetry_transform(pc)
#     # # symmetric_pc.save_to_bin("C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/symmetry_000000.bin")
#     # # label_processor.update_label("C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/txt/000000.txt", "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/txt/symmetry_000000.txt", "symmetry")
#     # # label_processor.update_calib("C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/calib/000000.txt", "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/calib/symmetry_000000.txt", "symmetry")
#     #
#     # # 2. 雨效变换示例
#     # rain_pc = transforms.rain_effect(pc)
#     # rain_pc.save_to_bin("C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/bin/rain_000000.bin")
#     # label_processor.update_label("C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/before/txt/000000.txt", "C:/Users/Lzd/Code/PythonCode/Li-LiDAR/LiDAR/after/txt/rain_000000.txt", "rain")
#     # #
#     # # # 3. 旋转变换示例（90度）
#     # # rotated_pc = transforms.rotation_transform(pc, ry=math.radians(90))
#     # # rotated_pc.save_to_bin("output_rotation.bin")
#     # # label_processor.update_label("label.txt", "label_rotation.txt", "rotation", ry=math.radians(90))
#     # # label_processor.update_calib("calib.txt", "calib_rotation.txt", "rotation", ry=math.radians(90))