import numpy as np
import json
from numpy.linalg import inv, svd, det

# 记录映射参数
dna_rendering_params = {
    "fx_coeffs": np.array([1.67851246e-15, 6.24209961e+02]),
    "fy_coeffs": np.array([5.37836198e-15, 6.26386047e+02]),
    "rotation_transform": np.array([[0.99999198, 0.00199136, 0.00347549],
                                    [-0.00196369, 0.99996649, -0.00794796],
                                    [-0.0034912, 0.00794107, 0.99996237]]),  # 旋转变换矩阵 (3x3)
    "trans_transform": np.array([[0.26710255, 0.02551325, 0.00703949],
                                 [0.00067834, 0.23504987, -0.01377058],
                                 [-0.00152665, -0.02451222, 0.26089812]]),  # 平移缩放矩阵 (3x3)
    "trans_bias": np.array([0.00199912, 0.00152352, -0.0088726])  # 平移偏置向量 (3,)
}


def load_camera_matrices(true_json, pred_json):
    """
    从JSON加载真实和预测相机矩阵

    参数:
        true_json: 真实相机矩阵的JSON数据或文件路径
        pred_json: 预测相机矩阵的JSON数据或文件路径

    返回:
        true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics
    """
    # 如果输入是文件路径，则加载JSON文件
    if isinstance(true_json, str):
        with open(true_json, 'r') as f:
            true_data = json.load(f)
    else:
        true_data = true_json

    if isinstance(pred_json, str):
        with open(pred_json, 'r') as f:
            pred_data = json.load(f)
    else:
        pred_data = pred_json

    # 提取真实矩阵
    true_intrinsics = np.array(true_data["intrinsic"])
    true_extrinsics = np.array(true_data["extrinsic"])

    # 提取预测矩阵
    pred_intrinsics = np.array(pred_data["intrinsics"])
    pred_extrinsics = np.array(pred_data["extrinsics"])

    return true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics


def map_camera_para_calculate(true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics):
    """计算相机参数映射关系"""
    # 1. 内参映射学习
    true_fx, true_fy = [], []
    pred_fx, pred_fy = [], []

    num_views = min(len(true_intrinsics), len(pred_intrinsics))
    for i in range(num_views):
        true_K = true_intrinsics[i]
        true_fx.append(true_K[0, 0])
        true_fy.append(true_K[1, 1])

        # 获取对应视角的预测内参
        if i < len(pred_intrinsics[0]):
            pred_K = pred_intrinsics[0][i]
        else:
            pred_K = pred_intrinsics[0][i % len(pred_intrinsics[0])]

        pred_fx.append(pred_K[0, 0])
        pred_fy.append(pred_K[1, 1])

    fx_coeffs = np.polyfit(true_fx, pred_fx, 1)
    fy_coeffs = np.polyfit(true_fy, pred_fy, 1)

    # 2. 外参映射学习 - 独立处理旋转和平移
    true_rotations = []
    pred_rotations = []
    true_translations = []
    pred_translations = []

    # 提取真实相机的旋转和平移
    for i in range(len(true_extrinsics)):
        # 真实外参是4x4矩阵
        R_true = true_extrinsics[i][:3, :3]
        t_true = true_extrinsics[i][:3, 3]
        true_rotations.append(R_true)
        true_translations.append(t_true)

    # 提取预测相机的旋转和平移
    for i in range(len(pred_extrinsics)):
        # 预测外参是3x4矩阵
        R_pred = pred_extrinsics[i][:3, :3]
        t_pred = pred_extrinsics[i][:3, 3]
        pred_rotations.append(R_pred)
        pred_translations.append(t_pred)

    # 确保相同数量的点
    min_count = min(len(true_rotations), len(pred_rotations))
    true_rotations = true_rotations[:min_count]
    pred_rotations = pred_rotations[:min_count]
    true_translations = true_translations[:min_count]
    pred_translations = pred_translations[:min_count]

    print(f"Processing {min_count} matching views...")

    # 1. 旋转矩阵的变换（使用Kabsch算法）
    # 计算旋转矩阵的平均值
    mean_true_rotation = np.mean(true_rotations, axis=0)
    mean_pred_rotation = np.mean(pred_rotations, axis=0)

    # 正确的Kabsch算法实现
    H = np.zeros((3, 3))
    for i in range(min_count):
        # 修正：使用矩阵乘法而不是outer
        H += pred_rotations[i] @ true_rotations[i].T

    # SVD分解
    U, S, Vt = svd(H)
    rotation_transform = U @ Vt

    # 确保右手坐标系
    if det(rotation_transform) < 0:
        # 修正：正确处理行列式负数的情况
        U[:, -1] *= -1
        rotation_transform = U @ Vt

    # 2. 平移向量的变换（最小二乘线性回归）
    true_trans = np.array(true_translations)
    pred_trans = np.array(pred_translations)

    # 添加偏置项
    X_trans = np.hstack([true_trans, np.ones((true_trans.shape[0], 1))])

    # 求解最小二乘问题
    B_with_bias, _, _, _ = np.linalg.lstsq(X_trans, pred_trans, rcond=None)

    # 提取变换矩阵（3x3）和偏置（3x1）
    trans_transform = B_with_bias[:3, :].T
    trans_bias = B_with_bias[3, :]

    return {
        "fx_coeffs": fx_coeffs,
        "fy_coeffs": fy_coeffs,
        "rotation_transform": rotation_transform,  # 旋转变换矩阵 (3x3)
        "trans_transform": trans_transform,  # 平移缩放矩阵 (3x3)
        "trans_bias": trans_bias  # 平移偏置向量 (3,)
    }


def mapping_camera(new_true_intrinsic, new_true_extrinsic, params=None):
    """预测新的相机参数"""
    if params is None:
        params = dna_rendering_params

    # 解包参数
    fx_coeffs = params["fx_coeffs"]
    fy_coeffs = params["fy_coeffs"]
    rotation_transform = params["rotation_transform"]
    trans_transform = params["trans_transform"]
    trans_bias = params["trans_bias"]

    # 1. 映射内参
    fx_pred = fx_coeffs[0] * new_true_intrinsic[0][0] + fx_coeffs[1]
    fy_pred = fy_coeffs[0] * new_true_intrinsic[1][1] + fy_coeffs[1]

    pred_intrinsic = np.array([
        [fx_pred, 0, 259],
        [0, fy_pred, 259],
        [0, 0, 1]
    ])

    # 2. 映射外参
    # 提取旋转和平移部分
    R_true = new_true_extrinsic[:3, :3]
    t_true = new_true_extrinsic[:3, 3]

    # 对旋转矩阵应用旋转变换
    transformed_rotation = rotation_transform @ R_true

    # 对平移向量应用独立的线性变换
    transformed_translation = trans_transform @ t_true + trans_bias

    # 组合成预测外参 (3x4)
    pred_extrinsic = np.column_stack((transformed_rotation, transformed_translation))
    pred_extrinsic = np.vstack([pred_extrinsic, [0, 0, 0, 1]])

    return pred_intrinsic, pred_extrinsic


# ======================== 使用示例 ========================
if __name__ == "__main__":
    # 示例1: 从JSON文件加载相机矩阵
    true_file = "./camera_para/true_cameras.json"
    pred_file = "./camera_para/pred_cameras.json"

    true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics = load_camera_matrices(
        true_file, pred_file
    )

    # 学习映射参数
    params = map_camera(true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics)

    # 定义新的真实相机参数
    new_true_intrinsic = np.array([  # 新的真实内参
        [642.4722900390625, 0.0, 253.94805908203125],
        [0.0, 644.1978149414062, 311.5732421875],
        [0, 0, 1]
    ])

    new_true_extrinsic = np.array([  # 新的真实外参
        [
            0.6926189661026001,
            0.24331846833229065,
            -0.6790251731872559,
            2.1045403480529785
        ],
        [
            -0.28157922625541687,
            0.9579005241394043,
            0.0560329332947731,
            -0.19780826568603516
        ],
        [
            0.6640723347663879,
            0.15238994359970093,
            0.7319735288619995,
            0.7948368787765503
        ],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 预测新的相机参数
    pred_K, pred_T = mapping_camera(new_true_intrinsic, new_true_extrinsic, params)

    print("预测内参:")
    print(pred_K)
    print("\n预测外参:")
    print(pred_T)
