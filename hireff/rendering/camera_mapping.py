import numpy as np
import json
from numpy.linalg import inv, svd, det

# Recorded mapping parameters
dna_rendering_params = {
    "fx_coeffs": np.array([1.67851246e-15, 6.24209961e+02]),
    "fy_coeffs": np.array([5.37836198e-15, 6.26386047e+02]),
    "rotation_transform": np.array([[0.99999198, 0.00199136, 0.00347549],
                                    [-0.00196369, 0.99996649, -0.00794796],
                                    [-0.0034912, 0.00794107, 0.99996237]]),  # Rotation transform matrix (3x3)
    "trans_transform": np.array([[0.26710255, 0.02551325, 0.00703949],
                                 [0.00067834, 0.23504987, -0.01377058],
                                 [-0.00152665, -0.02451222, 0.26089812]]),  # Translation scale matrix (3x3)
    "trans_bias": np.array([0.00199912, 0.00152352, -0.0088726])  # Translation bias vector (3,)
}


def load_camera_matrices(true_json, pred_json):
    """
    Load true and predicted camera matrices from JSON.

    Args:
        true_json: Ground truth camera matrix JSON data or file path
        pred_json: Predicted camera matrix JSON data or file path

    Returns:
        true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics
    """
    # If input is a file path, load the JSON file
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

    # Extract true matrices
    true_intrinsics = np.array(true_data["intrinsic"])
    true_extrinsics = np.array(true_data["extrinsic"])

    # Extract predicted matrices
    pred_intrinsics = np.array(pred_data["intrinsics"])
    pred_extrinsics = np.array(pred_data["extrinsics"])

    return true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics


def map_camera_para_calculate(true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics):
    """Compute the camera parameter mapping relationship."""
    # 1. Intrinsic mapping learning
    true_fx, true_fy = [], []
    pred_fx, pred_fy = [], []

    num_views = min(len(true_intrinsics), len(pred_intrinsics))
    for i in range(num_views):
        true_K = true_intrinsics[i]
        true_fx.append(true_K[0, 0])
        true_fy.append(true_K[1, 1])

        # Get the predicted intrinsics for the corresponding view
        if i < len(pred_intrinsics[0]):
            pred_K = pred_intrinsics[0][i]
        else:
            pred_K = pred_intrinsics[0][i % len(pred_intrinsics[0])]

        pred_fx.append(pred_K[0, 0])
        pred_fy.append(pred_K[1, 1])

    fx_coeffs = np.polyfit(true_fx, pred_fx, 1)
    fy_coeffs = np.polyfit(true_fy, pred_fy, 1)

    # 2. Extrinsic mapping learning - process rotation and translation independently
    true_rotations = []
    pred_rotations = []
    true_translations = []
    pred_translations = []

    # Extract rotation and translation from true cameras
    for i in range(len(true_extrinsics)):
        # True extrinsics are 4x4 matrices
        R_true = true_extrinsics[i][:3, :3]
        t_true = true_extrinsics[i][:3, 3]
        true_rotations.append(R_true)
        true_translations.append(t_true)

    # Extract rotation and translation from predicted cameras
    for i in range(len(pred_extrinsics)):
        # Predicted extrinsics are 3x4 matrices
        R_pred = pred_extrinsics[i][:3, :3]
        t_pred = pred_extrinsics[i][:3, 3]
        pred_rotations.append(R_pred)
        pred_translations.append(t_pred)

    # Ensure same number of points
    min_count = min(len(true_rotations), len(pred_rotations))
    true_rotations = true_rotations[:min_count]
    pred_rotations = pred_rotations[:min_count]
    true_translations = true_translations[:min_count]
    pred_translations = pred_translations[:min_count]

    print(f"Processing {min_count} matching views...")

    # 1. Rotation matrix transformation (using Kabsch algorithm)
    # Compute mean rotation matrices
    mean_true_rotation = np.mean(true_rotations, axis=0)
    mean_pred_rotation = np.mean(pred_rotations, axis=0)

    # Correct Kabsch algorithm implementation
    H = np.zeros((3, 3))
    for i in range(min_count):
        # Fix: Use matrix multiplication instead of outer product
        H += pred_rotations[i] @ true_rotations[i].T

    # SVD decomposition
    U, S, Vt = svd(H)
    rotation_transform = U @ Vt

    # Ensure right-handed coordinate system
    if det(rotation_transform) < 0:
        # Fix: Correctly handle negative determinant case
        U[:, -1] *= -1
        rotation_transform = U @ Vt

    # 2. Translation vector transformation (least-squares linear regression)
    true_trans = np.array(true_translations)
    pred_trans = np.array(pred_translations)

    # Add bias term
    X_trans = np.hstack([true_trans, np.ones((true_trans.shape[0], 1))])

    # Solve least-squares problem
    B_with_bias, _, _, _ = np.linalg.lstsq(X_trans, pred_trans, rcond=None)

    # Extract transform matrix (3x3) and bias (3x1)
    trans_transform = B_with_bias[:3, :].T
    trans_bias = B_with_bias[3, :]

    return {
        "fx_coeffs": fx_coeffs,
        "fy_coeffs": fy_coeffs,
        "rotation_transform": rotation_transform,  # Rotation transform matrix (3x3)
        "trans_transform": trans_transform,  # Translation scale matrix (3x3)
        "trans_bias": trans_bias  # Translation bias vector (3,)
    }


def mapping_camera(new_true_intrinsic, new_true_extrinsic, params=None):
    """Predict new camera parameters."""
    if params is None:
        params = dna_rendering_params

    # Unpack parameters
    fx_coeffs = params["fx_coeffs"]
    fy_coeffs = params["fy_coeffs"]
    rotation_transform = params["rotation_transform"]
    trans_transform = params["trans_transform"]
    trans_bias = params["trans_bias"]

    # 1. Map intrinsics
    fx_pred = fx_coeffs[0] * new_true_intrinsic[0][0] + fx_coeffs[1]
    fy_pred = fy_coeffs[0] * new_true_intrinsic[1][1] + fy_coeffs[1]

    pred_intrinsic = np.array([
        [fx_pred, 0, 259],
        [0, fy_pred, 259],
        [0, 0, 1]
    ])

    # 2. Map extrinsics
    # Extract rotation and translation parts
    R_true = new_true_extrinsic[:3, :3]
    t_true = new_true_extrinsic[:3, 3]

    # Apply rotation transform to the rotation matrix
    transformed_rotation = rotation_transform @ R_true

    # Apply independent linear transformation to the translation vector
    transformed_translation = trans_transform @ t_true + trans_bias

    # Assemble into predicted extrinsics (3x4)
    pred_extrinsic = np.column_stack((transformed_rotation, transformed_translation))
    pred_extrinsic = np.vstack([pred_extrinsic, [0, 0, 0, 1]])

    return pred_intrinsic, pred_extrinsic


# ======================== Usage example ========================
if __name__ == "__main__":
    # Example 1: Load camera matrices from JSON files
    true_file = "./camera_para/true_cameras.json"
    pred_file = "./camera_para/pred_cameras.json"

    true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics = load_camera_matrices(
        true_file, pred_file
    )

    # Learn mapping parameters
    params = map_camera(true_intrinsics, true_extrinsics, pred_intrinsics, pred_extrinsics)

    # Define new true camera parameters
    new_true_intrinsic = np.array([  # New true intrinsics
        [642.4722900390625, 0.0, 253.94805908203125],
        [0.0, 644.1978149414062, 311.5732421875],
        [0, 0, 1]
    ])

    new_true_extrinsic = np.array([  # New true extrinsics
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

    # Predict new camera parameters
    pred_K, pred_T = mapping_camera(new_true_intrinsic, new_true_extrinsic, params)

    print("Predicted intrinsics:")
    print(pred_K)
    print("\nPredicted extrinsics:")
    print(pred_T)
