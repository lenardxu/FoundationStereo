import numpy as np
from typing import Tuple, Dict
import open3d as o3d


def register_point_clouds_using_axis_point(
    source_axis: np.ndarray, 
    source_point: np.ndarray, 
    target_axis: np.ndarray, 
    target_point: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    print(f"Apply axis-point-to-axis-point alignment.")

    metric = {}
    # Step 0: Compute the translation from source point to target point
    source_point = np.squeeze(source_point)
    target_point = np.squeeze(target_point)
    translation = target_point - source_point

    # Step 1: Normalize the vectors
    source_axis = source_axis / np.linalg.norm(source_axis)
    if len(source_axis.shape) == 2:
        source_axis = np.squeeze(source_axis)
    target_axis = target_axis / np.linalg.norm(target_axis)
    if len(target_axis.shape) == 2:
        target_axis = np.squeeze(target_axis)

    # Step 2: Compute the axis of rotation (cross product)
    rotation_axis = np.cross(source_axis, target_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    # If the vectors are already aligned, no rotation is needed
    if rotation_axis_norm < 1e-6:
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        return transformation, metric

    rotation_axis = rotation_axis / rotation_axis_norm  # Normalize the axis

    # Step 3: Compute the rotation matrix
    angle = np.arccos(np.clip(np.dot(source_axis, target_axis), -1.0, 1.0))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    # Prepare the transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation

    print(f"Axis-point-to-axis-point alignment applied successfully.")

    return transformation, metric