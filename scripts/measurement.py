import numpy as np
import open3d as o3d


def calculate_point_cloud_principle_axes(
        pcd: o3d.geometry.PointCloud,
        use_pca: bool = True,
        axis_selection: str = "all"
    ) -> np.ndarray:
    """
    Calculate the principle axes of a point cloud using PCA.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        use_pca (bool): Whether to use PCA for calculating principal axes.
        axis_selection (str): Which principal axis to return ("first", "second", "third", or "all").

    Returns:
        np.ndarray: A 3x3 array where each column is a principle axis.
    """
    if use_pca:
        points = np.asarray(pcd.points)
        # Compute the centroid of the point cloud
        centroid = np.mean(points, axis=0)
        # Compute the covariance matrix for the centered points
        centered_points = points - centroid
        covariance_matrix = np.cov(centered_points.T)
        # Perform eigen decomposition to find the principal axes
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        # Sort the eigen_values and eigen_vectors in the order of decreasing eigen values
        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigenvalues = eigen_values[sorted_indices]
        sorted_eigenvectors = eigen_vectors[:, sorted_indices]
        # The eigen_vectors correspond to the principal axes
        measurement = sorted_eigenvectors
    else:
        obbox = pcd.get_oriented_bounding_box()
        # Get the rotation matrix of the OBB
        obbox_rot_mat = obbox.R
        # Get the extent (lengths along each axis)
        axes_extent = obbox.extent
        # Sort the axes based on the extent (variance)
        ## The larger the extent, the more variance the points have along that axis
        sorted_indices = np.argsort(axes_extent)[::-1]
        principal_axes_sorted = obbox_rot_mat[:, sorted_indices]
        measurement = principal_axes_sorted

    if axis_selection == "first":
        measurement = measurement[:, 0]
        print(f"First principal axis calculation: {measurement}")
    elif axis_selection == "second":
        measurement = measurement[:, 1]
        print(f"Second principal axis calculation: {measurement}")
    elif axis_selection == "third":
        measurement = measurement[:, 2]
        print(f"Third principal axis calculation: {measurement}")
    else:
        measurement = measurement
        print(f"Principal axes (R) calculation: {measurement}")

    return measurement