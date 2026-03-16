from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d
from pathlib import Path
import time
import sys
import zlib
from typing import Optional, Tuple, Dict
from typing_extensions import override
import copy

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


class BaseComponent(ABC):
    """Base class for all point cloud processing components.

    Provides common structure and methods for point cloud operations including
    execution, visualization, and data retrieval.

    Attributes:
        _class_name (str): The name of the component class.
        _params (dict): Dictionary of component parameters.
        _cloud (Optional[o3d.geometry.PointCloud]): The processed point cloud.
    """

    def __init__(self):
        """Initializes the BaseComponent with default attributes."""
        self._class_name = self.__class__.__name__
        self._params = {}
        self._cloud = None

    def _update(self):
        """Initializes component-specific settings.
        
        Override in subclasses if needed.
        """
        pass

    def update(self, **new_params):
        """Updates component parameters and reinitializes if changed.

        Args:
            **new_params: Arbitrary keyword arguments representing parameter
                names and their new values. Only existing parameters in _params
                will be updated.
        """
        updated = False
        for key, value in new_params.items():
            if key in self._params:
                if type(value) == np.ndarray:
                    if self._params[key] is None or (not np.ndarray_equal(self._params[key], value)):
                        self._params[key] = value
                        updated = True
                elif self._params[key] != value:
                    self._params[key] = value
                    updated = True
        if updated:
            self._update()

    @abstractmethod
    def execute(self, point_cloud: o3d.geometry.PointCloud):
        """Executes the point cloud processing operation.

        Args:
            point_cloud (o3d.geometry.PointCloud): Point cloud to be processed.

        Returns:
            o3d.geometry.PointCloud: The processed point cloud.
        """
        # Validate input type
        if not isinstance(point_cloud, o3d.geometry.PointCloud):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'point_cloud' expected o3d.geometry.PointCloud, got {type(point_cloud).__name__}"
            )
        
        raise NotImplementedError(f"[{self._class_name}.execute] Subclasses should implement this method.")
    
    @abstractmethod
    def get_data(self):
        """Retrieves the processed data.
        
        Override in subclasses.
        """
        raise NotImplementedError(f"[{self._class_name}.execute] Subclasses should implement this method.")

    def visualize(
            self, 
            save: bool = False, 
            dirpath: Optional[Path] = None, 
            name: Optional[str] = None
        ):
        """Visualizes the result point cloud.
        
        Displays the point cloud using either a socket connection (preferred) or
        Open3D's visualization tool. Optionally saves the point cloud to disk.

        Args:
            save (bool): If True, saves the point cloud to disk.
            dirpath (Optional[Path]): Directory path for saving. If None, uses default location.
            name (Optional[str]): Display name for the visualization.
            
        Raises:
            TypeError: If input parameters have incorrect types.
        """
        # Validate input types
        if not isinstance(save, bool):
            raise TypeError(
                f"Type validation failed in {self._class_name}.visualize: "
                f"parameter 'save' expected bool, got {type(save).__name__}"
            )
        if not isinstance(dirpath, (type(None), Path)):
            raise TypeError(
                f"Type validation failed in {self._class_name}.visualize: "
                f"parameter 'dirpath' expected Optional[Path], got {type(dirpath).__name__}"
            )
        if not isinstance(name, (type(None), str)):
            raise TypeError(
                f"Type validation failed in {self._class_name}.visualize: "
                f"parameter 'name' expected Optional[str], got {type(name).__name__}"
            )
        
        if self._cloud is not None:
            o3d.visualization.draw_geometries([self._cloud])

            if save:
                # Determine the dirpath for saving clouds
                if dirpath is None:
                    parent_dir = Path(sys.modules['__main__'].__file__).resolve().parent
                    dirpath = parent_dir / "images"

                # Create the folder if it does not exist
                dirpath.mkdir(parents=True, exist_ok=True)

                # Generate timestamp and file name
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cloud_name = f"{timestamp}_{self._class_name}.ply"
                cloud_path = dirpath / cloud_name

                # Save the point cloud
                o3d.io.write_point_cloud(str(cloud_path), self._cloud)
                print(f"[{self._class_name}] Saved cloud to {str(cloud_path)}")

        else:
            print(f"[Warning] [{self._class_name}.visualize] No point cloud data to visualize. Please run execute() first.")


class BaseRegistration(BaseComponent):
    """Base class for point cloud registration operations.

    Provides structure for aligning point clouds using various registration algorithms,
    computing transformation matrices between source and target clouds.

    Attributes:
        _transformation (Optional[np.ndarray]): The computed transformation matrix after registration.
        _cloud (Optional[o3d.geometry.PointCloud]): The transformed source point cloud combined with target for visualization.
    """
    
    def __init__(self):
        """Initializes the BaseRegistration."""
        super().__init__()
        self._transformation = None

    @override
    def get_data(self):
        """Retrieves the transformation matrix from registration.

        Returns:
            Optional[np.ndarray]: The transformation matrix as a numpy array, or None if not available.
        """
        if self._transformation is not None:
            return self._transformation
        else:
            print(f"[Warning] [{self._class_name}] No transformation available. Please run execute() first.")
            return None
        

class TranslationSampledPointToPointICPRegistration(BaseRegistration):
    """Base class for ICP with multiple translation initializations.

    Tries ICP from multiple initial translations to find the best alignment. This is an
    abstract base class extended by specific translation sampling strategies.

    Attributes:
        _params (dict): Dictionary containing ICP parameters and sampled transformations.
    """

    def __init__(self,
                 min_fitness_score: float = 0.5,
                 early_stop_fitness_score: float = 0.9,
                 max_iterations: int = 50,
                 max_correspondence_distance: float = 20,
                 estimate_scaling: bool = False):
        """Initializes the translation sampled ICP registration.

        Args:
            min_fitness_score (float): Lower bound fitness score threshold. Defaults to 0.5.
            early_stop_fitness_score (float): Higher bound fitness score for early termination. Defaults to 0.9.
            max_iterations (int): Maximum number of ICP iterations. Defaults to 50.
            max_correspondence_distance (float): Distance threshold for ICP correspondence. Defaults to 20.
            estimate_scaling (bool): If True, performs scaled transformation estimation. Defaults to False.
        """
        # Validate input types
        if not isinstance(min_fitness_score, (float, int)):
            raise TypeError(
                f"Type validation failed in TranslationSamplerPointToPointICPRegistration.__init__: "
                f"parameter 'min_fitness_score' expected float, got {type(min_fitness_score).__name__}"
            )
        if not isinstance(early_stop_fitness_score, (float, int)):
            raise TypeError(
                f"Type validation failed in TranslationSampledPointToPointICPRegistration.__init__: "
                f"parameter 'early_stop_fitness_score' expected float, got {type(early_stop_fitness_score).__name__}"
            )
        if not isinstance(max_iterations, int):
            raise TypeError(
                f"Type validation failed in TranslationSampledPointToPointICPRegistration.__init__: "
                f"parameter 'max_iterations' expected int, got {type(max_iterations).__name__}"
            )
        if not isinstance(max_correspondence_distance, (float, int)):
            raise TypeError(
                f"Type validation failed in TranslationSampledPointToPointICPRegistration.__init__: "
                f"parameter 'max_correspondence_distance' expected float, got {type(max_correspondence_distance).__name__}"
            )
        if not isinstance(estimate_scaling, bool):
            raise TypeError(
                f"Type validation failed in TranslationSampledPointToPointICPRegistration.__init__: "
                f"parameter 'estimate_scaling' expected bool, got {type(estimate_scaling).__name__}"
            )
        
        super().__init__()
        self._params = {
            "min_fitness_score": min_fitness_score,
            "early_stop_fitness_score": early_stop_fitness_score,
            "max_iterations": max_iterations,
            "max_correspondence_distance": max_correspondence_distance,
            "estimate_scaling": estimate_scaling,
            "sampled_source_T_target_list": []
        }

    def execute(self, 
                source_point_cloud: o3d.geometry.PointCloud, 
                target_point_cloud: o3d.geometry.PointCloud, 
                initial_transformation_matrix: np.ndarray = np.eye(4)
        ) -> tuple[np.ndarray, dict]:
        """Executes ICP with multiple translation samples to find the best alignment.

        Args:
            source_point_cloud (o3d.geometry.PointCloud): The source point cloud to be aligned.
            target_point_cloud (o3d.geometry.PointCloud): The target point cloud to align to.
            initial_transformation_matrix (np.ndarray): Initial 4x4 transformation matrix. Defaults to identity.

        Returns:
            tuple[np.ndarray, dict]: A tuple containing the best transformation matrix and metrics dictionary,
                or ('null', None) if all attempts fail.
        """
        # Validate input types
        if not isinstance(source_point_cloud, o3d.geometry.PointCloud):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'source_point_cloud' expected o3d.geometry.PointCloud, got {type(source_point_cloud).__name__}"
            )
        if not isinstance(target_point_cloud, o3d.geometry.PointCloud):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'target_point_cloud' expected o3d.geometry.PointCloud, got {type(target_point_cloud).__name__}"
            )
        if not isinstance(initial_transformation_matrix, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'initial_transformation_matrix' expected np.ndarray, got {type(initial_transformation_matrix).__name__}"
            )
        
        print(f"[{self._class_name}.execute] Applying translation sampled point-to-point ICP.")

        
        if initial_transformation_matrix is None or (isinstance(initial_transformation_matrix, str) and initial_transformation_matrix.lower() == "null"):
            print(f"[{self._class_name}.execute] Initial transformation is set as identity matrix when it is None or 'null'.")
            initial_transformation_matrix = np.eye(4)

        
        # Reset cloud as None to avoid visualization of old cloud coming from last run of execute
        self._cloud = None

        highest_fitness_score = -1
        best_result = None
        result_transformation = None
        metric = {}
        for sampled_source_T_target in self._params["sampled_source_T_target_list"]:
            updated_initial_transformation = initial_transformation_matrix @ sampled_source_T_target

            # Perform Point-to-Point ICP
            result = o3d.pipelines.registration.registration_icp(
                source_point_cloud,
                target_point_cloud,
                self._params["max_correspondence_distance"],
                updated_initial_transformation,  # Initial transformation
                o3d.pipelines.registration.TransformationEstimationPointToPoint(self._params["estimate_scaling"]),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self._params["max_iterations"])
            )
            fitness_score = result.fitness

            # Check if alignment was successful
            if fitness_score < self._params["min_fitness_score"]:
                continue

            if fitness_score > highest_fitness_score:
                print(
                    f"[{self._class_name}.execute] Init t with score {fitness_score}: {updated_initial_transformation[:3, 3]}")
                result_transformation = result.transformation
                metric["correspondence_set"] = result.correspondence_set
                metric["fitness"] = fitness_score
                metric["inlier_rmse"] = result.inlier_rmse
                best_result = result
                highest_fitness_score = fitness_score

            if fitness_score > self._params["early_stop_fitness_score"]:
                print(
                    f"[{self._class_name}.execute] Init t with score {fitness_score}: {updated_initial_transformation[:3, 3]}")
                print(f"[Success] [{self._class_name}.execute] ICP alignment succeeded in passing {self._params['early_stop_fitness_score']}.")
                print(f"[Success] [{self._class_name}.execute] {result}")
                result_transformation = result.transformation
                metric["correspondence_set"] = result.correspondence_set
                metric["fitness"] = fitness_score
                metric["inlier_rmse"] = result.inlier_rmse
                break

        if highest_fitness_score == -1:
            print(f"[Error] [{self._class_name}.execute] ICP alignment failed. Low fitness score.")
            return np.eye(4), None

        self._transformation = result_transformation
        if fitness_score <= self._params["early_stop_fitness_score"]:
            print(f"[{self._class_name}.execute] Best result: {best_result}")

        # To visualize self._cloud - transformed source_copy
        self._cloud = copy.deepcopy(source_point_cloud)
        # initial_cloud = copy.deepcopy(source)
        # initial_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # red for visibility

        self._cloud.paint_uniform_color([1.0, 0.0, 0.0])  # red for visibility
        self._cloud.transform(self._transformation)
        self._cloud += target_point_cloud  # to have transformed source cloud and target cloud visualized together
        # self._cloud += initial_cloud
        print(f"[{self._class_name}.execute] Translation sampled point-to-point ICP alignment applied successfully.")
        transformation = copy.deepcopy(self._transformation)

        return transformation, metric


class CuboidTranslationSamplerPointToPointICPRegistration(TranslationSampledPointToPointICPRegistration):
    """Performs ICP with translation samples uniformly distributed in a cuboid region.

    Samples initial translations on a regular 3D grid (cuboid) and tries ICP from each
    sampled position to find the best alignment.

    Attributes:
        _params (dict): Dictionary containing cuboid sampling parameters and ICP settings.
    """

    def __init__(self,
                 step_size: float = 0.001,
                 x_min: float = -0.01, 
                 x_max: float = 0.01,
                 y_min: float = -0.01, 
                 y_max: float = 0.01,
                 z_min: float = -0.01, 
                 z_max: float = 0.01,
                 min_fitness_score: float = 0.5,
                 early_stop_fitness_score: float = 0.9,
                 max_iterations: int = 50,
                 max_correspondence_distance: float = 20,
                 estimate_scaling: bool = False):
        """Initializes the translation cuboid sampled ICP registration.

        Args:
            step_size (float): Step size for sampling along each axis. Defaults to 0.001.
            x_min (float): Minimum translation along x-axis. Defaults to -0.01.
            x_max (float): Maximum translation along x-axis. Defaults to 0.01.
            y_min (float): Minimum translation along y-axis. Defaults to -0.01.
            y_max (float): Maximum translation along y-axis. Defaults to 0.01.
            z_min (float): Minimum translation along z-axis. Defaults to -0.01.
            z_max (float): Maximum translation along z-axis. Defaults to 0.01.
            min_fitness_score (float): Lower bound fitness score threshold. Defaults to 0.5.
            early_stop_fitness_score (float): Higher bound fitness score for early termination. Defaults to 0.9.
            max_iterations (int): Maximum number of ICP iterations. Defaults to 50.
            max_correspondence_distance (float): Distance threshold for ICP correspondence. Defaults to 20.
            estimate_scaling (bool): If True, performs scaled transformation estimation. Defaults to False.
        """
        # Validate input types
        if not isinstance(step_size, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'step_size' expected float, got {type(step_size).__name__}"
            )
        if not isinstance(x_min, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'x_min' expected float, got {type(x_min).__name__}"
            )
        if not isinstance(x_max, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'x_max' expected float, got {type(x_max).__name__}"
            )
        if not isinstance(y_min, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'y_min' expected float, got {type(y_min).__name__}"
            )
        if not isinstance(y_max, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'y_max' expected float, got {type(y_max).__name__}"
            )
        if not isinstance(z_min, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'z_min' expected float, got {type(z_min).__name__}"
            )
        if not isinstance(z_max, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'z_max' expected float, got {type(z_max).__name__}"
            )
        if not isinstance(min_fitness_score, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'min_fitness_score' expected float, got {type(min_fitness_score).__name__}"
            )
        if not isinstance(early_stop_fitness_score, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'early_stop_fitness_score' expected float, got {type(early_stop_fitness_score).__name__}"
            )
        if not isinstance(max_iterations, int):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'max_iterations' expected int, got {type(max_iterations).__name__}"
            )
        if not isinstance(max_correspondence_distance, (int, float)):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'max_correspondence_distance' expected float, got {type(max_correspondence_distance).__name__}"
            )
        if not isinstance(estimate_scaling, bool):
            raise TypeError(
                f"Type validation failed in TranslationCuboidSampledPointToPointICPRegistration.__init__: "
                f"parameter 'estimate_scaling' expected bool, got {type(estimate_scaling).__name__}"
            )
        
        super().__init__(min_fitness_score, early_stop_fitness_score, max_iterations, max_correspondence_distance, estimate_scaling)
        self._params = {
            "step_size": step_size,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "max_iterations": max_iterations,
            "max_correspondence_distance": max_correspondence_distance,
            "estimate_scaling": estimate_scaling,
            "min_fitness_score": min_fitness_score,
            "early_stop_fitness_score": early_stop_fitness_score
        }

        self._update()
    
    def _update(self):
        """Initializes translation samples on a 3D cuboid grid."""
        self._params["sampled_source_T_target_list"] = []
        x_range = np.arange(self._params["x_min"], self._params["x_max"] + self._params["step_size"], self._params["step_size"])
        y_range = np.arange(self._params["y_min"], self._params["y_max"] + self._params["step_size"], self._params["step_size"])
        z_range = np.arange(self._params["z_min"], self._params["z_max"] + self._params["step_size"], self._params["step_size"])

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    translation = np.eye(4)
                    translation[:3, 3] = np.array([x, y, z])
                    self._params["sampled_source_T_target_list"].append(translation)


class PointToPlaneICPRegistration(BaseRegistration):
    """Performs point-to-plane ICP registration with optional robust kernels.

    Uses point-to-plane distance metric which is more robust than point-to-point for
    surfaces with normals. Optionally uses robust loss functions to handle outliers.

    Attributes:
        _params (dict): Dictionary containing ICP parameters (iterations, distance, normals estimation, robust kernel settings).
    """

    def __init__(self,
                 max_iterations: int = 50,
                 max_correspondence_distance: float = 50,
                 normal_search_radius: float = 0.001,
                 normal_max_neighbors: int = 30,
                 use_robust_kernel: bool = False,
                 loss_type: str = "tukey_loss",
                 noise_standard_deviation: float = 0.001):
        """Initializes the point-to-plane ICP registration.
        
        Args:
            max_iterations (int): Maximum number of ICP iterations. Defaults to 50.
            max_correspondence_distance (float): Distance threshold for ICP correspondence. Defaults to 50.
            normal_search_radius (float): Search radius for normal estimation. Defaults to 0.001.
            normal_max_neighbors (int): Maximum neighbors for normal estimation. Defaults to 30.
            use_robust_kernel (bool): If True, uses robust loss function. Defaults to False.
            loss_type (str): Type of robust loss ("tukey_loss", "cauchy_loss", "huber_loss"). Defaults to "tukey_loss".
            noise_standard_deviation (float): Noise standard deviation for Tukey loss. Defaults to 0.001.
        """
        # Validate input types
        if not isinstance(max_iterations, int):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'max_iterations' expected int, got {type(max_iterations).__name__}"
            )
        if not isinstance(max_correspondence_distance, (float, int)):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'max_correspondence_distance' expected float, got {type(max_correspondence_distance).__name__}"
            )
        if not isinstance(normal_search_radius, (float, int)):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'normal_search_radius' expected float, got {type(normal_search_radius).__name__}"
            )
        if not isinstance(normal_max_neighbors, int):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'normal_max_neighbors' expected int, got {type(normal_max_neighbors).__name__}"
            )
        if not isinstance(use_robust_kernel, bool):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'use_robust_kernel' expected bool, got {type(use_robust_kernel).__name__}"
            )
        if not isinstance(loss_type, str):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'loss_type' expected str, got {type(loss_type).__name__}"
            )
        if not isinstance(noise_standard_deviation, (float, int)):
            raise TypeError(
                f"Type validation failed in PointToPlaneICPRegistration.__init__: "
                f"parameter 'noise_standard_deviation' expected float, got {type(noise_standard_deviation).__name__}"
            )
        super().__init__()
        self._params = {
            "max_iterations": max_iterations,
            "max_correspondence_distance": max_correspondence_distance,
            "normal_search_radius": normal_search_radius,
            "normal_max_neighbors": normal_max_neighbors,
            "use_robust_kernel": use_robust_kernel,
            "loss_type": loss_type,
            "noise_standard_deviation": noise_standard_deviation,
            "transformation_estimation_method": None
        }
       
        self._update()
        
    def execute(self, 
                source_point_cloud: o3d.geometry.PointCloud, 
                target_point_cloud: o3d.geometry.PointCloud, 
                initial_transformation_matrix: np.ndarray = np.eye(4)) -> tuple[np.ndarray, dict]:
        """Performs point-to-plane ICP alignment and returns the transformation matrix.

        Args:
            source_point_cloud (o3d.geometry.PointCloud): The source point cloud to be aligned.
            target_point_cloud (o3d.geometry.PointCloud): The target point cloud to align to (normals will be estimated).
            initial_transformation_matrix (np.ndarray): Initial 4x4 transformation matrix. Defaults to identity.

        Returns:
            tuple[np.ndarray, dict]: A tuple containing the 4x4 transformation matrix and metrics dictionary
                (correspondence_set, fitness, inlier_rmse), or (None, None) if ICP fails.

        Note:
            The transformation represents the alignment from source to target point cloud.
            Target normals are computed automatically if not present.
        """
        # Validate input types
        if not isinstance(source_point_cloud, o3d.geometry.PointCloud):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'source_point_cloud' expected o3d.geometry.PointCloud, got {type(source_point_cloud).__name__}"
            )
        if not isinstance(target_point_cloud, o3d.geometry.PointCloud):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'target_point_cloud' expected o3d.geometry.PointCloud, got {type(target_point_cloud).__name__}"
            )
        if not isinstance(initial_transformation_matrix, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'initial_transformation_matrix' expected np.ndarray, got {type(initial_transformation_matrix).__name__}"
            )
        
        print(f"[{self._class_name}.execute] Applying point-to-plane ICP registration.")

        if initial_transformation_matrix is None:
            print(f"[Error] [{self._class_name}] ICP alignment failed. Initial transformation is None.")
            return np.eye(4), None

        # Reset cloud as None to avoid visualization of old cloud coming from last run of execute
        self._cloud = None

        # Estimate normals for target
        # Reference: objective function = sigma(((p - Tq) * n_p)^2), where n_p is the normal of target point
        target_point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._params["normal_search_radius"],
                max_nn=self._params["normal_max_neighbors"]
            )
        )
        # o3d.visualization.draw_geometries([target], point_show_normal=True)

        # Perform ICP
        try:
            result = o3d.pipelines.registration.registration_icp(
            source_point_cloud,
            target_point_cloud,
            self._params["max_correspondence_distance"],
            initial_transformation_matrix,  # Initial transformation
            self._params["transformation_estimation_method"],
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self._params["max_iterations"])
        )
        except Exception as e:
            print(f"[Error] [{self._class_name}] Error logging initial_transformation_matrix: {e}")
        print(f"[{self._class_name}] {result}")

        # Check if alignment was successful
        if result.fitness < 0.5:
            print(f"[Error] [{self._class_name}] ICP alignment failed. Low fitness score of {result.fitness}.")
            return np.eye(4), None

        print(f"[Success] [{self._class_name}] ICP alignment successful.")
        self._transformation = result.transformation
        metric = {}
        metric["correspondence_set"] = result.correspondence_set
        metric["fitness"] = result.fitness
        metric["inlier_rmse"] = result.inlier_rmse

        # To visualize self._cloud - transformed source_copy
        source_copy = copy.deepcopy(source_point_cloud)
        source_copy.paint_uniform_color([0.0, 1.0, 0.0])
        self._cloud = source_copy.transform(self._transformation)
        self._cloud += target_point_cloud  # to have transformed source cloud and target cloud visualized together

        print(f"[{self._class_name}.execute] Point-to-plane ICP registration applied successfully.")
        # Return the transformation matrix (pose)

        # Deepcopy using copy.deepcopy
        transformation = copy.deepcopy(result.transformation)

        return transformation, metric

    def _update(self):
        """Initializes the point-to-plane transformation estimation method with optional robust kernel."""
        if self._params["use_robust_kernel"]:
            if self._params["loss_type"] == "tukey_loss":
                self._noise_standard_deviation = self._params["noise_standard_deviation"]
                loss = o3d.pipelines.registration.TukeyLoss(k=self._params['noise_standard_deviation'])
                self._params["transformation_estimation_method"] = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            elif self._params["loss_type"] == "cauchy_loss":
                raise NotImplementedError(f"[{self._class_name}] Cauchy loss not implemented.")
            elif self._params["loss_type"] == "huber_loss":
                raise NotImplementedError(f"[{self._class_name}] Huber loss not implemented.")
            else:
                print(f"[Error] [{self._class_name}] Unknown loss type.")
                raise ValueError(f"[{self._class_name}] Unknown loss type.")
        else:
            self._params["transformation_estimation_method"] = o3d.pipelines.registration.TransformationEstimationPointToPlane()