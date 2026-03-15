from typing import Tuple, Union
import os,sys
import cv2
import numpy as np
import open3d as o3d



class DepthEstimation:
    def __init__(self):
        self._class_name = self.__class__.__name__

    def execute(
            self, 
            img_left: np.ndarray, 
            img_right: np.ndarray, 
        ):
        pass

class DepthEstimationUsingFoundationStereo(DepthEstimation):
    def __init__(self):
        super().__init__()

    def execute(
            self, 
            img_left: np.ndarray, 
            img_right: np.ndarray, 
        ):
        pass



def _rectify_stereo_images(
        img_left: Union[str, np.ndarray], img_right: Union[str, np.ndarray],
        K1, D1, 
        K2, D2,
        R, T  
    ) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(img_left, str):
        img_left_arr = cv2.imread(img_left)
    else:
        img_left_arr = img_left
    if isinstance(img_right, str):
        img_right_arr = cv2.imread(img_right)
    else:
        img_right_arr = img_right
    h, w = img_left_arr.shape[:2]

    # --- Stereo rectification ---
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2,
        (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,  # align principal points (recommended)
        alpha=0,  # 0 = crop to valid pixels only; 1 = keep all pixels
    )

    # --- Build rectification maps ---
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    # --- Remap ---
    rect_left  = cv2.remap(img_left_arr,  map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right_arr, map2x, map2y, cv2.INTER_LINEAR)

    return rect_left, rect_right