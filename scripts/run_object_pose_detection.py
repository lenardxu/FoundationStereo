# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
python run_object_pose_detection.py \
--left_file /home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/left_hand_left_camera/01.png \
--right_file /home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/left_hand_right_camera/01.png \
--ckpt_dir /home/hillbot/rkx/models/FoundationStereo/model_best_bp2/model_best_bp2.pth --out_dir ./test_outputs/
"""

import json
from typing import Tuple, Union, List
import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo

from transformers import Sam2Processor, Sam2Model
from accelerate import Accelerator
import torch

from detection import QWENVDetector


os.environ["all_proxy"] = "https://socks.127.0.0.1:7890"

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = masks.squeeze(0).cpu().numpy().astype(np.uint8) * 255
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def rectify_stereo_images(
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

    return rect_left, rect_right, R1, R2, P1, P2

def load_intrinsics_and_extrinsics(
    path: Union[Tuple[str, str], List[Tuple[str, str]]],
    inverse_extrinsics: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics (K) and extrinsics (T_c2w) from one or more
    calib.json files.

    Each JSON file is expected to contain:
        ``K``      – 3×3 intrinsic matrix.
        ``T_c2w``  – 4×4 camera-to-world transformation matrix.

    Args:
        path: A single tuple of (intrinsics_path, extrinsics_path) or a list of such tuples (one per camera).
        inverse_extrinsics: If True, invert the loaded T_c2w matrices.

    Returns:
        intrinsics  : np.ndarray of shape (N, 3, 3), dtype float64.
        extrinsics  : np.ndarray of shape (N, 4, 4), dtype float64
                      containing T_c2w as stored in the file.
                      To obtain world-to-camera (w2c) matrices expected by
                      the DA3 API, invert each matrix:
                      ``w2c = np.linalg.inv(extrinsics)``.
    """
    if isinstance(path, tuple):
        path = [path]

    K_list, T_list = [], []
    for intrinsics_path, extrinsics_path in path:
        with open(intrinsics_path, "r") as f:
            calib = json.load(f)
        with open(extrinsics_path, "r") as f:
            extrinsics_data = json.load(f)
        if "K" not in calib:
            raise ValueError(f"Invalid calib.json format in {intrinsics_path}: missing 'K'")
        if "T_c2w" not in extrinsics_data:
            raise ValueError(f"Invalid cam_extrinsics.json format in {extrinsics_path}: missing 'T_c2w'")
        K_list.append(np.array(calib["K"], dtype=np.float64))
        T_list.append(np.array(extrinsics_data["T_c2w"], dtype=np.float64) \
                      if not inverse_extrinsics else np.linalg.inv(np.array(extrinsics_data["T_c2w"], dtype=np.float64)))

    intrinsics  = np.stack(K_list)   # (N, 3, 3)
    extrinsics  = np.stack(T_list)   # (N, 4, 4)
    return intrinsics, extrinsics

def segment_target_object(image: np.ndarray):
   # Initialize the object detector
    qwen_model_name = "Qwen/Qwen3-VL-2B-Instruct"
    detector = QWENVDetector(model_name=qwen_model_name)

    # Define the query for searching the target object
    query = "an object grasped by the robot gripper"  # Replace with your test query

    # Run detection
    results = detector.execute(image=image, text=query)

    # Print results
    print("Detection Results:")
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}:")
        print(f"  Bounding Box: {result['bbox']}")
        # print(f"  Confidence: {result['score']:.4f}")

    detector.visualize()

    # Initialize the SAM model for segmentation
    sam_model_name = "facebook/sam2.1-hiera-large"
    device = Accelerator().device
    model = Sam2Model.from_pretrained(sam_model_name).to(device)
    processor = Sam2Processor.from_pretrained(sam_model_name)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    bbox_xywh = results[0]['bbox']
    bbox_x0, bbox_y0, bbox_w, bbox_h = bbox_xywh
    bbox_x1 = bbox_x0 + bbox_w
    bbox_y1 = bbox_y0 + bbox_h
    input_boxes = [[[bbox_x0, bbox_y0, bbox_x1, bbox_y1]]]

    inputs = processor(images=pil_image, input_boxes=input_boxes, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    # The model outputs multiple mask predictions ranked by quality score
    print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")

    # Select the mask with the largest area
    masks_squeezed = masks.squeeze(0)  # (3, H, W)
    areas = masks_squeezed.sum(dim=(1, 2))  # sum of True pixels per mask
    best_idx = areas.argmax().item()
    print(f"Selected mask {best_idx} with area {areas[best_idx].item()} pixels")
    best_mask = masks_squeezed[best_idx:best_idx+1].unsqueeze(0)  # (1, 1, H, W)

    # Overlay masks on the original image
    overlayed_image = overlay_masks(pil_image, best_mask)
    overlayed_image.show()

    target_mask = best_mask[0][0].cpu().numpy().astype(bool)  # (H, W) boolean array
    return target_mask

if __name__=="__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

    code_dir = os.path.dirname(os.path.realpath(__file__))

    calib_rst_paths: List[Tuple[str, str]] = []
    for image_path in (args.left_file, args.right_file):
        # image:  <base>/<session>/<seq>/<camera_name>/<frame>.png
        # calib:  <base>/<session>/<camera_name>/calib.json
        image_dir = os.path.dirname(image_path)
        image_name_stem = os.path.splitext(os.path.basename(image_path))[0]
        camera_name = os.path.basename(os.path.dirname(image_path))           # e.g. left_hand_center_camera
        session_dir = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))  # up 3 levels → <session>
        calib_rst_paths.append((os.path.join(session_dir, camera_name, "calib.json"), os.path.join(image_dir, f"cam_extrinsics_{image_name_stem}.json")))
    intrinsics, extrinsics = load_intrinsics_and_extrinsics(calib_rst_paths)
    K1, K2 = intrinsics[0], intrinsics[1]
    D1, D2 = None, None
    T_left_2_w, T_right_2_w = extrinsics[0], extrinsics[1]
    T_left_2_right = np.linalg.inv(T_right_2_w) @ T_left_2_w
    R_left_2_right = T_left_2_right[:3,:3]
    t_left_2_right = T_left_2_right[:3,3]

    # Rectify stereo images to get the rectified images ready for disparity estimation
    left_img, right_img, R_left_orig2rect, R_right_orig2rect, P_left_rect, P_right_rect = rectify_stereo_images(
        args.left_file, args.right_file, 
        K1, 
        D1, 
        K2, 
        D2,
        R_left_2_right, 
        t_left_2_right  
    )
    
    # rectify_stereo_images uses cv2 (BGR); convert to RGB to match imageio convention
    img0 = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    scale = args.scale
    # TODO: now only consider the scale=1.0 case (i.e., no scaling)
    assert scale==1, "scale must be ==1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")
    target_mask = segment_target_object(image=img0)
    print(f"Target object mask with shape {target_mask.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    logging.info(f"Output saved to {args.out_dir}")

    if args.remove_invisible:
        yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = us_right<0
        disp[invalid] = np.inf

    # Retain only the disparity values corresponding to the target object mask, set the rest to infinity
    disp[~target_mask] = np.inf

    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            # Don't use the original (unrectified) K and baseline for point cloud generation, because they won't align with the rectified images and disparity.
            #   K = K1.copy()  # use the left camera's intrinsic for point cloud generation
            #   baseline = float(np.linalg.norm(t_left_2_right))  # use the actual baseline between the two cameras
            K = P_left_rect[:3, :3].copy()
            baseline = -P_right_rect[0, 3] / P_right_rect[0, 0]  # encoded directly in P2
            K[:2] *= scale
            depth = K[0,0]*baseline/disp
            np.save(f'{args.out_dir}/depth_meter.npy', depth)
            xyz_map = depth2xyzmap(depth, K)
            # Get points in the original left camera frame (e.g. for pose estimation relative to the physical camera)
            xyz_orig_map = (R_left_orig2rect.T @ xyz_map.reshape(-1, 3).T).T.reshape(H, W, 3)
            pcd = toOpen3dCloud(xyz_orig_map.reshape(-1,3), img0_ori.reshape(-1,3))
            keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
            logging.info(f"PCL saved to {args.out_dir}")

            if args.denoise_cloud:
                logging.info("[Optional step] denoise point cloud...")
                cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                inlier_cloud = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
                pcd = inlier_cloud

            logging.info("Visualizing point cloud. Press ESC to exit.")
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            vis.run()
            vis.destroy_window()

    # TODO: steps in the next for object pose estimation in the conventional way
    # 1) Load the mesh model of the target object (e.g. from a obj file)
    # 2) Convert to pcd and downsample it, and downsample the observed point cloud as well
    # 3) Ovelap both point clouds' centroid
    # 4) Run ICP to get the pose of the object in the camera frame with visualization of the alignment result to verify the pose estimation quality
    # 5) Visualize the coordinate frame of the estimated pose in the point cloud as well
    # 6) Transform the estimated pose from the camera frame to the world frame using the camera extrinsics
    # 7) Validate the estimated pose by projecting the object mesh with the estimated pose back to the image and check if the projection aligns well with the observed object in the image (e.g. using a silhouette IoU metric or visualizing the overlay)

