import os
import cv2
from PIL import Image
import numpy as np
import matplotlib

from transformers import Sam2Processor, Sam2Model
from accelerate import Accelerator
import torch

from detection import QWENVDetector


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

if __name__ == "__main__":
    # Initialize the detector
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    os.environ["all_proxy"] = "https://socks.127.0.0.1:7890"
    detector = QWENVDetector(model_name=model_name)

    # Test image and query
    # /home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/chest_left_camera/01.png
    # /home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/left_hand_center_camera/01.png
    image_path = "//home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/left_hand_center_camera/01.png"  # Replace with your test image path
    image = cv2.imread(image_path)
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


    device = Accelerator().device

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
    processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

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
