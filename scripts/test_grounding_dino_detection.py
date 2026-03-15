from detection import GroundingDINODetector
import os
import cv2


if __name__ == "__main__":
    # Initialize the detector
    model_name = "IDEA-Research/grounding-dino-base"
    # try:
    #     detector = GroundingDINODetector(model_name=model_name)
    # except ValueError as e:
    #     if "Unknown scheme for proxy URL" in str(e):
    #         os.environ["all_proxy"] = "https://socks.127.0.0.1:7890"
    #         detector = GroundingDINODetector(model_name=model_name)
    #     else:
    #         raise
    os.environ["all_proxy"] = "https://socks.127.0.0.1:7890"
    detector = GroundingDINODetector(model_name=model_name, use_sahi=True)

    # Test image and query
    # /home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/chest_left_camera/01.png
    # /home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/left_hand_center_camera/01.png
    image_path = "//home/hillbot/zyu/DemoHacker/apps/place/data/pick_and_capture_grasped_obj/260313100158/001/chest_left_camera/01.png"  # Replace with your test image path
    image = cv2.imread(image_path)
    query = "a robot gripper"  # Replace with your test query

    # Run detection
    results = detector.execute(image=image, text=query)

    # Print results
    print("Detection Results:")
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}:")
        print(f"  Bounding Box: {result['bbox']}")
        print(f"  Confidence: {result['score']:.4f}")

    detector.visualize()

    

