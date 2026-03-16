import os
import sys
import time
import json
import re
import base64
import cv2
import numpy as np
from typing import Optional, Union, Sequence, Any, Dict, List
from numbers import Real, Integral
from abc import ABC, abstractmethod
import torch
import torchvision
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Qwen3VLForConditionalGeneration
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class BaseComponent(ABC):
    def __init__(self):
        self._class_name = self.__class__.__name__
        self._params = {}
        self._visualization_images = []

    def _initialize(self):
        """To be overridden by subclasses if needed."""
        pass

    def update(self, **new_params):
        updated = False
        for k, v in new_params.items():
            if k in self._params:
                if isinstance(self._params[k], np.ndarray):
                    if not np.array_equal(self._params[k], v):
                        self._params[k] = v
                        updated = True
                elif self._params[k] != v:
                    self._params[k] = v
                    updated = True
        if updated:
            self._initialize()

    def visualize(
            self,
            save: bool = False,
            path: Optional[str] = None,
            name: Optional[str] = None):
        # Validate input types
        if not isinstance(save, bool):
            raise TypeError(
                f"Type validation failed in {self._class_name}.visualize: "
                f"parameter 'save' expected bool, got {type(save).__name__}"
            )
        if not isinstance(path, (str, type(None))):
            raise TypeError(
                f"Type validation failed in {self._class_name}.visualize: "
                f"parameter 'path' expected Optional[str], got {type(path).__name__}"
            )
        if not isinstance(name, (str, type(None))):
            raise TypeError(
                f"Type validation failed in {self._class_name}.visualize: "
                f"parameter 'name' expected Optional[str], got {type(name).__name__}"
            )

        if not self._visualization_images:
            print(
                f"[Warning] [{self._class_name}] No visualization images found.")
            return

        for img in self._visualization_images:
            cv2.imshow("IMAGE", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

    def _generate_object_detection_annotation_dict(
            self,
            id: int,
            image_id: int = 1,
            category_id: int = 1,
            segmentation: Optional[Union[dict, Sequence]] = None,  # RLE dict or list-of-polygons
            area: Optional[float] = None,
            bbox: Optional[Union[np.ndarray, Sequence[Real]]] = None,  # [x,y,width,height]
            score: Optional[float] = None,
        ) -> Dict[str, Any]:
        """
            Build a COCO-like annotation dictionary with optional extension fields.

            This function always includes the standard COCO keys:
                id, image_id, category_id, segmentation, area, bbox, score

            Standard COCO fields are filled with `None` when not provided.

            Parameters
            ----------
            id:
                Annotation id (required).
            image_id:
                Image id (required). Must be int.
            category_id:
                Category id (required). Must be int.
            segmentation:
                Either an RLE dict (COCO-style, typically with keys like "counts" and "size")
                or a polygon representation (list of polygons, where each polygon is a list of
                numbers [x1,y1,x2,y2,...]). This function does basic shape/type checks.
            area:
                Area value (optional). Must be real number if provided.
            bbox:
                COCO bbox [x, y, width, height]. Accepts numpy array or sequence of 4 real
                numbers.
            
            Extended fields not part of COCO standard
            score:
                Confidence score (optional). Must be real number if provided.

            Returns
            -------
            dict
                A JSON-serializable annotation dictionary (numpy arrays normalized to lists).

            Raises
            ------
            TypeError / ValueError
                If inputs have incorrect types or invalid shapes/lengths.
        """
        def _require_int(name: str, value: Any) -> int:
            # bool is a subclass of int; disallow it explicitly
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            return int(value)

        def _require_real(name: str, value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            return float(value)

        def _to_list_of_reals(name: str, value: Union[np.ndarray, Sequence], length: Optional[int] = None) -> List[float]:
            if isinstance(value, np.ndarray):
                value_list = value.reshape(-1).tolist()
            else:
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                    raise TypeError(f"{name} must be a sequence of numbers, got {type(value).__name__}")
                value_list = list(value)

            if length is not None and len(value_list) != length:
                raise ValueError(f"{name} must have length {length}, got {len(value_list)}")

            out: List[float] = []
            for i, v in enumerate(value_list):
                try:
                    out.append(_require_real(f"{name}[{i}]", v))
                except TypeError as e:
                    raise TypeError(str(e)) from None
            return out

        # Required field checks
        ann_id = _require_int("id", id)
        img_id = _require_int("image_id", image_id)
        cat_id = _require_int("category_id", category_id)

        # COCO extended format for object detection annotation
        annotation: Dict[str, Any] = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "segmentation": None,
            "area": 0,
            "bbox": None,
            "iscrowd": 0,
            "score": None,
        }

        if segmentation is not None:
            # Accept dict (RLE) or sequence (polygons).
            if isinstance(segmentation, dict):
                annotation["segmentation"] = segmentation
            elif isinstance(segmentation, Sequence) and not isinstance(segmentation, (str, bytes, bytearray)):
                # Basic polygon validation: list of polygons; polygon is list of reals with even length >= 6.
                seg_list = list(segmentation)
                if len(seg_list) == 0:
                    raise ValueError("segmentation polygon list cannot be empty")
                normalized_polys = []
                for pi, poly in enumerate(seg_list):
                    poly_list = _to_list_of_reals(f"segmentation[{pi}]", poly)
                    if len(poly_list) < 6 or (len(poly_list) % 2) != 0:
                        raise ValueError(
                            f"segmentation[{pi}] must have even length >= 6 (x,y pairs), got {len(poly_list)}"
                        )
                    normalized_polys.append(poly_list)
                annotation["segmentation"] = normalized_polys
            else:
                raise TypeError("segmentation must be a dict (RLE) or a sequence (polygons)")

        if bbox is not None:
            bbox_list = _to_list_of_reals("bbox", bbox, length=4)
            annotation["bbox"] = bbox_list

        if area is not None:
            annotation["area"] = _require_real("area", area)

        # Exteded field validation
        if score is not None:
            annotation["score"] = _require_real("score", score)

        return annotation

class BaseShapeDetector(BaseComponent):
    def __init__(self):
        super().__init__()
        self._results = None

    def execute(self, image: np.ndarray):
        # Validate input type
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'image' expected np.ndarray, got {type(image).__name__}"
            )

        raise NotImplementedError(
            f"[{self._class_name}] Subclasses should implement this method.")

    def get_results(self):
        if self._results is not None:
            return self._results
        print(
            f"[Warning] [{self._class_name}] No detected shapes available. Please run execute() first.")
        return None


class GroundingDINODetector(BaseShapeDetector):
    """
    Zero-shot object detection using Grounding DINO (HuggingFace).

    Supports optional SAHI-style sliced inference for detecting small
    objects in large / high-resolution images.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    text_threshold : float
        Minimum confidence for text / label matching.
    box_threshold : float
        Minimum confidence for bounding-box predictions.
    use_sahi : bool
        Enable SAHI sliced inference.
    slice_width : int
        Width of each image slice (only used when *use_sahi* is True).
    slice_height : int
        Height of each image slice (only used when *use_sahi* is True).
    overlap : float
        Overlap ratio between adjacent slices (0-1).
    nms_threshold : float
        IoU threshold for NMS when merging slice detections.
    full_image_also : bool
        When SAHI is enabled, also run a full-image pass to catch large objects.

    Usage
    -----
    >>> detector = GroundingDINODetector("IDEA-Research/grounding-dino-base")
    >>> results  = detector.execute(image_bgr, "a crack. a spall.")
    >>> detector.visualize()
    """

    _COLOURS_BGR = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
        (49, 130, 245), (180, 30, 145), (240, 240, 240), (128, 0, 0),
        (0, 128, 128), (203, 192, 255), (144, 238, 144), (130, 0, 75),
        (60, 20, 220), (0, 194, 255), (128, 128, 0),
    ]

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        text_threshold: float = 0.1,
        box_threshold: float = 0.1,
        use_sahi: bool = False,
        slice_width: int = 640,
        slice_height: int = 640,
        overlap: float = 0.25,
        nms_threshold: float = 0.1,
        full_image_also: bool = True,
    ):
        if not isinstance(model_name, str):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'model_name' expected str, got {type(model_name).__name__}"
            )
        if not isinstance(text_threshold, float):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'text_threshold' expected float, got {type(text_threshold).__name__}"
            )
        if not isinstance(box_threshold, float):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'box_threshold' expected float, got {type(box_threshold).__name__}"
            )
        if not isinstance(use_sahi, bool):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'use_sahi' expected bool, got {type(use_sahi).__name__}"
            )
        if not isinstance(slice_width, int):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'slice_width' expected int, got {type(slice_width).__name__}"
            )
        if not isinstance(slice_height, int):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'slice_height' expected int, got {type(slice_height).__name__}"
            )
        if not isinstance(overlap, float):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'overlap' expected float, got {type(overlap).__name__}"
            )
        if not isinstance(nms_threshold, float):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'nms_threshold' expected float, got {type(nms_threshold).__name__}"
            )
        if not isinstance(full_image_also, bool):
            raise TypeError(
                f"Type validation failed in GroundingDINODetector.__init__: "
                f"parameter 'full_image_also' expected bool, got {type(full_image_also).__name__}"
            )

        super().__init__()

        self._params = {
            "model_name": model_name,
            "text_threshold": text_threshold,
            "box_threshold": box_threshold,
            "use_sahi": use_sahi,
            "slice_width": slice_width,
            "slice_height": slice_height,
            "overlap": overlap,
            "nms_threshold": nms_threshold,
            "full_image_also": full_image_also,
        }

        self._initialize()

    def _initialize(self):
        model_name = self._params["model_name"]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{self._class_name}._initialize] Loading model: {model_name}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name
        ).to(self._device)
        print(
            f"[{self._class_name}._initialize] Model loaded on {self._device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, image: np.ndarray, text: str):
        """
        Run zero-shot object detection on *image*.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (OpenCV convention).
        text : str
            Dot-separated, lowercase text queries
            (e.g. ``"a crack. a spall."``).

        Returns
        -------
        list[dict]
            COCO-like annotation dictionaries sorted by area (descending).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'image' expected np.ndarray, got {type(image).__name__}"
            )
        if not isinstance(text, str):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'text' expected str, got {type(text).__name__}"
            )

        print(f"[{self._class_name}.execute] Executing GroundingDINODetector.")
        self._visualization_images = []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        if self._params["use_sahi"]:
            boxes, scores, labels = self._detect_sahi(pil_image, text)
        else:
            boxes, scores, labels = self._detect_full(pil_image, text)

        annotations = []
        label_to_cat_id: dict[str, int] = {}

        for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            x0, y0, x1, y1 = box.tolist()
            w = x1 - x0
            h = y1 - y0
            area = w * h
            if area <= 0:
                print(
                    f"[Warning] [{self._class_name}.execute] Skipping invalid bbox "
                    f"with area={area}"
                )
                continue

            if label not in label_to_cat_id:
                label_to_cat_id[label] = len(label_to_cat_id)

            annotation = self._generate_object_detection_annotation_dict(
                id=len(annotations),
                image_id=0,
                category_id=label_to_cat_id[label],
                bbox=[float(x0), float(y0), float(w), float(h)],
                area=float(area),
                score=float(score),
            )
            annotation["description"] = label
            annotations.append(annotation)

        annotations.sort(
            key=lambda a: a.get("area", 0.0) or 0.0, reverse=True
        )
        for idx, ann in enumerate(annotations):
            ann["id"] = idx

        self._create_visualization_image(image, annotations)

        self._results = annotations
        print(
            f"[{self._class_name}.execute] Detected {len(annotations)} object(s)."
        )
        return self._results

    # ------------------------------------------------------------------
    # Internal - full-image inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _detect_full(self, pil_image: Image.Image, text: str):
        """Run Grounding DINO on the full image."""
        inputs = self._processor(
            images=pil_image, text=text, return_tensors="pt"
        ).to(self._device)
        outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._params["box_threshold"],
            text_threshold=self._params["text_threshold"],
            target_sizes=[pil_image.size[::-1]],
        )[0]
        return result["boxes"], result["scores"], result["labels"]

    # ------------------------------------------------------------------
    # Internal - SAHI sliced inference
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_slices(img_w, img_h, slice_w, slice_h, overlap):
        """Return ``(x0, y0, x1, y1)`` crop boxes that tile the full image."""
        step_x = max(1, int(slice_w * (1 - overlap)))
        step_y = max(1, int(slice_h * (1 - overlap)))

        slices = []
        for y0 in range(0, img_h, step_y):
            for x0 in range(0, img_w, step_x):
                x1 = min(x0 + slice_w, img_w)
                y1 = min(y0 + slice_h, img_h)
                if x1 - x0 < slice_w and x0 > 0:
                    x0 = max(0, x1 - slice_w)
                if y1 - y0 < slice_h and y0 > 0:
                    y0 = max(0, y1 - slice_h)
                slices.append((x0, y0, x1, y1))
        return list(dict.fromkeys(slices))

    @torch.no_grad()
    def _detect_sahi(self, pil_image: Image.Image, text: str):
        """SAHI-style sliced inference with optional full-image pass + NMS."""
        all_boxes, all_scores, all_labels = [], [], []

        if self._params["full_image_also"]:
            boxes, scores, labels = self._detect_full(pil_image, text)
            if len(boxes):
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.extend(labels)

        img_w, img_h = pil_image.size
        slices = self._generate_slices(
            img_w,
            img_h,
            self._params["slice_width"],
            self._params["slice_height"],
            self._params["overlap"],
        )
        print(
            f"[{self._class_name}] SAHI: running on {len(slices)} slices "
            f"({self._params['slice_width']}x{self._params['slice_height']}, "
            f"overlap={self._params['overlap']})"
        )

        for x0, y0, x1, y1 in slices:
            patch = pil_image.crop((x0, y0, x1, y1))
            boxes, scores, labels = self._detect_full(patch, text)
            if len(boxes) == 0:
                continue
            offset = torch.tensor(
                [x0, y0, x0, y0], device=boxes.device
            )
            all_boxes.append(boxes + offset)
            all_scores.append(scores)
            all_labels.extend(labels)

        if not all_boxes:
            return (
                torch.zeros((0, 4)),
                torch.zeros((0,)),
                [],
            )

        merged_boxes = torch.cat(all_boxes)
        merged_scores = torch.cat(all_scores)

        keep = torchvision.ops.nms(
            merged_boxes, merged_scores, self._params["nms_threshold"]
        )
        kept_labels = [all_labels[i] for i in keep.tolist()]
        return merged_boxes[keep], merged_scores[keep], kept_labels

    # ------------------------------------------------------------------
    # Internal - visualization
    # ------------------------------------------------------------------
    def _create_visualization_image(
        self, image: np.ndarray, annotations: list
    ):
        """Draw bounding boxes with labels and scores onto *image* (BGR)."""
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}._create_visualization_image: "
                f"parameter 'image' expected np.ndarray, got {type(image).__name__}"
            )
        if not isinstance(annotations, list):
            raise TypeError(
                f"Type validation failed in {self._class_name}._create_visualization_image: "
                f"parameter 'annotations' expected list, got {type(annotations).__name__}"
            )

        vis = image.copy()

        unique_labels = list(
            dict.fromkeys(a.get("description", "") for a in annotations)
        )
        colour_map = {
            lbl: self._COLOURS_BGR[i % len(self._COLOURS_BGR)]
            for i, lbl in enumerate(unique_labels)
        }

        for ann in annotations:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            label = ann.get("description", "")
            score = ann.get("score")
            colour = colour_map.get(label, (0, 255, 0))

            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            text = f"{label} {score:.2f}" if score is not None else label
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                text, font_face, font_scale, thickness
            )
            cv2.rectangle(
                vis,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                vis,
                text,
                (x1 + 2, y1 - baseline - 2),
                font_face,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        self._visualization_images.append(vis)

class QWENVDetector(BaseShapeDetector):
    _COLOURS_BGR = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
        (49, 130, 245), (180, 30, 145), (240, 240, 240), (128, 0, 0),
        (0, 128, 128), (203, 192, 255), (144, 238, 144), (130, 0, 75),
        (60, 20, 220), (0, 194, 255), (128, 128, 0),
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        text_threshold: float = 0.1,
        box_threshold: float = 0.1,
    ):
        if not isinstance(model_name, str):
            raise TypeError(
                f"Type validation failed in QWENVDetector.__init__: "
                f"parameter 'model_name' expected str, got {type(model_name).__name__}"
            )
        if not isinstance(text_threshold, float):
            raise TypeError(
                f"Type validation failed in QWENVDetector.__init__: "
                f"parameter 'text_threshold' expected float, got {type(text_threshold).__name__}"
            )
        if not isinstance(box_threshold, float):
            raise TypeError(
                f"Type validation failed in QWENVDetector.__init__: "
                f"parameter 'box_threshold' expected float, got {type(box_threshold).__name__}"
            )

        super().__init__()

        self._params = {
            "model_name": model_name,
            "text_threshold": text_threshold,
            "box_threshold": box_threshold,
        }

        self._initialize()

    def _initialize(self):
        model_name = self._params["model_name"]
        print(f"[{self._class_name}._initialize] Loading model: {model_name}")
        # os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                # torch_dtype="auto",
                # device_map="auto",
            )
        except Exception:
            print(f"[{self._class_name}._initialize] Network unavailable, loading model from local cache")
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                # torch_dtype="auto",
                # device_map="auto",
                local_files_only=True,
            )
        self._model_on_gpu = False
        self._device = next(self._model.parameters()).device
        if self._device.type == "cuda":
            print(f"[{self._class_name}._initialize] Model is on GPU")
            self._model_on_gpu = True

        try:
            self._processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            print(f"[{self._class_name}._initialize] Network unavailable, loading processor from local cache")
            self._processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        print(f"[{self._class_name}._initialize] Model loaded on {self._device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, image: np.ndarray, text: str):
        """
        Run zero-shot object detection on *image*.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (OpenCV convention).
        text : str
            Lowercase text queries
            
        Returns
        -------
        list[dict]
            COCO-like annotation dictionaries sorted by area (descending).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'image' expected np.ndarray, got {type(image).__name__}"
            )
        if not isinstance(text, str):
            raise TypeError(
                f"Type validation failed in {self._class_name}.execute: "
                f"parameter 'text' expected str, got {type(text).__name__}"
            )

        print(f"[{self._class_name}.execute] Executing QWENVDetector.")
        self._visualization_images = []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        h, w = image.shape[:2]

        # Build a detection prompt asking for structured JSON output
        detection_prompt = (
            f"Detect all instances of \"{text}\" in this image. "
            f"The image is {w}x{h} pixels. "
            "For each detected object, output a JSON object with "
            "\"label\" (a short description string) and "
            "\"bbox\" (an array of [x_min, y_min, x_max, y_max] in pixel coordinates). "
            "Output ONLY a JSON array of detections, nothing else. "
            "If nothing is detected, output an empty array []."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": detection_prompt},
                ],
            }
        ]

        # Apply chat template
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info (images/videos from messages)
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [pil_image]
            video_inputs = None

        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"[{self._class_name}.execute] Raw model output: {output_text[:500]}")

        # Parse detections from model output
        annotations = self._parse_detection_output(output_text, w, h)

        # Sort by area descending and re-index
        annotations.sort(
            key=lambda a: a.get("area", 0.0) or 0.0, reverse=True
        )
        for idx, ann in enumerate(annotations):
            ann["id"] = idx

        self._create_visualization_image(image, annotations)

        self._results = annotations
        print(
            f"[{self._class_name}.execute] Detected {len(annotations)} object(s)."
        )
        return self._results

    # ------------------------------------------------------------------
    # Internal - parse model output
    # ------------------------------------------------------------------
    def _parse_detection_output(
        self, output_text: str, img_w: int, img_h: int
    ) -> List[Dict[str, Any]]:
        """
        Parse the VLM's free-form text output into COCO-like annotation dicts.

        Tries JSON extraction first, then falls back to regex-based parsing.
        """
        annotations: List[Dict[str, Any]] = []
        detections: list = []

        # --- Strategy 1: extract a JSON array from the text ---------------
        # Find the outermost [ ... ] in the output
        json_match = re.search(r"\[.*\]", output_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    detections = parsed
            except json.JSONDecodeError:
                pass

        # --- Strategy 2: try parsing individual JSON objects --------------
        if not detections:
            for obj_match in re.finditer(
                r"\{[^{}]*\}", output_text, re.DOTALL
            ):
                try:
                    obj = json.loads(obj_match.group())
                    if "bbox" in obj:
                        detections.append(obj)
                except json.JSONDecodeError:
                    continue

        # --- Strategy 3: regex for coordinate patterns --------------------
        if not detections:
            # Match patterns like (x1, y1, x2, y2) or [x1, y1, x2, y2]
            coord_pattern = re.compile(
                r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,"
                r"\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]"
            )
            for m in coord_pattern.finditer(output_text):
                x1, y1, x2, y2 = (
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                )
                detections.append({"label": "object", "bbox": [x1, y1, x2, y2]})

        # --- Build COCO-like annotations from parsed detections -----------
        for det in detections:
            bbox = det.get("bbox") or det.get("bounding_box") or det.get("box")
            if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue

            # If coordinates are normalised to [0, 1000], convert to pixels
            if all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
                max_coord = max(x1, y1, x2, y2)
                if max_coord <= 1.0:
                    # Normalised [0, 1]
                    x1 *= img_w
                    y1 *= img_h
                    x2 *= img_w
                    y2 *= img_h
                elif max_coord <= 1000:
                    # Qwen-VL style [0, 1000]
                    x1 = x1 / 1000.0 * img_w
                    y1 = y1 / 1000.0 * img_h
                    x2 = x2 / 1000.0 * img_w
                    y2 = y2 / 1000.0 * img_h

            # Clamp to image bounds
            x1 = max(0.0, min(x1, float(img_w)))
            y1 = max(0.0, min(y1, float(img_h)))
            x2 = max(0.0, min(x2, float(img_w)))
            y2 = max(0.0, min(y2, float(img_h)))

            coco_w = x2 - x1
            coco_h = y2 - y1
            area = coco_w * coco_h
            if area <= 0:
                continue

            label = det.get("label", "object")
            score = det.get("score") or det.get("confidence")
            if score is not None:
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = None

            annotation = self._generate_object_detection_annotation_dict(
                id=len(annotations),
                image_id=0,
                category_id=0,
                bbox=[float(x1), float(y1), float(coco_w), float(coco_h)],
                area=float(area),
                score=score,
            )
            annotation["description"] = str(label)
            annotations.append(annotation)

        return annotations

    # ------------------------------------------------------------------
    # Internal - visualization
    # ------------------------------------------------------------------
    def _create_visualization_image(
        self, image: np.ndarray, annotations: list
    ):
        """Draw bounding boxes with labels and scores onto *image* (BGR)."""
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Type validation failed in {self._class_name}._create_visualization_image: "
                f"parameter 'image' expected np.ndarray, got {type(image).__name__}"
            )
        if not isinstance(annotations, list):
            raise TypeError(
                f"Type validation failed in {self._class_name}._create_visualization_image: "
                f"parameter 'annotations' expected list, got {type(annotations).__name__}"
            )

        vis = image.copy()

        unique_labels = list(
            dict.fromkeys(a.get("description", "") for a in annotations)
        )
        colour_map = {
            lbl: self._COLOURS_BGR[i % len(self._COLOURS_BGR)]
            for i, lbl in enumerate(unique_labels)
        }

        for ann in annotations:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)

            label = ann.get("description", "")
            score = ann.get("score")
            colour = colour_map.get(label, (0, 255, 0))

            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            text = f"{label} {score:.2f}" if score is not None else label
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                text, font_face, font_scale, thickness
            )
            cv2.rectangle(
                vis,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                vis,
                text,
                (x1 + 2, y1 - baseline - 2),
                font_face,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        self._visualization_images.append(vis)