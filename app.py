"""
VisionGuard AI: Integrated Road Scene Understanding and Hazard Analysis System
==============================================================================
A beginner-friendly Streamlit dashboard that demonstrates many Computer Vision
ideas on road images and videos.

HOW THE APP IS ORGANISED
  * CORE (always works, needs only Streamlit, OpenCV, NumPy, Pillow):
      - image processing (grayscale, blur, edges)
      - image-patch visualisation and forward-diffusion noising demo
      - video frame sampling with a simple OpenCV motion detector
      - robustness comparison of image statistics
  * OPTIONAL (need extra libraries and model weights that download on demand):
      - YOLO object detection and tracking   (package: ultralytics)
      - DINOv2 embeddings and CLIP matching  (packages: torch, transformers)
      - Segment Anything style segmentation  (package: ultralytics)
      - Diffusion inpainting                 (packages: diffusers, torch, transformers)

NOTHING is downloaded when the app starts. A model is loaded only when you press
its "Load" button, and loaded models are cached so they are not loaded twice.

IMPORTANT HONESTY NOTES
  * The default detector is trained on the COCO dataset (80 everyday classes).
    It has NO pothole or road-damage class, and this app never claims otherwise.
  * "Hazard" information in this app is a simple, unvalidated heuristic summary
    (object counts and how large objects appear in the frame). It is NOT a safety
    system and must not be used for real driving or safety decisions.
  * No accuracy, precision, recall or F1 value is ever shown, because that needs
    labelled ground-truth data, which this app does not have.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import io
import os
import tempfile
import time
import traceback
import zlib
from collections import Counter
from typing import Any, Callable, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ----------------------------------------------------------------------------
# Settings (constants)
# ----------------------------------------------------------------------------
MIN_SIDE = 64                      # images smaller than this are rejected
MAX_ORIGINAL_PIXELS = 50_000_000   # reject absurdly large images (50 megapixels)
MAX_VIDEO_MB = 200                 # reject videos larger than this
FONT = cv2.FONT_HERSHEY_SIMPLEX

YOLO_CHOICES = ["yolov8n.pt", "yolo11n.pt"]   # small COCO-trained detectors
SAM_CHOICES = ["mobile_sam.pt", "sam_b.pt"]   # promptable segmentation weights
DINO_MODEL_ID = "facebook/dinov2-small"       # small DINOv2 ViT (patch size 14)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_INPAINT_MODEL_ID = "sd-legacy/stable-diffusion-inpainting"

PIP_HINTS = {
    "ultralytics": "pip install ultralytics lapx",
    "transformers": "pip install torch transformers",
    "diffusers": "pip install torch transformers diffusers accelerate",
}

# The 80 classes of the COCO dataset (what the default YOLO weights can detect)
COCO_CLASSES = (
    "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, "
    "fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, "
    "elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, "
    "skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, "
    "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, "
    "apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, "
    "potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, "
    "cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, "
    "scissors, teddy bear, hair drier, toothbrush"
)
VULNERABLE_CLASSES = ("person", "bicycle", "motorcycle")
VEHICLE_CLASSES = ("car", "bus", "truck", "train")
TRAFFIC_CONTROL_CLASSES = ("traffic light", "stop sign")

DEFAULT_CLIP_PROMPTS = "\n".join([
    "a photo of a busy city street with many cars",
    "a photo of an empty highway",
    "a photo of a road at night",
    "a photo of a road in heavy rain or fog",
    "a photo of pedestrians crossing a road",
    "a photo of a traffic jam",
])


class ImageProblem(Exception):
    """Raised when an uploaded image cannot be used. The message is shown to the user."""


# ----------------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------------
def has_module(name: str) -> bool:
    """True if a Python package is installed (without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def describe_load_error(exc: Exception, label: str, pip_hint: str) -> str:
    """Turn a technical exception into a friendly message for the user."""
    technical = f"{type(exc).__name__}: {exc}"
    lowered = str(exc).lower()
    if isinstance(exc, ImportError):
        return (f"{label} needs an optional library that is missing or failed to import. "
                f"Install it with:  {pip_hint}  (then restart the app). Details: {technical}")
    if (isinstance(exc, MemoryError) or "out of memory" in lowered
            or "not enough memory" in lowered or "paging file" in lowered):
        return (f"Not enough memory to load {label}. Close other programs, or use a smaller "
                f"model. Details: {technical}")
    network_words = ("connection", "timed out", "timeout", "resolve", "offline",
                     "max retries", "urlopen", "ssl", "http", "404", "401", "403",
                     "download", "network")
    if any(word in lowered for word in network_words):
        return (f"{label} could not be downloaded. Check your internet connection, firewall "
                f"or proxy, and try again. (Do not disable SSL verification.) "
                f"Details: {technical}")
    return f"{label} could not be loaded. Details: {technical}"


def request_model(key: str, loader: Callable[..., Any], args: tuple, label: str,
                  pip_hint: str) -> bool:
    """
    Load a model because the user pressed a button. On success, remember which
    arguments were used in st.session_state so the cached model can be reused.
    """
    try:
        with st.spinner(f"Loading {label}... the first use may download files. Please wait."):
            loader(*args)
    except Exception as exc:  # noqa: BLE001 - we want to show ANY loading problem nicely
        st.session_state.pop(key, None)
        st.error(describe_load_error(exc, label, pip_hint))
        return False
    st.session_state[key] = args
    return True


def get_loaded(key: str, loader: Callable[..., Any]) -> Any:
    """Return an already-requested model (from the cache) or None."""
    args = st.session_state.get(key)
    if args is None:
        return None
    try:
        return loader(*args)
    except Exception:  # noqa: BLE001
        st.session_state.pop(key, None)
        return None


def run_safely(section_name: str, func: Callable[..., Any], *args: Any) -> None:
    """Run one dashboard section so that an unexpected error cannot crash the whole app."""
    try:
        func(*args)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The '{section_name}' section hit an unexpected error: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


# ----------------------------------------------------------------------------
# Image loading and validation
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def load_image_cached(file_bytes: bytes, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Read uploaded bytes into an RGB NumPy array, validating and resizing safely.

    Returns (rgb_array, (original_width, original_height)).
    Raises ImageProblem with a friendly message if the file cannot be used.
    """
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        width, height = pil_image.size
        if width * height > MAX_ORIGINAL_PIXELS:
            raise ImageProblem(f"The image is too large ({width} x {height} pixels). "
                               "Please upload a photo under 50 megapixels.")
        pil_image = ImageOps.exif_transpose(pil_image)   # fix phone-photo rotation
        pil_image = pil_image.convert("RGB")
    except ImageProblem:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as error:
        raise ImageProblem("This file could not be read as an image. It may be corrupted "
                           "or not a real JPG/PNG file.") from error

    width, height = pil_image.size
    if min(width, height) < MIN_SIDE:
        raise ImageProblem(f"The image is too small ({width} x {height} pixels). "
                           f"Both sides must be at least {MIN_SIDE} pixels.")

    rgb = np.array(pil_image, dtype=np.uint8)
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    return rgb, (width, height)


def load_uploaded_image(uploaded: Any, max_side: int) -> Optional[tuple[np.ndarray, tuple[int, int]]]:
    """Wrapper that shows errors in the sidebar and never raises."""
    if uploaded is None:
        return None
    try:
        return load_image_cached(uploaded.getvalue(), max_side)
    except ImageProblem as problem:
        st.sidebar.error(str(problem))
    except Exception:  # noqa: BLE001
        st.sidebar.error("Something went wrong while opening the image. Try another file.")
    return None


# ----------------------------------------------------------------------------
# MODULE 1: classical image processing (OpenCV)
# ----------------------------------------------------------------------------
def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """RGB -> single-channel grayscale (weighted sum of R, G, B)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(rgb: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Gaussian blur. The kernel size must be odd; sigma 0 lets OpenCV choose it."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(rgb, (kernel_size, kernel_size), sigmaX=float(sigma))


def detect_edges(rgb: np.ndarray, method: str, low: int, high: int,
                 pre_blur: bool) -> np.ndarray:
    """Edge detection with Canny or Sobel. Returns a uint8 grayscale edge image."""
    gray = to_grayscale(rgb)
    if pre_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if method == "Canny":
        return cv2.Canny(gray, int(low), int(high))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    peak = float(magnitude.max())
    if peak <= 0:
        return np.zeros_like(gray)
    return (magnitude / peak * 255.0).astype(np.uint8)


def adjust_brightness(rgb: np.ndarray, beta: int) -> np.ndarray:
    """Add a constant to every pixel (clipped to 0..255). Negative = darker."""
    return cv2.convertScaleAbs(rgb, alpha=1.0, beta=float(beta))


def downscale_and_restore(rgb: np.ndarray, percent: int) -> np.ndarray:
    """Shrink the image to `percent` of its size, then enlarge it back (loses detail)."""
    height, width = rgb.shape[:2]
    percent = int(np.clip(percent, 5, 100))
    small_w = max(int(width * percent / 100), 8)
    small_h = max(int(height * percent / 100), 8)
    small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def sharpness_score(rgb: np.ndarray) -> float:
    """Variance of the Laplacian: a common simple sharpness measure (higher = sharper)."""
    return float(cv2.Laplacian(to_grayscale(rgb), cv2.CV_64F).var())


def mean_brightness(rgb: np.ndarray) -> float:
    """Average grayscale value (0 = black, 255 = white)."""
    return float(to_grayscale(rgb).mean())


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def color_for_label(label: str) -> tuple[int, int, int]:
    """A stable colour for each class name (same label = same colour every time)."""
    value = zlib.crc32(label.encode("utf-8"))
    return (55 + value % 170, 55 + (value // 7) % 170, 55 + (value // 49) % 170)


def draw_detections(rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw boxes, labels, confidence scores and track IDs on a copy of the image."""
    out = rgb.copy()
    short_side = min(out.shape[:2])
    thickness = max(2, short_side // 300)
    scale = max(0.5, short_side / 900)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        color = color_for_label(det["label"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = det["label"]
        if det.get("id") is not None:
            text = f"#{det['id']} {text}"
        if det.get("conf") is not None:
            text += f" {det['conf']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 1)
        top = max(y1 - text_h - baseline - 2, 0)
        cv2.rectangle(out, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 2), color, -1)
        cv2.putText(out, text, (x1 + 2, top + text_h), FONT, scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0),
                 alpha: float = 0.5) -> np.ndarray:
    """Colour a boolean mask on top of the image and draw its outline."""
    out = rgb.copy()
    selected = mask > 0
    if selected.any():
        blended = rgb[selected].astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha
        out[selected] = blended.astype(np.uint8)
        contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def draw_patch_grid(rgb: np.ndarray, patch_size: int = 14, image_size: int = 224,
                    zoom: int = 3) -> np.ndarray:
    """Show how a Vision Transformer cuts an image into square patches."""
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    big = cv2.resize(resized, (image_size * zoom, image_size * zoom),
                     interpolation=cv2.INTER_NEAREST)
    step = patch_size * zoom
    for position in range(0, image_size * zoom + 1, step):
        cv2.line(big, (position, 0), (position, image_size * zoom - 1), (255, 255, 0), 1)
        cv2.line(big, (0, position), (image_size * zoom - 1, position), (255, 255, 0), 1)
    return big


# ----------------------------------------------------------------------------
# MODULE 2 / 7 / 8: detection helpers (ultralytics YOLO, optional)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_name: str, purpose: str) -> Any:
    """
    Load YOLO weights. `purpose` ("detect" or "track") gives separate model objects,
    because tracking keeps internal state that should not leak into plain detection.
    Weights download automatically the first time (about 6 MB for the nano models).
    """
    from ultralytics import YOLO  # imported here so the app works without ultralytics
    return YOLO(weights_name)


def parse_ultralytics_result(result: Any) -> list[dict[str, Any]]:
    """Convert one ultralytics result into a simple list of dictionaries."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    names = result.names
    detections = []
    for i in range(len(xyxy)):
        detections.append({
            "label": str(names[int(class_ids[i])]),
            "conf": float(confidences[i]),
            "box": tuple(float(v) for v in xyxy[i]),
            "id": int(track_ids[i]) if track_ids is not None else None,
        })
    return detections


def run_detection(model: Any, rgb: np.ndarray, conf: float, imgsz: int
                  ) -> tuple[list[dict[str, Any]], float]:
    """Run the detector once. Returns (detections, inference time in milliseconds)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # ultralytics expects OpenCV's BGR order
    start = time.perf_counter()
    result = model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return parse_ultralytics_result(result), elapsed_ms


@st.cache_data(show_spinner=False, max_entries=16)
def cached_detection(_model: Any, weights_name: str, rgb: np.ndarray, conf: float,
                     imgsz: int) -> tuple[list[dict[str, Any]], float]:
    """Cached wrapper so moving unrelated widgets does not re-run the detector."""
    return run_detection(_model, rgb, conf, imgsz)


def summarize_scene(detections: list[dict[str, Any]], image_shape: tuple[int, ...]) -> dict[str, Any]:
    """
    A simple, UNVALIDATED summary of what was detected. It only counts objects and
    checks how much of the frame each box covers. It does not measure distance or risk.
    """
    counts = Counter(d["label"] for d in detections)
    frame_area = float(image_shape[0] * image_shape[1])
    large_objects = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        fraction = max(x2 - x1, 0) * max(y2 - y1, 0) / frame_area
        if fraction >= 0.15:
            large_objects.append((det["label"], fraction))
    return {
        "counts": counts,
        "vulnerable": sum(counts[c] for c in VULNERABLE_CLASSES),
        "vehicles": sum(counts[c] for c in VEHICLE_CLASSES),
        "traffic_control": sum(counts[c] for c in TRAFFIC_CONTROL_CLASSES),
        "large_objects": large_objects,
    }


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection-over-Union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def count_matches(reference: list[dict[str, Any]], other: list[dict[str, Any]],
                  iou_threshold: float = 0.5) -> int:
    """
    How many detections in `other` agree with `reference` (same class, IoU >= threshold,
    each reference box used once). This measures CONSISTENCY, not correctness.
    """
    used: set[int] = set()
    matches = 0
    for det in other:
        best_index, best_iou = -1, 0.0
        for index, ref in enumerate(reference):
            if index in used or ref["label"] != det["label"]:
                continue
            value = box_iou(det["box"], ref["box"])
            if value > best_iou:
                best_index, best_iou = index, value
        if best_index >= 0 and best_iou >= iou_threshold:
            used.add(best_index)
            matches += 1
    return matches


# ----------------------------------------------------------------------------
# MODULE 3 / 4: DINOv2 and CLIP (torch + transformers, optional)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_dino_model(model_id: str) -> Any:
    """Load a pretrained DINOv2 backbone (about 90 MB for the small version)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_clip_model(model_id: str) -> tuple[Any, Any]:
    """Load a pretrained CLIP model and its processor (roughly 600 MB)."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def extract_dino_features(model: Any, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run DINOv2 on the image resized to 224 x 224 (a 16 x 16 grid of 14-pixel patches).

    Returns (cls_embedding, patch_embeddings, grid_side).
      cls_embedding    - one vector describing the whole image
      patch_embeddings - one vector per patch (grid_side * grid_side vectors)
    """
    import torch
    size, patch = 224, 14
    grid = size // patch
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        output = model(pixel_values=tensor)
    tokens = output.last_hidden_state[0].cpu().numpy()   # (1 + patches, dimension)
    return tokens[0], tokens[-(grid * grid):], grid


@st.cache_data(show_spinner=False, max_entries=8)
def cached_dino_features(_model: Any, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Cached wrapper around extract_dino_features."""
    return extract_dino_features(_model, rgb)


def pca_feature_image(patch_embeddings: np.ndarray, grid: int, out_size: tuple[int, int]) -> np.ndarray:
    """
    Compress each patch vector to 3 numbers with PCA and show them as RGB colours.
    Patches that look alike to DINOv2 get similar colours. (The colours themselves
    have no fixed meaning; only the grouping matters.)
    """
    centered = patch_embeddings - patch_embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T                       # (patches, 3)
    for channel in range(3):
        low, high = np.percentile(projected[:, channel], [2, 98])
        projected[:, channel] = np.clip((projected[:, channel] - low) / (high - low + 1e-8), 0, 1)
    small = (projected.reshape(grid, grid, 3) * 255).astype(np.uint8)
    return cv2.resize(small, out_size, interpolation=cv2.INTER_NEAREST)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors: 1 = same direction, 0 = unrelated, -1 = opposite."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def clip_compare(model: Any, processor: Any, rgb: np.ndarray, prompts: tuple[str, ...]
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compare one image with several text prompts using CLIP.

    Returns (cosine_similarities, relative_scores, seconds).
      cosine_similarities - raw image-text similarity for each prompt
      relative_scores     - softmax across THESE prompts only (adds up to 1.0).
                            This is NOT a real-world probability.
    """
    import torch
    start = time.perf_counter()
    inputs = processor(text=list(prompts), images=Image.fromarray(rgb), return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    image_vec = output.image_embeds[0]
    text_vecs = output.text_embeds
    image_vec = image_vec / image_vec.norm()
    text_vecs = text_vecs / text_vecs.norm(dim=-1, keepdim=True)
    cosines = (text_vecs @ image_vec).cpu().numpy()
    logits = float(model.logit_scale.exp().item()) * cosines
    exp = np.exp(logits - logits.max())
    return cosines, exp / exp.sum(), time.perf_counter() - start


@st.cache_data(show_spinner=False, max_entries=8)
def cached_clip_compare(_bundle: Any, rgb: np.ndarray, prompts: tuple[str, ...]
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Cached wrapper around clip_compare."""
    model, processor = _bundle
    return clip_compare(model, processor, rgb, prompts)


# ----------------------------------------------------------------------------
# MODULE 5: diffusion (forward noising is plain NumPy; inpainting is optional)
# ----------------------------------------------------------------------------
def alpha_bar(t: int, total_steps: int = 1000, beta_start: float = 1e-4,
              beta_end: float = 0.02) -> float:
    """
    Fraction of the original signal left after t forward-diffusion steps, using the
    common linear noise schedule. alpha_bar(0) = 1 (clean); near 0 = pure noise.
    """
    if t <= 0:
        return 1.0
    betas = np.linspace(beta_start, beta_end, total_steps)
    return float(np.cumprod(1.0 - betas)[min(t, total_steps) - 1])


def forward_noise(rgb: np.ndarray, t: int, seed: int) -> np.ndarray:
    """
    The FORWARD diffusion step:  x_t = sqrt(a) * x_0 + sqrt(1 - a) * noise
    where a = alpha_bar(t). This only ADDS noise; it does not restore anything.
    """
    a = alpha_bar(t)
    x0 = rgb.astype(np.float32) / 127.5 - 1.0               # scale pixels to -1..1
    noise = np.random.default_rng(seed).standard_normal(x0.shape).astype(np.float32)
    xt = np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise
    return np.clip((xt + 1.0) * 127.5, 0, 255).astype(np.uint8)


@st.cache_resource(show_spinner=False)
def load_inpaint_pipeline(model_id: str) -> tuple[Any, str]:
    """
    Load a pretrained Stable Diffusion inpainting pipeline. THIS IS A MULTI-GIGABYTE
    DOWNLOAD and is only called when the user explicitly asks for it.
    """
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    device = "cuda" if use_cuda else "cpu"
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()          # lowers memory use
    return pipe, device


def inpaint_region(bundle: tuple[Any, str], rgb: np.ndarray, box_pct: tuple[int, int, int, int],
                   prompt: str, steps: int, guidance: float, seed: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill a rectangular region using real diffusion inference (iterative denoising).
    Returns (resized_input, mask, result). The filled content is GENERATED, so it is a
    plausible guess, not the true hidden content.
    """
    import torch
    pipe, _ = bundle
    height, width = rgb.shape[:2]
    scale = 512.0 / max(height, width)
    new_w = max(64, int(round(width * scale / 8.0)) * 8)
    new_h = max(64, int(round(height * scale / 8.0)) * 8)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    work = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)

    x_pct, y_pct, w_pct, h_pct = box_pct
    x0 = int(new_w * x_pct / 100.0)
    y0 = int(new_h * y_pct / 100.0)
    x1 = min(new_w, x0 + max(int(new_w * w_pct / 100.0), 8))
    y1 = min(new_h, y0 + max(int(new_h * h_pct / 100.0), 8))
    mask = np.zeros((new_h, new_w), np.uint8)
    mask[y0:y1, x0:x1] = 255                 # white = region to repaint

    generator = torch.Generator("cpu").manual_seed(int(seed))
    output = pipe(prompt=prompt, image=Image.fromarray(work), mask_image=Image.fromarray(mask),
                  height=new_h, width=new_w, num_inference_steps=int(steps),
                  guidance_scale=float(guidance), generator=generator).images[0]
    return work, mask, np.array(output.convert("RGB"))


# ----------------------------------------------------------------------------
# MODULE 6: promptable segmentation (ultralytics SAM, optional)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sam_model(weights_name: str) -> Any:
    """
    Load Segment Anything style weights through ultralytics. The checkpoint
    (mobile_sam.pt is about 40 MB, sam_b.pt about 375 MB) downloads on first use.
    """
    from ultralytics import SAM
    return SAM(weights_name)


def run_sam(model: Any, rgb: np.ndarray, mode: str, point: tuple[int, int],
            box: tuple[int, int, int, int]) -> tuple[Optional[np.ndarray], float]:
    """Segment with one point prompt or one box prompt. Returns (mask or None, seconds)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    start = time.perf_counter()
    if mode == "Point":
        results = model(bgr, points=[int(point[0]), int(point[1])], labels=[1], verbose=False)
    else:
        results = model(bgr, bboxes=[int(v) for v in box], verbose=False)
    elapsed = time.perf_counter() - start
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None, elapsed
    mask = result.masks.data[0].cpu().numpy() > 0.5
    height, width = rgb.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST) > 0
    return mask, elapsed


# ----------------------------------------------------------------------------
# MODULE 7: video processing
# ----------------------------------------------------------------------------
def resize_frame_even(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink a frame if it is large, and make width/height even (needed by video codecs)."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(int(width * scale), 2), max(int(height * scale), 2)),
                           interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


def open_video_writer(path: str, fps: float, size: tuple[int, int]) -> tuple[Any, str]:
    """Try an H.264 writer first (plays in browsers), then fall back to mp4v."""
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer, code
        writer.release()
    return None, ""


def detect_motion(subtractor: Any, bgr: np.ndarray, min_area_fraction: float = 0.002
                  ) -> list[dict[str, Any]]:
    """
    Very simple OpenCV motion detector (background subtraction). It finds MOVING regions.
    It does not know what the objects are and does not give track IDs.
    """
    mask = subtractor.apply(bgr)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)   # drop shadow pixels (127)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_fraction * bgr.shape[0] * bgr.shape[1]
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detections.append({"label": "motion", "conf": None, "box": (x, y, x + w, y + h), "id": None})
    return detections


def probe_video(path: str) -> dict[str, Any]:
    """Read basic video facts. Raises ValueError if OpenCV cannot open or read the file."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not open this video. Try a standard H.264 MP4 file.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("The video opened, but no frame could be read from it.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
        return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), "fps": float(fps),
                "width": int(frame.shape[1]), "height": int(frame.shape[0])}
    finally:
        cap.release()


def process_video(video_path: str, mode: str, model: Any, conf: float, imgsz: int, stride: int,
                  max_frames: int, max_side: int,
                  progress: Callable[[float, str], None]) -> dict[str, Any]:
    """
    Process a video ONE FRAME AT A TIME (the whole video is never loaded into memory).

    mode "yolo"   - YOLO detection + ByteTrack tracking (persistent track IDs)
    mode "motion" - OpenCV background subtraction (no AI, no IDs)
    stride        - process every `stride`-th frame (frame skipping saves time)
    """
    info = probe_video(video_path)
    total = info["frames"]
    planned = max_frames if total <= 0 else min(max_frames, (total + stride - 1) // stride)
    sample_every = max(1, planned // 6)
    out_fps = max(1.0, info["fps"] / stride)

    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = handle.name
    handle.close()

    cap = cv2.VideoCapture(video_path)
    writer = None
    codec = ""
    subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32,
                                                    detectShadows=True) if mode == "motion" else None
    class_track_ids: dict[str, set[int]] = {}
    class_frame_counts: Counter = Counter()
    per_frame_counts: list[int] = []
    samples: list[np.ndarray] = []
    processed = 0
    index = 0
    start = time.perf_counter()
    try:
        while processed < max_frames:
            wanted = index % stride == 0
            if wanted:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None      # skip without decoding fully
            index += 1
            if not ok:
                break
            if not wanted or frame is None:
                continue

            frame = resize_frame_even(frame, max_side)
            if mode == "yolo":
                result = model.track(frame, persist=processed > 0, conf=conf, imgsz=imgsz,
                                     tracker="bytetrack.yaml", verbose=False)[0]
                detections = parse_ultralytics_result(result)
            else:
                detections = detect_motion(subtractor, frame)

            annotated_rgb = draw_detections(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detections)
            if writer is None:
                writer, codec = open_video_writer(
                    out_path, out_fps, (annotated_rgb.shape[1], annotated_rgb.shape[0]))
            if writer is not None:
                writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

            for det in detections:
                class_frame_counts[det["label"]] += 1
                if det["id"] is not None:
                    class_track_ids.setdefault(det["label"], set()).add(det["id"])
            per_frame_counts.append(len(detections))
            if processed % sample_every == 0 and len(samples) < 6:
                samples.append(annotated_rgb)

            processed += 1
            fraction = min(processed / max(planned, 1), 1.0)
            progress(fraction, f"Processed {processed} of about {planned} frames")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start
    video_bytes = b""
    try:
        if processed > 0 and writer is not None and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as handle_in:
                video_bytes = handle_in.read()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

    return {"mode": mode, "processed": processed, "elapsed": elapsed, "codec": codec,
            "video_bytes": video_bytes, "samples": samples, "info": info, "stride": stride,
            "class_track_ids": {k: sorted(v) for k, v in class_track_ids.items()},
            "class_frame_counts": dict(class_frame_counts),
            "per_frame_counts": per_frame_counts, "out_fps": out_fps}


# ----------------------------------------------------------------------------
# UI building blocks
# ----------------------------------------------------------------------------
def need_image(image_data: Optional[tuple[np.ndarray, tuple[int, int]]]) -> bool:
    """Show a hint and return False when no image has been uploaded yet."""
    if image_data is None:
        st.info("Upload a road image in the sidebar (section 1) to use this module.")
        return False
    return True


def explain_missing_library(package: str) -> None:
    """Show install instructions for an optional library that is not installed."""
    st.warning(f"This optional feature needs the **{package}** library, which is not installed. "
               f"Install it in your virtual environment with:  `{PIP_HINTS[package]}`  "
               "and restart the app. The rest of the dashboard still works.")


def detector_ready() -> Optional[Any]:
    """The loaded detection model, or None."""
    return get_loaded("yolo_detect", load_yolo_model)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar() -> dict[str, Any]:
    """Draw the sidebar and return the chosen settings."""
    st.sidebar.title("🛡️ VisionGuard AI")
    st.sidebar.caption("Road scene understanding demo. Educational use only, not a safety system.")

    st.sidebar.header("1. Inputs")
    image_file = st.sidebar.file_uploader("Road image (JPG, JPEG, PNG)",
                                          type=["jpg", "jpeg", "png"], key="image_upload")
    video_file = st.sidebar.file_uploader("Road video (MP4, MOV, AVI, MKV)",
                                          type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    max_side = st.sidebar.slider("Max image side for processing (px)", 320, 1600, 960, 80,
                                 help="Bigger images are shrunk to this size to keep the app fast.")

    st.sidebar.header("2. Object detector (optional)")
    weights = st.sidebar.selectbox("Pretrained YOLO weights", YOLO_CHOICES,
                                   help="Small models trained on COCO (80 classes). "
                                        "Downloaded only when you press the Load button.")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.sidebar.select_slider("Inference image size", options=[320, 480, 640], value=640,
                                     help="Smaller = faster but less accurate on small objects.")
    if not has_module("ultralytics"):
        st.sidebar.warning("`ultralytics` is not installed, so detection, tracking and "
                           "segmentation are unavailable. See the Home tab.")
    elif st.sidebar.button("Load / switch detector"):
        request_model("yolo_detect", load_yolo_model, (weights, "detect"),
                      f"YOLO weights '{weights}'", PIP_HINTS["ultralytics"])
    loaded = st.session_state.get("yolo_detect")
    if loaded:
        note = "" if loaded[0] == weights else " (selection changed: press the button to switch)"
        st.sidebar.success(f"Detector loaded: {loaded[0]}{note}")
    elif has_module("ultralytics"):
        st.sidebar.info("No detector loaded yet.")

    st.sidebar.header("3. Library status")
    for package in ("ultralytics", "torch", "transformers", "diffusers"):
        st.sidebar.write(("✅ " if has_module(package) else "⬜ ") + package)

    return {"image_file": image_file, "video_file": video_file, "max_side": max_side,
            "weights": weights, "conf": conf, "imgsz": imgsz}


# ----------------------------------------------------------------------------
# Tab: Home
# ----------------------------------------------------------------------------
def render_home(settings: dict[str, Any]) -> None:
    st.title("🛡️ VisionGuard AI")
    st.subheader("Integrated Road Scene Understanding and Hazard Analysis (educational demo)")
    st.write(
        "Upload a road image or video in the sidebar and explore classical image processing, "
        "pretrained object detection, self-supervised features, image-text matching, "
        "diffusion, promptable segmentation, tracking and robustness testing.")
    st.warning(
        "**Please read:** the default detector knows only the 80 COCO classes. It has **no "
        "pothole or road-damage class**. The 'scene summary' is a simple heuristic, not a "
        "validated hazard or safety measure. No accuracy, precision, recall or F1 value is "
        "shown anywhere, because that needs labelled ground-truth data.")

    st.markdown("#### Workflow")
    st.code(
        "Upload image/video  ->  validate + resize safely\n"
        "      |-> Module 1  OpenCV filters (grayscale, blur, edges)\n"
        "      |-> Module 2  YOLO detection (optional model)\n"
        "      |-> Module 3  DINOv2 patch features + image similarity (optional model)\n"
        "      |-> Module 4  CLIP image-text matching (optional model)\n"
        "      |-> Module 5  Diffusion: forward-noise demo + optional inpainting\n"
        "      |-> Module 6  SAM-style segmentation from a point/box (optional model)\n"
        "      |-> Module 7  Video sampling + tracking (YOLO+ByteTrack) or motion-only\n"
        "      |-> Module 8  Robustness: blur / resize / brightness comparison\n"
        "      '-> Module 9  This dashboard", language="text")

    st.markdown("#### Module status on this computer")
    ultra = has_module("ultralytics")
    torch_stack = has_module("torch") and has_module("transformers")
    diffusion = has_module("diffusers") and has_module("torch")
    rows = [
        {"Module": "1 Image processing", "Needs": "OpenCV, NumPy", "Status": "Ready"},
        {"Module": "2 Object detection", "Needs": "ultralytics + YOLO weights",
         "Status": "Library found: load weights in sidebar" if ultra else "Optional: not installed"},
        {"Module": "3 DINOv2 features / similarity", "Needs": "torch, transformers",
         "Status": "Patch grid ready; model " + ("available" if torch_stack else "not installed")},
        {"Module": "4 CLIP text matching", "Needs": "torch, transformers",
         "Status": "Library found: press Load" if torch_stack else "Optional: not installed"},
        {"Module": "5 Diffusion", "Needs": "diffusers, torch (inpainting only)",
         "Status": "Forward-noise demo ready; inpainting " + ("available" if diffusion else "not installed")},
        {"Module": "6 Segmentation (SAM)", "Needs": "ultralytics + SAM checkpoint",
         "Status": "Library found: press Load" if ultra else "Optional: not installed"},
        {"Module": "7 Video", "Needs": "OpenCV (motion mode); ultralytics (tracking)",
         "Status": "Motion mode ready; tracking " + ("available" if ultra else "not installed")},
        {"Module": "8 Robustness", "Needs": "OpenCV; detector for detection comparison",
         "Status": "Image statistics ready; detection comparison " + ("available" if ultra else "not installed")},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("'Available' means the library is installed. Model weights are downloaded only "
               "when you press the corresponding Load button.")

    st.markdown("#### Quick start")
    st.markdown(
        "1. Upload an image in the sidebar.\n"
        "2. Try **Image processing** (works immediately).\n"
        "3. For detection, press **Load / switch detector** in the sidebar, then open **Object detection**.\n"
        "4. Optional modules each have their own Load button inside their tab.")


# ----------------------------------------------------------------------------
# Tab: Image processing (Module 1)
# ----------------------------------------------------------------------------
def render_image_processing(image_data: Optional[tuple[np.ndarray, tuple[int, int]]]) -> None:
    st.header("Module 1: Image processing")
    st.write("Classical image processing changes pixels with fixed mathematical rules (no AI "
             "model). These steps are often used before or alongside deep-learning models.")
    if not need_image(image_data):
        return
    rgb, original_size = image_data
    height, width = rgb.shape[:2]
    st.caption(f"Uploaded size: {original_size[0]} x {original_size[1]} px. "
               f"Processing size: {width} x {height} px.")

    left, right = st.columns(2)
    with left:
        blur_kernel = st.slider("Gaussian blur kernel size (odd number)", 1, 51, 9, 2)
        blur_sigma = st.slider("Blur sigma (0 = automatic)", 0.0, 15.0, 0.0, 0.5)
    with right:
        method = st.radio("Edge detector", ["Canny", "Sobel"], horizontal=True)
        low = st.slider("Canny low threshold", 0, 255, 80, disabled=method != "Canny")
        high = st.slider("Canny high threshold", 0, 255, 160, disabled=method != "Canny")
        pre_blur = st.checkbox("Blur slightly before finding edges (reduces noise)", value=True)
    if method == "Canny" and high <= low:
        st.warning("The high threshold should be larger than the low threshold. Using low + 1.")
        high = min(255, low + 1)

    gray = to_grayscale(rgb)
    blurred = apply_gaussian_blur(rgb, blur_kernel, blur_sigma)
    edges = detect_edges(rgb, method, low, high, pre_blur)

    row1 = st.columns(2)
    row1[0].image(rgb, caption="Original", use_container_width=True)
    row1[1].image(gray, caption="Grayscale", use_container_width=True)
    row2 = st.columns(2)
    row2[0].image(blurred, caption=f"Gaussian blur (kernel {blur_kernel}, sigma {blur_sigma})",
                  use_container_width=True)
    row2[1].image(edges, caption=f"{method} edges", use_container_width=True)

    with st.expander("What am I looking at?"):
        st.markdown(
            "- **Grayscale** keeps only brightness, which is simpler to analyse.\n"
            "- **Gaussian blur** averages each pixel with its neighbours (weighted like a bell "
            "curve). It removes noise but also fine detail.\n"
            "- **Canny edges** mark places where brightness changes sharply, using two thresholds. "
            "**Sobel** shows the strength of brightness change as a grey image.")


# ----------------------------------------------------------------------------
# Tab: Object detection (Module 2)
# ----------------------------------------------------------------------------
def render_detection(image_data: Optional[tuple[np.ndarray, tuple[int, int]]],
                     settings: dict[str, Any]) -> None:
    st.header("Module 2: Object detection (pretrained YOLO)")
    st.write("An object detector finds objects and draws a **bounding box**, a **class label** "
             "and a **confidence score** for each one.")
    with st.expander("Which classes can this model detect?", expanded=False):
        st.write("The default YOLO weights were trained on the **COCO** dataset, which has 80 "
                 "everyday classes:")
        st.caption(COCO_CLASSES)
        st.error("There is **no pothole, crack, lane-marking or road-damage class** in COCO. "
                 "This app will never claim to detect them. Detecting such things would need a "
                 "model trained and verified on labelled road-damage data.")

    if not has_module("ultralytics"):
        explain_missing_library("ultralytics")
        return
    model = detector_ready()
    if model is None:
        st.info("Press **Load / switch detector** in the sidebar (section 2). The small weights "
                "file downloads once.")
        return
    if not need_image(image_data):
        return

    rgb, _ = image_data
    weights_name = st.session_state["yolo_detect"][0]
    try:
        detections, elapsed_ms = cached_detection(model, weights_name, rgb, settings["conf"],
                                                  settings["imgsz"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Detection failed: {exc}")
        return

    annotated = draw_detections(rgb, detections)
    left, right = st.columns(2)
    left.image(rgb, caption="Original", use_container_width=True)
    right.image(annotated, caption=f"Detections (confidence >= {settings['conf']:.2f})",
                use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Objects detected", len(detections))
    m2.metric("Inference time (ms)", f"{elapsed_ms:.0f}")
    st.caption("Inference time depends on your computer. The first run after loading is often "
               "slower, and a cached result shows the time of the run that produced it.")

    if detections:
        table = [{"Class": d["label"], "Confidence": round(d["conf"], 3),
                  "Box (x1, y1, x2, y2)": ", ".join(str(int(v)) for v in d["box"])}
                 for d in sorted(detections, key=lambda d: -d["conf"])]
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("Nothing detected at this threshold. Try lowering the confidence threshold. "
                "Remember that objects outside the 80 COCO classes can never be detected.")

    summary = summarize_scene(detections, rgb.shape)
    st.markdown("#### Simple scene summary (heuristic, not validated)")
    s1, s2, s3 = st.columns(3)
    s1.metric("Vulnerable road users", summary["vulnerable"], help="person, bicycle, motorcycle")
    s2.metric("Vehicles", summary["vehicles"], help="car, bus, truck, train")
    s3.metric("Traffic lights / stop signs", summary["traffic_control"])
    if summary["large_objects"]:
        names = ", ".join(f"{label} ({fraction * 100:.0f}% of frame)"
                          for label, fraction in summary["large_objects"])
        st.warning(f"Objects covering a large part of the frame: {names}. A large box can mean "
                   "the object is close, but it can also just be a big object, so this is only a "
                   "rough hint and not a distance measurement.")
    st.caption("This summary only counts detections and compares box sizes with the frame. It "
               "does not predict collisions or measure real risk.")


# ----------------------------------------------------------------------------
# Tab: Embeddings, similarity, CLIP (Modules 3 and 4)
# ----------------------------------------------------------------------------
def render_embeddings(image_data: Optional[tuple[np.ndarray, tuple[int, int]]],
                      settings: dict[str, Any]) -> None:
    st.header("Modules 3 and 4: Visual embeddings, similarity and vision-language matching")

    st.subheader("Self-supervised learning and Vision Transformers")
    st.markdown(
        "**Self-supervised learning (SSL)** trains a model on lots of images **without human "
        "labels**. DINOv2 learns by comparing different views of the same image and by predicting "
        "hidden parts, so images and image parts that look alike end up with similar number "
        "vectors (**embeddings**). A **Vision Transformer (ViT)** cuts the image into square "
        "**patches**, turns each patch into a vector, and lets all patches exchange information "
        "through attention. This app only **uses** a pretrained DINOv2 model; it does **not** "
        "train DINOv2.")

    if need_image(image_data):
        rgb, _ = image_data
        st.image(draw_patch_grid(rgb), caption="The image resized to 224 x 224 and cut into "
                 "16 x 16 = 256 patches of 14 x 14 pixels (the grid DINOv2-small uses). "
                 "Drawn with OpenCV; no model needed.", width=480)

    st.markdown("##### DINOv2 features (optional, about 90 MB download)")
    if not (has_module("torch") and has_module("transformers")):
        explain_missing_library("transformers")
    else:
        if st.button("Load DINOv2 (small)"):
            request_model("dino", load_dino_model, (DINO_MODEL_ID,), "DINOv2", PIP_HINTS["transformers"])
        dino = get_loaded("dino", load_dino_model)
        if dino is None:
            st.info("DINOv2 is not loaded yet. Press the button above.")
        elif need_image(image_data):
            rgb, _ = image_data
            try:
                cls_vec, patches, grid = cached_dino_features(dino, rgb)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Feature extraction failed: {describe_load_error(exc, 'DINOv2', PIP_HINTS['transformers'])}")
            else:
                pca_img = pca_feature_image(patches, grid, (rgb.shape[1], rgb.shape[0]))
                c1, c2 = st.columns(2)
                c1.image(rgb, caption="Original", use_container_width=True)
                c2.image(pca_img, caption="PCA of patch embeddings: similar colours = patches "
                         "DINOv2 considers similar", use_container_width=True)
                st.write(f"Whole-image embedding: **{cls_vec.shape[0]} numbers**, "
                         f"L2 norm {float(np.linalg.norm(cls_vec)):.2f}.")
                st.bar_chart(cls_vec.astype(float).tolist())
                st.caption("Each bar is one dimension of the embedding. No single dimension has "
                           "an easy human meaning.")

                st.markdown("##### Image similarity")
                second_file = st.file_uploader("Upload a second image to compare",
                                               type=["jpg", "jpeg", "png"], key="second_image")
                if second_file is not None:
                    try:
                        rgb2, _ = load_image_cached(second_file.getvalue(), settings["max_side"])
                        cls2, _, _ = cached_dino_features(dino, rgb2)
                    except ImageProblem as problem:
                        st.error(str(problem))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not compare images: {exc}")
                    else:
                        similarity = cosine_similarity(cls_vec, cls2)
                        st.image(rgb2, caption="Second image", width=360)
                        st.metric("Cosine similarity of DINOv2 embeddings", f"{similarity:.3f}")
                        st.caption("1 means the embeddings point the same way. Compare values "
                                   "between image pairs; there is no universal 'same scene' cut-off.")

    st.divider()
    st.subheader("Vision-language model: CLIP image-text matching")
    st.markdown(
        "CLIP was trained on image-caption pairs so that an image and a matching sentence get "
        "similar embeddings. Enter your own descriptions; the app measures how close each one "
        "is to the uploaded image.")
    st.warning("CLIP similarity scores are **not guaranteed probabilities**. The 'relative score' "
               "below only compares your prompts with each other and always adds up to 100%, even "
               "if none of them describes the image. CLIP has **not been verified** for road-damage "
               "or safety judgements.")
    if not (has_module("torch") and has_module("transformers")):
        explain_missing_library("transformers")
        return
    if st.button("Load CLIP (about 600 MB)"):
        request_model("clip", load_clip_model, (CLIP_MODEL_ID,), "CLIP", PIP_HINTS["transformers"])
    clip_bundle = get_loaded("clip", load_clip_model)
    if clip_bundle is None:
        st.info("CLIP is not loaded yet. Press the button above.")
        return
    prompts_text = st.text_area("Candidate descriptions (one per line)", DEFAULT_CLIP_PROMPTS, height=150)
    prompts = tuple(line.strip() for line in prompts_text.splitlines() if line.strip())
    if not need_image(image_data):
        return
    if len(prompts) < 2:
        st.info("Enter at least two descriptions (one per line).")
        return
    if len(prompts) > 20:
        st.warning("Using the first 20 descriptions only.")
        prompts = prompts[:20]
    rgb, _ = image_data
    try:
        cosines, relative, seconds = cached_clip_compare(clip_bundle, rgb, prompts)
    except Exception as exc:  # noqa: BLE001
        st.error(describe_load_error(exc, "CLIP comparison", PIP_HINTS["transformers"]))
        return
    order = np.argsort(-cosines)
    table = [{"Description": prompts[i], "Cosine similarity": round(float(cosines[i]), 4),
              "Relative score among prompts (%)": round(float(relative[i]) * 100, 1)} for i in order]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.bar_chart({"Relative score (%)": {prompts[i]: float(relative[i]) * 100 for i in order}})
    st.caption(f"Computed in {seconds:.2f} s (cached results show the original time).")


# ----------------------------------------------------------------------------
# Tab: Diffusion (Module 5)
# ----------------------------------------------------------------------------
def render_diffusion(image_data: Optional[tuple[np.ndarray, tuple[int, int]]]) -> None:
    st.header("Module 5: Diffusion-based image restoration (partly optional)")
    st.markdown(
        "**Diffusion models** learn to remove noise. **Forward process:** noise is added to a "
        "clean image step by step until only noise remains. **Reverse process:** a neural network "
        "predicts the noise at each step and removes a little of it, turning random noise back "
        "into a clean image. For **inpainting**, the unmasked parts of the image guide the "
        "denoising so the model fills the masked region with plausible content.")
    st.info("**Honest labelling:** the OpenCV blur and filters elsewhere in this app are ordinary "
            "filters and are **not** diffusion. Only the optional inpainting below runs real "
            "diffusion inference. Section A is a plain-NumPy demonstration of the forward "
            "(noising) maths and does **not** restore anything.")

    if not need_image(image_data):
        return
    rgb, _ = image_data

    st.subheader("A. Forward noising demo (always available)")
    preview = rgb
    if max(rgb.shape[:2]) > 384:
        scale = 384.0 / max(rgb.shape[:2])
        preview = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
    t_value = st.slider("Diffusion step t (0 = clean, 1000 = almost pure noise)", 0, 1000, 300, 50)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    cols = st.columns(2)
    cols[0].image(preview, caption="Clean image (t = 0)", use_container_width=True)
    cols[1].image(forward_noise(preview, t_value, int(seed)),
                  caption=f"Noised at t = {t_value} (signal kept: {alpha_bar(t_value) * 100:.1f}%)",
                  use_container_width=True)
    strip = st.columns(5)
    for column, t in zip(strip, (0, 100, 300, 600, 999)):
        column.image(forward_noise(preview, t, int(seed)), caption=f"t = {t}",
                     use_container_width=True)
    st.caption("Uses x_t = sqrt(a) * x_0 + sqrt(1 - a) * noise with a linear noise schedule "
               "(beta from 0.0001 to 0.02 over 1000 steps).")

    st.subheader("B. Optional: diffusion inpainting (real inference)")
    st.warning("This downloads a **multi-gigabyte** Stable Diffusion inpainting model and needs "
               "roughly 4 to 8 GB of free RAM. On a CPU it can take several minutes per image. "
               "The model is **not** downloaded unless you tick the box and press Load. "
               "Inpainting **invents** the content of the masked area; it does not recover what "
               "was really there, so never treat it as evidence.")
    if not (has_module("diffusers") and has_module("torch")):
        explain_missing_library("diffusers")
        return
    agree = st.checkbox("I understand the download size and the memory/time needs.")
    model_id = st.text_input("Inpainting model id (Hugging Face)", DEFAULT_INPAINT_MODEL_ID,
                             help="Edit this if the repository has moved. Some models require "
                                  "accepting a licence on huggingface.co first.")
    if st.button("Load inpainting model", disabled=not agree):
        request_model("inpaint", load_inpaint_pipeline, (model_id,), "the inpainting model",
                      PIP_HINTS["diffusers"])
    bundle = get_loaded("inpaint", load_inpaint_pipeline)
    if bundle is None:
        st.info("The inpainting model is not loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    x_pct = c1.slider("Mask left (%)", 0, 90, 35)
    y_pct = c2.slider("Mask top (%)", 0, 90, 35)
    w_pct = c3.slider("Mask width (%)", 5, 60, 25)
    h_pct = c4.slider("Mask height (%)", 5, 60, 25)
    prompt = st.text_input("What should fill the masked area?", "an empty asphalt road surface")
    steps = st.slider("Denoising steps (fewer = faster, rougher)", 5, 30, 15)
    guidance = st.slider("Guidance scale", 1.0, 12.0, 7.5, 0.5)

    guide = rgb.copy()
    gx0, gy0 = int(rgb.shape[1] * x_pct / 100), int(rgb.shape[0] * y_pct / 100)
    gx1 = min(rgb.shape[1] - 1, gx0 + int(rgb.shape[1] * w_pct / 100))
    gy1 = min(rgb.shape[0] - 1, gy0 + int(rgb.shape[0] * h_pct / 100))
    cv2.rectangle(guide, (gx0, gy0), (gx1, gy1), (255, 0, 0), max(2, min(rgb.shape[:2]) // 200))
    st.image(guide, caption="Red rectangle = region that will be regenerated", width=480)

    if st.button("Run diffusion inpainting"):
        try:
            with st.spinner("Running diffusion... this can take several minutes on a CPU."):
                start = time.perf_counter()
                work, mask, result = inpaint_region(bundle, rgb, (x_pct, y_pct, w_pct, h_pct),
                                                    prompt, steps, guidance, int(seed))
                elapsed = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            st.error(describe_load_error(exc, "Diffusion inpainting", PIP_HINTS["diffusers"]))
        else:
            r1, r2 = st.columns(2)
            r1.image(work, caption="Input (resized for the model)", use_container_width=True)
            r2.image(result, caption="Inpainted result (generated content)", use_container_width=True)
            st.caption(f"Completed in {elapsed:.1f} s. If the result is completely black, the "
                       "model's built-in safety checker may have blocked the output.")


# ----------------------------------------------------------------------------
# Tab: Segmentation (Module 6)
# ----------------------------------------------------------------------------
def render_segmentation(image_data: Optional[tuple[np.ndarray, tuple[int, int]]]) -> None:
    st.header("Module 6: Promptable segmentation (Segment Anything style)")
    st.markdown(
        "**Segment Anything (SAM)** has three parts. The **image encoder** (a large Vision "
        "Transformer) turns the image into feature maps once. The **prompt encoder** turns your "
        "hint (a point or a box) into numbers. The **mask decoder** combines both and quickly "
        "outputs a mask for the object you pointed at. SAM outlines objects but does **not** "
        "say what they are.")
    st.info("A SAM checkpoint is **required**. `mobile_sam.pt` (about 40 MB) is the lightest; "
            "`sam_b.pt` (about 375 MB) is larger. The file downloads (through ultralytics) only "
            "when you press Load. Without it, segmentation is not available.")
    if not has_module("ultralytics"):
        explain_missing_library("ultralytics")
        return

    weights = st.selectbox("SAM checkpoint", SAM_CHOICES)
    if st.button("Load segmentation model"):
        request_model("sam", load_sam_model, (weights,), f"SAM weights '{weights}'", PIP_HINTS["ultralytics"])
    sam = get_loaded("sam", load_sam_model)
    if sam is None:
        st.info("No SAM model is loaded, so segmentation cannot run yet.")
        return
    if not need_image(image_data):
        return

    rgb, _ = image_data
    height, width = rgb.shape[:2]
    mode = st.radio("Prompt type", ["Point", "Box"], horizontal=True)
    guide = rgb.copy()
    thickness = max(2, min(height, width) // 200)
    point = (width // 2, height // 2)
    box = (0, 0, width - 1, height - 1)
    if mode == "Point":
        c1, c2 = st.columns(2)
        px = c1.slider("Point X (% of width)", 0, 100, 50)
        py = c2.slider("Point Y (% of height)", 0, 100, 50)
        point = (min(width - 1, int(width * px / 100)), min(height - 1, int(height * py / 100)))
        cv2.drawMarker(guide, point, (255, 0, 0), cv2.MARKER_CROSS, 30, thickness)
    else:
        c1, c2, c3, c4 = st.columns(4)
        x1 = c1.slider("Left (%)", 0, 95, 25)
        y1 = c2.slider("Top (%)", 0, 95, 25)
        x2 = c3.slider("Right (%)", 5, 100, 75)
        y2 = c4.slider("Bottom (%)", 5, 100, 75)
        if x2 <= x1 or y2 <= y1:
            st.warning("The box must have the right edge greater than the left and the "
                       "bottom greater than the top.")
            return
        box = (int(width * x1 / 100), int(height * y1 / 100),
               min(width - 1, int(width * x2 / 100)), min(height - 1, int(height * y2 / 100)))
        cv2.rectangle(guide, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness)
    st.image(guide, caption="Prompt preview (red)", width=520)

    if st.button("Run segmentation"):
        try:
            with st.spinner("Segmenting..."):
                mask, seconds = run_sam(sam, rgb, mode, point, box)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Segmentation failed: {exc}")
            return
        if mask is None:
            st.warning("The model returned no mask for this prompt. Try another point or box.")
            return
        left, right = st.columns(2)
        left.image(rgb, caption="Original", use_container_width=True)
        right.image(overlay_mask(rgb, mask), caption="Segmentation mask", use_container_width=True)
        area = float(mask.sum()) / mask.size * 100.0
        st.metric("Mask covers", f"{area:.1f}% of the image")
        st.caption(f"Computed in {seconds:.2f} s. The mask outlines one object or region; it "
                   "does not tell you what it is, and results can be wrong on unusual scenes.")
        st.image((mask.astype(np.uint8) * 255), caption="Binary mask (white = segmented)", width=360)


# ----------------------------------------------------------------------------
# Tab: Video (Module 7)
# ----------------------------------------------------------------------------
def render_video(settings: dict[str, Any]) -> None:
    st.header("Module 7: Video processing and object tracking")
    st.markdown(
        "**Detection** looks at each frame on its own: 'there is a car here'. **Tracking** links "
        "detections across frames so the same car keeps the same **ID number**. This app tracks "
        "with YOLO + ByteTrack. Without ultralytics you can still try a simple **motion-only** "
        "mode (OpenCV background subtraction), which has no class names and no IDs.")
    video_file = settings["video_file"]
    if video_file is None:
        st.info("Upload a video in the sidebar (section 1).")
        return
    size_mb = video_file.size / (1024 * 1024)
    if size_mb > MAX_VIDEO_MB:
        st.error(f"This video is {size_mb:.0f} MB. The limit here is {MAX_VIDEO_MB} MB.")
        return

    ultra = has_module("ultralytics")
    modes = ["Detection + tracking (YOLO + ByteTrack)", "Motion only (OpenCV, no AI)"]
    mode_label = st.radio("Mode", modes, index=0 if ultra else 1)
    use_yolo = mode_label == modes[0]
    tracker_model = None
    if use_yolo:
        if not ultra:
            explain_missing_library("ultralytics")
            return
        if st.button(f"Load tracker weights ({settings['weights']})"):
            request_model("yolo_track", load_yolo_model, (settings["weights"], "track"),
                          f"YOLO weights '{settings['weights']}'", PIP_HINTS["ultralytics"])
        tracker_model = get_loaded("yolo_track", load_yolo_model)
        if tracker_model is None:
            st.info("Load the tracker weights first. Note: the first tracking run may also "
                    "install a small helper package (lap/lapx) automatically.")
            return

    c1, c2, c3 = st.columns(3)
    stride = c1.slider("Process every Nth frame", 1, 30, 5, help="Higher = faster, but tracking "
                       "can get worse because objects move further between processed frames.")
    max_frames = c2.slider("Maximum frames to process", 10, 600, 120, 10)
    video_side = c3.select_slider("Max frame side (px)", options=[320, 480, 640, 960], value=640)

    if st.button("Process video"):
        suffix = os.path.splitext(video_file.name)[1] or ".mp4"
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = handle.name
        try:
            handle.write(video_file.getbuffer())
            handle.close()
            bar = st.progress(0.0, text="Starting...")
            result = process_video(
                temp_path, "yolo" if use_yolo else "motion", tracker_model, settings["conf"],
                settings["imgsz"], stride, max_frames, video_side,
                lambda fraction, text: bar.progress(fraction, text=text))
            bar.empty()
            st.session_state["video_result"] = result
        except ValueError as problem:
            st.error(str(problem))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Video processing failed: {exc}")
        finally:
            try:
                handle.close()
                os.remove(temp_path)
            except OSError:
                pass

    result = st.session_state.get("video_result")
    if not result:
        return
    info = result["info"]
    st.markdown("#### Results of the last run")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames processed", result["processed"])
    m2.metric("Processing time (s)", f"{result['elapsed']:.1f}")
    m3.metric("Speed (frames/s)", f"{result['processed'] / result['elapsed']:.1f}" if result["elapsed"] > 0 else "n/a")
    m4.metric("Source", f"{info['width']}x{info['height']} @ {info['fps']:.0f} fps")
    if result["processed"] == 0:
        st.warning("No frames were processed. The video may be unsupported or empty.")
        return
    if result["samples"]:
        st.markdown("Sample processed frames:")
        columns = st.columns(min(3, len(result["samples"])))
        for i, frame in enumerate(result["samples"]):
            columns[i % len(columns)].image(frame, use_container_width=True)
    if result["mode"] == "yolo":
        rows = [{"Class": label, "Unique track IDs": len(ids), "IDs": ", ".join(map(str, ids[:15])) + (" ..." if len(ids) > 15 else "")}
                for label, ids in sorted(result["class_track_ids"].items())]
        if rows:
            st.markdown("Unique tracked objects per class:")
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("Counting unique IDs is NOT the same as counting real objects: ID switches "
                       "and missed detections can split one object into several IDs or merge "
                       "different ones. With frame skipping this effect is stronger.")
        else:
            st.info("No tracked objects were found at this confidence threshold.")
    else:
        st.info(f"Motion mode found moving regions in "
                f"{sum(1 for n in result['per_frame_counts'] if n > 0)} of "
                f"{len(result['per_frame_counts'])} processed frames. These boxes mark movement, "
                "not recognised objects, and the first frames are unreliable while the background "
                "model learns.")
    if result["video_bytes"]:
        if result["codec"] == "avc1":
            st.video(result["video_bytes"])
        else:
            st.caption("The video was written with the 'mp4v' codec, which many browsers cannot "
                       "play inline. Use the download button and open it in a desktop player.")
        st.download_button("⬇️ Download annotated video (MP4)", result["video_bytes"],
                           file_name="visionguard_annotated.mp4", mime="video/mp4")
    else:
        st.warning("The annotated video file could not be written on this computer. The sample "
                   "frames above are still valid.")


# ----------------------------------------------------------------------------
# Tab: Robustness (Module 8)
# ----------------------------------------------------------------------------
def build_variants(rgb: np.ndarray, blur_kernel: int, scale_percent: int,
                   brightness: int) -> list[tuple[str, np.ndarray]]:
    """The four image versions compared in the robustness module."""
    return [
        ("Original", rgb),
        (f"Gaussian blur (kernel {blur_kernel})", apply_gaussian_blur(rgb, blur_kernel, 0.0)),
        (f"Downscaled to {scale_percent}% and restored", downscale_and_restore(rgb, scale_percent)),
        (f"Brightness {brightness:+d}", adjust_brightness(rgb, brightness)),
    ]


def render_robustness(image_data: Optional[tuple[np.ndarray, tuple[int, int]]],
                      settings: dict[str, Any]) -> None:
    st.header("Module 8: Robustness and evaluation")
    st.markdown(
        "A model that works on a clean photo can fail when conditions change. Here you degrade "
        "the image and see what changes. **Lighting** (dark or glare) hides detail; **blur** (motion, "
        "rain, focus) removes edges; **low resolution** makes small, far-away objects vanish; "
        "**viewpoint** changes object shape; **occlusion** hides part of an object.")
    if not need_image(image_data):
        return
    rgb, _ = image_data

    c1, c2, c3 = st.columns(3)
    blur_kernel = c1.slider("Blur kernel size (odd)", 3, 51, 15, 2)
    scale_percent = c2.slider("Resize: keep this % of resolution", 10, 100, 30, 5)
    brightness = c3.slider("Brightness change", -150, 150, -80, 10)
    variants = build_variants(rgb, blur_kernel, scale_percent, brightness)

    stats_rows = [{"Version": name, "Sharpness (Laplacian variance)": round(sharpness_score(img), 1),
                   "Mean brightness (0-255)": round(mean_brightness(img), 1)} for name, img in variants]
    st.markdown("#### Image statistics (always computed)")
    st.dataframe(stats_rows, use_container_width=True, hide_index=True)

    st.markdown("#### Detector comparison (needs a loaded detector)")
    model = detector_ready() if has_module("ultralytics") else None
    if model is None:
        st.info("Load a YOLO detector (sidebar, section 2) to compare detections across versions.")
    else:
        if st.button("Run detection on all four versions"):
            try:
                with st.spinner("Running the detector on four images..."):
                    run_detection(model, rgb, settings["conf"], settings["imgsz"])   # warm-up, not timed
                    outcomes = []
                    for name, img in variants:
                        detections, elapsed_ms = run_detection(model, img, settings["conf"], settings["imgsz"])
                        outcomes.append((name, img, detections, elapsed_ms))
                st.session_state["robust_result"] = outcomes
            except Exception as exc:  # noqa: BLE001
                st.error(f"Detector comparison failed: {exc}")
        outcomes = st.session_state.get("robust_result")
        if outcomes:
            reference = outcomes[0][2]
            rows = []
            for name, _, detections, elapsed_ms in outcomes:
                matched = count_matches(reference, detections)
                mean_conf = float(np.mean([d["conf"] for d in detections])) if detections else None
                rows.append({
                    "Version": name,
                    "Detections": len(detections),
                    "Mean confidence": round(mean_conf, 3) if mean_conf is not None else "n/a",
                    "Inference ms": round(elapsed_ms),
                    "Agree with original": matched,
                    "Missing vs original": len(reference) - matched,
                    "New vs original": len(detections) - matched,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            grid = st.columns(2)
            for i, (name, img, detections, _) in enumerate(outcomes):
                grid[i % 2].image(draw_detections(img, detections), caption=name, use_container_width=True)
            st.caption("'Agree with original' = same class and box overlap (IoU >= 0.5) with the "
                       "detections on the original image. This measures CONSISTENCY, not "
                       "correctness: the original detections may themselves be wrong. Mean confidence "
                       "is the model's own score, not accuracy. Timing varies with your hardware "
                       "and background programs (a warm-up run is done first and not counted). "
                       "The settings applied are the sliders at the time you pressed the button.")

    with st.expander("False positives, false negatives and why there are no accuracy numbers here"):
        st.markdown(
            "- **False positive:** the model reports something that is not there (for example a "
            "shadow called a 'person').\n"
            "- **False negative:** the model misses something that is there (for example a dark, "
            "distant pedestrian).\n"
            "- **Precision** = correct detections / all detections. **Recall** = correct detections / "
            "all real objects. **F1** combines both. **mAP** summarises precision across thresholds.\n"
            "- All of these require **labelled ground truth** (humans marking every real object). "
            "This app has none, so it deliberately shows **no** precision, recall, F1 or accuracy. "
            "Counts, timings and consistency checks are shown instead because they can really "
            "be calculated.")


# ----------------------------------------------------------------------------
# Tab: Concepts and limitations
# ----------------------------------------------------------------------------
def render_concepts() -> None:
    st.header("Concepts and limitations")
    with st.expander("Image processing", expanded=True):
        st.write("Images are grids of pixels. Filters (blur, edges) compute new pixel values from "
                 "neighbours using fixed formulas. They are fast and explainable, but they do not "
                 "'understand' the scene.")
    with st.expander("Object detection and tracking"):
        st.write("A detector (YOLO) predicts boxes, labels and confidence in one pass. Tracking "
                 "(ByteTrack) matches boxes between frames, mostly using motion and overlap, to give "
                 "persistent IDs. Trackers can swap or lose IDs when objects cross or are hidden.")
    with st.expander("Self-supervised learning and Vision Transformers"):
        st.write("SSL learns from unlabelled images. A ViT splits an image into patches, embeds "
                 "each as a vector, and uses attention so every patch can look at every other patch. "
                 "DINOv2 is a pretrained ViT whose embeddings are useful for similarity and "
                 "segmentation-like tasks. This app does not train it.")
    with st.expander("Image similarity and CLIP"):
        st.write("Embeddings are vectors; cosine similarity measures how aligned two vectors are. "
                 "CLIP maps images and text into the same space, which allows 'zero-shot' matching "
                 "with arbitrary descriptions. Scores are relative and can be misleading, especially "
                 "for rare concepts or fine details.")
    with st.expander("Diffusion models"):
        st.write("Forward: add noise gradually. Reverse: a trained network removes noise step by step. "
                 "Inpainting uses the same idea with a mask. Generated pixels are guesses. They can "
                 "look realistic and still be wrong, so they are unsuitable as evidence.")
    with st.expander("Promptable segmentation (SAM)"):
        st.write("Image encoder: computes features of the whole image. Prompt encoder: encodes a point "
                 "or box. Mask decoder: predicts a mask from both. SAM is class-agnostic: it outlines "
                 "regions but does not name them.")
    with st.expander("Known limitations of this project"):
        st.markdown(
            "- The default detector has **no pothole or road-damage class**.\n"
            "- The scene summary is a heuristic, **not** a validated hazard or safety system.\n"
            "- No model here has been evaluated by this project on labelled data.\n"
            "- Night, rain, fog, glare, blur, occlusion and unusual camera angles can cause missed "
            "or wrong detections.\n"
            "- CPU speed is limited; large images or videos can be slow.\n"
            "- Optional modules depend on third-party libraries and downloads that can change.\n"
            "- Do not use this app for driving, surveillance of people, or any safety-critical decision.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="VisionGuard AI", page_icon="🛡️", layout="wide")
    settings = render_sidebar()
    image_data = load_uploaded_image(settings["image_file"], settings["max_side"])

    tabs = st.tabs(["🏠 Home", "🖼️ Image processing", "🎯 Object detection",
                    "🧬 Embeddings & similarity", "🌫️ Diffusion", "✂️ Segmentation",
                    "🎞️ Video & tracking", "🧪 Robustness & evaluation",
                    "📚 Concepts & limitations"])
    with tabs[0]:
        run_safely("Home", render_home, settings)
    with tabs[1]:
        run_safely("Image processing", render_image_processing, image_data)
    with tabs[2]:
        run_safely("Object detection", render_detection, image_data, settings)
    with tabs[3]:
        run_safely("Embeddings & similarity", render_embeddings, image_data, settings)
    with tabs[4]:
        run_safely("Diffusion", render_diffusion, image_data)
    with tabs[5]:
        run_safely("Segmentation", render_segmentation, image_data)
    with tabs[6]:
        run_safely("Video & tracking", render_video, settings)
    with tabs[7]:
        run_safely("Robustness & evaluation", render_robustness, image_data, settings)
    with tabs[8]:
        run_safely("Concepts & limitations", render_concepts)


if __name__ == "__main__":
    main()
