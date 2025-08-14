# notebooks/yolo11_predict_display_openvino.py
"""
Notebook cell to run predictions with a trained Ultralytics YOLOv11 model optimized with OpenVINO.
- Uses OpenVINO for accelerated inference on Intel hardware
- Uses model's .plot() for boxes/masks and shows via matplotlib.
- Saves annotated frames to disk.
- Prints a concise table of detections with performance metrics.

Why: Keep a single, reusable entrypoint for images/URLs/directories/videos with OpenVINO optimization.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import matplotlib.pyplot as plt

# Optional: only needed if you prefer manual OpenCV saving; matplotlib save works too
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from ultralytics import YOLO

# -------------------------------
# 1) OpenVINO Model Management
# -------------------------------
def export_to_openvino(
    weights: str = "best.pt",
    imgsz: int = 640,
) -> str:
    """Export YOLO model to OpenVINO format for optimized inference.
    
    Returns:
        Path to the exported OpenVINO model directory
    """
    model = YOLO(weights, task='detect')
    
    # Export to OpenVINO format
    try:
        ov_model_path = model.export(
            format="openvino", 
            imgsz=imgsz,
            half=False,  # Use FP32 for better compatibility
            int8=False,  # Set to True for further optimization if supported
        )
        
        print(f"[INFO] Model exported to OpenVINO: {ov_model_path}")
        return ov_model_path
    except Exception as e:
        print(f"[ERROR] Failed to export model to OpenVINO: {e}")
        raise


def load_optimized_model(
    weights: str = "best.pt",
    imgsz: int = 640,
    use_openvino: bool = True,
) -> tuple[YOLO, str]:
    """Load YOLO model with optional OpenVINO optimization.
    
    Returns:
        Tuple of (model, engine_type)
    """
    if use_openvino:
        try:
            # Check if OpenVINO model already exists
            ov_model_path = str(Path(weights).with_suffix('')) + "_openvino_model"
            ov_xml_path = Path(ov_model_path) / f"{Path(weights).stem}.xml"
            
            if not ov_xml_path.exists():
                print("[INFO] Exporting model to OpenVINO format...")
                export_to_openvino(weights, imgsz)
            
            # Load OpenVINO model - use the directory path, not the .xml file
            model = YOLO(str(ov_model_path), task='detect')
            print(f"[INFO] Loaded OpenVINO model: {ov_model_path}")
            return model, "OpenVINO"
        except Exception as e:
            print(f"[WARN] Failed to load OpenVINO model: {e}")
            print("[INFO] Falling back to PyTorch model...")
    
    # Load original PyTorch model
    model = YOLO(weights)
    print(f"[INFO] Loaded PyTorch model: {weights}")
    return model, "PyTorch"


# -------------------------------
# 2) Load your trained model with OpenVINO optimization
# -------------------------------
MODEL_WEIGHTS = "best.pt"  # adjust if your run name differs
model, engine_type = load_optimized_model(MODEL_WEIGHTS, use_openvino=True)

# -------------------------------
# 3) Helper: show image (RGB np.ndarray) inline
# -------------------------------
def _imshow_rgb(img: np.ndarray, title: str | None = None) -> None:
    if img.ndim == 2:
        disp = img
    else:
        # Ensure RGB for matplotlib (Ultralytics returns BGR from .plot())
        disp = img[:, :, ::-1] if img.shape[-1] == 3 else img
    plt.figure(figsize=(12, 8))
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.imshow(disp)
    plt.show()


# -------------------------------
# 4) Core runner: predict + display + save with performance metrics
# -------------------------------
SourceLike = Union[str, Path, int, np.ndarray, List[Union[str, Path]]]

def predict_and_show(
    source: SourceLike,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Union[str, int] = 'cpu',
    save_dir: Union[str, Path] = "runs/predict_display",
    show_inline: bool = True,
    save_images: bool = True,
    verbose: bool = True,
) -> dict:
    """Run inference and visualize results with performance tracking.

    Args:
        source: Image path/URL, directory, video path, webcam index, ndarray, or list of paths.
        imgsz: Inference size.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        device: CUDA index or 'cpu'.
        save_dir: Where annotated images/frames will be saved.
        show_inline: Display annotated image(s) inline (matplotlib).
        save_images: Save annotated outputs to files.
        verbose: Print detailed performance information.
    
    Returns:
        Dictionary with performance metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Measure total inference time
    start_total = time.time()
    
    results = model(
        source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        stream=False,  # set True for very large batches/streams
    )
    
    end_total = time.time()
    total_inference_time = (end_total - start_total) * 1000  # Convert to ms

    # Performance tracking
    inference_times = []
    total_detections = 0
    processed_images = 0

    # Ultralytics returns a list-like of Result objects
    for idx, r in enumerate(results):
        # Measure per-image processing time
        start_img = time.time()
        
        # r.plot() returns an annotated BGR image (np.ndarray)
        annotated_bgr = r.plot()  # handles boxes/masks/keypoints if present
        
        end_img = time.time()
        img_process_time = (end_img - start_img) * 1000
        inference_times.append(img_process_time)

        # Compose a readable title and filename
        orig_name = Path(r.path).name if hasattr(r, "path") and r.path else f"frame_{idx:06d}.jpg"
        
        # Count detections
        num_detections = len(r.boxes) if r.boxes is not None else 0
        total_detections += num_detections
        processed_images += 1
        
        title = f"{orig_name} | Engine: {engine_type} | {num_detections} detections | {img_process_time:.1f}ms"
        out_path = save_dir / orig_name

        # Save annotated image
        if save_images:
            if _HAS_CV2:
                cv2.imwrite(str(out_path), annotated_bgr)
            else:
                # Fall back to matplotlib save (expects RGB)
                plt.imsave(str(out_path), annotated_bgr[:, :, ::-1])

        # Inline display
        if show_inline:
            _imshow_rgb(annotated_bgr, title=title)

        # Compact, per-image summary
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            names = [r.names[int(c)] for c in cls]
            # Tabulate
            print(f"\nDetections for: {orig_name}")
            print("idx  class          conf    x1     y1     x2     y2")
            for i, (n, p, box) in enumerate(zip(names, confs, xyxy)):
                x1, y1, x2, y2 = box
                print(f"{i:>3}  {n:<13}  {p:>5.2f}  {x1:>5.0f}  {y1:>5.0f}  {x2:>5.0f}  {y2:>5.0f}")
        else:
            print(f"\nNo detections for: {orig_name}")

        # Optional: show mask stats when available
        if getattr(r, "masks", None) is not None:
            n_masks = len(r.masks)
            print(f"Masks: {n_masks} instance(s)")

    # Performance summary
    performance_metrics = {
        'engine_type': engine_type,
        'total_inference_time_ms': total_inference_time,
        'avg_per_image_ms': np.mean(inference_times) if inference_times else 0,
        'min_per_image_ms': np.min(inference_times) if inference_times else 0,
        'max_per_image_ms': np.max(inference_times) if inference_times else 0,
        'total_detections': total_detections,
        'processed_images': processed_images,
        'avg_detections_per_image': total_detections / processed_images if processed_images > 0 else 0,
        'fps': 1000 / np.mean(inference_times) if inference_times else 0,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Engine: {performance_metrics['engine_type']}")
        print(f"Total inference time: {performance_metrics['total_inference_time_ms']:.1f}ms")
        print(f"Average per image: {performance_metrics['avg_per_image_ms']:.1f}ms")
        print(f"Min/Max per image: {performance_metrics['min_per_image_ms']:.1f}ms / {performance_metrics['max_per_image_ms']:.1f}ms")
        print(f"Estimated FPS: {performance_metrics['fps']:.1f}")
        print(f"Total detections: {performance_metrics['total_detections']}")
        print(f"Images processed: {performance_metrics['processed_images']}")
        print(f"Avg detections/image: {performance_metrics['avg_detections_per_image']:.1f}")
        print(f"{'='*60}")
    
    return performance_metrics


# -------------------------------
# 5) Example usage with your URL (runs inline)
# -------------------------------
if __name__ == "__main__":
    # Test with a simple image - you can change this to any local image path
    import urllib.request
    import os
    
    # Download a test image if it doesn't exist
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("Downloading test image...")
        try:
            urllib.request.urlretrieve(
                "https://cf.shopee.ph/file/e50eba5b2aa1a1cf32d0cb6b39630ce1", 
                test_image_path
            )
            print("Test image downloaded successfully!")
        except Exception as e:
            print(f"Could not download test image: {e}")
            print("Please provide a local image path instead.")
            sys.exit(1)
    
    # Run prediction with OpenVINO optimization
    metrics = predict_and_show(
        test_image_path,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device='cpu',  # OpenVINO works best with CPU
        save_dir="runs/predict_display",
        show_inline=True,
        save_images=True,
        verbose=True,
    )


# -------------------------------
# 6) Comparison function: PyTorch vs OpenVINO
# -------------------------------
def compare_engines(
    source: SourceLike,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = 'cpu',
    runs: int = 5,
) -> dict:
    """Compare PyTorch vs OpenVINO performance.
    
    Args:
        source: Test image or video
        imgsz: Input size
        conf: Confidence threshold
        iou: IoU threshold
        device: Device to use
        runs: Number of test runs for averaging
    
    Returns:
        Performance comparison dictionary
    """
    print(f"Running performance comparison ({runs} runs each)...")
    
    results = {
        'PyTorch': [],
        'OpenVINO': []
    }
    
    # Test PyTorch
    print("\nTesting PyTorch engine...")
    pytorch_model = YOLO("best.pt")
    for i in range(runs):
        start = time.time()
        _ = pytorch_model(source, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        end = time.time()
        results['PyTorch'].append((end - start) * 1000)
    
    # Test OpenVINO
    print("Testing OpenVINO engine...")
    try:
        ov_model, _ = load_optimized_model("best.pt", imgsz=imgsz, use_openvino=True)
        for i in range(runs):
            start = time.time()
            _ = ov_model(source, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
            end = time.time()
            results['OpenVINO'].append((end - start) * 1000)
    except Exception as e:
        print(f"OpenVINO test failed: {e}")
        results['OpenVINO'] = [0] * runs
    
    # Calculate statistics
    pytorch_avg = np.mean(results['PyTorch'])
    pytorch_std = np.std(results['PyTorch'])
    openvino_avg = np.mean(results['OpenVINO'])
    openvino_std = np.std(results['OpenVINO'])
    
    speedup = pytorch_avg / openvino_avg if openvino_avg > 0 else 0
    
    comparison = {
        'pytorch_avg_ms': pytorch_avg,
        'pytorch_std_ms': pytorch_std,
        'openvino_avg_ms': openvino_avg,
        'openvino_std_ms': openvino_std,
        'speedup_factor': speedup,
        'pytorch_fps': 1000 / pytorch_avg if pytorch_avg > 0 else 0,
        'openvino_fps': 1000 / openvino_avg if openvino_avg > 0 else 0,
    }
    
    print(f"\n{'='*50}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*50}")
    print(f"PyTorch:  {pytorch_avg:.1f}±{pytorch_std:.1f}ms ({comparison['pytorch_fps']:.1f} FPS)")
    print(f"OpenVINO: {openvino_avg:.1f}±{openvino_std:.1f}ms ({comparison['openvino_fps']:.1f} FPS)")
    if speedup > 0:
        print(f"Speedup:  {speedup:.2f}x faster with OpenVINO")
    print(f"{'='*50}")
    
    return comparison


# -------------------------------
# 7) (Optional) Manual draw for full control
# -------------------------------
def manual_draw_boxes(
    result,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a manually annotated BGR image from a single Ultralytics Result.
    Why: gives you control over styling beyond .plot().
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for manual drawing.")

    img = result.orig_img.copy()  # BGR
    if result.boxes is None or len(result.boxes) == 0:
        return img

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.int().cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    names = [result.names[int(c)] for c in cls]

    for (x1, y1, x2, y2), name, conf in zip(xyxy, names, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"{name} {conf:.2f}"
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img

# Example usage:
# Run performance comparison
# comparison_results = compare_engines("test_image.jpg", runs=3)
