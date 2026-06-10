"""
Fire + FireZone — Integrated Pipeline
======================================
Cascade demo: Keras classifier triggers, YOLO detector localizes.

Stage 1 (cheap, fast):  Keras EfficientNet/etc → "fire / none / smoke" + Grad-CAM
Stage 2 (heavier):      YOLO11 → bounding boxes around fire/smoke regions
                        (run only when Stage 1 triggers an alert)

USAGE
-----
Single image:
    python fire_pipeline.py --image path/to/img.jpg

Folder of images:
    python fire_pipeline.py --folder path/to/test_images/

Video file:
    python fire_pipeline.py --video path/to/test.mp4

Live webcam (Jetson demo):
    python fire_pipeline.py --webcam
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import keras
import tensorflow as tf
import matplotlib.cm as cm
from ultralytics import YOLO

# Candidate preprocess_input functions — needed because the .keras file
# was saved with a Lambda layer wrapping preprocess_input, which Keras
# can't deserialize without us passing it as a custom object.
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_v2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess


# =====================================================================
# CONFIG — edit these paths
# =====================================================================
KERAS_MODEL_PATH = Path("best_model.keras")
YOLO_WEIGHTS = Path("/Users/ghofranemerhbene/Work/HACK-IA/hackia/firezone_work/runs/firezone_v1/weights/best.pt")

# Class order of the Keras classifier — confirmed via inspect_classifier_v3 probe
KERAS_CLASS_NAMES = ["fire", "none", "smoke"]

# YOLO classes (from training YAML)
YOLO_CLASS_NAMES = {0: "smoke", 1: "fire"}

# Cascade thresholds — Stage 2 (YOLO) runs only if classifier is confident enough
CLASSIFIER_TRIGGER_CLASSES = {"fire", "smoke"}
CLASSIFIER_MIN_CONF = 0.50

# YOLO inference settings
YOLO_CONF = 0.25
YOLO_IOU = 0.45
DEVICE = "mps"

# Drawing — BGR colors
BOX_COLORS = {"fire": (0, 0, 255), "smoke": (0, 165, 255)}


# =====================================================================
# Stage 1 — Keras classifier (with Grad-CAM XAI)
# =====================================================================
class FireClassifier:
    """Wraps the Keras classifier with prediction + Grad-CAM."""

    def __init__(self, model_path: Path):
        # Try each known preprocess_input variant in turn, like inspect_classifier_v3
        candidates = [
            ("efficientnet_v2", effnet_v2_preprocess),
            ("efficientnet",    effnet_preprocess),
            ("resnet_v2",       resnet_v2_preprocess),
            ("vgg16",           vgg16_preprocess),
            ("mobilenet_v2",    mobilenet_v2_preprocess),
        ]

        self.model = None
        last_err = None
        for name, fn in candidates:
            try:
                self.model = keras.models.load_model(
                    model_path, compile=False,
                    custom_objects={"preprocess_input": fn},
                )
                print(f"[FireClassifier] loaded with {name} preprocess_input")
                break
            except Exception as e:
                last_err = e
        if self.model is None:
            raise RuntimeError(f"Could not load model. Last error:\n{last_err}")

        self.input_size = self.model.input_shape[1:3]   # (H, W)

        # Has embedded preprocessing? (training notebook style)
        self.has_embedded_pp = any(
            l.name.lower() == "preprocessing" for l in self.model.layers
        )

        # Locate the last 4D conv layer for Grad-CAM
        self.base_model_layer = None
        self.last_conv_name = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                self.base_model_layer = layer
                break
        if self.base_model_layer is not None:
            for layer in reversed(self.base_model_layer.layers):
                if len(layer.output.shape) == 4:
                    self.last_conv_name = layer.name
                    break
        else:
            for layer in reversed(self.model.layers):
                if len(layer.output.shape) == 4:
                    self.last_conv_name = layer.name
                    break

        print(f"[FireClassifier] loaded {model_path.name}")
        print(f"   input size: {self.input_size}")
        print(f"   classes:    {KERAS_CLASS_NAMES}")
        print(f"   last conv:  {self.last_conv_name}")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize + format BGR frame into model input tensor."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]))
        arr = resized.astype(np.float32)
        if not self.has_embedded_pp:
            arr = arr / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, frame_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Returns (label, confidence, full probability vector)."""
        x = self._preprocess(frame_bgr)
        raw = self.model.predict(x, verbose=0)[0]
        if not np.isclose(raw.sum(), 1.0, atol=1e-3):
            probs = tf.nn.softmax(raw).numpy()
        else:
            probs = raw
        idx = int(np.argmax(probs))
        return KERAS_CLASS_NAMES[idx], float(probs[idx]), probs

    def gradcam(self, frame_bgr: np.ndarray, target_class_idx: Optional[int] = None) -> np.ndarray:
        """Returns a heatmap (H_small, W_small) normalized 0–1."""
        x = self._preprocess(frame_bgr)
        x_tensor = tf.convert_to_tensor(x)

        if self.base_model_layer is not None:
            conv_extractor = keras.Model(
                inputs=self.base_model_layer.input,
                outputs=self.base_model_layer.get_layer(self.last_conv_name).output,
            )
            try:
                pp_layer = self.model.get_layer("preprocessing")
            except Exception:
                pp_layer = None

            with tf.GradientTape() as tape:
                x_in = pp_layer(x_tensor) if pp_layer is not None else x_tensor
                conv_outputs = conv_extractor(x_in)
                tape.watch(conv_outputs)
                try:
                    pooled = self.model.get_layer("global_avg_pool")(conv_outputs)
                except Exception:
                    pooled = tf.reduce_mean(conv_outputs, axis=[1, 2])
                try:
                    logits = self.model.get_layer("predictions")(pooled)
                except Exception:
                    logits = pooled
                    start = False
                    for layer in self.model.layers:
                        if isinstance(layer, keras.Model):
                            start = True
                            continue
                        if start and layer.name not in ("global_avg_pool",):
                            logits = layer(logits)
                if target_class_idx is None:
                    target_class_idx = int(tf.argmax(logits[0]))
                channel = logits[:, target_class_idx]
        else:
            grad_model = keras.Model(
                inputs=self.model.input,
                outputs=[self.model.get_layer(self.last_conv_name).output, self.model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, logits = grad_model(x_tensor)
                if target_class_idx is None:
                    target_class_idx = int(tf.argmax(logits[0]))
                channel = logits[:, target_class_idx]

        grads = tape.gradient(channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()


# =====================================================================
# Stage 2 — YOLO detector
# =====================================================================
class FireZoneDetector:
    def __init__(self, weights_path: Path):
        self.model = YOLO(str(weights_path))
        print(f"[FireZoneDetector] loaded {weights_path.name}")

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Returns list of {class_name, conf, box=(x1,y1,x2,y2)}."""
        results = self.model.predict(
            source=frame_bgr, conf=YOLO_CONF, iou=YOLO_IOU,
            device=DEVICE, verbose=False,
        )[0]
        out = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            out.append({
                "class_name": YOLO_CLASS_NAMES.get(cls_idx, f"cls{cls_idx}"),
                "conf": float(box.conf[0]),
                "box": tuple(int(v) for v in box.xyxy[0].tolist()),
            })
        return out


# =====================================================================
# Drawing helpers
# =====================================================================
def overlay_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    hm_u8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_bgr = (jet[:, ::-1] * 255).astype(np.uint8)
    jet_hm = jet_bgr[hm_u8]
    jet_hm = cv2.resize(jet_hm, (w, h))
    return cv2.addWeighted(frame_bgr, 1.0, jet_hm, alpha, 0)


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = BOX_COLORS.get(d["class_name"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['conf']:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def annotate_header(frame_bgr: np.ndarray, classifier_label: str, classifier_conf: float,
                    n_detections: int, latency_ms: float) -> np.ndarray:
    out = frame_bgr.copy()
    txt = (f"Class: {classifier_label} ({classifier_conf*100:.0f}%)"
           f"  |  Detections: {n_detections}"
           f"  |  {latency_ms:.0f} ms")
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, txt, (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return out


# =====================================================================
# Main per-frame logic
# =====================================================================
def process_frame(frame_bgr: np.ndarray, classifier: FireClassifier,
                  detector: FireZoneDetector) -> dict:
    t0 = time.perf_counter()
    label, conf, probs = classifier.predict(frame_bgr)

    detections, gradcam_heat = [], None
    if label in CLASSIFIER_TRIGGER_CLASSES and conf >= CLASSIFIER_MIN_CONF:
        detections = detector.detect(frame_bgr)
        try:
            gradcam_heat = classifier.gradcam(frame_bgr)
        except Exception as e:
            print(f"[warn] gradcam failed: {e}")
            gradcam_heat = None

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "classifier_label": label,
        "classifier_conf": conf,
        "classifier_probs": probs,
        "detections": detections,
        "gradcam": gradcam_heat,
        "latency_ms": elapsed_ms,
    }


def render_frame(frame_bgr: np.ndarray, result: dict, show_gradcam: bool = True) -> np.ndarray:
    out = frame_bgr
    if show_gradcam and result["gradcam"] is not None:
        out = overlay_heatmap(out, result["gradcam"], alpha=0.30)
    if result["detections"]:
        out = draw_detections(out, result["detections"])
    out = annotate_header(out, result["classifier_label"], result["classifier_conf"],
                          len(result["detections"]), result["latency_ms"])
    return out


# =====================================================================
# Entry points
# =====================================================================
def run_on_image(path: Path, classifier: FireClassifier, detector: FireZoneDetector,
                 out_dir: Path):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[skip] couldn't read {path}")
        return
    result = process_frame(frame, classifier, detector)
    rendered = render_frame(frame, result)
    out_path = out_dir / f"{path.stem}_pipeline.jpg"
    cv2.imwrite(str(out_path), rendered)
    print(f"  {path.name}: {result['classifier_label']} "
          f"({result['classifier_conf']:.2f}), "
          f"{len(result['detections'])} boxes, "
          f"{result['latency_ms']:.0f} ms  →  {out_path.name}")


def run_on_folder(folder: Path, classifier, detector, out_dir):
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    print(f"\nProcessing {len(files)} images from {folder}")
    for p in files:
        run_on_image(p, classifier, detector, out_dir)


def run_on_video(video_path: Path, classifier, detector, out_path: Path,
                 every_n_frames: int = 1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"\nProcessing {video_path.name} → {out_path.name}")
    n = 0
    last_result = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every_n_frames == 0:
            last_result = process_frame(frame, classifier, detector)
        rendered = render_frame(frame, last_result) if last_result else frame
        writer.write(rendered)
        n += 1
        if n % 30 == 0:
            print(f"  frame {n}: {last_result['classifier_label']} "
                  f"({last_result['classifier_conf']:.2f}), "
                  f"{len(last_result['detections'])} boxes")
    cap.release()
    writer.release()
    print(f"Done. Wrote {n} frames.")


def run_on_webcam(classifier, detector):
    cap = cv2.VideoCapture(0)
    print("\nWebcam mode — press 'q' to quit, 'g' to toggle Grad-CAM overlay.")
    show_gradcam = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = process_frame(frame, classifier, detector)
        rendered = render_frame(frame, result, show_gradcam=show_gradcam)
        cv2.imshow("Fire + FireZone Pipeline", rendered)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            show_gradcam = not show_gradcam
    cap.release()
    cv2.destroyAllWindows()


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  type=Path)
    src.add_argument("--folder", type=Path)
    src.add_argument("--video",  type=Path)
    src.add_argument("--webcam", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("pipeline_outputs"))
    parser.add_argument("--keras-model", type=Path, default=KERAS_MODEL_PATH)
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHTS)
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True, parents=True)

    classifier = FireClassifier(args.keras_model)
    detector = FireZoneDetector(args.yolo_weights)

    if args.image:
        run_on_image(args.image, classifier, detector, args.out)
    elif args.folder:
        run_on_folder(args.folder, classifier, detector, args.out)
    elif args.video:
        run_on_video(args.video, classifier, detector,
                     args.out / f"{args.video.stem}_pipeline.mp4")
    elif args.webcam:
        run_on_webcam(classifier, detector)


if __name__ == "__main__":
    main()