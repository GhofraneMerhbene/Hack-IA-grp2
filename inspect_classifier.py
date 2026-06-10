"""
inspect_classifier_v3.py
------------------------
Extends v2 with a --probe mode that runs the model on real fire/smoke/none
images to determine class order. After this script, you'll know which output
index corresponds to which label.

Usage:
    python inspect_classifier_v3.py /path/to/model.keras
    python inspect_classifier_v3.py /path/to/model.keras --probe FIRE_IMG SMOKE_IMG NONE_IMG
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_v2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

CANDIDATES = [
    ("efficientnet_v2", effnet_v2_preprocess),
    ("efficientnet",    effnet_preprocess),
    ("resnet_v2",       resnet_v2_preprocess),
    ("vgg16",           vgg16_preprocess),
    ("mobilenet_v2",    mobilenet_v2_preprocess),
]


def try_load(model_path: Path):
    last_err = None
    for name, fn in CANDIDATES:
        try:
            print(f"  Trying preprocess_input from {name}...", end=" ")
            model = keras.models.load_model(
                model_path, compile=False,
                custom_objects={"preprocess_input": fn},
            )
            print("OK")
            return model, name
        except Exception as e:
            print("nope")
            last_err = e
    raise RuntimeError(f"None worked. Last error:\n{last_err}")


def get_input_hw(model):
    shape = model.input_shape
    if isinstance(shape, list): shape = shape[0]
    return shape[1], shape[2]


def predict_image(model, img_path: Path, input_hw):
    """Load an image, resize, predict. Returns probability vector."""
    h, w = input_hw
    img = keras.utils.load_img(img_path, target_size=(h, w))
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # Model has embedded 'preprocessing' layer → feed raw 0–255
    raw = model.predict(arr, verbose=0)[0]
    # Softmax if outputs aren't probs
    if not np.isclose(raw.sum(), 1.0, atol=1e-3):
        probs = tf.nn.softmax(raw).numpy()
    else:
        probs = raw
    return probs


def probe(model, fire_path: Path, smoke_path: Path, none_path: Path, input_hw):
    """Run model on three labeled images and report which index wins."""
    print("\n" + "=" * 70)
    print("PROBE — running model on labeled images")
    print("=" * 70)

    n_classes = model.output_shape[-1]
    truth_to_path = [("fire", fire_path), ("smoke", smoke_path), ("none", none_path)]

    print(f"\n{'true label':>12}  |  argmax  |  {'probabilities':>40}")
    print("-" * 75)
    results = {}
    for true_label, img_path in truth_to_path:
        if not img_path.exists():
            print(f"  {true_label:>12}  |   ???   |  FILE NOT FOUND: {img_path}")
            continue
        probs = predict_image(model, img_path, input_hw)
        idx = int(np.argmax(probs))
        probs_str = "  ".join(f"[{i}]={p:.3f}" for i, p in enumerate(probs))
        print(f"  {true_label:>12}  |   {idx}    |  {probs_str}")
        results[true_label] = idx

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if len(results) == 3 and len(set(results.values())) == 3:
        # All three predictions distinct → we have a clear mapping
        idx_to_label = {v: k for k, v in results.items()}
        ordered = [idx_to_label[i] for i in range(n_classes)]
        print(f"\n  ✅ Class order (index → label):")
        for i, lbl in enumerate(ordered):
            print(f"     index {i} = {lbl}")
        print(f"\n  Use in your pipeline:")
        print(f"     KERAS_CLASS_NAMES = {ordered}")
    else:
        print("\n  ⚠️  The model didn't predict three distinct classes on these probes.")
        print("     This could mean:")
        print("       - The model is weak/wrong (check accuracy with verify notebook)")
        print("       - Your chosen probe images aren't clearly fire/smoke/none")
        print("       - Two labels map to the same prediction (model confusion)")
        print("\n  Try with different images that are visually unambiguous.")
        print(f"\n  Raw predictions: {results}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path, nargs="?",
                        default=Path("/Users/ghofranemerhbene/Work/HACK-IA/hackia/best_model.keras"))
    parser.add_argument("--probe", nargs=3, metavar=("FIRE_IMG", "SMOKE_IMG", "NONE_IMG"),
                        help="Three image paths: one fire, one smoke, one neither")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: {model_path} does not exist."); sys.exit(1)

    print(f"Loading: {model_path}")
    print("Trying preprocess_input candidates:\n")
    model, base_kind = try_load(model_path)

    print(f"\n✅ Loaded with preprocess_input from: {base_kind}")
    input_hw = get_input_hw(model)
    print(f"Input size: {input_hw}")
    print(f"Output width: {model.output_shape[-1]} classes")

    if args.probe:
        probe(model, Path(args.probe[0]), Path(args.probe[1]), Path(args.probe[2]), input_hw)
    else:
        print("\nTo determine class order, re-run with three real images:")
        print("  python inspect_classifier_v3.py <model_path> \\")
        print("      --probe <fire_img> <smoke_img> <none_img>")
        print("\nFor example, pick one image from each class folder of D-Fire test set:")
        print("  --probe /path/to/D-Fire-classifier/test/fire/IMG_X.jpg \\")
        print("          /path/to/D-Fire-classifier/test/smoke/IMG_Y.jpg \\")
        print("          /path/to/D-Fire-classifier/test/none/IMG_Z.jpg")


if __name__ == "__main__":
    main()