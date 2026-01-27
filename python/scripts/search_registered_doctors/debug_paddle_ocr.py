import os
import cv2
from paddleocr import PaddleOCR


# -----------------------------
# PaddleOCR setup
# -----------------------------
# If there is any Chinese on receipts, use "ch". Otherwise "en".
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
)


# -----------------------------
# Helpers
# -----------------------------
def norm_spaces(s: str) -> str:
    return " ".join(s.split())


def crop_region(img, region):
    h, w = img.shape[:2]
    if region == "top":
        return img[0 : int(h * 0.33), :]
    if region == "bottom":
        return img[int(h * 0.67) : h, :]
    return img


def _get_paddle_result(img_bgr):
    """
    PaddleOCR API compatibility:
    Newer versions: use ocr.predict(img)
    Older versions: ocr.ocr(img, cls=True) exists
    Your version: ocr.ocr is deprecated and forwards to predict; predict() does NOT accept cls.
    """
    # Ensure BGR
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # Prefer predict() for new PaddleOCR
    if hasattr(ocr, "predict"):
        return ocr.predict(img_bgr)

    # Fallback for older PaddleOCR
    return ocr.ocr(img_bgr, cls=True)


def ocr_lines_from_paddle(img_bgr):
    if img_bgr is None:
        return []

    result = _get_paddle_result(img_bgr)

    # PaddleOCR outputs can differ slightly by version.
    # Common shape: [ [ [box, (text, conf)], ... ] ]
    # Sometimes: result itself is already the list of dets.
    dets = None

    if not result:
        return []

    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], list):
            # likely [ [dets] ]
            dets = result[0]
        else:
            # already det list
            dets = result
    else:
        return []

    if not dets:
        return []

    items = []
    for det in dets:
        try:
            box, (text, conf) = det
        except Exception:
            # unexpected det shape
            continue

        if not text:
            continue

        text = str(text).strip()
        if not text:
            continue

        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min = min(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = (y_min + y_max) / 2.0

        items.append(
            {
                "text": text,
                "conf": conf,
                "x_min": x_min,
                "y_center": y_center,
                "height": max(1.0, y_max - y_min),
            }
        )

    if not items:
        return []

    # Sort by vertical then horizontal
    items.sort(key=lambda x: (x["y_center"], x["x_min"]))

    # Group into lines
    lines = []
    current = []
    current_y = None
    current_h = 0.0

    for it in items:
        if current_y is None:
            current = [it]
            current_y = it["y_center"]
            current_h = it["height"]
            continue

        thresh = max(14.0, 0.9 * max(current_h, it["height"]))
        if abs(it["y_center"] - current_y) <= thresh:
            current.append(it)
            current_y = (current_y * (len(current) - 1) + it["y_center"]) / len(current)
            current_h = max(current_h, it["height"])
        else:
            lines.append(current)
            current = [it]
            current_y = it["y_center"]
            current_h = it["height"]

    if current:
        lines.append(current)

    # Join line items left-to-right
    out_lines = []
    for li in lines:
        li.sort(key=lambda x: x["x_min"])
        out_lines.append(norm_spaces(" ".join(x["text"] for x in li)))

    return out_lines


def debug_image(image_path: str):
    print("\n==============================")
    print("DEBUG OCR FOR IMAGE:")
    print(image_path)
    print("==============================")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Failed to load image")
        return

    for region in ["top", "full", "bottom"]:
        region_img = crop_region(img, region)
        lines = ocr_lines_from_paddle(region_img)

        print(f"\n--- REGION: {region.upper()} | lines={len(lines)} ---")
        for i, ln in enumerate(lines[:120]):
            print(f"[{i:03d}] {ln}")


if __name__ == "__main__":
    INPUT_DIR = "preprocessed_out"

    files = sorted(os.listdir(INPUT_DIR))
    for filename in files:
        if "__ocr" not in filename:
            continue

        path = os.path.join(INPUT_DIR, filename)
        debug_image(path)

        print("\nPress ENTER for next image...")
        input()
