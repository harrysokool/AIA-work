import os
from typing import Dict, Optional

import cv2
import numpy as np


def _resize_up(img: np.ndarray, target_min_dim: int = 1800) -> np.ndarray:
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim >= target_min_dim:
        return img
    scale = target_min_dim / float(min_dim)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _resize_down_max_side(img: np.ndarray, max_side: int = 3000) -> np.ndarray:
    h, w = img.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_side:
        return img
    scale = max_side / float(cur_max)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    bg = np.clip(bg, 1, 255)
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm


def _mild_sharpen(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    # toned down vs your original to avoid "edge soup"
    sharp = cv2.addWeighted(gray, 1.10, blur, -0.10, 0)
    return sharp


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def _deskew_if_needed(gray: np.ndarray, max_abs_angle_deg: float = 6.0) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2000:
        return gray

    coords = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.8 or abs(angle) > max_abs_angle_deg:
        return gray

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_for_paddleocr(
    input_path: str,
    save_dir: Optional[str] = None,
    target_min_dim: int = 1800,
    max_side_after_processing: int = 3000,
    make_binary_fallback: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Produces outputs that work well with PaddleOCR predict() on mac:
      - ocr_gray: cleaned grayscale uint8
      - ocr_rgb: BGR (3-channel) version for safe saving (JPEG recommended)
      - ocr_binary (optional): fallback only (generally NOT recommended for PaddleOCR v5)

    If save_dir is provided, saves:
      - {base}__ocr_gray.png
      - {base}__ocr_rgb.jpg   (this is the one you should feed to PaddleOCR)
      - {base}__ocr_binary.png (optional)
    """
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {input_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray = _resize_up(gray, target_min_dim=target_min_dim)
    gray = _normalize_illumination(gray)
    gray = _denoise(gray)
    gray = _mild_sharpen(gray)
    gray = _deskew_if_needed(gray)
    gray = _resize_down_max_side(gray, max_side=max_side_after_processing)

    ocr_gray = gray
    ocr_rgb = cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2BGR)

    outputs: Dict[str, np.ndarray] = {"ocr_gray": ocr_gray, "ocr_rgb": ocr_rgb}

    if make_binary_fallback:
        # keep as fallback only; don’t use this as your default for PaddleOCR v5
        ocr_binary = cv2.threshold(
            ocr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        outputs["ocr_binary"] = ocr_binary

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]

        gray_path = os.path.join(save_dir, f"{base}__ocr_gray.png")
        rgb_path = os.path.join(save_dir, f"{base}__ocr_rgb.jpg")

        cv2.imwrite(gray_path, ocr_gray)

        # JPEG is intentionally used here to “sanitize” output for PaddleOCR
        cv2.imwrite(rgb_path, ocr_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        if make_binary_fallback:
            bin_path = os.path.join(save_dir, f"{base}__ocr_binary.png")
            cv2.imwrite(bin_path, outputs["ocr_binary"])

    return outputs


def preprocess_folder(
    input_folder: str, output_folder: str
) -> Dict[str, Dict[str, np.ndarray]]:
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(exts):
            continue

        input_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")

        outputs[filename] = preprocess_for_paddleocr(
            input_path=input_path,
            save_dir=output_folder,
            target_min_dim=1800,
            max_side_after_processing=3000,
            make_binary_fallback=True,
        )

    return outputs


if __name__ == "__main__":
    in_dir = "doctor_receipts"
    out_dir = "preprocessed_out"

    all_outputs = preprocess_folder(in_dir, out_dir)
    print("Finished processing:", list(all_outputs.keys()))
    print("Tip: feed PaddleOCR the __ocr_rgb.jpg files from preprocessed_out/")
