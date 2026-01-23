import cv2
import numpy as np


def _resize_up(img: np.ndarray, target_min_dim: int = 1800) -> np.ndarray:
    """
    Upscale small scans so text is easier for OCR.
    Keeps aspect ratio. If already large enough, returns unchanged.
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim >= target_min_dim:
        return img
    scale = target_min_dim / float(min_dim)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Correct uneven lighting (common in photos/scans).
    Uses a large-kernel background estimate and division normalization.
    Keeps grayscale (OCR-friendly).
    """
    # Estimate background (smooth large-scale illumination)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)

    # Avoid divide by zero
    bg = np.clip(bg, 1, 255)

    # Normalize: gray / bg * 255
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm


def _mild_sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Mild edge enhancement without harsh halos.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.25, blur, -0.25, 0)
    return sharp


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Light denoising that preserves text strokes well.
    """
    return cv2.fastNlMeansDenoising(gray, h=10)


def _deskew_if_needed(gray: np.ndarray, max_abs_angle_deg: float = 6.0) -> np.ndarray:
    """
    Robust deskew that tries to estimate skew from strong text edges.
    Only applies if angle is small and meaningful. Avoids catastrophic rotations.
    """
    # Edge map of text
    edges = cv2.Canny(gray, 50, 150)

    # Find coordinates of edges
    ys, xs = np.where(edges > 0)
    if len(xs) < 2000:  # too few edges, skip
        return gray

    coords = np.column_stack((xs, ys)).astype(np.float32)

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect angle quirks
    # angle in (-90, 0]; convert to a small rotation around 0
    if angle < -45:
        angle = 90 + angle  # e.g. -80 -> 10
    # Now angle is roughly in [-45, 45]
    # We want to rotate by -angle
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


def preprocess_for_ocr_best(input_path: str, save_dir: str | None = None) -> dict:
    """
    Returns two images:
      - 'ocr_gray': best default for modern OCR (recommended)
      - 'ocr_binary': fallback for some stubborn scans

    If save_dir is provided, saves both versions as PNG.
    """
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) Upscale if needed
    gray = _resize_up(gray, target_min_dim=1800)

    # 2) Illumination normalization (helps shadows / uneven scan)
    gray = _normalize_illumination(gray)

    # 3) Light denoise + mild sharpen
    gray = _denoise(gray)
    gray = _mild_sharpen(gray)

    # 4) Optional, safe deskew
    gray = _deskew_if_needed(gray)

    # This is your primary OCR input
    ocr_gray = gray

    # Fallback: binarized version (do NOT use morphology closing here)
    # Otsu is safer than adaptive for OCR fallback
    ocr_binary = cv2.threshold(ocr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{base}__ocr_gray.png"), ocr_gray)
        cv2.imwrite(os.path.join(save_dir, f"{base}__ocr_binary.png"), ocr_binary)

    return {"ocr_gray": ocr_gray, "ocr_binary": ocr_binary}


if __name__ == "__main__":
    out = preprocess_for_ocr_best(
        "doctor_receipts/sample5.jpg", save_dir="preprocessed_out"
    )
    print("Saved:", list(out.keys()))
