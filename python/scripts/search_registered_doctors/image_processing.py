import cv2
import numpy as np


def preprocess_receipt(input_path, output_path=None):
    # 1. Read in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    # 2. Upscale for small text
    img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # 3. Gentle contrast boost
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=10)

    # 4. Very light denoising (median filter preserves edges better than Gaussian)
    img = cv2.medianBlur(img, 3)

    # 5. Adaptive threshold with smaller block size and lower C
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )

    # 6. Morphological closing to reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 7. Deskew
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # 8. Save processed image
    if output_path:
        cv2.imwrite(output_path, img)

    return img


# Example usage
if __name__ == "__main__":
    processed = preprocess_receipt("sample_5.jpg", "processed_receipt.jpg")
    print("Preprocessing complete.")
