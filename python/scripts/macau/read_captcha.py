import base64
import json
import urllib.request
import io
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
import time


# ========================
# Paths
# ========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_PNG = BASE_DIR / "data" / "captcha" / "captcha_macau.png"
DECODED_IMAGE_PNG = BASE_DIR / "data" / "captcha" / "decoded_captcha.png"


# ========================
# Ollama config
# ========================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2-vision"  # Or try "minicpm-v" if better


PROMPT = (
    "4-character distorted CAPTCHA. Read left to right. "
    "Lowercase for curves (z not Z, s not S, g not G). "
    "Output ONLY the 4 chars: e.g. 'z3sg'"
    "Output ONLY the 4 chars, DO NOT output anything else"
)


def preprocess_image(image_path: Path) -> bytes:
    """Sharpen/denoise for better OCR accuracy"""
    img = Image.open(image_path).convert("RGB")

    # Grayscale + heavy contrast boost
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)  # Crank contrast

    # Sharpen + median filter for noise
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.5)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # Save as PNG bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def read_image_with_llama32vision(image_path: Path, model: str = MODEL) -> str:
    img_bytes = preprocess_image(image_path)
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "images": [image_b64],
        "options": {
            "temperature": 0.1,
            "num_predict": 20,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


def main():
    if not IMAGE_PNG.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PNG}")

    text = read_image_with_llama32vision(IMAGE_PNG)
    print(f"'{text}'")

    elements = text.split()
    print(elements)
    letters = []
    for i in range(4, len(elements)):
        letters.append(elements[i][0])
    ans = "".join(letters)
    print(ans)


if __name__ == "__main__":
    main()
