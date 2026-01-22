import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import easyocr

# Optional: Set Tesseract path on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("L")  # Grayscale
    img = img.filter(ImageFilter.MedianFilter())  # Reduce noise
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Boost contrast
    return img


def ocr_with_tesseract(img):
    return pytesseract.image_to_string(img, lang="eng")


def ocr_with_easyocr(image_path):
    reader = easyocr.Reader(["en"], gpu=False)  # Offline
    result = reader.readtext(image_path, detail=0)
    return "\n".join(result)


if __name__ == "__main__":
    image_path = "doctor_receipt.jpg"
    img = preprocess_image(image_path)

    print("=== Tesseract OCR ===")
    print(ocr_with_tesseract(img))

    print("\n=== EasyOCR ===")
    print(ocr_with_easyocr(image_path))
