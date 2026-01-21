import re
import pytesseract
from PIL import Image
import os

# Path to Tesseract executable (adjust for your OS)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_doctor_from_receipt(image_path):
    try:
        # OCR: Convert image to text
        text = pytesseract.image_to_string(Image.open(image_path))

        # Regex to find "Doctor:" followed by a name
        doctor_pattern = r"Doctor:\s*(Dr\.?\s*)?[A-Za-z]+(?:\s+[A-Za-z]+)*"
        match = re.search(doctor_pattern, text, re.IGNORECASE)

        doctor_name = match.group(0).replace("Doctor:", "").strip() if match else None

        return {
            "file": os.path.basename(image_path),
            "doctor_name": doctor_name,
            "raw_text": text,
        }
    except Exception as e:
        return {"file": os.path.basename(image_path), "error": str(e)}


# Example: Process all receipts in a folder
def process_receipts(folder_path):
    results = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
            results.append(extract_doctor_from_receipt(os.path.join(folder_path, file)))
    return results


# Example usage
folder = r"C:\Users\hiadgr8\desktop\AIA-WORK\python\data\doctor_receipts"
data = process_receipts(folder)

for entry in data:
    print(entry)
