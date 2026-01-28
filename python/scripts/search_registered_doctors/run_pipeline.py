import os
from fetch_doctor import load_doctors_set
from extract_doctors import extract_doctor_name

# Input directory (raw images only)
RAW_DIR = "doctor_receipts"


def _is_valid_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def run_pipeline():
    print("========================================")
    print("      HK Doctor Receipt Verification     ")
    print("========================================\n")

    # 1) Load doctor list
    print("[1] Loading HKMC registered doctors list...")
    hk_doctors = load_doctors_set()
    print(f"[INFO] Loaded {len(hk_doctors)} doctors.\n")

    # 2) Extract doctor names directly from RAW receipts (no preprocessing)
    print("[2] Extracting doctor names from raw images...\n")

    results = []

    for filename in os.listdir(RAW_DIR):
        if not _is_valid_image_file(filename):
            continue

        image_path = os.path.join(RAW_DIR, filename)

        print(f"--- Processing {filename} ---")

        try:
            result = extract_doctor_name(image_path, debug=False)
        except Exception as e:
            result = None
            print(f"[ERROR] OCR failed: {e}")

        if result:
            doctor_name = result.get("doctor_name")
            confidence = float(result.get("confidence", 0.0) or 0.0)
        else:
            doctor_name = None
            confidence = 0.0

        print("Extracted name:  ", doctor_name)
        print("Confidence:      ", confidence)

        # 3) Check if doctor is in HK Doctor list
        is_registered = doctor_name in hk_doctors if doctor_name else False

        print("HK Registered?:  ", is_registered)
        print("")

        results.append(
            {
                "file": filename,
                "doctor_name": doctor_name,
                "confidence": confidence,
                "registered": is_registered,
            }
        )

    print("\n========================================")
    print("            Pipeline Completed          ")
    print("========================================")

    return results


if __name__ == "__main__":
    output = run_pipeline()

    print("\nSummary:")
    for item in output:
        print(item)
