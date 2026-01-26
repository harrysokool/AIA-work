import os
from fetch_doctor import load_doctors_set
from image_processing import preprocess_folder
from extract_doctors import extract_doctor_name

# Input/output directories
RAW_DIR = "doctor_receipts"
PRE_DIR = "preprocessed_out"


def run_pipeline():
    print("========================================")
    print("      HK Doctor Receipt Verification     ")
    print("========================================\n")

    # -------------------------------------------
    # 1. Load HK doctors registry (from pickle)
    # -------------------------------------------
    print("[1] Loading HKMC registered doctors list...")
    hk_doctors = load_doctors_set()
    print(f"[INFO] Loaded {len(hk_doctors)} doctors.\n")

    # -------------------------------------------
    # 2. Preprocess raw images
    # -------------------------------------------
    print("[2] Preprocessing all receipt images...")
    preprocess_folder(RAW_DIR, PRE_DIR)
    print("[INFO] Preprocessing done.\n")

    # -------------------------------------------
    # 3. Extract doctor names from processed files
    # -------------------------------------------
    print("[3] Extracting doctor names...\n")

    results = []

    for filename in os.listdir(PRE_DIR):
        if "__ocr_binary" not in filename:
            continue  # skip non-binary files

        binary_path = os.path.join(PRE_DIR, filename)
        gray_path = binary_path.replace("__ocr_binary", "__ocr_gray")

        print(f"--- Processing {filename} ---")

        # Try Binary First
        try:
            result = extract_doctor_name(binary_path, debug=False)
        except Exception:
            result = None

        # Fallback to grayscale
        if (not result) or (not result.get("doctor_name")):
            if os.path.exists(gray_path):
                print("Binary failed → trying grayscale...")
                try:
                    result = extract_doctor_name(gray_path, debug=False)
                except Exception:
                    result = None

        if result:
            doctor_name = result.get("doctor_name")
            confidence = result.get("confidence")
        else:
            doctor_name = None
            confidence = 0

        print("Extracted name:  ", doctor_name)
        print("Confidence:      ", confidence)

        # -------------------------------------------
        # 4. Check if doctor is in HK Doctor list
        # -------------------------------------------
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
