import os
from pathlib import Path
import csv
import time

from fetch_doctor import load_doctors_set
from extract_doctors import extract_doctor_name

# Input directory (raw images only)
RAW_DIR = Path("doctor_receipts")
OUT_CSV = Path("result.csv")  # or Path("out") / "result.csv"


def write_csv(rows: list[dict], filepath: Path) -> None:
    if not rows:
        print("No rows to write.")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so Excel opens Chinese correctly
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {filepath.resolve()}")


def _is_valid_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _norm_name(s: str | None) -> str | None:
    if not s:
        return None
    # normalize whitespace, remove common prefixes, lowercase for matching
    t = s.strip().replace("\u3000", " ")
    t = " ".join(t.split())
    # remove 'Dr' prefixes commonly found in OCR outputs
    for p in ("Dr. ", "Dr ", "DR. ", "DR "):
        if t.startswith(p):
            t = t[len(p) :]
            break
    return t.casefold()


def run_pipeline():
    print("========================================")
    print("      HK Doctor Receipt Verification     ")
    print("========================================\n")

    # 1) Load doctor list
    print("[1] Loading HKMC registered doctors list...")
    hk_doctors = load_doctors_set()  # expected to be an iterable (e.g., set)
    # normalize registry upfront for robust matching
    hk_doctors_norm = {_norm_name(n) for n in hk_doctors if _norm_name(n)}
    print(f"[INFO] Loaded {len(hk_doctors)} doctors.\n")

    # 2) Extract doctor names directly from RAW receipts (no preprocessing)
    print("[2] Extracting doctor names from raw images...\n")

    results: list[dict] = []

    if not RAW_DIR.exists():
        print(f"[ERROR] Input directory not found: {RAW_DIR.resolve()}")
        return results

    # Sort files for deterministic processing
    for filename in sorted(os.listdir(RAW_DIR)):
        if not _is_valid_image_file(filename):
            continue

        image_path = RAW_DIR / filename
        print(f"--- Processing {filename} ---")

        try:
            result = extract_doctor_name(str(image_path), debug=False)
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

        # 3) Check if doctor is in HK Doctor lists
        norm_doc = _norm_name(doctor_name)
        is_registered = norm_doc in hk_doctors_norm if norm_doc else False

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

    # 4) Export CSV for Excel  ✅ FIXED: pass results and a Path
    write_csv(results, OUT_CSV)

    print("\n========================================")
    print("            Pipeline Completed          ")
    print("========================================")

    return results


if __name__ == "__main__":

    tic = time.perf_counter()
    output = run_pipeline()
    toc = time.perf_counter()

    print(f"Time took: {toc - tic:0.4f}s")
