import re
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2

from paddleocr import PaddleOCR


# -----------------------------
# Config: anchors + normalization
# -----------------------------
ANCHOR_PATTERNS = [
    r"doctor\s*name",
    r"\bdoctor\b",
    r"\bphysician\b",
    r"\battending\b",
    r"\bconsultant\b",
    r"\bsurgeon\b",
    r"\bdr\b\.?",  # dr / dr.
    r"醫生",
    r"医生",
]

NON_DOCTOR_ROLE_PATTERNS = [
    r"\bphysiotherapist\b",
    r"\bphysio\b",
    r"\btherapist\b",
    r"\bnurse\b",
    r"\bchiropractor\b",
]

STOPWORDS_AFTER_NAME = [
    "patient",
    "code",
    "date",
    "transaction",
    "receipt",
    "amount",
    "total",
    "paid",
    "tel",
    "fax",
    "address",
    "room",
    "reg",
    "registration",
]

DEGREE_SUFFIX_PATTERNS = [
    r"\bmbbs\b",
    r"\bmd\b",
    r"\bms\b",
    r"\bphd\b",
    r"\bfrcs\b",
    r"\bmrcp\b",
    r"\bf\.h\.k\.a\.m\b",
    r"\bf\.c\.s\b",
    r"\bf\.r\.c\.s\b",
]


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s: str) -> str:
    """
    Normalize common OCR typos + casing for anchor matching.
    """
    t = s.lower()
    t = (
        t.replace("doetor", "doctor")
        .replace("doct0r", "doctor")
        .replace("docter", "doctor")
        .replace("docior", "doctor")
    )
    t = t.replace("phvsician", "physician").replace("physlcian", "physician")
    return t


# -----------------------------
# PaddleOCR singleton
# -----------------------------
# lang="en" is usually best for English receipts.
# If you truly have mixed Chinese/English and want OCR to read Chinese too, try lang="ch".
_PADDLE_OCR = PaddleOCR(
    use_angle_cls=True,
    lang="en",
)


# -----------------------------
# OCR helpers (PaddleOCR)
# -----------------------------
def ocr_lines_from_paddle(img_bgr, min_conf: float = 0.0) -> List[str]:
    """
    Run PaddleOCR and reconstruct text lines by grouping boxes with similar y.
    Input: BGR or grayscale image (we convert to BGR if needed).
    Output: list of reconstructed lines (strings).
    """
    if img_bgr is None:
        return []

    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    result = _PADDLE_OCR.ocr(img_bgr, cls=True)

    items = []
    # result is typically: [ [ [box, (text, conf)], ... ] ]
    if not result or not result[0]:
        return []

    for det in result[0]:
        box, (text, conf) = det
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        if conf < min_conf:
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min, _ = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = (y_min + y_max) / 2.0

        items.append(
            {
                "text": text,
                "conf": conf,
                "x_min": x_min,
                "y_min": y_min,
                "y_center": y_center,
                "height": max(1.0, (y_max - y_min)),
            }
        )

    if not items:
        return []

    # Sort by vertical position first
    items.sort(key=lambda d: (d["y_center"], d["x_min"]))

    # Group into lines
    lines: List[List[dict]] = []
    current: List[dict] = []
    current_y: Optional[float] = None
    current_h: float = 0.0

    for it in items:
        if current_y is None:
            current = [it]
            current_y = it["y_center"]
            current_h = it["height"]
            continue

        # Threshold adapts to text size; receipts vary a lot
        thresh = max(10.0, 0.6 * max(current_h, it["height"]))
        if abs(it["y_center"] - current_y) <= thresh:
            current.append(it)
            # update running average y for stability
            current_y = (current_y * (len(current) - 1) + it["y_center"]) / len(current)
            current_h = max(current_h, it["height"])
        else:
            lines.append(current)
            current = [it]
            current_y = it["y_center"]
            current_h = it["height"]

    if current:
        lines.append(current)

    # Sort each line left-to-right and join
    out_lines: List[str] = []
    for line_items in lines:
        line_items.sort(key=lambda d: d["x_min"])
        s = _norm_spaces(" ".join(d["text"] for d in line_items))
        if len(s) >= 2:
            out_lines.append(s)

    return out_lines


def score_ocr_text(lines: List[str]) -> int:
    """
    Score OCR output based on whether we see strong anchors.
    """
    joined = "\n".join(lines)
    t = normalize_text(joined)
    score = 0

    if "doctor name" in t:
        score += 6
    if re.search(r"\bdoctor\b", t):
        score += 4
    if re.search(r"\bphysician\b", t):
        score += 3
    if re.search(r"\bdr\b\.?", t):
        score += 2
    if "醫生" in joined or "医生" in joined:
        score += 2

    if len(lines) < 5:
        score -= 2

    return score


def multi_pass_ocr_lines(img) -> List[str]:
    """
    With PaddleOCR, multi-PSM isn't relevant.
    We can still do a couple passes with different min_conf if you want.
    Here we just do one pass (keeps behavior stable).
    """
    return ocr_lines_from_paddle(img, min_conf=0.0)


# -----------------------------
# Candidate extraction + scoring
# -----------------------------
@dataclass
class Candidate:
    name: str
    evidence: str
    source: str  # "doctor_field" or "dr_line"
    region: str  # "top", "full", "bottom"
    score: int


def looks_like_non_doctor_line(line: str) -> bool:
    t = normalize_text(line)
    return any(re.search(p, t) for p in NON_DOCTOR_ROLE_PATTERNS)


def clean_name(raw: str) -> str:
    """
    Clean and normalize extracted name.
    Keeps LASTNAME FIRST order as it appears in the receipt.
    Does NOT attempt to reorder tokens.
    """
    s = raw.strip()

    # Stop at common stopwords if OCR glued multiple fields
    s_lower = normalize_text(s)
    for w in STOPWORDS_AFTER_NAME:
        idx = s_lower.find(f" {w}")
        if idx != -1:
            s = s[:idx].strip()
            break

    # Remove degree suffixes
    for pat in DEGREE_SUFFIX_PATTERNS:
        m = re.search(pat, normalize_text(s))
        if m:
            s = s[: m.start()].strip()
            break

    # Keep letters, spaces, dots, commas
    s = re.sub(r"[^A-Za-z\s\.,]", " ", s)
    s = _norm_spaces(s)

    # Normalize leading Dr
    s = re.sub(r"^\s*dr\s*\.\s*", "Dr. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*dr\s+", "Dr. ", s, flags=re.IGNORECASE)

    return s


def extract_from_doctor_field(line: str) -> Optional[str]:
    """
    Extract name from explicit Doctor / Doctor Name field lines.
    Example: "Doctor: Lee Dai Shing"
    """
    m = re.search(r"\bdoctor\s*name\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bdoctor\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        return None

    rest = m.group(1).strip()
    name = clean_name(rest)

    if len(name.replace("Dr. ", "").split()) < 2:
        return None

    return name


def extract_from_dr_line(line: str) -> Optional[str]:
    """
    Extract from lines starting with Dr / DR.
    """
    m = re.search(r"^\s*dr\.?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        return None

    rest = m.group(1).strip()
    name = clean_name("Dr. " + rest)

    if len(name.replace("Dr. ", "").split()) < 2:
        return None

    return name


def generate_candidates(lines: List[str], region: str) -> List[Candidate]:
    cands: List[Candidate] = []

    for line in lines:
        if looks_like_non_doctor_line(line):
            continue

        t = normalize_text(line)

        # Explicit doctor field lines
        if any(
            re.search(p, t)
            for p in [
                r"\bdoctor\b",
                r"doctor\s*name",
                r"\bphysician\b",
                r"醫生",
                r"医生",
            ]
        ):
            nm = extract_from_doctor_field(line)
            if nm:
                sc = 0
                if "doctor name" in t:
                    sc += 80
                elif re.search(r"\bdoctor\b", t):
                    sc += 65
                elif (
                    re.search(r"\bphysician\b", t)
                    or ("醫生" in line)
                    or ("医生" in line)
                ):
                    sc += 55

                sc += {"top": 15, "full": 10, "bottom": 5}.get(region, 0)
                cands.append(Candidate(nm, line, "doctor_field", region, sc))

        # Fallback DR lines
        if re.search(r"^\s*dr\.?\s+", t) or re.search(r"\bdr\.\s*[A-Za-z]", line):
            nm = extract_from_dr_line(line)
            if nm:
                sc = 40
                sc += {"top": 12, "full": 8, "bottom": 10}.get(region, 0)
                cands.append(Candidate(nm, line, "dr_line", region, sc))

    return cands


def dedupe_candidates(cands: List[Candidate]) -> List[Candidate]:
    """
    Merge duplicates and boost score if repeated.
    """
    by_key: Dict[str, Candidate] = {}

    def key_of(name: str) -> str:
        k = normalize_text(name)
        k = k.replace(".", "").replace(",", "").strip()
        return _norm_spaces(k)

    for c in cands:
        k = key_of(c.name)
        if k in by_key:
            existing = by_key[k]
            existing.score += 15
            if existing.source != "doctor_field" and c.source == "doctor_field":
                c.score = existing.score
                by_key[k] = c
            else:
                by_key[k] = existing
        else:
            by_key[k] = c

    return sorted(by_key.values(), key=lambda x: x.score, reverse=True)


# -----------------------------
# Region cropping
# -----------------------------
def crop_region(img, region: str):
    h, w = img.shape[:2]
    if region == "top":
        return img[0 : int(h * 0.33), :]
    if region == "bottom":
        return img[int(h * 0.67) : h, :]
    return img


def clean_name1(name: str) -> str:
    # remove punctuation
    name = re.sub(r"[.,]", "", name)
    # normalize spaces + uppercase, remove DR prefix token
    return " ".join(name.split()).upper().replace("DR ", "")


# -----------------------------
# Public API
# -----------------------------
def extract_doctor_name(image_path: str, debug: bool = False) -> Dict:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    all_candidates: List[Candidate] = []

    for region in ["top", "full", "bottom"]:
        region_img = crop_region(img, region)
        # PaddleOCR expects BGR or will still work with gray; we convert inside
        lines = multi_pass_ocr_lines(region_img)

        if debug:
            print(f"\n--- OCR ({region}) first 40 lines ---")
            print("\n".join(lines[:40]))

        all_candidates.extend(generate_candidates(lines, region=region))

    cands = dedupe_candidates(all_candidates)

    if not cands:
        return {
            "doctor_name": None,
            "confidence": 0.0,
            "evidence": [],
            "reason": "No explicit doctor anchor found in OCR output",
        }

    best = cands[0]

    if best.source == "doctor_field" and best.score >= 70:
        conf = 0.95
    elif best.source == "doctor_field":
        conf = 0.90
    elif best.source == "dr_line" and best.score >= 55:
        conf = 0.85
    else:
        conf = 0.80

    # IMPORTANT: we do NOT reorder tokens; we keep "Lee Dai Shing" as-is.
    return {
        "doctor_name": clean_name1(best.name),
        "confidence": conf,
        "evidence": [best.evidence],
        "meta": {"source": best.source, "region": best.region, "score": best.score},
    }


if __name__ == "__main__":
    input_dir = "preprocessed_out"

    for filename in os.listdir(input_dir):
        if "__ocr_binary" not in filename:
            continue

        p_bin = os.path.join(input_dir, filename)
        p_gray = p_bin.replace("__ocr_binary", "__ocr_gray")

        print(f"Processing {filename}...")

        doctor = None

        # Try binary
        try:
            doctor = extract_doctor_name(p_bin, debug=False)
        except Exception:
            doctor = None

        # Fallback to grayscale if missing name
        if (not doctor) or (not doctor.get("doctor_name")):
            if os.path.exists(p_gray):
                print("Binary failed → trying grayscale")
                try:
                    doctor = extract_doctor_name(p_gray, debug=False)
                except Exception:
                    doctor = None

        print("Result:", doctor)
        print()
