import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import easyocr


# =========================================================
# Config: anchors + normalization (same as your original)
# =========================================================
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


# =========================================================
# EasyOCR: reader cache + helpers
# =========================================================

# Global cache so we don't re-initialize EasyOCR.Reader repeatedly.
_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool], easyocr.Reader] = {}


def get_easyocr_reader(
    langs: Tuple[str, ...] = ("en",), gpu: bool = False
) -> easyocr.Reader:
    """
    Lazily create and cache an EasyOCR Reader for the given languages.
    """
    key = (tuple(langs), bool(gpu))
    if key not in _READER_CACHE:
        _READER_CACHE[key] = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
    return _READER_CACHE[key]


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensure a 3-channel RGB image for EasyOCR.
    """
    if img is None:
        return img
    if img.ndim == 2:
        # grayscale -> RGB
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # BGR -> RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _group_boxes_into_lines(
    results: List[Tuple[List[List[float]], str, float]], img_h: int
) -> List[str]:
    """
    Convert EasyOCR detections into line-wise strings by grouping on y-centers
    and ordering left->right within each line.

    results: list of (bbox, text, conf) where bbox is 4 points [[x1,y1],...[x4,y4]]
    """
    if not results:
        return []

    # Compute y-center and x-left for each detection
    tokens = []
    for bbox, text, conf in results:
        if not text or (isinstance(conf, (int, float)) and conf < 0):
            continue
        pts = np.array(bbox, dtype=float)
        y_center = float(pts[:, 1].mean())
        x_left = float(pts[:, 0].min())
        tokens.append((y_center, x_left, text.strip()))

    if not tokens:
        return []

    # Cluster into lines by y with an adaptive threshold
    tokens.sort(key=lambda x: (x[0], x[1]))
    y_thresh = max(8.0, img_h * 0.012)  # ~1.2% of height or 8px, whichever larger

    lines: List[List[Tuple[float, float, str]]] = []
    for y, x, t in tokens:
        if not lines:
            lines.append([(y, x, t)])
            continue
        # If close to last line's average y, append; else start new line
        last = lines[-1]
        last_y = sum(p[0] for p in last) / len(last)
        if abs(y - last_y) <= y_thresh:
            last.append((y, x, t))
        else:
            lines.append([(y, x, t)])

    # Sort each line left->right; join tokens
    out_lines = []
    for line in lines:
        line.sort(key=lambda p: p[1])
        sent = _norm_spaces(" ".join(p[2] for p in line if p[2]))
        if len(sent) >= 2:
            out_lines.append(sent)
    return out_lines


def ocr_lines_from_easyocr(
    img: np.ndarray,
    langs: Tuple[str, ...] = ("en",),
    gpu: bool = False,
    min_conf: float = 0.0,
    paragraph: bool = False,
) -> List[str]:
    """
    Run EasyOCR and reconstruct line strings.
    paragraph=True lets EasyOCR merge detections into longer chunks; we still re-group to lines.
    """
    reader = get_easyocr_reader(langs=langs, gpu=gpu)
    rgb = _to_rgb(img)
    results = reader.readtext(rgb, detail=True, paragraph=paragraph)
    # Keep entries >= min_conf (or when conf not provided)
    filtered = []
    for item in results:
        if len(item) == 3:
            bbox, text, conf = item
        else:
            # Some versions return (bbox, text, conf, ...). We only care first 3.
            bbox, text, conf = item[0], item[1], item[2] if len(item) > 2 else 0.0
        if (isinstance(conf, (int, float)) and conf < min_conf) or not text:
            continue
        filtered.append((bbox, text, conf))

    h = rgb.shape[0]
    return _group_boxes_into_lines(filtered, img_h=h)


# =========================================================
# OCR scoring (same logic you had)
# =========================================================
def score_ocr_text(lines: List[str]) -> int:
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


# =========================================================
# Multi-pass OCR strategy (replacing your Tesseract PSM tries)
# =========================================================
def _autocontrast(gray: np.ndarray, clip_hist_percent: float = 1.0) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cdf = hist.cumsum()
    total = float(cdf[-1])
    low = int(np.searchsorted(cdf, total * clip_hist_percent / 100.0))
    high = int(np.searchsorted(cdf, total * (1 - clip_hist_percent / 100.0)))
    if high <= low:
        return gray
    scale = 255.0 / (high - low)
    adjusted = ((gray - low) * scale).clip(0, 255).astype(np.uint8)
    return adjusted


def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return th


def _prepare_variants(img: np.ndarray) -> List[np.ndarray]:
    """
    Prepare a few pre-processing variants to improve recognition.
    Returns color images (BGR) suitable for conversion to RGB later.
    """
    # Start from color BGR
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ac = _autocontrast(gray, 1.0)
    bin_img = _binarize(ac)

    # Create 3-channel versions for EasyOCR
    v1 = bgr  # original
    v2 = cv2.cvtColor(ac, cv2.COLOR_GRAY2BGR)  # autocontrast
    v3 = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)  # Otsu binary

    # Also try a mild upscale for small text
    h, w = gray.shape[:2]
    upscale = cv2.resize(
        v2, (int(w * 1.25), int(h * 1.25)), interpolation=cv2.INTER_CUBIC
    )

    return [v1, v2, v3, upscale]


def multi_pass_ocr_lines(
    img: np.ndarray,
    langs: Tuple[str, ...] = ("en",),
    gpu: bool = False,
) -> List[str]:
    """
    Try several pre-processing variants and paragraph modes; pick the best by anchor score.
    """
    variants = _prepare_variants(img)
    best_lines: List[str] = []
    best_score = -10

    for paragraph in (False, True):
        for v in variants:
            lines = ocr_lines_from_easyocr(
                v, langs=langs, gpu=gpu, min_conf=0.0, paragraph=paragraph
            )
            s = score_ocr_text(lines)
            if s > best_score:
                best_score = s
                best_lines = lines

    return best_lines


# =========================================================
# Candidate extraction + scoring (same as your original)
# =========================================================
@dataclass
class Candidate:
    name: str
    evidence: str
    source: str  # "doctor_field" or "dr_line"
    region: str  # "top", "full", or "bottom"
    score: int


def looks_like_non_doctor_line(line: str) -> bool:
    t = normalize_text(line)
    return any(re.search(p, t) for p in NON_DOCTOR_ROLE_PATTERNS)


def clean_name(raw: str) -> str:
    """
    Clean and normalize extracted name.
    Supports title case and ALL CAPS with commas.
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
    Extract name from explicit Doctor / Doctor Name / Physician field lines.
    """
    m = re.search(r"\bdoctor\s*name\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bdoctor\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bphysician\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        # Chinese terms
        m = re.search(r"(醫生|医生)\s*[:：\-]?\s*(.+)$", line)
        if m:
            rest = m.group(2).strip()
        else:
            return None
    else:
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
            # Prefer doctor_field over dr_line if duplicate
            if existing.source != "doctor_field" and c.source == "doctor_field":
                c.score = existing.score
                by_key[k] = c
            else:
                by_key[k] = existing
        else:
            by_key[k] = c

    return sorted(by_key.values(), key=lambda x: x.score, reverse=True)


# =========================================================
# Region cropping (same as your original)
# =========================================================
def crop_region(img: np.ndarray, region: str) -> np.ndarray:
    h, w = img.shape[:2]
    if region == "top":
        return img[0 : int(h * 0.33), :]
    if region == "bottom":
        return img[int(h * 0.67) : h, :]
    return img


def clean_name_for_output(name: str) -> str:
    """
    Output format: uppercase, no punctuation, no leading DR.
    """
    name = re.sub(r"[.,]", "", name)
    name = " ".join(name.split()).upper()
    name = name.replace("DR ", "").strip()
    return name


# =========================================================
# Public API
# =========================================================
def extract_doctor_name(
    image_path: str,
    debug: bool = False,
    langs: Tuple[str, ...] = ("en",),  # add 'ch_tra','ch_sim' if needed
    gpu: bool = False,
) -> Dict:
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    all_candidates: List[Candidate] = []

    for region in ["top", "full", "bottom"]:
        region_img = crop_region(img_color, region)
        lines = multi_pass_ocr_lines(region_img, langs=langs, gpu=gpu)

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

    return {
        "doctor_name": clean_name_for_output(best.name),
        "confidence": conf,
        "evidence": [best.evidence],
        "meta": {"source": best.source, "region": best.region, "score": best.score},
    }


# =========================================================
# CLI / batch runner (same folder layout/behavior as yours)
# =========================================================
def _is_valid_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


if __name__ == "__main__":
    # Adjust languages here if your docs include Chinese:
    # langs = ("en", "ch_tra", "ch_sim")
    langs = ("en",)
    gpu = False

    input_dir = "preprocessed_out"

    for filename in os.listdir(input_dir):
        if "__ocr_binary" not in filename:
            continue
        if not _is_valid_image_file(filename):
            continue

        p_bin = os.path.join(input_dir, filename)
        p_gray = p_bin.replace("__ocr_binary", "__ocr_gray")

        print(f"Processing {filename}...")

        # Try binary first (as your original code does)
        doctor: Optional[Dict] = None
        try:
            doctor = extract_doctor_name(p_bin, debug=False, langs=langs, gpu=gpu)
        except Exception as e:
            # print("Binary error:", e)
            doctor = None

        # Decide if fallback is needed
        doctor_name = None
        conf = 0.0
        if isinstance(doctor, dict):
            doctor_name = doctor.get("doctor_name")
            conf = float(doctor.get("confidence", 0.0) or 0.0)

        # Fallback to grayscale if no name (or low confidence)
        if (not doctor_name) and os.path.exists(p_gray):
            print("Binary failed → trying grayscale")
            try:
                doctor = extract_doctor_name(p_gray, debug=False, langs=langs, gpu=gpu)
            except Exception as e:
                print("Gray error:", e)
                doctor = None

        print("Result:", doctor)
        print()
