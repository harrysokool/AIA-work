import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
from paddleocr import PaddleOCR


# -----------------------------
# IMPORTANT: disable the model hoster connectivity check
# -----------------------------
# You saw: "Checking connectivity to the model hosters..."
# In locked-down environments, this can hang.
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"


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
# NOTE: start with use_angle_cls=False to avoid stalls.
# If this works, you can switch it back to True later.
_PADDLE_OCR = PaddleOCR(
    use_angle_cls=False,
    lang="en",  # switch to "ch" if you have mixed Chinese in receipts
)


def _ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _paddle_run(img) -> object:
    """
    PaddleOCR API compatibility:
    Newer versions: predict(img) and do NOT pass cls
    Older versions: ocr(img, cls=True)
    """
    img = _ensure_bgr(img)
    if img is None:
        return None

    # Diagnostic timing
    t0 = time.time()

    if hasattr(_PADDLE_OCR, "predict"):
        out = _PADDLE_OCR.predict(img)
    else:
        out = _PADDLE_OCR.ocr(img, cls=True)

    dt = time.time() - t0
    return out, dt


# -----------------------------
# OCR helpers (PaddleOCR)
# -----------------------------
def ocr_lines_from_paddle(img, min_conf: float = 0.0, debug_tag: str = "") -> List[str]:
    if img is None:
        return []

    print(f"[OCR] start {debug_tag}")
    (result, dt) = _paddle_run(img)
    print(f"[OCR] done  {debug_tag}  time={dt:.2f}s")

    if not result:
        return []

    # Normalize output into list of dets
    dets = None
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], list):
            dets = result[0]
        else:
            dets = result
    else:
        return []

    if not dets:
        return []

    items = []
    for det in dets:
        try:
            box, (text, conf) = det
        except Exception:
            continue

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
        x_min = min(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = (y_min + y_max) / 2.0

        items.append(
            {
                "text": text,
                "conf": conf,
                "x_min": x_min,
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

        thresh = max(14.0, 0.9 * max(current_h, it["height"]))
        if abs(it["y_center"] - current_y) <= thresh:
            current.append(it)
            current_y = (current_y * (len(current) - 1) + it["y_center"]) / len(current)
            current_h = max(current_h, it["height"])
        else:
            lines.append(current)
            current = [it]
            current_y = it["y_center"]
            current_h = it["height"]

    if current:
        lines.append(current)

    out_lines: List[str] = []
    for line_items in lines:
        line_items.sort(key=lambda d: d["x_min"])
        s = _norm_spaces(" ".join(d["text"] for d in line_items))
        if len(s) >= 2:
            out_lines.append(s)

    return out_lines


def multi_pass_ocr_lines(img, debug_tag: str = "") -> List[str]:
    return ocr_lines_from_paddle(img, min_conf=0.0, debug_tag=debug_tag)


# -----------------------------
# Candidate extraction + scoring
# -----------------------------
@dataclass
class Candidate:
    name: str
    evidence: str
    source: str
    region: str
    score: int


def looks_like_non_doctor_line(line: str) -> bool:
    t = normalize_text(line)
    return any(re.search(p, t) for p in NON_DOCTOR_ROLE_PATTERNS)


def clean_name(raw: str) -> str:
    s = raw.strip()

    s_lower = normalize_text(s)
    for w in STOPWORDS_AFTER_NAME:
        idx = s_lower.find(f" {w}")
        if idx != -1:
            s = s[:idx].strip()
            break

    for pat in DEGREE_SUFFIX_PATTERNS:
        m = re.search(pat, normalize_text(s))
        if m:
            s = s[: m.start()].strip()
            break

    s = re.sub(r"[^A-Za-z\s\.,]", " ", s)
    s = _norm_spaces(s)

    s = re.sub(r"^\s*dr\s*\.\s*", "Dr. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*dr\s+", "Dr. ", s, flags=re.IGNORECASE)

    return s


def extract_from_doctor_field(line: str) -> Optional[str]:
    patterns = [
        r"\bdoctor\s*name\b\s*[:\-]?\s*(.+)$",
        r"\battending\s*doctor\b\s*[:\-]?\s*(.+)$",
        r"\bdoctor\b\s*[:\-]?\s*(.+)$",
        r"\bphysician\b\s*[:\-]?\s*(.+)$",
        r"\bconsultant\b\s*[:\-]?\s*(.+)$",
        r"\bdr\s*name\b\s*[:\-]?\s*(.+)$",
        r"(醫生|医生)\s*[:：\-]?\s*(.+)$",
    ]

    for pat in patterns:
        m = re.search(pat, line, flags=re.IGNORECASE)
        if not m:
            continue

        rest = m.group(m.lastindex).strip()
        name = clean_name(rest)
        if len(name.replace("Dr. ", "").split()) < 2:
            return None
        return name

    return None


def extract_from_dr_line(line: str) -> Optional[str]:
    m = re.search(r"^\s*dr\.?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        return None

    rest = m.group(1).strip()
    name = clean_name("Dr. " + rest)

    if len(name.replace("Dr. ", "").split()) < 2:
        return None

    return name


def extract_doctor_name_from_neighbors(lines: List[str], i: int) -> Optional[str]:
    t = normalize_text(lines[i])
    if not re.search(r"\bdoctor\b|\bphysician\b|醫生|医生", t):
        return None

    direct = extract_from_doctor_field(lines[i])
    if direct:
        return direct

    for j in (i + 1, i + 2):
        if j >= len(lines):
            break
        cand_line = lines[j].strip()
        if not cand_line:
            continue
        if looks_like_non_doctor_line(cand_line):
            continue

        name = clean_name(cand_line)
        if len(name.replace("Dr. ", "").split()) >= 2:
            return name

    return None


def generate_candidates(lines: List[str], region: str) -> List[Candidate]:
    cands: List[Candidate] = []

    for i, line in enumerate(lines):
        if looks_like_non_doctor_line(line):
            continue

        t = normalize_text(line)

        nm = extract_doctor_name_from_neighbors(lines, i)
        if nm:
            sc = 70
            if "doctor name" in t:
                sc += 20
            elif re.search(r"\bdoctor\b", t):
                sc += 10
            sc += {"top": 15, "full": 10, "bottom": 5}.get(region, 0)

            evidence = line
            if i + 1 < len(lines):
                evidence = f"{line} | NEXT: {lines[i+1]}"
            cands.append(Candidate(nm, evidence, "doctor_field", region, sc))

        if re.search(r"^\s*dr\.?\s+", t) or re.search(r"\bdr\.\s*[A-Za-z]", line):
            nm2 = extract_from_dr_line(line)
            if nm2:
                sc = 40 + {"top": 12, "full": 8, "bottom": 10}.get(region, 0)
                cands.append(Candidate(nm2, line, "dr_line", region, sc))

    return cands


def dedupe_candidates(cands: List[Candidate]) -> List[Candidate]:
    by_key: Dict[str, Candidate] = {}

    def key_of(name: str) -> str:
        k = normalize_text(name).replace(".", "").replace(",", "").strip()
        return _norm_spaces(k)

    for c in cands:
        k = key_of(c.name)
        if k in by_key:
            by_key[k].score += 15
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
    name = re.sub(r"[.,]", "", name)
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
        lines = multi_pass_ocr_lines(
            region_img, debug_tag=f"{os.path.basename(image_path)}::{region}"
        )

        if debug:
            print(f"\n--- OCR ({region}) first 80 lines ---")
            print("\n".join(lines[:80]))

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
    conf = 0.90 if best.source == "doctor_field" else 0.80

    return {
        "doctor_name": clean_name1(best.name),
        "confidence": conf,
        "evidence": [best.evidence],
        "meta": {"source": best.source, "region": best.region, "score": best.score},
    }


if __name__ == "__main__":
    input_dir = "preprocessed_out"

    # Only run one file first so we can see exactly where it hangs
    test_file = "sample2__ocr_gray.png"
    test_path = os.path.join(input_dir, test_file)

    print(f"Testing single file: {test_path}")
    out = extract_doctor_name(test_path, debug=True)
    print("Result:", out)
