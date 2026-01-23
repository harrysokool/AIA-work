import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import pytesseract


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
# OCR helpers
# -----------------------------
def ocr_lines_from_tesseract(img, psm: int, min_conf: float = 0.0) -> List[str]:
    """
    OCR via image_to_data and reconstruct lines using (block, par, line).
    Keeps low-confidence words to avoid losing key fields on noisy scans.
    """
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT
    )

    n = len(data["text"])
    lines: Dict[Tuple[int, int, int], List[str]] = {}

    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue

        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        # Keep almost everything; only drop if explicitly below min_conf
        if conf != -1.0 and conf < min_conf:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    out_lines = [_norm_spaces(" ".join(lines[k])) for k in sorted(lines.keys())]
    out_lines = [ln for ln in out_lines if len(ln) >= 2]
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
    Try multiple PSMs and choose the OCR output that best captures anchors.
    """
    psm_candidates = [6, 4, 11, 3]
    best_lines: List[str] = []
    best_score = -10

    for psm in psm_candidates:
        lines = ocr_lines_from_tesseract(img, psm=psm, min_conf=0.0)
        s = score_ocr_text(lines)
        if s > best_score:
            best_score = s
            best_lines = lines

    return best_lines


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

    # Normalize leading Dr (NO inline flags)
    s = re.sub(r"^\s*dr\s*\.\s*", "Dr. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*dr\s+", "Dr. ", s, flags=re.IGNORECASE)

    return s


def extract_from_doctor_field(line: str) -> Optional[str]:
    """
    Extract name from explicit Doctor / Doctor Name field lines.
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

    return {
        "doctor_name": best.name,
        "confidence": conf,
        "evidence": [best.evidence],
        "meta": {"source": best.source, "region": best.region, "score": best.score},
    }


if __name__ == "__main__":
    for p in [
        "preprocessed_out/sample1__ocr_binary.png",
        "preprocessed_out/sample2__ocr_binary.png",
        "preprocessed_out/sample3__ocr_binary.png",
        "preprocessed_out/sample4__ocr_binary.png",
        "preprocessed_out/sample5__ocr_binary.png",
    ]:
        try:
            print(p, "=>", extract_doctor_name(p, debug=False))
            print("")
        except Exception as e:
            print(p, "=> ERROR:", e)
