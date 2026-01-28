import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import easyocr


# =========================================================
# Config
# =========================================================
ANCHOR_PATTERNS = [
    r"doctor\s*name",
    r"\bdoctor\b",
    r"\bphysician\b",
    r"\battending\b",
    r"\bconsultant\b",
    r"\bsurgeon\b",
    r"\bdr\b\.?",
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


# =========================================================
# EasyOCR: reader cache
# =========================================================
_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool], easyocr.Reader] = {}


def get_easyocr_reader(
    langs: Tuple[str, ...] = ("en",), gpu: bool = False
) -> easyocr.Reader:
    key = (tuple(langs), bool(gpu))
    if key not in _READER_CACHE:
        _READER_CACHE[key] = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
    return _READER_CACHE[key]


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================================================
# OCR grouping (same idea as yours)
# =========================================================
def _group_boxes_into_lines(
    results: List[Tuple[List[List[float]], str, float]], img_h: int
) -> List[str]:
    if not results:
        return []

    tokens = []
    for bbox, text, conf in results:
        if not text:
            continue
        pts = np.array(bbox, dtype=float)
        y_center = float(pts[:, 1].mean())
        x_left = float(pts[:, 0].min())
        tokens.append((y_center, x_left, text.strip()))

    if not tokens:
        return []

    tokens.sort(key=lambda x: (x[0], x[1]))
    y_thresh = max(8.0, img_h * 0.012)

    lines: List[List[Tuple[float, float, str]]] = []
    for y, x, t in tokens:
        if not lines:
            lines.append([(y, x, t)])
            continue
        last = lines[-1]
        last_y = sum(p[0] for p in last) / len(last)
        if abs(y - last_y) <= y_thresh:
            last.append((y, x, t))
        else:
            lines.append([(y, x, t)])

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
    min_conf: float = 0.2,
    paragraph: bool = False,
    allowlist: Optional[str] = None,
) -> List[str]:
    """
    Faster defaults:
    - min_conf bumped (filters junk)
    - allowlist optional (can speed recognition for Latin text receipts)
    """
    reader = get_easyocr_reader(langs=langs, gpu=gpu)
    rgb = _to_rgb(img)

    # Some EasyOCR versions support allowlist/blocklist kwargs; safe-guard by try/except.
    try:
        results = reader.readtext(
            rgb, detail=True, paragraph=paragraph, allowlist=allowlist
        )
    except TypeError:
        results = reader.readtext(rgb, detail=True, paragraph=paragraph)

    filtered = []
    for item in results:
        bbox, text, conf = item[0], item[1], item[2] if len(item) > 2 else 0.0
        if not text:
            continue
        if isinstance(conf, (int, float)) and conf < min_conf:
            continue
        filtered.append((bbox, text, conf))

    h = rgb.shape[0]
    return _group_boxes_into_lines(filtered, img_h=h)


# =========================================================
# Scoring
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
# Preprocess: fewer, cheaper variants (optimized)
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
    return ((gray - low) * scale).clip(0, 255).astype(np.uint8)


def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _variants_fast(img: np.ndarray) -> List[np.ndarray]:
    """
    FAST path: only 2 variants.
    """
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ac = _autocontrast(gray, 1.0)
    v1 = bgr
    v2 = cv2.cvtColor(ac, cv2.COLOR_GRAY2BGR)
    return [v1, v2]


def _variants_fallback(img: np.ndarray) -> List[np.ndarray]:
    """
    SLOWER fallback: add binarize and small upscale.
    Only used if fast path fails.
    """
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ac = _autocontrast(gray, 1.0)
    bin_img = _binarize(ac)

    v1 = bgr
    v2 = cv2.cvtColor(ac, cv2.COLOR_GRAY2BGR)
    v3 = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]
    upscale = cv2.resize(
        v2, (int(w * 1.25), int(h * 1.25)), interpolation=cv2.INTER_CUBIC
    )
    return [v1, v2, v3, upscale]


# =========================================================
# Candidate extraction (same logic)
# =========================================================
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
    m = re.search(r"\bdoctor\s*name\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bdoctor\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bphysician\b\s*[:\-]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
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

        if any(
            re.search(p, t)
            for p in [r"\bdoctor\b", r"doctor\s*name", r"\bphysician\b", r"醫生", r"医生"]
        ):
            nm = extract_from_doctor_field(line)
            if nm:
                sc = 0
                if "doctor name" in t:
                    sc += 80
                elif re.search(r"\bdoctor\b", t):
                    sc += 65
                elif re.search(r"\bphysician\b", t) or ("醫生" in line) or ("医生" in line):
                    sc += 55
                sc += {"top": 15, "full": 10, "bottom": 5}.get(region, 0)
                cands.append(Candidate(nm, line, "doctor_field", region, sc))

        if re.search(r"^\s*dr\.?\s+", t) or re.search(r"\bdr\.\s*[A-Za-z]", line):
            nm = extract_from_dr_line(line)
            if nm:
                sc = 40 + {"top": 12, "full": 8, "bottom": 10}.get(region, 0)
                cands.append(Candidate(nm, line, "dr_line", region, sc))

    return cands


def dedupe_candidates(cands: List[Candidate]) -> List[Candidate]:
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


# =========================================================
# Region cropping (optimized: smaller first)
# =========================================================
def crop_region(img: np.ndarray, region: str) -> np.ndarray:
    h, w = img.shape[:2]
    if region == "top25":
        return img[0 : int(h * 0.25), :]
    if region == "top40":
        return img[0 : int(h * 0.40), :]
    if region == "bottom":
        return img[int(h * 0.67) : h, :]
    return img


def clean_name_for_output(name: str) -> str:
    name = re.sub(r"[.,]", "", name)
    name = " ".join(name.split()).upper()
    name = name.replace("DR ", "").strip()
    return name


def _best_from_lines(lines: List[str], region: str) -> List[Candidate]:
    return generate_candidates(lines, region=region)


def _ocr_try(
    img: np.ndarray,
    region_label: str,
    langs: Tuple[str, ...],
    gpu: bool,
    paragraph: bool,
    variants: List[np.ndarray],
    allowlist: Optional[str],
) -> Tuple[List[str], List[Candidate], int]:
    best_lines: List[str] = []
    best_cands: List[Candidate] = []
    best_score = -10

    for v in variants:
        lines = ocr_lines_from_easyocr(
            v,
            langs=langs,
            gpu=gpu,
            min_conf=0.2,
            paragraph=paragraph,
            allowlist=allowlist,
        )
        s = score_ocr_text(lines)
        cands = _best_from_lines(lines, region=region_label)

        # Prefer "has candidates" strongly; then fall back to anchor score
        key = (1 if cands else 0, s)
        best_key = (1 if best_cands else 0, best_score)

        if key > best_key:
            best_lines, best_cands, best_score = lines, cands, s

        # Early exit: if we already got a strong doctor_field candidate, stop this stage
        if cands:
            top = sorted(cands, key=lambda x: x.score, reverse=True)[0]
            if top.source == "doctor_field" and top.score >= 80:
                return best_lines, best_cands, best_score

    return best_lines, best_cands, best_score


# =========================================================
# Public API (optimized staged pipeline)
# =========================================================
def extract_doctor_name(
    image_path: str,
    debug: bool = False,
    langs: Tuple[str, ...] = ("en",),
    gpu: bool = False,
) -> Dict:
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Allowlist for English-ish receipts (optional). You can set to None to disable.
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-()/"

    all_candidates: List[Candidate] = []
    evidence_lines: List[str] = []

    # Stage plan:
    # 1) top25, fast variants, paragraph=False
    # 2) top40, fast variants, paragraph=False
    # 3) full, fast variants, paragraph=False
    # 4) fallback variants + paragraph=True only if still nothing
    stages = [
        ("top25", False, "fast"),
        ("top40", False, "fast"),
        ("full", False, "fast"),
        ("full", True, "fallback"),
        ("bottom", False, "fallback"),
    ]

    for region, paragraph, speed in stages:
        region_img = crop_region(img_color, region)
        variants = _variants_fast(region_img) if speed == "fast" else _variants_fallback(region_img)

        lines, cands, s = _ocr_try(
            img=region_img,
            region_label=region if region != "full" else "full",
            langs=langs,
            gpu=gpu,
            paragraph=paragraph,
            variants=variants,
            allowlist=allowlist if langs == ("en",) or "en" in langs else None,
        )

        if debug:
            print(f"\n--- Stage region={region} paragraph={paragraph} speed={speed} anchorScore={s} ---")
            print("\n".join(lines[:40]))

        if cands:
            all_candidates.extend(cands)
            evidence_lines.extend([c.evidence for c in cands])

            # Early stop if we already have a high-confidence best candidate
            best_now = dedupe_candidates(all_candidates)[0]
            if best_now.source == "doctor_field" and best_now.score >= 80:
                break
            if best_now.source == "dr_line" and best_now.score >= 60:
                break

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
# CLI / batch runner
# =========================================================
def _is_valid_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


if __name__ == "__main__":
    # If you have Chinese receipts, use:
    # langs = ("en", "ch_tra", "ch_sim")
    langs = ("en",)

    # Set True only if you have CUDA-capable GPU properly set up
    gpu = False

    input_dir = "doctor_receipts"

    for filename in os.listdir(input_dir):
        if not _is_valid_image_file(filename):
            continue

        p = os.path.join(input_dir, filename)
        print(f"Processing {filename}...")

        try:
            result = extract_doctor_name(p, debug=False, langs=langs, gpu=gpu)
        except Exception as e:
            result = {"doctor_name": None, "confidence": 0.0, "error": str(e)}

        print("Result:", result)
        print()
