from playwright.sync_api import sync_playwright
from pathlib import Path
import base64

URL = "https://iiep.amcm.gov.mo/cpdportal-public/person/?language=zh-hant"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CAPTCHA_DIR = BASE_DIR / "data" / "captcha"
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CAPTCHA_DIR / "captcha_macau.png"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")

    img_sel = 'img[alt="驗證碼"]'
    page.wait_for_selector(img_sel, timeout=15000)

    data_url = page.get_attribute(img_sel, "src")
    if not data_url:
        raise RuntimeError("Found the captcha <img> but src was empty.")

    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise RuntimeError(
            f"src is not a base64 PNG data URL. src starts with: {data_url[:40]}"
        )

    b64 = data_url[len(prefix) :]  # noqa
    png_bytes = base64.b64decode(b64)

    OUT_PATH.write_bytes(png_bytes)
    print(f"Saved {OUT_PATH}")

    browser.close()
