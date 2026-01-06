from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CAPTCHA_DIR = BASE_DIR / "data" / "captcha"
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://iir.ia.org.hk/#/search/individual", wait_until="networkidle")

    src = page.get_attribute("#stickyImg", "src")
    full_url = urljoin("https://iir.ia.org.hk/", src.lstrip("./"))

    img_bytes = page.request.get(full_url).body()

    out_path = CAPTCHA_DIR / "captcha.png"
    out_path.write_bytes(img_bytes)

    print("Saved captcha.png")

    browser.close()
