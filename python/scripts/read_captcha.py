import cv2
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
CAPTCHA_PNG = BASE_DIR / "data" / "captcha" / "captcha.png"
DECODED_CAPTCHA_PNG = BASE_DIR / "data" / "captcha" / "decoded_captcha.png"
