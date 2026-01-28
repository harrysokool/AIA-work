import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")  # no extra flags needed

print("predicting...")
res = ocr.ocr("preprocessed_out/sample2__ocr_rgb.jpg")

print("done")
for line in res[0]:
    print(line[1][0])
