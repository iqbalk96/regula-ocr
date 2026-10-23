
from pathlib import Path

import cv2
import pymupdf
from paddleocr import PaddleOCR

# =========================
# KONFIGURASI
# =========================
PDF_PATH = Path("data/dataset/2007_Peraturan_BPK_02.pdf")
OUTPUT_DIR = Path("output")
PAGE_NUMBER = 1  # Nomor halaman dimulai dari 1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# VALIDASI FILE
# =========================
if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF tidak ditemukan: {PDF_PATH.resolve()}"
    )

# =========================
# PDF -> GAMBAR
# =========================
doc = pymupdf.open(str(PDF_PATH))

if PAGE_NUMBER < 1 or PAGE_NUMBER > len(doc):
    doc.close()
    raise ValueError("Nomor halaman berada di luar rentang PDF.")

page = doc[PAGE_NUMBER - 1]

# Render pada skala 2x
pix = page.get_pixmap(
    matrix=pymupdf.Matrix(2, 2),
    alpha=False
)

image_path = OUTPUT_DIR / f"halaman_{PAGE_NUMBER:02d}.png"
pix.save(str(image_path))
doc.close()

print("Gambar tersimpan:", image_path.resolve())

# =========================
# BACA GAMBAR
# =========================
image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Gagal membaca gambar hasil render.")

print("Resolusi gambar:", image.shape[1], "x", image.shape[0])

# =========================
# INISIALISASI OCR
# =========================
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# =========================
# JALANKAN OCR
# =========================
results = ocr.predict(input=image)

texts = []

for result in results:
    data = result.json

    if isinstance(data, dict):
        data = data.get("res", data)

    recognized_texts = data.get("rec_texts", [])
    texts.extend(recognized_texts)

recognized_text = "\n".join(texts)

# =========================
# SIMPAN HASIL
# =========================
text_path = OUTPUT_DIR / f"hasil_ocr_{PAGE_NUMBER:02d}.txt"
text_path.write_text(recognized_text, encoding="utf-8")

print("\n========== HASIL OCR ==========\n")
print(recognized_text[:3000])
print("\nJumlah baris teks:", len(texts))
print("Hasil lengkap:", text_path.resolve())
